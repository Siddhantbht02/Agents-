"""
Sales Department Agent for the Capstone multi-agent system.

Responsibilities (per spec):
  1. Lead generation            (Tavily Search)
  2. Lead enrichment            (Firecrawl)
  3. CRM management             (PostgreSQL upsert, ON CONFLICT DO UPDATE)
  4. Outreach email drafting    (LLM; Gmail draft best-effort)
  5. Reply classification       (keyword rules + LLM fallback)
  6. Company discovery          (Tavily Search)
  7. Qualification              (rule-based: Hot/Warm/Cold/Reject)
  8. Sales intelligence         (Tavily news search)
  9. Department memory          (department_logs + optional Redis batch cache)
 10. Structured JSON results    (returned to the CEO Agent)

Run:
  py sales_agent.py --demo
  py sales_agent.py --task '{"action": "generate_leads", "criteria": "AI startups in the US", "count": 5}'
  py sales_agent.py --gmail-login        (authorize Gmail draft creation)

Or import:  from sales_agent import SalesAgent; print(SalesAgent().run(task))

Environment variables:
  TAVILY_API_KEY, FIRECRAWL_API_KEY          required for search/scrape
  OPENAI_API_KEY, LLM_MODEL, LLM_BASE_URL    LLM for email writing / reply fallback
  DATABASE_URL                                postgresql://user:pass@host:5432/db
  REDIS_URL                                   optional, batch cache only
  GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET        run `py sales_agent.py --gmail-login` to authorize
  GMAIL_ACCESS_TOKEN                          optional, pre-fetched token (skips login)
  GMAIL_REFRESH_TOKEN                         optional, auto-refreshes an expired access token
  DEFAULT_SALES_EMAIL, SENDER_NAME            from-line for drafts

Third-party deps (everything else is stdlib):
  pip install psycopg2-binary redis
"""

import base64
import http.server
import json
import os
import sys
import webbrowser
from datetime import datetime, timezone
from email.message import EmailMessage
from urllib.parse import urlparse, urlencode, parse_qs

from agent_base import BaseAgent, Json, _domain_of, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS companies (
  id BIGSERIAL PRIMARY KEY,
  domain TEXT UNIQUE NOT NULL,
  name TEXT,
  website TEXT,
  industry TEXT,
  employees TEXT,
  tech_stack JSONB NOT NULL DEFAULT '[]'::jsonb,
  funding_stage TEXT,
  description TEXT,
  location TEXT,
  intelligence JSONB NOT NULL DEFAULT '{}'::jsonb,
  last_contacted TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS leads (
  id BIGSERIAL PRIMARY KEY,
  company_id BIGINT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
  contact_name TEXT,
  contact_email TEXT,
  title TEXT,
  linkedin TEXT,
  status TEXT NOT NULL DEFAULT 'new',
  qualification TEXT,
  outreach_history JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_leads_company_email
  ON leads(company_id, contact_email) WHERE contact_email IS NOT NULL;
"""

REPLY_LABELS = ("interested", "not_interested", "wants_demo", "needs_follow_up",
                "out_of_office", "spam", "already_customer")

# ponytail: keyword rules first (free, deterministic), LLM only for ambiguous replies.
RULES = [
    ("out_of_office", ["out of office", "o.o.o.", "on leave", "on vacation",
                       "currently out", "away until", "annual leave", "vacation"]),
    ("already_customer", ["already use", "already a customer", "existing customer",
                          "current customer", "already have an account",
                          "i use your", "we use your", "long-time customer"]),
    ("not_interested", ["not interested", "no thanks", "not for us", "not relevant",
                        "not a fit", "no need", "doesn't fit", "not now",
                        "not right now", "no thank you"]),
    ("spam", ["unsubscribe", "spam", "remove me", "do not email",
              "stop emailing", "please don't contact"]),
    ("wants_demo", ["demo", "walkthrough", "book a call", "schedule",
                    "let's talk", "set up a call", "show me", "screen share"]),
    ("interested", ["interested", "sounds great", "tell me more", "more info",
                    "send me", "looks great", "looks good", "excited",
                    "happy to chat", "great timing", "love to learn"]),
    ("needs_follow_up", ["maybe", "later", "not sure", "busy", "soon",
                         "next week", "a bit", "somewhat"]),
]

FIREWALL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "industry": {"type": "string"},
        "employees": {"type": "string"},
        "funding_stage": {"type": "string"},
        "description": {"type": "string"},
        "location": {"type": "string"},
        "tech_stack": {"type": "array", "items": {"type": "string"}},
        "contact_email": {"type": "string"},
        "linkedin": {"type": "string"},
    },
}


class _AuthHandler(http.server.BaseHTTPRequestHandler):
    code = None

    def do_GET(self):
        _AuthHandler.code = parse_qs(urlparse(self.path).query).get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h3>Authorization received. Close this tab.</h3></body></html>")

    def log_message(self, *args):
        pass


class SalesAgent(BaseAgent):
    DEPARTMENT = "sales"
    SCHEMA_SQL = SCHEMA_SQL

    # ---------- 1/6. lead generation & company discovery ----------

    def discover_companies(self, criteria, n=10):
        return self.tavily_search(str(criteria), max(1, min(int(n or 10), 100)))

    def generate_leads(self, criteria, count=50, batch_id=None):
        cached = self.active_batch(batch_id)
        if cached:
            return {"leads": cached, "emails_drafted": [], "crm_updates": [],
                    "human_approval_required": False, "cached": True}
        if not criteria:
            raise ValueError("criteria required")
        count = max(1, min(int(count or 50), 200))

        # 9. department memory: never contact a company twice, no duplicate leads
        contacted = self._contacted_domains()
        leads, crm_updates, seen = [], [], set()
        for c in self.discover_companies(criteria, n=count * 3):
            if len(leads) >= count:
                break
            domain = c.get("domain")
            if not domain or domain in contacted or domain in seen:
                continue
            seen.add(domain)
            try:
                data = self.firecrawl_scrape(c["url"])
            except Exception as e:
                self._log("enrich_failed", {"url": c["url"], "error": str(e)})
                continue
            data["domain"] = domain
            company_id, inserted = self.upsert_company(data)
            lead_id, _ = self.upsert_lead(company_id, {
                "contact_email": data.get("contact_email"),
                "contact_name": None, "title": None, "linkedin": data.get("linkedin"),
                "status": "new", "qualification": self.qualify(data)})
            crm_updates.append({"domain": domain, "company_id": company_id,
                                "lead_id": lead_id, "inserted": inserted})
            leads.append({"id": lead_id, "company_id": company_id, "domain": domain,
                          "company_name": data.get("name") or domain,
                          "website": data.get("url"), "description": data.get("description"),
                          "contact_email": data.get("contact_email"),
                          "qualification": self.qualify(data)})
        self._log("generate_leads", {"criteria": criteria, "count": len(leads)})
        self.mark_batch(batch_id, leads)
        return {"leads": leads, "emails_drafted": [], "crm_updates": crm_updates,
                "human_approval_required": False}

    # ---------- 2. lead enrichment ----------

    def firecrawl_scrape(self, url):
        key = os.getenv("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("FIRECRAWL_API_KEY not set")
        data = self._http_json(
            "POST", "https://api.firecrawl.dev/v1/scrape",
            {"url": url, "formats": ["markdown", "extract"], "onlyMainContent": True,
             "extract": {"schema": FIREWALL_SCHEMA,
                         "prompt": "Extract company facts from the page."}},
            {"Authorization": f"Bearer {key}"}, timeout=90)
        if not (data or {}).get("success"):
            raise RuntimeError(f"firecrawl failed: {data}")
        meta = (data.get("data") or {}).get("metadata") or {}
        ex = (data.get("data") or {}).get("llm_extraction") or {}
        emp = ex.get("employees")
        return {"url": meta.get("ogUrl") or url,
                "name": ex.get("name") or meta.get("ogSiteName"),
                "description": (ex.get("description") or meta.get("description") or "")[:1000],
                "industry": ex.get("industry"),
                "employees": str(emp) if emp else None,
                "funding_stage": ex.get("funding_stage"),
                "location": ex.get("location"),
                "tech_stack": ex.get("tech_stack") or [],
                "contact_email": ex.get("contact_email"),
                "linkedin": ex.get("linkedin")}

    def enrich_company(self, url):
        if not url:
            raise ValueError("url required")
        data = self.firecrawl_scrape(url)
        data["domain"] = _domain_of(url) or _domain_of(data.get("url"))
        self._log("enrich", {"domain": data["domain"], "url": url})
        return data

    # ---------- 3. CRM management (upsert) ----------

    def upsert_company(self, data):
        domain = data.get("domain") or _domain_of(data.get("url"))
        if not domain:
            raise ValueError("company domain/url required")
        with self._db().cursor() as cur:
            cur.execute("""
                INSERT INTO companies
                  (domain, name, website, industry, employees, tech_stack,
                   funding_stage, description, location, intelligence)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (domain) DO UPDATE SET
                  name = COALESCE(EXCLUDED.name, companies.name),
                  website = COALESCE(EXCLUDED.website, companies.website),
                  industry = COALESCE(EXCLUDED.industry, companies.industry),
                  employees = COALESCE(EXCLUDED.employees, companies.employees),
                  tech_stack = COALESCE(EXCLUDED.tech_stack, companies.tech_stack),
                  funding_stage = COALESCE(EXCLUDED.funding_stage, companies.funding_stage),
                  description = COALESCE(EXCLUDED.description, companies.description),
                  location = COALESCE(EXCLUDED.location, companies.location),
                  intelligence = COALESCE(EXCLUDED.intelligence, companies.intelligence),
                  updated_at = now()
                RETURNING id, (xmax = 0) AS inserted
            """, (domain, data.get("name"), data.get("website") or data.get("url"),
                  data.get("industry"), data.get("employees"),
                  Json(data.get("tech_stack") or []), data.get("funding_stage"),
                  data.get("description"), data.get("location"),
                  Json(data.get("intelligence") or {})))
            row = cur.fetchone()
            return row[0], bool(row[1])

    def upsert_lead(self, company_id, data):
        with self._db().cursor() as cur:
            cur.execute("""
                INSERT INTO leads
                  (company_id, contact_name, contact_email, title, linkedin, status, qualification)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (company_id, contact_email) WHERE contact_email IS NOT NULL
                DO UPDATE SET
                  contact_name = COALESCE(EXCLUDED.contact_name, leads.contact_name),
                  title = COALESCE(EXCLUDED.title, leads.title),
                  linkedin = COALESCE(EXCLUDED.linkedin, leads.linkedin),
                  status = COALESCE(EXCLUDED.status, leads.status),
                  qualification = COALESCE(EXCLUDED.qualification, leads.qualification)
                RETURNING id, (xmax = 0) AS inserted
            """, (company_id, data.get("contact_name"), data.get("contact_email"),
                  data.get("title"), data.get("linkedin"),
                  data.get("status", "new"), data.get("qualification")))
            row = cur.fetchone()
            return row[0], bool(row[1])

    def update_crm(self, company, lead):
        cid, inserted = self.upsert_company(company)
        lead_id, _ = self.upsert_lead(cid, lead)
        self._log("update_crm", {"company_id": cid, "lead_id": lead_id})
        return {"crm_updates": [{"company_id": cid, "lead_id": lead_id,
                                 "inserted": inserted}]}

    def _contacted_domains(self):
        try:
            with self._db().cursor() as cur:
                cur.execute("SELECT domain FROM companies WHERE last_contacted IS NOT NULL")
                return {r[0] for r in cur.fetchall()}
        except Exception:
            return set()

    # ---------- 7. qualification ----------

    def qualify(self, lead):
        if not isinstance(lead, dict) or not lead:
            return "Reject"
        score = 0.0
        fs = str(lead.get("funding_stage") or "").lower()
        if "series" in fs:
            score += 2
        elif "seed" in fs:
            score += 1
        emp = str(lead.get("employees") or "").lower()
        for band, pts in (("1-10", .5), ("11-50", 1), ("51-200", 1.5), ("201-500", 1.5),
                          ("501-1000", 1), ("1001-5000", 1), ("5001", .5)):
            if band in emp:
                score += pts
                break
        if lead.get("description"):
            score += .5
        if lead.get("tech_stack"):
            score += .5
        if lead.get("intelligence"):
            score += 1
        if score >= 3:
            return "Hot Lead"
        if score >= 2:
            return "Warm Lead"
        if score >= 1:
            return "Cold Lead"
        return "Reject"

    # ---------- 8. sales intelligence ----------

    def sales_intelligence(self, domain):
        if not domain:
            raise ValueError("domain required")
        items = [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                 for r in self.tavily_search(f"{domain} funding news hiring launch", 5)]
        intel = {"source": "tavily", "items": items}
        try:
            with self._db().cursor() as cur:
                cur.execute("UPDATE companies SET intelligence = %s WHERE domain = %s",
                            (Json(intel), domain))
        except Exception:
            pass
        return {"domain": domain, "intelligence": intel}

    # ---------- 5. reply classification ----------

    def classify_reply(self, text):
        text = (text or "").strip()
        if not text:
            return "spam"
        low = text.lower()
        for label, words in RULES:
            if any(w in low for w in words):
                return label
        return self._llm_classify(text) or "needs_follow_up"

    def _llm_classify(self, text):
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")):
            return None
        labels = ", ".join(REPLY_LABELS)
        out = self._chat(
            f"Classify this sales email reply. Reply with exactly one label from: {labels}. "
            "Nothing else.", text)
        out = (out or "").strip().lower()
        for lab in REPLY_LABELS:
            if lab in out:
                return lab
        return None

    # ---------- 4. outreach email drafting (draft only, human approval) ----------

    def draft_email(self, lead, intent="cold_email", context=None):
        to = lead.get("contact_email")
        name = lead.get("contact_name") or "there"
        company = lead.get("company_name") or lead.get("name") or lead.get("domain") or "your company"
        system = (
            "You are a senior B2B sales copywriter. Write a short, personalized sales email. "
            "Return ONLY valid JSON: {\"subject\": \"...\", \"body\": \"...\"}. "
            "Body: plain text, 4-6 short sentences, no placeholders, mention the "
            "recipient's company naturally, end with one clear call to action.")
        user = f"Intent: {intent}\nRecipient: {name}\nCompany: {company}\nContext: {context or ''}"
        j = self._chat_json(system, user)
        subject = j.get("subject") or f"Quick question for {company}"
        body = j.get("body") or ""
        drafted = self._gmail_draft(to, subject, body)
        if lead.get("company_id"):
            try:
                with self._db().cursor() as cur:
                    cur.execute(
                        "UPDATE leads SET outreach_history = outreach_history || %s::jsonb "
                        "WHERE id = %s",
                        (json.dumps([{"type": "draft", "intent": intent, "subject": subject,
                                      "at": datetime.now(timezone.utc).isoformat()}]),
                         lead["company_id"]))
            except Exception:
                pass
        return {"to": to, "subject": subject, "body": body, "intent": intent,
                "drafted_in_gmail": bool(drafted),
                "human_approval_required": True}

    def _gmail_draft(self, to, subject, body):
        # ponytail: creates a draft only, never sends. Token refreshed when possible.
        token = self._gmail_token()
        if not token:
            return None
        msg = EmailMessage()
        msg["To"] = to
        msg["From"] = os.getenv("DEFAULT_SALES_EMAIL", "sales@example.com")
        msg["Subject"] = subject
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        self._http_json("POST", "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                        {"message": {"raw": raw}},
                        {"Authorization": f"Bearer {token}"})
        return True

    def _gmail_token(self):
        token = os.getenv("GMAIL_ACCESS_TOKEN")
        if token:
            return token
        refresh = os.getenv("GMAIL_REFRESH_TOKEN")
        cid, secret = os.getenv("GMAIL_CLIENT_ID"), os.getenv("GMAIL_CLIENT_SECRET")
        if refresh and cid and secret:
            data = self._http_json(
                "POST", "https://oauth2.googleapis.com/token",
                {"grant_type": "refresh_token", "client_id": cid,
                 "client_secret": secret, "refresh_token": refresh},
                form=True)
            return (data or {}).get("access_token")
        return None

    def gmail_login(self):
        """Loopback OAuth flow: opens the browser, catches the redirect, prints tokens."""
        cid, secret = os.getenv("GMAIL_CLIENT_ID"), os.getenv("GMAIL_CLIENT_SECRET")
        if not cid or not secret:
            raise RuntimeError("set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET")
        port = 8888
        while True:
            try:
                server = http.server.HTTPServer(("127.0.0.1", port), _AuthHandler)
                break
            except OSError:
                port += 1
        redirect_uri = f"http://127.0.0.1:{port}"
        auth_url = ("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
            "client_id": cid, "redirect_uri": redirect_uri,
            "response_type": "code", "access_type": "offline", "prompt": "consent",
            "scope": "https://www.googleapis.com/auth/gmail.compose"}))
        print(f"Register this redirect URI in Google Cloud Console, then open:\n{redirect_uri}")
        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()
        if not _AuthHandler.code:
            raise RuntimeError("no authorization code received")
        data = self._http_json(
            "POST", "https://oauth2.googleapis.com/token",
            {"code": _AuthHandler.code, "client_id": cid, "client_secret": secret,
             "redirect_uri": redirect_uri, "grant_type": "authorization_code"},
            form=True)
        return {"access_token": (data or {}).get("access_token"),
                "refresh_token": (data or {}).get("refresh_token")}

    # ---------- 10. structured results for the CEO Agent ----------

    def run(self, task):
        return self.handle(task, {
            "generate_leads": lambda: self.generate_leads(
                task.get("criteria"), task.get("count", 50), task.get("batch_id")),
            "discover_companies": lambda: self.discover_companies(
                task.get("criteria"), task.get("count", 10)),
            "enrich_company": lambda: self.enrich_company(task.get("url")),
            "update_crm": lambda: self.update_crm(
                task.get("company") or {}, task.get("lead") or {}),
            "draft_email": lambda: self.draft_email(
                task.get("lead") or {}, task.get("intent", "cold_email"),
                task.get("context")),
            "classify_reply": lambda: self.classify_reply(
                task.get("email") or task.get("reply") or ""),
            "sales_intelligence": lambda: self.sales_intelligence(task.get("domain")),
            "qualify": lambda: self.qualify(task.get("lead") or {}),
        })


def demo():
    a = SalesAgent()
    assert a.qualify({"funding_stage": "Series A", "employees": "51-200",
                      "description": "SaaS", "tech_stack": ["aws"]}) == "Hot Lead"
    assert a.qualify({"funding_stage": "Seed", "employees": "11-50"}) == "Warm Lead"
    assert a.qualify({"funding_stage": "Bootstrapped", "employees": "1-10",
                      "tech_stack": ["python"]}) == "Cold Lead"
    assert a.qualify({}) == "Reject"
    assert a.classify_reply("Out of office until Thursday") == "out_of_office"
    assert a.classify_reply("Please unsubscribe me") == "spam"
    assert a.classify_reply("We already use your product") == "already_customer"
    assert a.classify_reply("Not interested, thanks") == "not_interested"
    assert a.classify_reply("I'd love a demo") == "wants_demo"
    assert a.classify_reply("Interested! Send me more info") == "interested"
    assert a.classify_reply("Maybe later") == "needs_follow_up"
    assert _domain_of("https://www.Acme.com/team") == "acme.com"
    assert _domain_of("acme.io") == "acme.io"
    assert _domain_of("") is None
    print("SalesAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--gmail-login" in sys.argv:
        t = SalesAgent().gmail_login()
        print("\nSet these environment variables:")
        print(f'$env:GMAIL_ACCESS_TOKEN="{t.get("access_token")}"')
        if t.get("refresh_token"):
            print(f'$env:GMAIL_REFRESH_TOKEN="{t.get("refresh_token")}"')
        print("$env:GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET stay set for auto-refresh.")
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(SalesAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
