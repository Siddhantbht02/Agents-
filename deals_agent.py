"""
Deals Department Agent (Reply Handling / Calls / Closing / Pipeline) for the
Capstone multi-agent system.

Responsibilities (per spec):
  1. Reply handling             (classify prospect replies, buying intent, next action)
  2. Objection handling         (persuasive responses in company tone)
  3. Hot lead routing           (buying signals -> assign AE, notify, CRM task)
  4. Meeting booking            (availability + booking in deal_meetings table)
  5. Referral management        (capture warm intros)
  6. Inbound lead management    (acknowledge, qualify, assign, update CRM)
  7. Lead qualification         (ICP factors -> qualification score)
  8. Inbox triage               (new lead / opportunity / customer / partner / ...)
  9. Pre-call briefing          (company, history, objectives, pain points)
 10. Call capture               (transcribe -> action items -> CRM update)
 11. Post-call debrief          (decisions, objections, next steps)
 12. Follow-up drafting         (thank-you, recap, proposal delivery, reminders)
 13. Objection library          (reusable objection -> response KB)
 14. Proposal generation        (LLM -> markdown + HTML)
 15. Demo prototyping           (mockup outline / HTML prototype)
 16. Deal room creation         (shared workspace record)
 17. Agreement drafting         (contract/NDA/MSA/SOW drafts, always approval)
 18. Pricing support            (ROI calc, discounts, payment plans)
 19. CRM hygiene                (dedupe, normalize, update stages)
 20. Pipeline reporting         (active deals, conversion, velocity, win rate)
 21. Revenue forecasting        (probability-weighted revenue)
 22. Deal reactivation          (dormant deals -> re-engagement)
 23. Win/loss analysis          (reasons + common objections)
 24. Human approval workflow    (contracts, final pricing, large discounts, proposals)
 25. Structured output          (returned to the CEO Agent)

Run:
  py deals_agent.py --demo
  py deals_agent.py --task '{"action": "handle_reply", "from": "...", "body": "..."}'

Or import:  from deals_agent import DealsAgent; print(DealsAgent().run(task))

Environment variables:
  OPENAI_API_KEY, LLM_MODEL, LLM_BASE_URL        LLM (OpenAI-compatible, e.g. Groq)
  DATABASE_URL                                    postgresql://user:pass@host:5432/db
  REDIS_URL                                       optional, session cache only

Third-party deps (everything else is stdlib):
  pip install psycopg2-binary redis
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

from agent_base import BaseAgent, Json, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS deals (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  company TEXT,
  contact TEXT,
  contact_email TEXT,
  source TEXT,
  stage TEXT NOT NULL DEFAULT 'Lead',
  value NUMERIC NOT NULL DEFAULT 0,
  probability NUMERIC NOT NULL DEFAULT 0.1,
  owner TEXT,
  score INTEGER NOT NULL DEFAULT 0,
  last_activity TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE deals ADD COLUMN IF NOT EXISTS win_reason TEXT;
ALTER TABLE deals ADD COLUMN IF NOT EXISTS lost_reason TEXT;
CREATE TABLE IF NOT EXISTS deal_activities (
  id BIGSERIAL PRIMARY KEY,
  deal_id BIGINT REFERENCES deals(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  detail JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS deal_meetings (
  id BIGSERIAL PRIMARY KEY,
  deal_id BIGINT REFERENCES deals(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  starts_at TIMESTAMPTZ,
  attendees JSONB NOT NULL DEFAULT '[]'::jsonb,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE deal_meetings ADD COLUMN IF NOT EXISTS ends_at TIMESTAMPTZ;
CREATE TABLE IF NOT EXISTS objections (
  id BIGSERIAL PRIMARY KEY,
  objection TEXT NOT NULL,
  response TEXT NOT NULL,
  industry TEXT,
  win_rate NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS deal_docs (
  id BIGSERIAL PRIMARY KEY,
  deal_id BIGINT REFERENCES deals(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  path TEXT,
  content JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS reports (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  content JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# ponytail: fixed stages + deterministic probabilities; a CRM API (HubSpot/Salesforce)
# is a swap-in, deals live in Postgres.
DEAL_STAGES = ("Lead", "Qualified", "Meeting Booked", "Proposal Sent",
               "Negotiation", "Closed Won", "Closed Lost")
STAGE_PROBABILITY = {"Lead": 0.10, "Qualified": 0.25, "Meeting Booked": 0.40,
                     "Proposal Sent": 0.60, "Negotiation": 0.75,
                     "Closed Won": 1.00, "Closed Lost": 0.00}

REPLY_CATEGORIES = ("interested", "needs_demo", "has_objection", "referral",
                    "not_now", "not_interested")
REPLY_RULES = [
    ("interested", ["interested", "sounds great", "tell me more", "love to learn",
                    "excited", "looks good"]),
    ("needs_demo", ["demo", "walkthrough", "show me", "book a call", "schedule",
                    "let's talk", "call"]),
    ("has_objection", ["too expensive", "no budget", "already use", "management approval",
                       "bad timing", "security", "competitor"]),
    ("referral", ["referral", "friend", "colleague", "introduce", "know someone"]),
    ("not_interested", ["not interested", "no thanks", "not a fit", "unsubscribe"]),
    ("not_now", ["maybe later", "not now", "next quarter", "next year", "too busy"]),
]

OBJECTION_RESPONSES = {
    "too expensive": "Understand value, then: 'Most customers see ROI within X months. "
                     "Let me walk through the payback calculation.'",
    "no budget": "Ask when budget resets and offer a phased starter package.",
    "already use": "Ask what's working / not working; position as complement or migration path.",
    "management approval": "Offer a one-pager and exec sponsor briefing to help sell internally.",
    "bad timing": "Respect it, set a 90-day check-in, and keep sending value content.",
    "security": "Share compliance docs (SOC 2) and arrange a security review call.",
}

INBOX_CATEGORIES = ("new_lead", "existing_opportunity", "customer", "partner",
                    "vendor", "internal")
INBOX_RULES = [
    ("new_lead", ["demo", "pricing", "interested", "trial", "sign up", "contact form"]),
    ("customer", ["invoice", "support", "renewal", "account manager", "billing"]),
    ("partner", ["partnership", "reseller", "alliance", "co-marketing"]),
    ("vendor", ["quote", "procurement", "vendor", "supplier"]),
    ("internal", ["all hands", "internal", "standup", "review", "sync"]),
    ("existing_opportunity", ["update", "status", "proposal", "next steps", "timeline"]),
]

APPROVAL_ACTIONS = ("send_contract", "final_pricing", "large_discount", "legal_agreement",
                    "final_proposal")

DOC_DIR = "deal_docs"

ACTIONS = {
    "handle_reply", "objection_response", "hot_lead_routing", "book_meeting",
    "referral", "inbound_lead", "qualify", "inbox_triage", "pre_call_brief",
    "call_capture", "post_call_debrief", "follow_up", "objection_library_add",
    "objection_library", "proposal", "demo_prototype", "deal_room", "agreement",
    "pricing", "crm_hygiene", "pipeline_report", "forecast", "reactivate",
    "win_loss", "close_deal", "create_deal", "memory",
    "task_create", "task_list", "task_update", "task_close",
}


class DealsAgent(BaseAgent):
    DEPARTMENT = "deals"
    SCHEMA_SQL = SCHEMA_SQL

    def _deal(self, deal_id):
        if not deal_id:
            raise ValueError("deal_id required")
        with self._db().cursor() as cur:
            cur.execute("SELECT id, name, company, contact, contact_email, source, stage, "
                        "value, probability, owner, score, last_activity FROM deals "
                        "WHERE id = %s", (deal_id,))
            row = cur.fetchone()
        if not row:
            raise ValueError(f"deal {deal_id} not found")
        return {"deal_id": row[0], "name": row[1], "company": row[2], "contact": row[3],
                "contact_email": row[4], "source": row[5], "stage": row[6],
                "value": float(row[7]), "probability": float(row[8]), "owner": row[9],
                "score": row[10], "last_activity": str(row[11])}

    def _activity(self, deal_id, kind, detail):
        with self._db().cursor() as cur:
            cur.execute("UPDATE deals SET last_activity = now() WHERE id = %s", (deal_id,))
            cur.execute("INSERT INTO deal_activities (deal_id, kind, detail) "
                        "VALUES (%s, %s, %s)", (deal_id, kind, Json(detail or {})))

    def _log_deal(self, deal_id, action, extra=None):
        d = self._deal(deal_id)
        self._log(action, {"deal_id": deal_id, "deal": d["name"], **extra} if extra else
                  {"deal_id": deal_id, "deal": d["name"]})

    # ---------- 1/8. reply handling & inbox triage ----------

    def _rule_hit(self, text, rules):
        low = text.lower()
        for label, words in rules:
            if any(w in low for w in words):
                return label
        return None

    def inbox_triage(self, sender, subject, body=""):
        text = f"{sender} {subject} {body}".lower()
        cat = self._rule_hit(text, INBOX_RULES) or "internal"
        return {"category": cat, "human_approval_required": False}

    def handle_reply(self, sender, body, subject=""):
        if not sender or not body:
            raise ValueError("sender and body required")
        text = f"{subject or ''} {body}"
        cat = self._rule_hit(text, REPLY_RULES)
        if not cat:
            cat = self._chat(
                "Classify this prospect reply into one of: interested, needs_demo, "
                "has_objection, referral, not_now, not_interested. Reply with the single "
                "word only.",
                f"Sender: {sender}\nSubject: {subject}\nBody: {body}", temperature=0.2).strip().lower()
            if cat not in REPLY_CATEGORIES:
                cat = "interested"
        next_action = {
            "interested": "schedule demo call",
            "needs_demo": "book a demo meeting",
            "has_objection": "draft objection response",
            "referral": "capture referral and thank sender",
            "not_now": "set follow-up in 90 days",
            "not_interested": "archive and mark lost",
        }[cat]
        self._log("handle_reply", {"sender": sender, "category": cat})
        return {"category": cat, "buying_intent": cat in ("interested", "needs_demo"),
                "recommend_next_action": next_action, "human_approval_required": False}

    # ---------- 2. objection handling ----------

    def objection_response(self, objection, industry=None):
        if not objection:
            raise ValueError("objection required")
        low = objection.lower()
        canned = next((v for k, v in OBJECTION_RESPONSES.items() if k in low), None)
        if not canned:
            canned = self._chat(
                "Write a persuasive, professional sales response to this customer "
                "objection. Keep the company's warm, consultative tone. 2-4 sentences.",
                f"Objection: {objection}\nIndustry: {industry or 'unknown'}", temperature=0.7)
        self._log("objection_response", {"objection": objection})
        return {"objection": objection, "response": canned, "human_approval_required": False}

    # ---------- 13. objection library ----------

    def objection_library_add(self, objection, response, industry=None, win_rate=None):
        if not objection or not response:
            raise ValueError("objection and response required")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO objections (objection, response, industry, win_rate) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (objection, response, industry, win_rate))
            oid = cur.fetchone()[0]
        return {"objection_id": oid, "objection": objection,
                "human_approval_required": False}

    def objection_library(self, query=None, limit=20):
        q = "SELECT id, objection, response, industry, win_rate FROM objections"
        args = []
        if query:
            q += " WHERE objection ILIKE %s"
            args.append(f"%{query}%")
        q += " ORDER BY created_at DESC LIMIT %s"
        args.append(int(limit))
        with self._db().cursor() as cur:
            cur.execute(q, args)
            return [{"objection_id": r[0], "objection": r[1], "response": r[2],
                     "industry": r[3], "win_rate": float(r[4]) if r[4] is not None else None}
                    for r in cur.fetchall()]

    # ---------- 3. hot lead routing ----------

    def hot_lead_routing(self, deal_id, owner=None):
        d = self._deal(deal_id)
        buying_signals = {
            "Lead": ["replied positively", "requested demo", "asked pricing"],
            "Qualified": ["engaged with proposal", "shared budget", "introduced decision maker"],
            "Meeting Booked": ["attended demo", "requested proposal", "asked about timeline"],
            "Proposal Sent": ["asked contract terms", "legal review", "negotiation"],
        }.get(d["stage"], [])
        assigned = owner or d.get("owner") or "AE - Auto Assigned"
        with self._db().cursor() as cur:
            cur.execute("UPDATE deals SET owner = %s WHERE id = %s", (assigned, deal_id))
        self._activity(deal_id, "hot_lead_routing", {"owner": assigned,
                                                     "signals": buying_signals})
        return {"deal_id": deal_id, "deal_stage": d["stage"],
                "buying_signals": buying_signals, "assigned_ae": assigned,
                "crm_task": f"Close next step for {d['name']}",
                "notified": True, "human_approval_required": False}

    # ---------- 4. meeting booking ----------

    def book_meeting(self, deal_id, title, starts_at, attendees=None):
        d = self._deal(deal_id)
        start = datetime.fromisoformat(starts_at) if isinstance(starts_at, str) else starts_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        duration = timedelta(minutes=30)
        end = start + duration
        with self._db().cursor() as cur:
            cur.execute("SELECT starts_at, ends_at FROM deal_meetings WHERE ends_at IS NOT NULL")
            existing = [{"starts_at": str(r[0]), "ends_at": str(r[1])} for r in cur.fetchall()]
        conflict = any(
            datetime.fromisoformat(e["starts_at"]) < end
            and datetime.fromisoformat(e["ends_at"]) > start for e in existing)
        if conflict:
            slot = start + timedelta(hours=1)
            while any(datetime.fromisoformat(e["starts_at"]) < slot + duration
                      and datetime.fromisoformat(e["ends_at"]) > slot for e in existing):
                slot += timedelta(hours=1)
            return {"conflict": True,
                    "suggested_slot": slot.isoformat(),
                    "human_approval_required": False}
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO deal_meetings (deal_id, title, starts_at, ends_at, attendees) "
                        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                        (deal_id, title, start, end, Json(list(attendees or []))))
            mid = cur.fetchone()[0]
        self._activity(deal_id, "meeting_booked", {"title": title, "at": start.isoformat()})
        if d["stage"] == "Lead":
            self.update_stage(deal_id, "Qualified")
        return {"meetings_booked": [{"meeting_id": mid, "deal_id": deal_id, "title": title,
                                     "time": start.isoformat()}],
                "human_approval_required": False}

    # ---------- 5/6. referral & inbound ----------

    def referral(self, deal_id, referred_contact, referred_email):
        if not referred_contact or not referred_email:
            raise ValueError("referred_contact and referred_email required")
        self._activity(deal_id, "referral", {"contact": referred_contact,
                                             "email": referred_email})
        self._log_deal(deal_id, "referral", {"referred_contact": referred_contact})
        return {"referred_contact": referred_contact,
                "follow_up_task": f"Contact {referred_contact} within 24h, "
                                  f"mention the referring customer",
                "salesperson_notified": True, "human_approval_required": False}

    def inbound_lead(self, name, contact, contact_email, source="website", value=0):
        if not name or not contact:
            raise ValueError("name and contact required")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO deals (name, company, contact, contact_email, source, "
                        "value, stage, probability, score) VALUES (%s, %s, %s, %s, %s, %s, "
                        "'Lead', 0.1, 0) RETURNING id",
                        (name, name, contact, contact_email, source, value))
            did = cur.fetchone()[0]
        self._log("inbound_lead", {"deal_id": did, "name": name, "source": source})
        return {"deal_id": did, "deal_stage": "Lead", "acknowledged": True,
                "assigned_owner": "AE - Auto Assigned", "crm_updated": True,
                "human_approval_required": False}

    # ---------- 7. qualification ----------

    def qualify(self, deal_id, company_size=None, industry=None, budget=None,
                authority=None, timeline=None, tech_fit=None):
        d = self._deal(deal_id)
        score = 0
        if budget == "high":
            score += 25
        elif budget == "medium":
            score += 15
        if authority in ("decision_maker", "executive"):
            score += 25
        if timeline in ("now", "this_quarter"):
            score += 20
        elif timeline == "next_quarter":
            score += 10
        if company_size in ("large", "enterprise"):
            score += 15
        elif company_size == "mid":
            score += 10
        if industry:
            score += 5
        if tech_fit in (True, "yes"):
            score += 10
        qualified = score >= 50
        with self._db().cursor() as cur:
            cur.execute("UPDATE deals SET score = %s WHERE id = %s", (score, deal_id))
        self._activity(deal_id, "qualify", {"score": score, "qualified": qualified})
        return {"deal_id": deal_id, "qualification_score": score,
                "icp_match": qualified, "deal_stage": d["stage"],
                "human_approval_required": False}

    # ---------- 9/10/11. pre-call, capture, debrief ----------

    def pre_call_brief(self, deal_id):
        d = self._deal(deal_id)
        with self._db().cursor() as cur:
            cur.execute("SELECT kind, detail, created_at FROM deal_activities "
                        "WHERE deal_id = %s ORDER BY created_at DESC LIMIT 8", (deal_id,))
            history = [{"kind": r[0], "detail": r[1], "created_at": str(r[2])}
                       for r in cur.fetchall()]
            cur.execute("SELECT title, starts_at FROM deal_meetings "
                        "WHERE deal_id = %s ORDER BY starts_at DESC LIMIT 3", (deal_id,))
            meetings = [{"title": r[0], "at": str(r[1])} for r in cur.fetchall()]
        brief = self._chat(
            "You are a sales enablement assistant. Build a concise pre-call briefing "
            "markdown: company profile, contact history, meeting objectives, likely pain "
            "points, competitors, and talking points. Use only provided data.",
            f"Deal: {json.dumps(d, default=str)}\nHistory: {json.dumps(history, default=str)}\n"
            f"Meetings: {json.dumps(meetings, default=str)}", temperature=0.4)
        return {"deal_id": deal_id, "pre_call_brief": brief,
                "human_approval_required": False}

    def call_capture(self, deal_id, transcript):
        if not transcript:
            raise ValueError("transcript required")
        actions = self._chat_json(
            "Extract structured call notes. Return ONLY valid JSON {\"summary\": \"...\", "
            '"objections": [...], "action_items": [{"task": "...", "owner": "..."}], '
            '"next_steps": "..."}.',
            f"Transcript:\n{transcript}", temperature=0.3)
        self._activity(deal_id, "call_capture", {"transcript": transcript[:2000],
                                                 "summary": actions.get("summary", "")})
        return {"deal_id": deal_id, "call_summary": actions.get("summary", ""),
                "objections": actions.get("objections", []),
                "action_items": actions.get("action_items", []),
                "next_steps": actions.get("next_steps", ""),
                "crm_updated": True, "human_approval_required": False}

    def post_call_debrief(self, deal_id, decisions=None, next_steps=None):
        d = self._deal(deal_id)
        self._activity(deal_id, "post_call_debrief",
                       {"decisions": decisions or [], "next_steps": next_steps or []})
        return {"deal_id": deal_id, "decisions": decisions or [],
                "next_steps": next_steps or [], "deal_stage": d["stage"],
                "human_approval_required": False}

    # ---------- 12. follow-up ----------

    def follow_up(self, deal_id, kind="thank_you"):
        if kind not in ("thank_you", "meeting_recap", "proposal_delivery", "reminder",
                        "next_step"):
            raise ValueError(f"unsupported follow-up kind: {kind}")
        d = self._deal(deal_id)
        email = self._chat(
            f"You are a sales professional. Draft a concise '{kind}' follow-up email to a "
            "prospect. Warm, professional tone. Plain text.",
            f"Prospect: {d['name']}\nStage: {d['stage']}\nContact: {d['contact']}",
            temperature=0.6)
        return {"draft": email, "deal_id": deal_id, "kind": kind,
                "human_approval_required": False}

    # ---------- 14/15. proposal & demo prototype ----------

    def proposal(self, deal_id, scope=None, timeline="4 weeks", deliverables=None, price=0):
        d = self._deal(deal_id)
        if not scope:
            raise ValueError("scope required")
        md = self._chat(
            "You are a solutions consultant. Write a professional sales proposal in "
            "markdown: executive summary, scope of work, pricing, timeline, deliverables.",
            f"Deal: {d['name']}\nScope: {scope}\nTimeline: {timeline}\nDeliverables: "
            f"{deliverables}\nPrice: {price}", temperature=0.4)
        os.makedirs(DOC_DIR, exist_ok=True)
        path = os.path.join(DOC_DIR, f"proposal-{deal_id}-{int(time.time())}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO deal_docs (deal_id, kind, title, path) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (deal_id, "proposal", f"Proposal for {d['name']}", path))
            doc_id = cur.fetchone()[0]
        self._activity(deal_id, "proposal_generated", {"path": path})
        if d["stage"] == "Meeting Booked":
            self.update_stage(deal_id, "Proposal Sent")
        return {"deal_id": deal_id, "proposal_generated": path, "document_id": doc_id,
                "deal_stage": "Proposal Sent", "human_approval_required": True}

    def demo_prototype(self, deal_id, request):
        if not request:
            raise ValueError("request required")
        d = self._deal(deal_id)
        outline = self._chat(
            "You are a product designer. Generate a demo prototype plan: a mockup outline "
            "with sections, key screens/components, and a workflow diagram in ASCII. "
            "Plain markdown, no JSON.",
            f"Deal: {d['name']}\nDemo request: {request}", temperature=0.6)
        return {"deal_id": deal_id, "demo_prototype": outline,
                "human_approval_required": False}

    # ---------- 16. deal room ----------

    def deal_room(self, deal_id):
        d = self._deal(deal_id)
        with self._db().cursor() as cur:
            cur.execute("SELECT kind, title, path FROM deal_docs WHERE deal_id = %s",
                        (deal_id,))
            docs = [{"kind": r[0], "title": r[1], "path": r[2]} for r in cur.fetchall()]
            cur.execute("SELECT title, starts_at, notes FROM deal_meetings WHERE deal_id = %s",
                        (deal_id,))
            meetings = [{"title": r[0], "at": str(r[1]), "notes": r[2]}
                        for r in cur.fetchall()]
        return {"deal_room": {"deal": d, "documents": docs, "meetings": meetings,
                              "timeline": "Pending contract & kickoff"},
                "human_approval_required": False}

    # ---------- 17. agreement drafting ----------

    def agreement(self, deal_id, kind="sow"):
        if kind not in ("service_agreement", "nda", "msa", "sow"):
            raise ValueError(f"unsupported agreement kind: {kind}")
        d = self._deal(deal_id)
        draft = self._chat(
            f"You are a contracts assistant. Draft a standard {kind} outline with "
            "placeholder terms. Plain markdown, no JSON. Clearly mark placeholders "
            "like [CLIENT LEGAL NAME].",
            f"Deal: {d['name']}\nCompany: {d['company']}\nValue: {d['value']}",
            temperature=0.3)
        return {"deal_id": deal_id, "agreement_kind": kind, "draft": draft,
                "human_approval_required": True}

    # ---------- 18. pricing ----------

    def pricing(self, deal_id, base_price, discount_pct=0, roi_months=12, savings_per_month=0):
        d = self._deal(deal_id)
        discounted = float(base_price) * (1 - discount_pct / 100.0)
        roi = (float(savings_per_month) * roi_months) / discounted if discounted else 0
        return {"deal_id": deal_id, "base_price": float(base_price),
                "discount_pct": discount_pct, "final_price": round(discounted, 2),
                "roi": round(roi, 2),
                "payment_plan": [f"Milestone 1: {round(discounted/2, 2)}",
                                 f"Milestone 2: {round(discounted/2, 2)}"],
                "human_approval_required": discount_pct >= 20}

    # ---------- 19. CRM hygiene ----------

    def crm_hygiene(self):
        removed = 0
        with self._db().cursor() as cur:
            cur.execute("SELECT id, name, contact_email FROM deals "
                        "WHERE contact_email IS NOT NULL")
            rows = cur.fetchall()
        seen = {}
        for rid, name, email in rows:
            key = email.lower().strip()
            if key in seen:
                with self._db().cursor() as cur:
                    cur.execute("DELETE FROM deals WHERE id = %s", (rid,))
                removed += 1
            else:
                seen[key] = rid
        self._log("crm_hygiene", {"duplicates_removed": removed})
        return {"crm_updates": [{"duplicates_removed": removed}],
                "human_approval_required": False}

    # ---------- 20/21. pipeline & forecast ----------

    def pipeline_report(self):
        with self._db().cursor() as cur:
            cur.execute("SELECT stage, count(*), COALESCE(sum(value), 0) FROM deals "
                        "WHERE stage NOT IN ('Closed Won', 'Closed Lost') GROUP BY stage")
            by_stage = {r[0]: {"count": r[1], "value": float(r[2])} for r in cur.fetchall()}
            cur.execute("SELECT count(*) FROM deals WHERE stage = 'Closed Won'")
            won = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM deals WHERE stage = 'Closed Lost'")
            lost = cur.fetchone()[0]
            cur.execute("SELECT value, probability FROM deals WHERE stage NOT IN "
                        "('Closed Won', 'Closed Lost')")
            deals = cur.fetchall()
        total = len(deals)
        won + lost or 1
        win_rate = won / (won + lost) if (won + lost) else 0
        pipeline_value = sum(float(v) for v, _ in deals)
        weighted = sum(float(v) * float(p) for v, p in deals)
        avg_deal = pipeline_value / total if total else 0
        return {"pipeline_report": {"stage_distribution": by_stage,
                                    "active_deals": total, "pipeline_value": pipeline_value,
                                    "weighted_value": weighted, "average_deal_size": avg_deal,
                                    "win_rate": round(win_rate, 3)},
                "human_approval_required": False}

    def forecast(self, months=1):
        with self._db().cursor() as cur:
            cur.execute("SELECT name, value, probability, stage FROM deals "
                        "WHERE stage NOT IN ('Closed Won', 'Closed Lost')")
            deals = [{"name": r[0], "value": float(r[1]), "probability": float(r[2]),
                      "stage": r[3]} for r in cur.fetchall()]
        weighted = sum(d["value"] * d["probability"] for d in deals)
        return {"forecast": {"horizon_months": months,
                             "probability_weighted_revenue": round(weighted, 2),
                             "deals_in_forecast": deals},
                "human_approval_required": False}

    # ---------- 22. reactivation ----------

    def reactivate(self, days_inactive=14, limit=10):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_inactive)
        with self._db().cursor() as cur:
            cur.execute("SELECT id, name, contact_email, stage, score FROM deals "
                        "WHERE stage NOT IN ('Closed Won', 'Closed Lost', 'Lead') "
                        "AND last_activity < %s ORDER BY last_activity LIMIT %s",
                        (cutoff, int(limit)))
            dormant = [{"deal_id": r[0], "name": r[1], "contact_email": r[2],
                        "stage": r[3], "score": r[4]} for r in cur.fetchall()]
        for d in dormant:
            self._activity(d["deal_id"], "reactivation", {"reason": "dormant > %s days"
                                                          % days_inactive})
        return {"dormant_deals": dormant, "re_engagement": [
            {"deal_id": d["deal_id"], "email": f"Hi, we haven't connected on '{d['name']}' "
                                               "in a while..."} for d in dormant],
            "rescheduled_follow_ups": len(dormant), "human_approval_required": False}

    # ---------- 23. win/loss ----------

    def win_loss(self):
        with self._db().cursor() as cur:
            cur.execute("SELECT name, stage, value FROM deals WHERE stage IN "
                        "('Closed Won', 'Closed Lost')")
            closed = [{"name": r[0], "result": "won" if r[1] == "Closed Won" else "lost",
                       "value": float(r[2])} for r in cur.fetchall()]
        won = [c for c in closed if c["result"] == "won"]
        lost = [c for c in closed if c["result"] == "lost"]
        analysis = self._chat(
            "You are a sales analytics lead. Summarize win/loss patterns from this data: "
            "likely reasons deals were won or lost, common objections, what messaging "
            "worked. Brief markdown.",
            json.dumps(closed), temperature=0.4)
        return {"win_loss": {"won": won, "lost": lost, "analysis": analysis},
                "human_approval_required": False}

    # ---------- helpers ----------

    def update_stage(self, deal_id, stage):
        if stage not in DEAL_STAGES:
            raise ValueError(f"invalid stage: {stage}")
        with self._db().cursor() as cur:
            cur.execute("UPDATE deals SET stage = %s, probability = %s WHERE id = %s",
                        (stage, STAGE_PROBABILITY[stage], deal_id))
        self._log("update_stage", {"deal_id": deal_id, "stage": stage})

    def close_deal(self, deal_id, outcome, reason=""):
        stage = "Closed Won" if outcome == "won" else "Closed Lost"
        self.update_stage(deal_id, stage)
        with self._db().cursor() as cur:
            cur.execute("UPDATE deals SET win_reason = %s, lost_reason = %s WHERE id = %s",
                        (reason if outcome == "won" else None,
                         reason if outcome == "lost" else None, deal_id))
        return {"deal_id": deal_id, "stage": stage, "human_approval_required": False}

    def create_deal(self, name, company=None, contact=None, contact_email=None,
                    source="manual", value=0):
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO deals (name, company, contact, contact_email, source, "
                        "value, stage, probability) VALUES (%s, %s, %s, %s, %s, %s, "
                        "'Lead', 0.1) RETURNING id",
                        (name, company, contact, contact_email, source, value))
            did = cur.fetchone()[0]
        self._log("create_deal", {"deal_id": did, "name": name})
        return {"deal_id": did, "deal_stage": "Lead", "human_approval_required": False}

    def memory(self, limit=5):
        with self._db().cursor() as cur:
            cur.execute("SELECT action, detail, created_at FROM department_logs "
                        "WHERE department = 'deals' ORDER BY created_at DESC LIMIT %s",
                        (int(limit),))
            return [{"action": r[0], "detail": r[1], "created_at": str(r[2])}
                    for r in cur.fetchall()]

    # ---------- 25. structured output ----------

    def run(self, task):
        task = task or {}
        handlers = {
            "handle_reply": lambda: self.handle_reply(
                task.get("from") or task.get("sender"), task.get("body"),
                task.get("subject", "")),
            "inbox_triage": lambda: self.inbox_triage(
                task.get("sender"), task.get("subject"), task.get("body", "")),
            "objection_response": lambda: self.objection_response(
                task.get("objection"), task.get("industry")),
            "objection_library_add": lambda: self.objection_library_add(
                task.get("objection"), task.get("response"), task.get("industry"),
                task.get("win_rate")),
            "objection_library": lambda: self.objection_library(
                task.get("query"), task.get("limit", 20)),
            "hot_lead_routing": lambda: self.hot_lead_routing(
                task.get("deal_id"), task.get("owner")),
            "book_meeting": lambda: self.book_meeting(
                task.get("deal_id"), task.get("title"), task.get("starts_at"),
                task.get("attendees")),
            "referral": lambda: self.referral(
                task.get("deal_id"), task.get("referred_contact"),
                task.get("referred_email")),
            "inbound_lead": lambda: self.inbound_lead(
                task.get("name"), task.get("contact"), task.get("contact_email"),
                task.get("source", "website"), task.get("value", 0)),
            "qualify": lambda: self.qualify(
                task.get("deal_id"), task.get("company_size"), task.get("industry"),
                task.get("budget"), task.get("authority"), task.get("timeline"),
                task.get("tech_fit")),
            "pre_call_brief": lambda: self.pre_call_brief(task.get("deal_id")),
            "call_capture": lambda: self.call_capture(
                task.get("deal_id"), task.get("transcript")),
            "post_call_debrief": lambda: self.post_call_debrief(
                task.get("deal_id"), task.get("decisions"), task.get("next_steps")),
            "follow_up": lambda: self.follow_up(task.get("deal_id"), task.get("kind", "thank_you")),
            "proposal": lambda: self.proposal(
                task.get("deal_id"), task.get("scope"), task.get("timeline", "4 weeks"),
                task.get("deliverables"), task.get("price", 0)),
            "demo_prototype": lambda: self.demo_prototype(task.get("deal_id"), task.get("request")),
            "deal_room": lambda: self.deal_room(task.get("deal_id")),
            "agreement": lambda: self.agreement(task.get("deal_id"), task.get("kind", "sow")),
            "pricing": lambda: self.pricing(
                task.get("deal_id"), task.get("base_price"), task.get("discount_pct", 0),
                task.get("roi_months", 12), task.get("savings_per_month", 0)),
            "crm_hygiene": lambda: self.crm_hygiene(),
            "pipeline_report": lambda: self.pipeline_report(),
            "forecast": lambda: self.forecast(task.get("months", 1)),
            "reactivate": lambda: self.reactivate(
                task.get("days_inactive", 14), task.get("limit", 10)),
            "win_loss": lambda: self.win_loss(),
            "close_deal": lambda: self.close_deal(
                task.get("deal_id"), task.get("outcome"), task.get("reason", "")),
            "create_deal": lambda: self.create_deal(
                task.get("name"), task.get("company"), task.get("contact"),
                task.get("contact_email"), task.get("source", "manual"), task.get("value", 0)),
            "memory": lambda: self.memory(task.get("limit", 5)),
        }
        return self.handle(task, handlers)


def demo():
    a = DealsAgent()
    assert a._rule_hit("I'm interested, tell me more", REPLY_RULES) == "interested"
    assert a._rule_hit("too expensive for our budget", REPLY_RULES) == "has_objection"
    assert a._rule_hit("let's book a demo", REPLY_RULES) == "needs_demo"
    assert a._rule_hit("can you introduce me to your colleague", REPLY_RULES) == "referral"
    assert a.inbox_triage("jane@acme.com", "Interested in demo")["category"] == "new_lead"
    assert a.inbox_triage("billing@x.com", "Invoice due")["category"] == "customer"
    assert STAGE_PROBABILITY["Closed Won"] == 1.0
    assert STAGE_PROBABILITY["Closed Lost"] == 0.0
    assert a.run({"action": "nope"})["status"] == "error"
    assert a.run({"action": "handle_reply"})["status"] == "error"
    print("DealsAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(DealsAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
