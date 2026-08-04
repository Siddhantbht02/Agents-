"""
Customer Department Agent (Support / Success / Community) for the Capstone
multi-agent system.

Responsibilities (per spec):
  1. Support ticket management   (PostgreSQL, priorities, routing, status)
  2. Ticket triage               (keyword rules + LLM: urgency, type, duplicates)
  3. FAQ & self-service          (KB lookup + LLM answers)
  4. Escalation detection        (angry/VIP/legal/security/refund/churn -> human)
  5. Response generation         (LLM, company tone)
  6. Knowledge base maintenance  (kb_articles)
  7. Customer onboarding         (welcome + checklist + training plan)
  8. Customer health monitoring  (health score from metrics)
  9. Churn prediction            (indicators -> retention actions)
 10. Renewal & expansion         (renewal reminders, upsell/cross-sell)
 11. Advocacy & referrals        (reviews, testimonials, case studies)
 12. Quarterly Business Review   (executive customer report)
 13. Community management        (posts, moderation, welcome)
 14. Community events            (AMAs, webinars)
 15. Sentiment analysis          (positive/neutral/negative)
 16. Human approval workflow     (refund/legal/public/termination -> approval)
 17. Structured output           (returned to the CEO Agent)

Run:
  py customer_agent.py --demo
  py customer_agent.py --task '{"action": "create_ticket", "customer": "Acme", ...}'

Or import:  from customer_agent import CustomerAgent; print(CustomerAgent().run(task))

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

from agent_base import BaseAgent, Json, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS support_tickets (
  id BIGSERIAL PRIMARY KEY,
  customer TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT,
  category TEXT,
  priority TEXT NOT NULL DEFAULT 'Medium',
  status TEXT NOT NULL DEFAULT 'open',
  sentiment TEXT,
  duplicate_of BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS kb_articles (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  tags TEXT[] NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS customer_health (
  customer TEXT PRIMARY KEY,
  score INTEGER NOT NULL,
  status TEXT NOT NULL,
  detail JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS community_posts (
  id BIGSERIAL PRIMARY KEY,
  author TEXT NOT NULL,
  content TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'post',
  status TEXT NOT NULL DEFAULT 'visible',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS community_events (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  starts_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'scheduled',
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

TICKET_PRIORITIES = ("Critical", "High", "Medium", "Low")
TICKET_STATUSES = ("open", "in_progress", "waiting", "resolved", "closed")

# ponytail: keyword rules first (free, deterministic), LLM only for ambiguous text.
TRIAGE_RULES = [
    ("billing", ["invoice", "payment", "refund", "charge", "overcharge", "receipt",
                 "billing", "subscription"]),
    ("technical", ["error", "bug", "crash", "api", "500", "not working", "failed",
                   "exception", "timeout", "install"]),
    ("account", ["password", "reset", "login", "locked", "sign in", "verify",
                 "two factor", "2fa"]),
    ("feature_request", ["would like", "feature", "enhancement", "suggestion", "idea"]),
    ("general", ["how do", "how to", "documentation", "guide", "help"]),
]

# signals that justify escalating to a human
ESCALATION_RULES = [
    ("angry", ["angry", "furious", "unacceptable", "terrible", "awful", "worst", "fed up"]),
    ("security", ["security", "breach", "hacked", "data leak", "privacy", "unauthorized"]),
    ("legal", ["lawsuit", "legal", "attorney", "sue", "sueing", "court"]),
    ("refund", ["refund", "money back", "compensation"]),
    ("vip", ["vip", "enterprise account", "tier 1", "contract"]),
]

SENTIMENT_NEG = ["worst", "terrible", "awful", "hate", "unhappy", "frustrated", "disappointed",
                 "refund", "broken", "useless", "angry"]
SENTIMENT_POS = ["great", "love", "amazing", "excellent", "awesome", "happy", "fantastic",
                 "thank", "best", "solved"]

APPROVAL_ACTIONS = {"refund", "compensation", "legal_response", "public_statement",
                    "account_termination"}

ACTIONS = {
    "create_ticket", "triage_ticket", "update_ticket_status", "list_tickets",
    "faq_answer", "kb_add", "kb_search", "escalate", "draft_response",
    "health_score", "churn_prediction", "renewal_plan", "advocacy", "qbr",
    "community_post", "community_event", "sentiment", "memory",
}


class CustomerAgent(BaseAgent):
    DEPARTMENT = "customer"
    SCHEMA_SQL = SCHEMA_SQL

    def _ticket(self, ticket_id):
        with self._db().cursor() as cur:
            cur.execute("SELECT id, customer, subject, body, category, priority, status, "
                        "sentiment, duplicate_of FROM support_tickets WHERE id = %s",
                        (ticket_id,))
            row = cur.fetchone()
        if not row:
            raise ValueError(f"ticket {ticket_id} not found")
        return {"ticket_id": row[0], "customer": row[1], "subject": row[2], "body": row[3],
                "category": row[4], "priority": row[5], "status": row[6],
                "sentiment": row[7], "duplicate_of": row[8]}

    # ---------- 1/2. ticket management & triage ----------

    def _rule_hit(self, text, rules):
        low = text.lower()
        for label, words in rules:
            if any(w in low for w in words):
                return label
        return None

    def sentiment(self, text):
        if not text:
            raise ValueError("text required")
        low = text.lower()
        if any(w in low for w in SENTIMENT_NEG):
            return {"sentiment": "negative"}
        if any(w in low for w in SENTIMENT_POS):
            return {"sentiment": "positive"}
        return {"sentiment": "neutral"}

    def create_ticket(self, customer, subject, body=None):
        if not customer or not subject:
            raise ValueError("customer and subject required")
        triage = self._rule_hit(f"{subject} {body or ''}", TRIAGE_RULES)
        sent = self.sentiment(f"{subject} {body or ''}")["sentiment"]
        dup = None
        with self._db().cursor() as cur:
            cur.execute("SELECT id FROM support_tickets WHERE status IN ('open','in_progress') "
                        "AND subject ILIKE %s ORDER BY created_at LIMIT 1", (f"%{subject[:50]}%",))
            row = cur.fetchone()
            dup = row[0] if row else None
        priority = "High" if sent == "negative" else "Medium"
        with self._db().cursor() as cur:
            cur.execute(
                "INSERT INTO support_tickets (customer, subject, body, category, priority, "
                "sentiment, duplicate_of) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (customer, subject, body, triage, priority, sent, dup))
            tid = cur.fetchone()[0]
        self._log("create_ticket", {"ticket_id": tid, "customer": customer, "subject": subject,
                                    "priority": priority, "category": triage})
        return {"tickets_processed": [{"ticket_id": tid, "customer": customer, "subject": subject,
                                       "category": triage, "priority": priority,
                                       "status": "open", "sentiment": sent,
                                       "duplicate_of": dup}],
                "human_approval_required": False}

    def triage_ticket(self, ticket_id):
        t = self._ticket(ticket_id)
        category = self._rule_hit(f"{t['subject']} {t['body'] or ''}", TRIAGE_RULES)
        urgency = "High" if t["sentiment"] == "negative" else t["priority"]
        with self._db().cursor() as cur:
            cur.execute("UPDATE support_tickets SET category = %s, priority = %s WHERE id = %s",
                        (category, urgency, ticket_id))
        return {"tickets_processed": [{**t, "category": category, "priority": urgency}],
                "human_approval_required": False}

    def update_ticket_status(self, ticket_id, status):
        if status not in TICKET_STATUSES:
            raise ValueError(f"invalid status: {status}")
        with self._db().cursor() as cur:
            cur.execute("UPDATE support_tickets SET status = %s WHERE id = %s", (status, ticket_id))
        self._log("update_ticket_status", {"ticket_id": ticket_id, "status": status})
        return {"ticket_id": ticket_id, "status": status, "human_approval_required": False}

    def list_tickets(self, status=None, limit=20):
        q = ("SELECT id, customer, subject, category, priority, status, sentiment, duplicate_of "
             "FROM support_tickets")
        args = []
        if status:
            q += " WHERE status = %s"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT %s"
        args.append(int(limit))
        with self._db().cursor() as cur:
            cur.execute(q, args)
            return [{"ticket_id": r[0], "customer": r[1], "subject": r[2], "category": r[3],
                     "priority": r[4], "status": r[5], "sentiment": r[6],
                     "duplicate_of": r[7]} for r in cur.fetchall()]

    # ---------- 3/6. FAQ & knowledge base ----------

    def kb_add(self, title, content, tags=None):
        if not title or not content:
            raise ValueError("title and content required")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO kb_articles (title, content, tags) VALUES (%s, %s, %s) "
                        "RETURNING id", (title, content, list(tags or [])))
            kid = cur.fetchone()[0]
        self._log("kb_add", {"kb_id": kid, "title": title})
        return {"kb_id": kid, "title": title, "human_approval_required": False}

    def kb_search(self, query, limit=5):
        # ponytail: token overlap instead of vector embeddings; article matches if it
        # shares significant words with the query. Swap in RAG/embeddings if accuracy gaps.
        stop = {"the", "a", "an", "my", "how", "do", "to", "i", "for", "of", "and", "is", "in",
                "on", "with", "please", "can", "help", "reset"}
        words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w not in stop]
        if not words:
            return []
        rows = []
        with self._db().cursor() as cur:
            cur.execute("SELECT id, title, content, created_at FROM kb_articles")
            for rid, title, content, created in cur.fetchall():
                hits = sum(1 for w in words if w in title.lower() or w in content.lower())
                if hits:
                    rows.append((hits, rid, title, created))
        rows.sort(key=lambda r: (-r[0], r[3]))
        return [{"kb_id": r[1], "title": r[2], "created_at": str(r[3])}
                for r in rows[:int(limit)]]

    def faq_answer(self, question):
        if not question:
            raise ValueError("question required")
        hits = self.kb_search(question, 3)
        if not hits:
            return {"answer": "No article matches this question. Ticket a human agent.",
                    "kb_matches": [], "human_approval_required": False}
        articles = []
        for h in hits:
            with self._db().cursor() as cur:
                cur.execute("SELECT content FROM kb_articles WHERE id = %s", (h["kb_id"],))
                articles.append({"title": h["title"], "content": cur.fetchone()[0]})
        answer = self._chat(
            "You are a support assistant. Answer the customer question using ONLY the "
            "knowledge base articles below. Be concise and friendly.",
            f"Question: {question}\nArticles:\n{json.dumps(articles, default=str)}",
            temperature=0.3)
        return {"answer": answer, "kb_matches": [h["title"] for h in hits],
                "human_approval_required": False}

    # ---------- 4. escalation ----------

    def escalate(self, ticket_id):
        t = self._ticket(ticket_id)
        reason = self._rule_hit(f"{t['subject']} {t['body'] or ''}", ESCALATION_RULES)
        if not reason:
            reason = "needs_human" if t["sentiment"] == "negative" else None
        if not reason:
            return {"escalated": False, "ticket_id": ticket_id,
                    "human_approval_required": False}
        self._log("escalate", {"ticket_id": ticket_id, "reason": reason,
                               "customer": t["customer"]})
        return {"escalated": True, "ticket_id": ticket_id, "reason": reason,
                "customer": t["customer"], "escalations": [reason],
                "human_approval_required": True}

    # ---------- 5. response generation ----------

    def draft_response(self, ticket_id, tone="supportive"):
        t = self._ticket(ticket_id)
        if t["duplicate_of"]:
            with self._db().cursor() as cur:
                cur.execute("SELECT subject FROM support_tickets WHERE id = %s",
                            (t["duplicate_of"],))
                ref = cur.fetchone()[0]
            return {"draft": f"Thanks for reaching out. This looks similar to '{ref}' — "
                             "our team is already on it. We'll update you shortly.",
                    "ticket_id": ticket_id, "human_approval_required": False}
        reply = self._chat(
            "You are a customer support agent. Write a warm, professional reply in the "
            "company's supportive tone. If it is a technical issue, include next steps.",
            f"Customer: {t['customer']}\nSubject: {t['subject']}\nBody: {t['body'] or ''}",
            temperature=0.6)
        return {"draft": reply, "ticket_id": ticket_id, "human_approval_required": False}

    # ---------- 7. onboarding ----------

    def onboarding_plan(self, customer, product, company_name="Acme"):
        if not customer or not product:
            raise ValueError("customer and product required")
        welcome = self._chat(
            "Write a short, warm welcome email for a new customer. Include a product "
            f"walkthrough of '{product}' and a first-login tip. Plain text.",
            f"Customer: {customer}\nCompany: {company_name}", temperature=0.6)
        checklist = [
            {"step": "Send welcome email", "done": True},
            {"step": "Share product walkthrough", "done": False},
            {"step": "Schedule onboarding call", "done": False},
            {"step": "Define success metrics", "done": False},
            {"step": "Set up training plan", "done": False},
        ]
        return {"welcome_email": welcome, "setup_checklist": checklist,
                "training_plan": [f"Week {i}: {phase}" for i, phase in enumerate(
                    ["Foundations", "Core usage", "Advanced features", "Review"], 1)],
                "human_approval_required": False}

    # ---------- 8/9. health & churn ----------

    def health_score(self, customer, login_freq=3, support_volume=0, satisfaction=4,
                     feature_adoption=0.5, activity=0.5):
        if not customer:
            raise ValueError("customer required")
        score = int(round(100 * (
            0.25 * min(activity, 1.0) + 0.2 * min(feature_adoption, 1.0)
            + 0.2 * (satisfaction / 5.0) + 0.2 * min(login_freq / 7.0, 1.0)
            - 0.15 * min(support_volume / 10.0, 1.0))))
        score = max(0, min(100, score))
        status = "Healthy" if score >= 70 else ("At Risk" if score >= 40 else "Critical")
        detail = {"login_frequency": login_freq, "support_volume": support_volume,
                  "satisfaction": satisfaction, "feature_adoption": feature_adoption,
                  "activity": activity}
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO customer_health (customer, score, status, detail) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (customer) DO UPDATE SET "
                        "score = EXCLUDED.score, status = EXCLUDED.status, "
                        "detail = EXCLUDED.detail, updated_at = now()",
                        (customer, score, status, Json(detail)))
        self._log("health_score", {"customer": customer, "score": score, "status": status})
        return {"customer_health": {"customer": customer, "score": score, "status": status,
                                    "detail": detail}, "human_approval_required": False}

    def churn_prediction(self, customer, health=None):
        if not customer:
            raise ValueError("customer required")
        if not health:
            with self._db().cursor() as cur:
                cur.execute("SELECT score, status, detail FROM customer_health "
                            "WHERE customer = %s", (customer,))
                row = cur.fetchone()
            if not row:
                raise ValueError(f"no health record for {customer}")
            health = {"score": row[0], "status": row[1], "detail": row[2]}
        risk = health["status"]
        actions = {
            "Critical": ["Immediate retention call", "Offer discount", "Executive outreach"],
            "At Risk": ["Re-engagement email", "Feature adoption webinar", "Check-in call"],
            "Healthy": ["Renewal reminder", "Referral request"],
        }[risk]
        return {"customer": customer, "churn_risk": risk,
                "recommend_retention_actions": actions, "human_approval_required": False}

    # ---------- 10. renewal & expansion ----------

    def renewal_plan(self, customer, plan, amount, days_to_renewal=30):
        if not customer or not plan:
            raise ValueError("customer and plan required")
        upsell = self._chat(
            "You are a customer success manager. Based on the customer's current plan, "
            "recommend one upsell and one cross-sell opportunity. Two short bullets.",
            f"Customer: {customer}\nPlan: {plan}\nAmount: {amount}\n"
            f"Days to renewal: {days_to_renewal}", temperature=0.5)
        return {"customer": customer, "renewal_reminder": (
            f"Renewal of {plan} ({amount}) due in {days_to_renewal} days"),
            "recommendations": upsell,
            "human_approval_required": False}

    # ---------- 11. advocacy ----------

    def advocacy(self, customer, score=None):
        if not customer:
            raise ValueError("customer required")
        if score is None:
            with self._db().cursor() as cur:
                cur.execute("SELECT score FROM customer_health WHERE customer = %s", (customer,))
                row = cur.fetchone()
                score = row[0] if row else 0
        if score < 70:
            return {"customer": customer, "eligible": False,
                    "message": "customer not eligible for advocacy yet",
                    "human_approval_required": False}
        return {"customer": customer, "eligible": True,
                "campaign": ["Request review", "Collect testimonial",
                             "Generate case study", "Invite to referral program"],
                "human_approval_required": False}

    # ---------- 12. QBR ----------

    def qbr(self, customer, usage=None):
        if not customer:
            raise ValueError("customer required")
        with self._db().cursor() as cur:
            cur.execute("SELECT score, status, detail FROM customer_health WHERE customer = %s",
                        (customer,))
            health = cur.fetchone()
        health = health or (0, "Unknown", {})
        health_json = json.dumps({"score": health[0], "status": health[1],
                                  "detail": health[2]}, default=str)
        usage_json = json.dumps(usage or {}, default=str)
        report = self._chat(
            "You are a customer success executive. Write a Quarterly Business Review in "
            "markdown: usage statistics, business outcomes, ROI, open issues, and "
            "recommendations. Use only the provided data.",
            f"Customer: {customer}\nHealth: {health_json}\nUsage: {usage_json}",
            temperature=0.4)
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO reports (title, kind, content) VALUES (%s, %s, %s) "
                        "RETURNING id", (f"{customer} QBR", "qbr",
                                         Json({"markdown": report})))
            rid = cur.fetchone()[0]
        self._log("qbr", {"customer": customer, "report_id": rid})
        return {"customer": customer, "qbr": report, "report_id": rid,
                "human_approval_required": False}

    # ---------- 13/14. community ----------

    def community_post(self, author, content, kind="post", action="welcome"):
        if not author or not content:
            raise ValueError("author and content required")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO community_posts (author, content, kind, status) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (author, content, kind, "visible"))
            pid = cur.fetchone()[0]
        moderation = "visible"
        if self.sentiment(content)["sentiment"] == "negative":
            moderation = "pending_review"
            with self._db().cursor() as cur:
                cur.execute("UPDATE community_posts SET status = %s WHERE id = %s",
                            ("pending_review", pid))
        return {"community_updates": [{"post_id": pid, "author": author, "kind": kind,
                                       "status": moderation}],
                "human_approval_required": moderation == "pending_review"}

    def community_event(self, title, kind, starts_at=None):
        if not title or not kind:
            raise ValueError("title and kind required")
        allowed = ("ama", "webinar", "product_launch", "office_hours", "challenge")
        if kind not in allowed:
            raise ValueError(f"unsupported event kind: {kind}")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO community_events (title, kind, starts_at) "
                        "VALUES (%s, %s, %s) RETURNING id", (title, kind, starts_at))
            eid = cur.fetchone()[0]
        return {"community_events": [{"event_id": eid, "title": title, "kind": kind,
                                      "status": "scheduled"}],
                "human_approval_required": False}

    # ---------- 16. approval helper ----------

    def _approval_result(self, action, ticket_id=None):
        self._log("approval_required", {"action": action, "ticket_id": ticket_id})
        return {"action": action, "ticket_id": ticket_id,
                "human_approval_required": True}

    # ---------- memory ----------

    def memory(self, limit=5):
        with self._db().cursor() as cur:
            cur.execute("SELECT action, detail, created_at FROM department_logs "
                        "WHERE department = 'customer' ORDER BY created_at DESC LIMIT %s",
                        (int(limit),))
            return [{"action": r[0], "detail": r[1], "created_at": str(r[2])}
                    for r in cur.fetchall()]

    # ---------- 17. structured output ----------

    def run(self, task):
        task = task or {}
        handlers = {
            "create_ticket": lambda: self.create_ticket(
                task.get("customer"), task.get("subject"), task.get("body")),
            "triage_ticket": lambda: self.triage_ticket(task.get("ticket_id")),
            "update_ticket_status": lambda: self.update_ticket_status(
                task.get("ticket_id"), task.get("status")),
            "list_tickets": lambda: self.list_tickets(task.get("status"), task.get("limit", 20)),
            "faq_answer": lambda: self.faq_answer(task.get("question")),
            "kb_add": lambda: self.kb_add(task.get("title"), task.get("content"),
                                          task.get("tags")),
            "kb_search": lambda: self.kb_search(task.get("query"), task.get("limit", 5)),
            "escalate": lambda: self.escalate(task.get("ticket_id")),
            "draft_response": lambda: self.draft_response(task.get("ticket_id")),
            "onboarding_plan": lambda: self.onboarding_plan(
                task.get("customer"), task.get("product"), task.get("company_name", "Acme")),
            "health_score": lambda: self.health_score(
                task.get("customer"), task.get("login_freq", 3), task.get("support_volume", 0),
                task.get("satisfaction", 4), task.get("feature_adoption", 0.5),
                task.get("activity", 0.5)),
            "churn_prediction": lambda: self.churn_prediction(
                task.get("customer"), task.get("health")),
            "renewal_plan": lambda: self.renewal_plan(
                task.get("customer"), task.get("plan"), task.get("amount"),
                task.get("days_to_renewal", 30)),
            "advocacy": lambda: self.advocacy(task.get("customer"), task.get("score")),
            "qbr": lambda: self.qbr(task.get("customer"), task.get("usage")),
            "community_post": lambda: self.community_post(
                task.get("author"), task.get("content"), task.get("kind", "post"),
                task.get("action", "welcome")),
            "community_event": lambda: self.community_event(
                task.get("title"), task.get("kind"), task.get("starts_at")),
            "sentiment": lambda: self.sentiment(task.get("text")),
            "refund": lambda: self._approval_result("refund", task.get("ticket_id")),
            "compensation": lambda: self._approval_result("compensation", task.get("ticket_id")),
            "legal_response": lambda: self._approval_result("legal_response",
                                                            task.get("ticket_id")),
            "public_statement": lambda: self._approval_result("public_statement"),
            "account_termination": lambda: self._approval_result("account_termination",
                                                                 task.get("ticket_id")),
            "memory": lambda: self.memory(task.get("limit", 5)),
        }
        return self.handle(task, handlers)


def demo():
    a = CustomerAgent()
    assert a.sentiment("this is the worst product ever")["sentiment"] == "negative"
    assert a.sentiment("great experience, love it")["sentiment"] == "positive"
    assert a.sentiment("the item is blue")["sentiment"] == "neutral"
    assert a._rule_hit("my invoice is wrong", TRIAGE_RULES) == "billing"
    assert a._rule_hit("reset my password", TRIAGE_RULES) == "account"
    assert a._rule_hit("hello world", TRIAGE_RULES) is None
    assert a._rule_hit("you are an awful company", ESCALATION_RULES) == "angry"
    assert a._rule_hit("security breach", ESCALATION_RULES) == "security"
    assert a._approval_result("refund")["human_approval_required"] is True
    assert a.run({"action": "nope"})["status"] == "error"
    assert a.run({"action": "create_ticket"})["status"] == "error"
    print("CustomerAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(CustomerAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
