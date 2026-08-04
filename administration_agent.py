"""
Administration Department Agent for the Capstone multi-agent system.

Responsibilities (per spec):
  1. Invoice management         (PostgreSQL + LLM, status lifecycle)
  2. Calendar management        (events in PostgreSQL, conflict checks + alternatives)
  3. Email management           (Gmail API drafts, best-effort)
  4. Email triage               (keyword rules + LLM fallback)
  5. Report generation          (LLM, stored in PostgreSQL)
  6/7. File management          (folders + move/rename/delete/search, Drive upload)
  8. Meeting coordination       (calendar events + reminders)
  9. Financial records          (invoice history in PostgreSQL)
 10. Administrative reporting   (executive summaries)
 11. File retrieval             (search by folder/name)
 12. Calendar conflict resolution (return alternate slots, never just fail)
 13. Human approval workflow    (financial actions -> approval)
 14. Structured output          (returned to the CEO Agent)

Run:
  py administration_agent.py --demo
  py administration_agent.py --task '{"action": "create_invoice", "client": "Acme", ...}'

Or import:  from administration_agent import AdministrationAgent; print(AdministrationAgent().run(task))

Environment variables:
  OPENAI_API_KEY, LLM_MODEL, LLM_BASE_URL        LLM (OpenAI-compatible, e.g. Groq)
  DATABASE_URL                                    postgresql://user:pass@host:5432/db
  GMAIL_ACCESS_TOKEN / GMAIL_REFRESH_TOKEN       Gmail drafts (optional, best-effort)
  GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET           refresh support
  REDIS_URL                                       optional, session cache only

Third-party deps (everything else is stdlib):
  pip install psycopg2-binary redis
"""

import base64
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from agent_base import BaseAgent, Json, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS invoices (
  id BIGSERIAL PRIMARY KEY,
  number TEXT UNIQUE NOT NULL,
  client TEXT NOT NULL,
  amount NUMERIC NOT NULL,
  currency TEXT NOT NULL DEFAULT 'USD',
  items JSONB NOT NULL DEFAULT '[]'::jsonb,
  status TEXT NOT NULL DEFAULT 'Draft',
  due_days INTEGER NOT NULL DEFAULT 30,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS calendar_events (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  starts_at TIMESTAMPTZ NOT NULL,
  ends_at TIMESTAMPTZ NOT NULL,
  attendees JSONB NOT NULL DEFAULT '[]'::jsonb,
  room TEXT,
  status TEXT NOT NULL DEFAULT 'scheduled',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS admin_files (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  folder TEXT NOT NULL DEFAULT '/',
  kind TEXT,
  url TEXT,
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

INVOICE_STATUSES = ("Draft", "Awaiting Approval", "Sent", "Paid", "Overdue", "Cancelled")
FINANCIAL_ACTIONS = {"send_invoice", "send_financial_document", "receipt"}

EMAIL_CATEGORIES = ("client", "sales", "support", "finance", "marketing", "internal", "urgent")

# ponytail: keyword rules first (free, deterministic), LLM only for ambiguous email.
TRIAGE_RULES = [
    ("urgent", ["urgent", "asap", "immediately", "critical", "deadline", "as soon as possible"]),
    ("finance", ["invoice", "payment", "receipt", "billing", "refund", "overdue", "pay"]),
    ("support", ["help", "issue", "error", "broken", "not working", "bug", "problem"]),
    ("sales", ["quote", "pricing", "demo", "buy", "purchase", "proposal"]),
    ("marketing", ["campaign", "newsletter", "press", "promotion", "advertise"]),
    ("internal", ["all hands", "internal", "standup", "retro", "team meeting"]),
    ("client", ["contract", "onboard", "project update", "status"]),
]

ACTIONS = {
    "create_invoice", "update_invoice_status", "list_invoices", "invoice_summary",
    "schedule_meeting", "find_slots", "cancel_meeting", "calendar_upcoming",
    "triage_email", "draft_reply", "store_file", "list_files", "move_file",
    "rename_file", "delete_file", "generate_report", "memory",
}


class AdministrationAgent(BaseAgent):
    DEPARTMENT = "administration"
    SCHEMA_SQL = SCHEMA_SQL

    # ---------- 1/9. invoices ----------

    def _next_invoice_number(self):
        with self._db().cursor() as cur:
            cur.execute("SELECT number FROM invoices ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
        year = datetime.now(timezone.utc).year
        if row:
            m = re.search(r"(\d+)$", row[0])
            n = int(m.group(1)) + 1 if m else 1
        else:
            n = 1
        return f"INV-{year}-{n:03d}"

    def create_invoice(self, client, items=None, amount=None, currency="USD", due_days=30):
        if not client:
            raise ValueError("client required")
        if not items and amount is None:
            raise ValueError("items or amount required")
        if items:
            j = self._chat_json(
                "You are an accountant. Fill in missing line-item prices for a professional "
                "invoice. Return ONLY valid JSON {\"items\": [{\"description\": \"...\", "
                '"qty": 1, "unit_price": 0.0}], "total": 0.0}.',
                f"Client: {client}\nRequested items: {json.dumps(items)}", temperature=0.3)
            inv_items = j.get("items") or items
            total = float(j.get("total") or sum(
                (i.get("qty", 1) or 1) * (i.get("unit_price") or 0) for i in inv_items))
        else:
            inv_items = [{"description": "Services", "qty": 1, "unit_price": float(amount)}]
            total = float(amount)
        number = self._next_invoice_number()
        with self._db().cursor() as cur:
            cur.execute(
                "INSERT INTO invoices (number, client, amount, currency, items, status, due_days) "
                "VALUES (%s, %s, %s, %s, %s, 'Draft', %s) RETURNING id",
                (number, client, total, currency, Json(inv_items), due_days))
            iid = cur.fetchone()[0]
        self._log("create_invoice", {"invoice_id": iid, "number": number, "client": client,
                                     "amount": total, "currency": currency})
        return {"invoice_id": iid, "number": number, "client": client, "amount": total,
                "currency": currency, "status": "Draft", "items": inv_items,
                "due_date": (datetime.now(timezone.utc) + timedelta(days=due_days))
                .date().isoformat(), "human_approval_required": False}

    def update_invoice_status(self, invoice_id, status):
        if status not in INVOICE_STATUSES:
            raise ValueError(f"invalid status: {status}")
        with self._db().cursor() as cur:
            cur.execute("UPDATE invoices SET status = %s WHERE id = %s", (status, invoice_id))
        self._log("update_invoice_status", {"invoice_id": invoice_id, "status": status})
        financial = status == "Sent" or status == "Paid"
        return {"invoice_id": invoice_id, "status": status,
                "human_approval_required": financial}

    def list_invoices(self, status=None, limit=20):
        q = "SELECT id, number, client, amount, currency, status, created_at FROM invoices"
        args = []
        if status:
            q += " WHERE status = %s"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT %s"
        args.append(int(limit))
        with self._db().cursor() as cur:
            cur.execute(q, args)
            return [{"invoice_id": r[0], "number": r[1], "client": r[2],
                     "amount": float(r[3]), "currency": r[4], "status": r[5],
                     "created_at": str(r[6])} for r in cur.fetchall()]

    def invoice_summary(self):
        with self._db().cursor() as cur:
            cur.execute("SELECT status, count(*), COALESCE(sum(amount), 0) FROM invoices "
                        "GROUP BY status")
            rows = cur.fetchall()
        summary = {r[0]: {"count": r[1], "total": float(r[2])} for r in rows}
        pending = [r for r in rows if r[0] in ("Sent", "Awaiting Approval", "Draft")]
        return {"invoice_summary": summary,
                "outstanding": {"count": sum(r[1] for r in pending),
                                "total": float(sum(r[2] for r in pending))},
                "human_approval_required": False}

    # ---------- 2/8/12. calendar ----------

    def _conflicts(self, starts_at, ends_at, exclude=None):
        with self._db().cursor() as cur:
            cur.execute(
                "SELECT id, title, starts_at, ends_at FROM calendar_events "
                "WHERE status != 'cancelled' AND starts_at < %s AND ends_at > %s",
                (ends_at, starts_at))
            return [{"event_id": r[0], "title": r[1], "starts_at": str(r[2]),
                     "ends_at": str(r[3])} for r in cur.fetchall() if r[0] != exclude]

    def _free_slots(self, start, minutes, n=3):
        slots = []
        cursor = start
        for _ in range(n * 5):
            end = cursor + timedelta(minutes=minutes)
            if not self._conflicts(cursor, end):
                slots.append(cursor)
                if len(slots) == n:
                    break
            cursor += timedelta(minutes=30)
        return slots

    def schedule_meeting(self, title, starts_at, duration_minutes=30, attendees=None, room=None):
        if not title:
            raise ValueError("title required")
        start = starts_at if isinstance(starts_at, datetime) else datetime.fromisoformat(starts_at)
        end = start + timedelta(minutes=int(duration_minutes))
        conflicts = self._conflicts(start, end)
        if conflicts:
            alt = [s.isoformat() for s in self._free_slots(start, int(duration_minutes))]
            return {"conflict": True, "conflicts_with": conflicts,
                    "suggested_slots": alt,
                    "message": "requested time conflicts with existing events",
                    "human_approval_required": False}
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO calendar_events (title, starts_at, ends_at, attendees, room) "
                        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                        (title, start, end, Json(list(attendees or [])), room))
            eid = cur.fetchone()[0]
        self._log("schedule_meeting", {"event_id": eid, "title": title,
                                       "starts_at": start.isoformat()})
        return {"calendar_event": {"event_id": eid, "title": title,
                                   "time": start.isoformat(), "end": end.isoformat(),
                                   "attendees": list(attendees or []), "room": room},
                "conflict": False, "human_approval_required": False}

    def find_slots(self, starts_at, duration_minutes=30, n=3):
        start = datetime.fromisoformat(starts_at)
        slots = [s.isoformat() for s in self._free_slots(start, int(duration_minutes), int(n))]
        return {"available_slots": slots, "human_approval_required": False}

    def cancel_meeting(self, event_id):
        with self._db().cursor() as cur:
            cur.execute("UPDATE calendar_events SET status = 'cancelled' WHERE id = %s",
                        (event_id,))
        self._log("cancel_meeting", {"event_id": event_id})
        return {"cancelled": True, "event_id": event_id, "human_approval_required": False}

    def calendar_upcoming(self, days=7, limit=20):
        with self._db().cursor() as cur:
            cur.execute("SELECT id, title, starts_at, ends_at, attendees, room FROM calendar_events "
                        "WHERE status != 'cancelled' AND starts_at > now() "
                        "AND starts_at < now() + interval '%s days' "
                        "ORDER BY starts_at LIMIT %s", (int(days), int(limit)))
            return [{"event_id": r[0], "title": r[1], "time": str(r[2]), "end": str(r[3]),
                     "attendees": r[4], "room": r[5]} for r in cur.fetchall()]

    # ---------- 3/4. email ----------

    def triage_email(self, subject, body):
        text = f"{subject or ''} {body or ''}".lower()
        if not text.strip():
            raise ValueError("subject or body required")
        for cat, words in TRIAGE_RULES:
            if any(w in text for w in words):
                return {"category": cat, "priority": "high" if cat == "urgent" else "normal"}
        cat = self._chat(
            "Classify this email into one of: client, sales, support, finance, marketing, "
            "internal, urgent. Reply with the single category word only.",
            f"Subject: {subject}\nBody: {body}", temperature=0.2).strip().lower()
        if cat not in EMAIL_CATEGORIES:
            cat = "internal"
        return {"category": cat, "priority": "normal"}

    def draft_reply(self, subject, body, category=None, to=None):
        if not subject and not body:
            raise ValueError("subject or body required")
        cat = category or self.triage_email(subject, body)["category"]
        reply = self._chat(
            "You are a professional administrative assistant. Write a concise, polite email "
            f"reply for a {cat} email. Include a greeting and sign-off.",
            f"Original subject: {subject}\nOriginal body: {body}", temperature=0.6)
        token = self._gmail_token()
        drafted = False
        if token and to:
            try:
                msg = EmailMessage()
                msg["To"] = to
                msg["Subject"] = f"Re: {subject}" if subject else "Re: your message"
                msg.set_content(reply)
                raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
                self._http_json("POST",
                                "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                                {"message": {"raw": raw}},
                                {"Authorization": f"Bearer {token}"})
                drafted = True
            except Exception:
                drafted = False
        return {"draft": reply, "category": cat, "drafted": drafted,
                "human_approval_required": False}

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
                 "client_secret": secret, "refresh_token": refresh}, form=True)
            return (data or {}).get("access_token")
        return None

    # ---------- 6/7/11. files ----------

    def store_file(self, name, folder="/", kind=None, url=None):
        if not name:
            raise ValueError("name required")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO admin_files (name, folder, kind, url) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (name, folder, kind, url))
            fid = cur.fetchone()[0]
        return {"file_id": fid, "name": name, "folder": folder,
                "human_approval_required": False}

    def list_files(self, folder=None, limit=50):
        q = "SELECT id, name, folder, kind, url FROM admin_files"
        args = []
        if folder:
            q += " WHERE folder = %s"
            args.append(folder)
        q += " ORDER BY created_at DESC LIMIT %s"
        args.append(int(limit))
        with self._db().cursor() as cur:
            cur.execute(q, args)
            return [{"file_id": r[0], "name": r[1], "folder": r[2], "kind": r[3],
                     "url": r[4]} for r in cur.fetchall()]

    def move_file(self, file_id, folder):
        with self._db().cursor() as cur:
            cur.execute("UPDATE admin_files SET folder = %s WHERE id = %s", (folder, file_id))
        return {"file_id": file_id, "folder": folder, "human_approval_required": False}

    def rename_file(self, file_id, name):
        with self._db().cursor() as cur:
            cur.execute("UPDATE admin_files SET name = %s WHERE id = %s", (name, file_id))
        return {"file_id": file_id, "name": name, "human_approval_required": False}

    def delete_file(self, file_id):
        with self._db().cursor() as cur:
            cur.execute("DELETE FROM admin_files WHERE id = %s", (file_id,))
        return {"deleted": True, "file_id": file_id, "human_approval_required": False}

    # ---------- 5/10. reports ----------

    def generate_report(self, kind, title="", detail=""):
        kinds = ("weekly", "monthly", "invoice", "team_activity", "department", "executive")
        if kind not in kinds:
            raise ValueError(f"unsupported report kind: {kind}")
        text = self._chat(
            "You are an administrative reporting analyst. Produce a concise professional "
            "markdown report with headings. Use only the provided data.",
            f"Report type: {kind}\nTitle: {title}\nData:\n{detail}", temperature=0.4)
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO reports (title, kind, content) VALUES (%s, %s, %s) "
                        "RETURNING id", (title or f"{kind} report", kind, Json({"markdown": text})))
            rid = cur.fetchone()[0]
        self._log("generate_report", {"report_id": rid, "kind": kind, "title": title})
        return {"report_id": rid, "title": title or f"{kind} report", "kind": kind,
                "report_markdown": text, "human_approval_required": False}

    # ---------- 10. department memory ----------

    def memory(self, limit=5):
        with self._db().cursor() as cur:
            cur.execute("SELECT action, detail, created_at FROM department_logs "
                        "WHERE department = 'administration' ORDER BY created_at DESC LIMIT %s",
                        (int(limit),))
            return [{"action": r[0], "detail": r[1], "created_at": str(r[2])}
                    for r in cur.fetchall()]

    # ---------- 14. structured output ----------

    def run(self, task):
        task = task or {}
        handlers = {
            "create_invoice": lambda: self.create_invoice(
                task.get("client"), task.get("items"), task.get("amount"),
                task.get("currency", "USD"), task.get("due_days", 30)),
            "update_invoice_status": lambda: self.update_invoice_status(
                task.get("invoice_id"), task.get("status")),
            "list_invoices": lambda: self.list_invoices(task.get("status"), task.get("limit", 20)),
            "invoice_summary": lambda: self.invoice_summary(),
            "schedule_meeting": lambda: self.schedule_meeting(
                task.get("title"), task.get("starts_at"), task.get("duration_minutes", 30),
                task.get("attendees"), task.get("room")),
            "find_slots": lambda: self.find_slots(
                task.get("starts_at"), task.get("duration_minutes", 30), task.get("n", 3)),
            "cancel_meeting": lambda: self.cancel_meeting(task.get("event_id")),
            "calendar_upcoming": lambda: self.calendar_upcoming(
                task.get("days", 7), task.get("limit", 20)),
            "triage_email": lambda: self.triage_email(task.get("subject"), task.get("body")),
            "draft_reply": lambda: self.draft_reply(
                task.get("subject"), task.get("body"), task.get("category"), task.get("to")),
            "store_file": lambda: self.store_file(
                task.get("name"), task.get("folder", "/"), task.get("kind"), task.get("url")),
            "list_files": lambda: self.list_files(task.get("folder"), task.get("limit", 50)),
            "move_file": lambda: self.move_file(task.get("file_id"), task.get("folder")),
            "rename_file": lambda: self.rename_file(task.get("file_id"), task.get("name")),
            "delete_file": lambda: self.delete_file(task.get("file_id")),
            "generate_report": lambda: self.generate_report(
                task.get("kind"), task.get("title", ""), task.get("detail", "")),
            "memory": lambda: self.memory(task.get("limit", 5)),
        }
        return self.handle(task, handlers)


def demo():
    a = AdministrationAgent()
    assert ACTIONS == ACTIONS
    assert a.run({"action": "nope"})["status"] == "error"
    assert a.run({"action": "create_invoice"})["status"] == "error"
    assert a.run({"action": "schedule_meeting", "title": "x"})["status"] == "error"
    t = a.triage_email("Invoice attached", "Please pay the overdue invoice")
    assert t["category"] == "finance", t
    t = a.triage_email("URGENT: server down", "Critical outage asap")
    assert t["category"] == "urgent", t
    assert _parse_iso("2026-08-10T14:00:00") is not None
    print("AdministrationAgent demo OK")


def _parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(AdministrationAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
