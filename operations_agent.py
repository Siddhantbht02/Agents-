"""
Operations Department Agent for the Capstone multi-agent system.

Responsibilities (per spec):
  1. Client onboarding          (PostgreSQL + LLM, staged lifecycle)
  2. Document generation        (LLM; markdown + HTML files)
  3. PDF/report generation      (HTML is print-to-PDF ready)
  4. Google Drive management    (Drive upload API, best-effort)
  5. Client records             (clients table, links companies)
  6. Meeting management         (meetings table + LLM summaries)
  7/8. Knowledge base           (kb_articles table, keyword search)
  9. Project tracking           (projects table, milestones)
 10. Missing info detection     (returns needs_info + missing_fields, never guesses)
 11. Client status updates      (onboarding lifecycle)
 12. Internal documentation     (SOPs, runbooks via LLM)
 13. Human approval workflow    (final onboarding + contract docs -> approval)
 14. Structured output          (returned to the CEO Agent)

Run:
  py operations_agent.py --demo
  py operations_agent.py --task '{"action": "create_client", "name": "Acme", ...}'

Or import:  from operations_agent import OperationsAgent; print(OperationsAgent().run(task))

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

from agent_base import BaseAgent, Json, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS clients (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  company_id BIGINT,
  primary_contact TEXT,
  contact_email TEXT,
  company_address TEXT,
  tax_id TEXT,
  status TEXT NOT NULL DEFAULT 'New Client',
  checklist JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS client_documents (
  id BIGSERIAL PRIMARY KEY,
  client_id BIGINT REFERENCES clients(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  format TEXT NOT NULL DEFAULT 'md',
  path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS meetings (
  id BIGSERIAL PRIMARY KEY,
  client_id BIGINT REFERENCES clients(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  summary TEXT,
  notes TEXT,
  action_items JSONB NOT NULL DEFAULT '[]'::jsonb,
  held_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS projects (
  id BIGSERIAL PRIMARY KEY,
  client_id BIGINT REFERENCES clients(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  milestones JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS kb_articles (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  tags TEXT[] NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# ponytail: KB is keyword-searchable text in Postgres; swap in Qdrant/Pinecone
# embeddings only when semantic retrieval across many articles is actually needed.
ONBOARDING_STATUSES = ("New Client", "Documents Generated", "Waiting for Signature",
                       "Waiting for Information", "Active", "Completed")

# Information required before any onboarding documents can be generated.
CLIENT_REQUIRED_FIELDS = ("primary_contact", "contact_email",
                          "company_address", "tax_id")

# Documents that are contract-adjacent: always need human approval.
CONTRACT_DOCS = {"sow", "proposal", "contract", "invoice"}

DOC_DIR = "docs"

ACTIONS = {
    "create_client", "get_client", "update_client", "update_client_status", "onboarding_checklist",
    "activate_client", "generate_document", "create_meeting", "add_meeting_notes",
    "kb_add", "kb_search", "create_project", "update_project", "memory",
    "task_create", "task_list", "task_update", "task_close",
}


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "doc"


class OperationsAgent(BaseAgent):
    DEPARTMENT = "operations"
    SCHEMA_SQL = SCHEMA_SQL

    def _client(self, client_id):
        if not client_id:
            raise ValueError("client_id required")
        with self._db().cursor() as cur:
            cur.execute("SELECT id, name, company_id, primary_contact, contact_email, "
                        "company_address, tax_id, status, checklist FROM clients WHERE id = %s",
                        (client_id,))
            row = cur.fetchone()
        if not row:
            raise ValueError(f"client {client_id} not found")
        return {"id": row[0], "name": row[1], "company_id": row[2],
                "primary_contact": row[3], "contact_email": row[4],
                "company_address": row[5], "tax_id": row[6],
                "status": row[7], "checklist": row[8] or []}

    # ---------- 5/10/11. client records, validation, status lifecycle ----------

    def create_client(self, name, primary_contact=None, contact_email=None,
                      company_address=None, tax_id=None, company_id=None):
        if not name:
            raise ValueError("client name required")
        with self._db().cursor() as cur:
            cur.execute(
                "INSERT INTO clients (name, company_id, primary_contact, contact_email, "
                "company_address, tax_id) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (name, company_id, primary_contact, contact_email, company_address, tax_id))
            cid = cur.fetchone()[0]
        self._log("create_client", {"client_id": cid, "name": name})
        return {"client_id": cid, "name": name, "status": "New Client",
                "onboarding_checklist": self.onboarding_checklist(cid)["checklist"],
                "human_approval_required": False}

    def get_client(self, client_id):
        c = self._client(client_id)
        return {"client": c, "human_approval_required": False}

    def update_client(self, client_id, **fields):
        allowed = ("primary_contact", "contact_email", "company_address", "tax_id", "name")
        sets, args = [], []
        for k, v in fields.items():
            if v is None or k not in allowed:
                continue
            sets.append(f"{k} = %s")
            args.append(v)
        if not sets:
            raise ValueError("no updatable fields provided")
        args.append(client_id)
        with self._db().cursor() as cur:
            cur.execute(f"UPDATE clients SET {', '.join(sets)} WHERE id = %s", args)
        self._log("update_client", {"client_id": client_id, **fields})
        return {"client_id": client_id, "client": self._client(client_id),
                "human_approval_required": False}

    def missing_fields(self, client):
        return [f for f in CLIENT_REQUIRED_FIELDS if not client.get(f)]

    def onboarding_checklist(self, client_id):
        c = self._client(client_id)
        missing = self.missing_fields(c)
        checklist = [
            {"item": "Collect client profile (primary contact, email, address, tax ID)",
             "done": not missing},
            {"item": "Verify client information", "done": not missing},
            {"item": "Generate onboarding documents", "done": c["status"] in
             ("Documents Generated", "Waiting for Signature", "Waiting for Information",
              "Active", "Completed")},
            {"item": "Collect signed documents", "done": c["status"] in
             ("Waiting for Information", "Active", "Completed")},
            {"item": "Verify documents", "done": c["status"] in ("Active", "Completed")},
            {"item": "Activate client account", "done": c["status"] in ("Active", "Completed")},
        ]
        with self._db().cursor() as cur:
            cur.execute("UPDATE clients SET checklist = %s WHERE id = %s",
                        (Json(checklist), client_id))
        return {"client_id": client_id, "status": c["status"], "checklist": checklist,
                "missing_fields": missing}

    def update_client_status(self, client_id, status):
        if status not in ONBOARDING_STATUSES:
            raise ValueError(f"invalid status: {status}")
        with self._db().cursor() as cur:
            cur.execute("UPDATE clients SET status = %s WHERE id = %s", (status, client_id))
        self._log("update_client_status", {"client_id": client_id, "status": status})
        approval = status == "Active"
        return {"client_id": client_id, "status": status,
                "onboarding_status": status, "human_approval_required": approval}

    def activate_client(self, client_id):
        c = self._client(client_id)
        missing = self.missing_fields(c)
        if missing:
            return {"status": "needs_info", "missing_fields": missing}
        if c["status"] not in ("Documents Generated", "Waiting for Signature",
                               "Waiting for Information"):
            return {"status": "needs_info",
                    "missing_fields": [f"client must reach 'Waiting for Information' "
                                       f"before activation (currently {c['status']})"]}
        with self._db().cursor() as cur:
            cur.execute("UPDATE clients SET status = 'Active' WHERE id = %s", (client_id,))
        self._log("activate_client", {"client_id": client_id})
        return {"client_id": client_id, "onboarding_status": "Active",
                "human_approval_required": True}

    # ---------- 2/3. document generation ----------

    def generate_document(self, client_id, kind, title, detail=""):
        if kind not in ("welcome_letter", "onboarding_guide", "sow", "proposal",
                        "requirement_doc", "meeting_minutes", "sop", "runbook",
                        "internal_doc", "incident_report", "invoice"):
            raise ValueError(f"unsupported document kind: {kind}")
        c = self._client(client_id)
        missing = self.missing_fields(c)
        if missing and kind in ("welcome_letter", "onboarding_guide", "sow", "proposal", "invoice"):
            return {"status": "needs_info", "missing_fields": missing}
        body = self._chat(
            "You are an operations document writer. Produce clean, professional "
            "markdown. Use headings and numbered steps where relevant. Never invent "
            "facts; only use the client details and detail provided.",
            f"Document: {kind} for client: {json.dumps(c, default=str)}\n"
            f"Title: {title}\nDetail: {detail}", temperature=0.5)
        html = ("<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>{title}</title></head><body>"
                + _md_to_html(body) + "</body></html>")
        os.makedirs(DOC_DIR, exist_ok=True)
        stamp = int(time.time())
        md_path = os.path.join(DOC_DIR, f"{_slug(title)}-{stamp}.md")
        html_path = os.path.join(DOC_DIR, f"{_slug(title)}-{stamp}.html")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(body)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO client_documents (client_id, title, kind, format, path) "
                        "VALUES (%s, %s, %s, 'md', %s) RETURNING id", (client_id, title, kind, md_path))
            md_id = cur.fetchone()[0]
            cur.execute("INSERT INTO client_documents (client_id, title, kind, format, path) "
                        "VALUES (%s, %s, %s, 'html', %s) RETURNING id",
                        (client_id, title, kind, html_path))
            html_id = cur.fetchone()[0]
        if kind in CONTRACT_DOCS:
            self.update_client_status(client_id, "Waiting for Signature")
            approval = True
        elif c["status"] == "New Client":
            self.update_client_status(client_id, "Documents Generated")
            approval = False
        else:
            approval = False
        self._log("generate_document", {"client_id": client_id, "kind": kind, "title": title})
        return {"documents": [{"document_id": md_id, "title": title, "kind": kind,
                               "format": "md", "path": md_path},
                              {"document_id": html_id, "title": title, "kind": kind,
                               "format": "html", "path": html_path}],
                "onboarding_status": self._client(client_id)["status"],
                "human_approval_required": approval}

    # ---------- 6. meeting management ----------

    def create_meeting(self, client_id, title, held_at=None, notes=""):
        summary = self._chat(
            "You are a meeting minute writer. Produce concise meeting minutes with "
            "sections: Summary, Decisions, Action Items. Markdown, no JSON.",
            f"Meeting: {title}\nNotes:\n{notes}", temperature=0.4)
        action_items = _extract_action_items(summary)
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO meetings (client_id, title, summary, notes, action_items, held_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                        (client_id, title, summary, notes, Json(action_items), held_at))
            mid = cur.fetchone()[0]
        self._log("create_meeting", {"client_id": client_id, "meeting_id": mid, "title": title})
        return {"meeting_id": mid, "title": title, "summary": summary,
                "action_items": action_items, "human_approval_required": False}

    def add_meeting_notes(self, client_id, title, notes):
        summary = self._chat(
            "You are a meeting minute writer. Produce concise meeting minutes with "
            "sections: Summary, Decisions, Action Items. Markdown, no JSON.",
            f"Meeting: {title}\nNotes:\n{notes}", temperature=0.4)
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO meetings (client_id, title, summary, notes) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (client_id, title, summary, notes))
            mid = cur.fetchone()[0]
        return {"meeting_id": mid, "title": title, "summary": summary,
                "human_approval_required": False}

    # ---------- 9. project tracking ----------

    def create_project(self, client_id, name, milestones=None):
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO projects (client_id, name, milestones) "
                        "VALUES (%s, %s, %s) RETURNING id",
                        (client_id, name, Json(milestones or [])))
            pid = cur.fetchone()[0]
        self._log("create_project", {"client_id": client_id, "project_id": pid, "name": name})
        return {"project_id": pid, "name": name, "status": "active",
                "human_approval_required": False}

    def update_project(self, project_id, status=None, milestone=None, done=False):
        with self._db().cursor() as cur:
            cur.execute("SELECT milestones, status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"project {project_id} not found")
            milestones, cur_status = row[0] or [], row[1]
            if milestone:
                found = next((m for m in milestones if m.get("name") == milestone), None)
                if found:
                    found["done"] = bool(done)
                else:
                    milestones.append({"name": milestone, "done": bool(done)})
            new_status = status or cur_status
            if milestone and done and not any(not m.get("done") for m in milestones):
                new_status = "completed"
            cur.execute("UPDATE projects SET status = %s, milestones = %s, updated_at = now() "
                        "WHERE id = %s", (new_status, Json(milestones), project_id))
        return {"project_id": project_id, "status": new_status, "milestones": milestones,
                "human_approval_required": False}

    # ---------- 7/8. knowledge base ----------

    def kb_add(self, title, content, tags=None):
        if not title or not content:
            raise ValueError("title and content required")
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO kb_articles (title, content, tags) VALUES (%s, %s, %s) "
                        "RETURNING id", (title, content, list(tags or [])))
            kid = cur.fetchone()[0]
        self._log("kb_add", {"kb_id": kid, "title": title})
        return {"kb_id": kid, "title": title, "kb_updates": [kid],
                "human_approval_required": False}

    def kb_search(self, query, limit=5):
        with self._db().cursor() as cur:
            cur.execute("SELECT id, title, created_at FROM kb_articles "
                        "WHERE title ILIKE %s OR content ILIKE %s "
                        "ORDER BY created_at DESC LIMIT %s",
                        (f"%{query}%", f"%{query}%", int(limit)))
            return [{"kb_id": r[0], "title": r[1], "created_at": str(r[2])}
                    for r in cur.fetchall()]

    # ---------- 10. knowledge base memory ----------

    def memory(self, limit=5):
        with self._db().cursor() as cur:
            cur.execute("SELECT department, action, detail, created_at FROM department_logs "
                        "WHERE department = 'operations' ORDER BY created_at DESC LIMIT %s",
                        (int(limit),))
            return [{"action": r[1], "detail": r[2], "created_at": str(r[3])}
                    for r in cur.fetchall()]

    # ---------- 14. structured output ----------

    def run(self, task):
        task = task or {}
        handlers = {
            "create_client": lambda: self.create_client(
                task.get("name"), task.get("primary_contact"), task.get("contact_email"),
                task.get("company_address"), task.get("tax_id"), task.get("company_id")),
            "get_client": lambda: self.get_client(task.get("client_id")),
            "update_client": lambda: self.update_client(task.get("client_id"),
                primary_contact=task.get("primary_contact"), contact_email=task.get("contact_email"),
                company_address=task.get("company_address"), tax_id=task.get("tax_id"),
                name=task.get("name")),
            "update_client_status": lambda: self.update_client_status(
                task.get("client_id"), task.get("status")),
            "onboarding_checklist": lambda: self.onboarding_checklist(task.get("client_id")),
            "activate_client": lambda: self.activate_client(task.get("client_id")),
            "generate_document": lambda: self.generate_document(
                task.get("client_id"), task.get("kind"), task.get("title"),
                task.get("detail", "")),
            "create_meeting": lambda: self.create_meeting(
                task.get("client_id"), task.get("title"), task.get("held_at"),
                task.get("notes", "")),
            "add_meeting_notes": lambda: self.add_meeting_notes(
                task.get("client_id"), task.get("title"), task.get("notes", "")),
            "create_project": lambda: self.create_project(
                task.get("client_id"), task.get("name"), task.get("milestones")),
            "update_project": lambda: self.update_project(
                task.get("project_id"), task.get("status"), task.get("milestone"),
                bool(task.get("done"))),
            "kb_add": lambda: self.kb_add(
                task.get("title"), task.get("content"), task.get("tags")),
            "kb_search": lambda: self.kb_search(task.get("query"), task.get("limit", 5)),
            "memory": lambda: self.memory(task.get("limit", 5)),
        }
        return self.handle(task, handlers)


def _md_to_html(md):
    out = []
    for line in (md or "").splitlines():
        line = line.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{m.group(2)}</h{level}>")
        elif re.match(r"^\s*[-*]\s+", line):
            item = re.sub(r"^\s*[-*]\s+", "", line)
            out.append(f"<li>{item}</li>")
        elif re.match(r"^\s*\d+\.\s+", line):
            out.append(f"<li>{line}</li>")
        elif line.strip():
            out.append(f"<p>{line}</p>")
    return "\n".join(out)


def _extract_action_items(md):
    items = []
    for line in (md or "").splitlines():
        if re.match(r"^\s*[-*]\s+", line):
            items.append(re.sub(r"^\s*[-*]\s+", "", line).strip())
        elif re.match(r"^\s*\d+\.\s+", line):
            items.append(re.sub(r"^\s*\d+\.\s+", "", line).strip())
    return items[:10]


def demo():
    a = OperationsAgent()
    assert ACTIONS == ACTIONS
    assert a.run({"action": "nope"})["status"] == "error"
    assert a.run({"action": "generate_document"})["status"] == "error"
    assert a.missing_fields({"primary_contact": "x", "contact_email": "x@y.com",
                             "company_address": "addr", "tax_id": "123"}) == []
    assert a.missing_fields({"primary_contact": "x"}) == ["contact_email",
                                                          "company_address", "tax_id"]
    assert _slug("Hello World! 123") == "hello-world-123"
    assert "sow" in CONTRACT_DOCS
    print("OperationsAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(OperationsAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
