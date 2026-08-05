"""
CEO Agent for the Capstone multi-agent system.

Executive coordinator: plans, delegates, monitors, and synthesizes the work of
seven department agents. Never does department work itself.

Run:
  py ceo_agent.py --demo
  py ceo_agent.py --task '{"request": "Find AI startups in India and email them."}'

Or import:  from ceo_agent import CEOAgent; print(CEOAgent().run(task))

Environment variables: same as department agents
  LLM_API_KEY, LLM_MODEL, LLM_BASE_URL          planning + synthesis
  DATABASE_URL                                   department_logs (ceo memory)

Third-party deps: none beyond the department agents.
"""

import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

from agent_base import BaseAgent
from sales_agent import SalesAgent
from marketing_agent import MarketingAgent
from research_agent import ResearchAgent
from operations_agent import OperationsAgent
from administration_agent import AdministrationAgent
from customer_agent import CustomerAgent
from deals_agent import DealsAgent

# Dependency ranks: higher rank runs strictly after lower. Independent
# departments (research + marketing) share a rank and run in parallel.
DEP_RANK = {
    "research": 0,
    "marketing": 0,
    "sales": 1,
    "deals": 2,
    "operations": 3,
    "customer": 4,
    "administration": 5,
}

CATALOG = {
    "research": {
        "company_research": ["company"],
        "competitor_analysis": ["company", "industry"],
        "market_research": ["industry"],
        "industry_intelligence": ["industry"],
        "news_monitoring": ["topic", "days"],
        "company_discovery": ["criteria"],
        "deep_website_analysis": ["url"],
        "timeline": ["company"],
        "swot": ["company"],
        "tech_analysis": ["company"],
    },
    "sales": {
        "generate_leads": ["criteria", "count"],
        "discover_companies": ["criteria", "count"],
        "enrich_company": ["url"],
        "update_crm": ["company", "lead"],
        "draft_email": ["lead", "intent", "context"],
        "classify_reply": ["email"],
        "sales_intelligence": ["domain"],
        "qualify": ["lead"],
    },
    "deals": {
        "handle_reply": ["from", "body", "subject"],
        "inbox_triage": ["sender", "subject", "body"],
        "objection_response": ["objection", "industry"],
        "objection_library_add": ["objection", "response", "industry", "win_rate"],
        "objection_library": ["query", "limit"],
        "hot_lead_routing": ["deal_id", "owner"],
        "book_meeting": ["deal_id", "title", "starts_at", "attendees"],
        "referral": ["deal_id", "referred_contact", "referred_email"],
        "inbound_lead": ["name", "contact", "contact_email", "source", "value"],
        "qualify": ["deal_id", "company_size", "industry", "budget", "authority", "timeline", "tech_fit"],
        "pre_call_brief": ["deal_id"],
        "call_capture": ["deal_id", "transcript"],
        "post_call_debrief": ["deal_id", "decisions", "next_steps"],
        "follow_up": ["deal_id", "kind"],
        "proposal": ["deal_id", "scope", "timeline", "deliverables", "price"],
        "demo_prototype": ["deal_id", "request"],
        "deal_room": ["deal_id"],
        "agreement": ["deal_id", "kind"],
        "pricing": ["deal_id", "base_price", "discount_pct", "roi_months", "savings_per_month"],
        "crm_hygiene": [],
        "pipeline_report": [],
        "forecast": ["months"],
        "reactivate": ["days_inactive", "limit"],
        "win_loss": [],
        "close_deal": ["deal_id", "outcome", "reason"],
        "create_deal": ["name", "company", "contact", "contact_email", "source", "value"],
    },
    "marketing": {
        "trend_research": ["topic", "n"],
        "trend_summary": ["topic", "n"],
        "content_strategy": ["topic", "days", "goal"],
        "write_copy": ["platform", "topic", "tone", "audience"],
        "generate_image": ["prompt"],
        "create_campaign": ["name", "goal", "product", "channels"],
        "set_brand_voice": ["voice", "example"],
        "get_brand_voice": ["voice"],
        "social_plan": ["topic", "days"],
        "analytics_summary": ["metrics"],
        "competitor_analysis": ["company"],
        "store_asset": ["name", "kind", "url"],
    },
    "operations": {
        "create_client": ["name", "primary_contact", "contact_email", "company_address", "tax_id", "company_id"],
        "get_client": ["client_id"],
        "update_client": ["client_id", "name", "primary_contact", "contact_email", "company_address", "tax_id"],
        "update_client_status": ["client_id", "status"],
        "onboarding_checklist": ["client_id"],
        "activate_client": ["client_id"],
        "generate_document": ["client_id", "kind", "title", "detail"],
        "create_meeting": ["client_id", "title", "held_at", "notes"],
        "add_meeting_notes": ["client_id", "title", "notes"],
        "create_project": ["client_id", "name", "milestones"],
        "update_project": ["project_id", "status", "milestone", "done"],
        "kb_add": ["title", "content", "tags"],
        "kb_search": ["query", "limit"],
    },
    "customer": {
        "create_ticket": ["customer", "subject", "body"],
        "triage_ticket": ["ticket_id"],
        "update_ticket_status": ["ticket_id", "status"],
        "list_tickets": ["status", "limit"],
        "faq_answer": ["question"],
        "kb_add": ["title", "content", "tags"],
        "kb_search": ["query", "limit"],
        "escalate": ["ticket_id"],
        "draft_response": ["ticket_id"],
        "onboarding_plan": ["customer", "product", "company_name"],
        "health_score": ["customer", "login_freq", "support_volume", "satisfaction", "feature_adoption", "activity"],
        "churn_prediction": ["customer", "health"],
        "renewal_plan": ["customer", "plan", "amount", "days_to_renewal"],
        "advocacy": ["customer", "score"],
        "qbr": ["customer", "usage"],
        "community_post": ["author", "content", "kind", "action"],
        "community_event": ["title", "kind", "starts_at"],
        "sentiment": ["text"],
        "refund": ["ticket_id"],
        "compensation": ["ticket_id"],
        "legal_response": ["ticket_id"],
        "public_statement": [],
        "account_termination": ["ticket_id"],
    },
    "administration": {
        "create_invoice": ["client_name", "amount", "due_days", "items"],
        "update_invoice_status": ["invoice_id", "status"],
        "list_invoices": ["status", "limit"],
        "invoice_summary": [],
        "schedule_meeting": ["title", "starts_at", "attendees"],
        "find_slots": ["date", "duration_min", "calendar_id"],
        "cancel_meeting": ["event_id"],
        "calendar_upcoming": ["days", "calendar_id"],
        "triage_email": ["subject", "body"],
        "draft_reply": ["email_from", "subject", "body"],
        "store_file": ["name", "kind", "url", "folder"],
        "list_files": ["folder", "limit"],
        "move_file": ["file_id", "folder"],
        "rename_file": ["file_id", "name"],
        "delete_file": ["file_id"],
        "generate_report": ["kind", "params"],
    },
}

PLANNER_SYSTEM = """You are the planning layer of a CEO agent. The user gives one request; you
decompose it into the minimum number of department steps.

Available departments (dependency order: research & marketing first, then
sales, deals, operations, customer, administration). Use only these department
names: research, sales, deals, marketing, operations, customer, administration.

Department selection guide (user intent -> department):
  find/discover companies, competitor, market trends, industry, news, swot, tech -> research
  generate leads, cold outreach, cold email, prospecting, crm -> sales
  prospect reply, proposal, demo, pricing, pipeline, forecast, close deal -> deals
  social media post, linkedin post, campaign, copy, brand voice, image, ad, content -> marketing
  client onboarding, documents, projects, meetings with client, knowledge base -> operations
  support ticket, refund, customer health, churn, community, qbr -> customer
  invoice, calendar, scheduling, email management, reports, files -> administration

Action disambiguation (research):
  "top N companies", "list of companies", "companies in <country/industry>",
  "find startups" -> research/company_discovery with params {"criteria": "<the list criteria>"}
  deep research on ONE named company -> research/company_research with params {"company": "<company name>"}
  A request for a list is always company_discovery, never company_research.

Each step must pick ONE action from the catalog below and ONLY include the
listed parameters for that action. Omit parameters whose value you don't know;
NEVER use placeholders like "your industry", "unknown", or "number of days".
Never invent parameters or actions. Chain dependent steps in order (research
before sales, sales before deals, deals before operations).

Catalog (department: action -> params):
{catalog}

Return ONLY valid JSON: {{"steps": [{{"department": "...", "action": "...", "params": {{...}}}}]}}.
If the request needs no department action, return {{"steps": []}}."""


class CEOAgent(BaseAgent):
    DEPARTMENT = "ceo"

    def __init__(self):
        super().__init__()
        self.agents = {
            "research": ResearchAgent(),
            "sales": SalesAgent(),
            "deals": DealsAgent(),
            "marketing": MarketingAgent(),
            "operations": OperationsAgent(),
            "customer": CustomerAgent(),
            "administration": AdministrationAgent(),
        }

    # ---------- planning ----------

    def _plan(self, request):
        text = self._chat(
            PLANNER_SYSTEM.replace("{catalog}", json.dumps(CATALOG)),
            str(request), temperature=0.2)
        # some models split the plan into multiple JSON objects; merge their steps
        steps = []
        for obj in self._all_json(text):
            if isinstance(obj, dict):
                steps.extend(obj.get("steps") or [])
        validated = []
        for s in steps:
            dept = s.get("department")
            action = s.get("action")
            params = {k: v for k, v in (s.get("params") or {}).items()
                      if k in CATALOG.get(dept, {}).get(action, []) and v is not None}
            if dept in CATALOG and action in CATALOG[dept]:
                validated.append({"department": dept, "action": action, "params": params})
        return validated

    # ---------- execution ----------

    def _execute_step(self, step):
        agent = self.agents[step["department"]]
        task = {"action": step["action"], **step["params"]}
        try:
            return agent.run(task)
        except Exception as e:
            return {"status": "error", "action": step["action"], "error": str(e)}

    def _execute(self, steps, correlation_id):
        results = []
        for rank in sorted(set(DEP_RANK[s["department"]] for s in steps)):
            batch = [s for s in steps if DEP_RANK[s["department"]] == rank]
            # different departments in a rank are independent -> parallel;
            # steps within one department are dependent -> sequential
            groups = {}
            for s in batch:
                groups.setdefault(s["department"], []).append(s)
            with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                futures = {}
                for dept, group in groups.items():
                    futures[pool.submit(self._run_group, group, correlation_id)] = dept
                for fut in futures:
                    results.extend(fut.result())
        return results

    def _run_group(self, group, correlation_id):
        out = []
        for step in group:
            result = self._execute_step(step)
            self._log(step["action"], {"correlation_id": correlation_id,
                                       "department": step["department"],
                                       "params": step["params"],
                                       "status": result.get("status")})
            out.append({"step": step, "result": result})
        return out

    # ---------- synthesis ----------

    def _synthesize(self, request, results):
        approvals = [r["result"]["action"] for r in results
                     if r["result"].get("human_approval_required")]
        failures = [r for r in results if r["result"].get("status") == "error"]
        lines = []
        for r in results:
            step, res = r["step"], r["result"]
            if res.get("status") == "error":
                lines.append(f"- {step['department']}/{step['action']} FAILED: {res.get('error')}")
            else:
                kept = {k: v for k, v in res.items() if k not in ("status", "action")}
                lines.append(f"- {step['department']}/{step['action']}: {json.dumps(kept)[:500]}")
        brief = self._chat(
            "You are a CEO summarizing completed department work for the user. "
            "Present a single coherent summary in plain language. Do NOT mention "
            "department names, action names, or JSON. If anything is pending human "
            "approval, say 'Draft created and awaiting approval.' If something failed, "
            "say what was not completed and why, without fabricating results.",
            f"User request: {request}\n\nDepartment outputs:\n" + "\n".join(lines),
            temperature=0.3)
        return {"summary": brief, "approvals_pending": approvals,
                "failures": [f"{r['step']['department']}/{r['step']['action']}"
                             for r in failures]}

    # ---------- entry ----------

    def run(self, task):
        task = task or {}
        request = task.get("request")
        if not request:
            return {"status": "error", "action": "plan", "error": "request is required"}
        correlation_id = str(uuid.uuid4())
        try:
            steps = self._plan(request)
        except Exception as e:
            return {"status": "error", "action": "plan", "error": str(e)}
        if not steps:
            return {"status": "ok", "action": "plan", "correlation_id": correlation_id,
                    "steps": [], "summary": "No department work was required for this request."}
        results = self._execute(steps, correlation_id)
        out = self._synthesize(request, results)
        return {"status": "ok", "action": "plan", "correlation_id": correlation_id,
                "steps": [s["step"] for s in results], **out}


def demo():
    a = CEOAgent()
    steps = a._plan("Find AI startups in India and send outreach emails")
    depts = [s["department"] for s in steps]
    assert "sales" in depts, f"expected sales in plan, got {depts}"
    assert all(d in CATALOG for d in depts)
    for s in steps:
        assert s["action"] in CATALOG[s["department"]]
        assert set(s["params"]) <= set(CATALOG[s["department"]][s["action"]])
    assert a.run({"request": ""})["status"] == "error"
    assert a.run({"request": "nothing actionable"})["status"] == "ok"
    print("CEOAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(CEOAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
