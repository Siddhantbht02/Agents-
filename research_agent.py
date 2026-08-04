"""
Research & Intelligence Department Agent for the Capstone multi-agent system.

Responsibilities (per spec):
  1. Company research          (Tavily + Firecrawl + LLM)
  2. Competitor analysis       (Tavily + LLM)
  3. Market research           (Tavily + LLM)
  4. Industry intelligence     (Tavily + LLM)
  5. News monitoring           (Tavily)
  6. Company discovery         (Tavily)
  7. Deep website analysis     (Firecrawl)
  8. Research report generation (stored in PostgreSQL)
  9. Source verification        (attach URLs; low_confidence if <2 corroborating sources)
 10. Knowledge base / cache     (reports table; reuse instead of duplicating)
 11. Company timeline          (LLM)
 12. SWOT analysis             (LLM)
 13. Technology analysis       (LLM)
 14. Structured output          (returned to the CEO Agent)

Run:
  py research_agent.py --demo
  py research_agent.py --task '{"action": "company_research", "company": "OpenAI"}'

Or import:  from research_agent import ResearchAgent; print(ResearchAgent().run(task))

Environment variables:
  TAVILY_API_KEY, FIRECRAWL_API_KEY              search / scrape
  OPENAI_API_KEY, LLM_MODEL, LLM_BASE_URL        LLM (OpenAI-compatible, e.g. Groq)
  DATABASE_URL                                    postgresql://user:pass@host:5432/db
  REDIS_URL                                       optional, session cache only

Third-party deps (everything else is stdlib):
  pip install psycopg2-binary redis
"""

import json
import os
import sys

from agent_base import BaseAgent, Json, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS reports (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  content JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# ponytail: "knowledge base" is the reports table queried by title keyword match
# instead of Qdrant/Pinecone embeddings; swap in a vector DB only when fuzzy
# semantic retrieval across thousands of reports is actually needed.
CACHE_DAYS = 7

ACTIONS = {
    "company_research", "competitor_analysis", "market_research",
    "industry_intelligence", "news_monitoring", "company_discovery",
    "deep_website_analysis", "timeline", "swot", "tech_analysis", "memory",
    "task_create", "task_list", "task_update", "task_close",
}

# Firecrawl extraction schema for company profiling.
RIC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "industry": {"type": "string"},
        "description": {"type": "string"},
        "location": {"type": "string"},
        "founded": {"type": "string"},
        "employees": {"type": "string"},
        "leadership": {"type": "array", "items": {"type": "string"}},
        "products": {"type": "array", "items": {"type": "string"}},
        "tech_stack": {"type": "array", "items": {"type": "string"}},
        "customers": {"type": "array", "items": {"type": "string"}},
        "pricing": {"type": "string"},
    },
}


class ResearchAgent(BaseAgent):
    DEPARTMENT = "research"
    SCHEMA_SQL = SCHEMA_SQL

    # ---------- shared research core ----------

    def _cached_report(self, kind, topic):
        try:
            with self._db().cursor() as cur:
                cur.execute(
                    "SELECT id, title, content, created_at FROM reports "
                    "WHERE kind = %s AND title ILIKE %s "
                    "AND created_at > now() - interval '%s days' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (kind, f"%{topic[:60]}%", CACHE_DAYS))
                return cur.fetchone()
        except Exception:
            return None

    def _verify(self, sources):
        domains = {s.get("domain") for s in sources if s.get("domain")}
        return len(domains) < 2

    def _cached_response(self, kind, topic):
        row = self._cached_report(kind, topic)
        if not row:
            return None
        self._log("cache_hit", {"kind": kind, "topic": topic, "report_id": row[0]})
        return {"report_id": row[0], "title": row[1], "cached": True,
                "report_markdown": (row[2] or {}).get("markdown"),
                "sources": [s.get("url") for s in (row[2] or {}).get("sources", [])],
                "low_confidence": False, "reused": True}

    def _research(self, kind, topic, system_prompt, sources, extra=None):
        hit = self._cached_response(kind, topic)
        if hit:
            return hit
        digest = "\n".join(f"- {s['title']}: {s['snippet']}\n  ({s['url']})"
                           for s in sources) or "No sources found."
        markdown = self._chat(
            f"{system_prompt}\n"
            "Write a structured markdown report with clear section headings. "
            "Attach source URLs inline as [n] with a numbered Sources section. "
            "Do not state facts that are not in the material. Plain markdown, no JSON.",
            f"Topic: {topic}\nMaterial:\n{digest}", temperature=0.5)
        content = {"markdown": markdown, "sources": sources, **(extra or {})}
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO reports (title, kind, content) VALUES (%s, %s, %s) "
                        "RETURNING id", (f"{topic} research report", kind, Json(content)))
            rid = cur.fetchone()[0]
        self._log(kind, {"topic": topic, "report_id": rid,
                         "sources": len(sources)})
        return {"report_id": rid, "title": f"{topic} research report",
                "report_markdown": markdown,
                "sources": [s["url"] for s in sources],
                "low_confidence": self._verify(sources)}

    # ---------- 1. company research ----------

    def firecrawl_extract(self, url):
        key = os.getenv("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("FIRECRAWL_API_KEY not set")
        data = self._http_json(
            "POST", "https://api.firecrawl.dev/v1/scrape",
            {"url": url, "formats": ["markdown", "extract"], "onlyMainContent": True,
             "extract": {"schema": RIC_SCHEMA,
                         "prompt": "Extract company facts from the page."}},
            {"Authorization": f"Bearer {key}"}, timeout=90)
        if not (data or {}).get("success"):
            raise RuntimeError(f"firecrawl failed: {data}")
        d = data.get("data") or {}
        meta = d.get("metadata") or {}
        ex = d.get("llm_extraction") or {}
        return {"url": meta.get("ogUrl") or url, "name": ex.get("name") or meta.get("ogSiteName"),
                "description": (ex.get("description") or meta.get("description") or "")[:1000],
                "industry": ex.get("industry"), "founded": ex.get("founded"),
                "employees": str(ex.get("employees")) if ex.get("employees") else None,
                "location": ex.get("location"),
                "leadership": ex.get("leadership") or [],
                "products": ex.get("products") or [],
                "tech_stack": ex.get("tech_stack") or [],
                "customers": ex.get("customers") or [],
                "pricing": ex.get("pricing")}

    def company_research(self, company, scrape_site=True):
        hit = self._cached_response("company", str(company))
        if hit:
            return hit
        sources = self.tavily_search(str(company), 10)
        site_url = None
        for s in sources:
            dom = s.get("domain") or ""
            if company.lower().split()[0] in dom and dom not in ("crunchbase.com", "wikipedia.org"):
                site_url = s["url"]
                break
        site = self.firecrawl_extract(site_url) if (scrape_site and site_url) else None
        extra = {"company_profile": site} if site else {}
        src = ([{"title": "official website", "url": site["url"],
                 "domain": site["url"].split("/")[2], "snippet": site["description"]}]
               if site else []) + sources
        return self._research(
            "company", str(company),
            "You are a business research analyst. Build a comprehensive company profile: "
            "overview, industry, products & services, headquarters, employees, funding history, "
            "customers, leadership, technology stack, recent developments.",
            src, extra)

    # ---------- 2. competitor analysis ----------

    def competitor_analysis(self, company, competitors=None):
        competitors = competitors or []
        if not competitors:
            raise ValueError("competitors list required")
        topic = f"{company} vs {', '.join(competitors[:3])}"
        hit = self._cached_response("competitor", topic)
        if hit:
            return hit
        sources = self.tavily_search(topic + " comparison", 10)
        return self._research(
            "competitor", topic,
            "You are a competitive intelligence analyst. Compare the companies on: strengths, "
            "weaknesses, pricing, products, features, market positioning. End with a verdict.",
            sources, {"company": company, "competitors": competitors})

    # ---------- 3. market research ----------

    def market_research(self, industry):
        hit = self._cached_response("market", str(industry))
        if hit:
            return hit
        sources = self.tavily_search(f"{industry} market size growth trends 2026", 10)
        return self._research(
            "market", str(industry),
            "You are a market research analyst. Analyze: market size, growth rate, emerging "
            "technologies, key players, opportunities, risks.",
            sources)

    # ---------- 4. industry intelligence ----------

    def industry_intelligence(self, industry):
        hit = self._cached_response("industry", str(industry))
        if hit:
            return hit
        sources = self.tavily_search(f"{industry} regulations technology adoption market shifts", 10)
        return self._research(
            "industry", str(industry),
            "You are an industry intelligence analyst. Track: new regulations, technology "
            "adoption, market shifts, consumer behavior, investment trends.",
            sources)

    # ---------- 5. news monitoring ----------

    def news_monitoring(self, topic, days=7):
        sources = self.tavily_search(f"{topic} news funding product launches acquisitions", 8)
        return {"topic": str(topic), "news": [
            {"title": s["title"], "url": s["url"], "domain": s["domain"],
             "snippet": s["snippet"]} for s in sources]}

    # ---------- 6. company discovery ----------

    def company_discovery(self, criteria):
        sources = self.tavily_search(str(criteria), 10)
        companies = [{"name": s["title"].split(" - ")[0], "url": s["url"],
                      "domain": s["domain"], "snippet": s["snippet"][:200]}
                     for s in sources if s.get("domain") not in ("crunchbase.com", "wikipedia.org")]
        return {"criteria": str(criteria), "companies_found": companies,
                "sources": [s["url"] for s in sources],
                "low_confidence": self._verify(sources)}

    # ---------- 7. deep website analysis ----------

    def deep_website_analysis(self, url):
        data = self.firecrawl_extract(url)
        report = {"url": data["url"], "extracted": data}
        self.store_report(url, "website", report)
        return {"report": report, "report_saved": True}

    # ---------- 8/12. timeline / swot / tech ----------

    def timeline(self, company):
        hit = self._cached_response("timeline", str(company))
        if hit:
            return hit
        sources = self.tavily_search(f"{company} funding rounds acquisitions product launches history", 10)
        return self._research(
            "timeline", str(company),
            "You are a business historian. Build a chronological timeline of key events: funding "
            "rounds, product launches, CEO changes, acquisitions, partnerships, expansion.",
            sources)

    def swot(self, company):
        hit = self._cached_response("swot", str(company))
        if hit:
            return hit
        sources = self.tavily_search(f"{company} SWOT analysis strengths weaknesses", 10)
        return self._research(
            "swot", str(company),
            "You are a strategy analyst. Produce a SWOT analysis: Strengths, Weaknesses, "
            "Opportunities, Threats. Use markdown sections.",
            sources)

    def tech_analysis(self, company):
        hit = self._cached_response("tech", str(company))
        if hit:
            return hit
        sources = self.tavily_search(f"{company} technology stack AI models cloud infrastructure", 10)
        return self._research(
            "tech", str(company),
            "You are a technology analyst. Analyze: AI models, cloud providers, APIs, programming "
            "languages, infrastructure, security practices.",
            sources)

    # ---------- 10. knowledge base / memory ----------

    def store_report(self, title, kind, content):
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO reports (title, kind, content) VALUES (%s, %s, %s) "
                        "RETURNING id", (title, kind, Json(content or {})))
            rid = cur.fetchone()[0]
        return rid

    def recent_reports(self, kind=None, limit=5):
        q = "SELECT title, kind, created_at FROM reports"
        args = []
        if kind:
            q += " WHERE kind = %s"
            args.append(kind)
        q += " ORDER BY created_at DESC LIMIT %s"
        args.append(int(limit))
        with self._db().cursor() as cur:
            cur.execute(q, args)
            return [{"title": r[0], "kind": r[1], "created_at": str(r[2])} for r in cur.fetchall()]

    # ---------- 14. structured output ----------

    def run(self, task):
        task = task or {}
        handlers = {
            "company_research": lambda: self.company_research(
                task.get("company"), bool(task.get("scrape_site", True))),
            "competitor_analysis": lambda: self.competitor_analysis(
                task.get("company"), task.get("competitors") or []),
            "market_research": lambda: self.market_research(task.get("industry")),
            "industry_intelligence": lambda: self.industry_intelligence(task.get("industry")),
            "news_monitoring": lambda: self.news_monitoring(task.get("topic"), task.get("days", 7)),
            "company_discovery": lambda: self.company_discovery(task.get("criteria")),
            "deep_website_analysis": lambda: self.deep_website_analysis(task.get("url")),
            "timeline": lambda: self.timeline(task.get("company")),
            "swot": lambda: self.swot(task.get("company")),
            "tech_analysis": lambda: self.tech_analysis(task.get("company")),
            "memory": lambda: self.recent_reports(task.get("kind"), task.get("limit", 5)),
        }
        return self.handle(task, handlers)


def demo():
    a = ResearchAgent()
    expected = ACTIONS
    assert ACTIONS == expected
    assert a.run({"action": "nope"})["status"] == "error"
    assert a.run({"action": "company_research"})["status"] == "error"
    assert a._verify([{"domain": "x.com"}, {"domain": "x.com"}]) is True
    assert a._verify([{"domain": "x.com"}, {"domain": "y.com"}]) is False
    print("ResearchAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(ResearchAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
