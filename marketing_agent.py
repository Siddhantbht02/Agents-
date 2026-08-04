"""
Marketing Department Agent for the Capstone multi-agent system.

Responsibilities (per spec):
  1. Trend research             (Tavily Search + LLM)
  2. Content strategy           (LLM)
  3. Copywriting                (LLM)
  4. Image generation           (Image Generation API)
  5. Campaign creation          (LLM, stored as reports)
  6. Brand voice management     (PostgreSQL brand_voice table)
  7. Social media planning      (LLM)
  8. Analytics summary          (LLM)
  9. Competitor analysis        (Tavily + LLM)
 10. Trend summarization        (Tavily + LLM)
 11. Asset management           (files table)
 12. Department memory          (reports + department_logs + Redis)
 13. Human approval workflow    (never publishes; approval flag in results)
 14. Structured output          (returned to the CEO Agent)

Run:
  py marketing_agent.py --demo
  py marketing_agent.py --task '{"action": "write_copy", "platform": "linkedin", "topic": "AI Agents"}'

Or import:  from marketing_agent import MarketingAgent; print(MarketingAgent().run(task))

Environment variables:
  TAVILY_API_KEY, FIRECRAWL_API_KEY              search
  OPENAI_API_KEY, LLM_MODEL, LLM_BASE_URL        LLM (OpenAI-compatible, e.g. Groq)
  DATABASE_URL                                    postgresql://user:pass@host:5432/db
  REDIS_URL                                       optional, batch cache only
   HF_API_KEY, IMAGE_MODEL                            image generation (default: Hugging Face / SDXL)
  SLACK_WEBHOOK_URL                               optional, review notifications

Third-party deps (everything else is stdlib):
  pip install psycopg2-binary redis
"""

import http.client
import json
import os
import sys
import time
from urllib.parse import quote

from agent_base import BaseAgent, Json, DEPARTMENT_LOGS_SQL

SCHEMA_SQL = DEPARTMENT_LOGS_SQL + """
CREATE TABLE IF NOT EXISTS reports (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  content JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS files (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS brand_voice (
  voice TEXT PRIMARY KEY,
  example TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

ALLOWED_PLATFORMS = {"linkedin", "x", "twitter", "instagram", "facebook",
                     "reddit", "medium", "blog"}

# ponytail: brand voice is stored as plain examples in Postgres; swap in a vector
# DB (Qdrant/Pinecone) only when fuzzy semantic tone-matching is actually needed.
ACTIONS = {
    "trend_research", "trend_summary", "content_strategy", "write_copy",
    "generate_image", "create_campaign", "set_brand_voice", "get_brand_voice",
    "social_plan", "analytics_summary", "competitor_analysis", "store_asset",
    "memory", "task_create", "task_list", "task_update", "task_close",
}


class MarketingAgent(BaseAgent):
    DEPARTMENT = "marketing"
    SCHEMA_SQL = SCHEMA_SQL

    # ---------- 1/10. trend research & summarization ----------

    def trend_research(self, topic, n=8):
        return {"topic": topic, "trends": self.tavily_search(str(topic), n)}

    def trend_summary(self, topic, n=8):
        trends = self.tavily_search(str(topic), n)
        digest = "\n".join(f"- {t['title']}: {t['snippet']}" for t in trends)
        text = self._chat(
            "Summarize market research into actionable insights. Output: a bullet list of "
            "the top trends and one 'Recommendation:' line. Plain text, no JSON.",
            f"Topic: {topic}\nResearch:\n{digest}", temperature=0.7)
        self.store_report(f"{topic} trend summary", "trend_summary",
                          {"topic": topic, "trends": [t["title"] for t in trends],
                           "summary": text})
        return {"trend_summary": text,
                "trends": [t["title"] for t in trends],
                "report_saved": True}

    # ---------- 2. content strategy ----------

    def content_strategy(self, topic, days=7, goal=""):
        j = self._chat_json(
            "You are a content strategist. Create a content calendar. Return ONLY valid JSON "
            '{"theme": "...", "calendar": [{"day": 1, "platform": "...", "post": "..."}]}.',
            f"Topic: {topic}\nDays: {days}\nGoal: {goal}", temperature=0.7)
        self.store_report(f"{topic} content strategy", "strategy", j)
        return {"strategy": j, "human_approval_required": False}

    # ---------- 3. copywriting ----------

    def write_copy(self, platform, topic, tone=None, audience=""):
        if platform.lower() not in ALLOWED_PLATFORMS:
            raise ValueError(f"unsupported platform: {platform}")
        j = self._chat_json(
            "You are an expert social/marketing copywriter. "
            f"{self._voice_block(tone)}"
            'Return ONLY valid JSON {"text": "...", "hashtags": [...], "cta": "..."}.',
            f"Platform: {platform}\nTopic: {topic}\nAudience: {audience}", temperature=0.8)
        post = {"platform": platform, "topic": topic, **j}
        self._log("write_copy", post)
        return {"posts": [post], "human_approval_required": False}

    # ---------- 4. image generation ----------

    def generate_image(self, prompt):
        key = os.getenv("HF_API_KEY")
        if not key:
            raise RuntimeError("HF_API_KEY not set")
        model = os.getenv("IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium-diffusers")
        conn = http.client.HTTPSConnection("router.huggingface.co", timeout=120)
        conn.request("POST", "/hf-inference/models/" + quote(model, safe="/"),
                     body=json.dumps({"inputs": prompt}).encode(),
                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status != 200:
            raise RuntimeError(f"image generation -> HTTP {resp.status}: {raw[:200].decode(errors='replace')}")
        ext = (resp.getheader("Content-Type") or "image/png").split("/")[-1] or "png"
        outdir = os.getenv("ASSET_DIR", os.path.join(os.path.dirname(__file__), "media"))
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{int(time.time())}.{ext}")
        with open(path, "wb") as f:
            f.write(raw)
        self.store_asset(prompt[:50], "image", path)
        return {"prompt": prompt, "image_file": path, "bytes": len(raw)}

    # ---------- 5. campaign creation ----------

    def create_campaign(self, name, goal, product, channels=None):
        channels = channels or ["linkedin", "blog"]
        j = self._chat_json(
            "You are a growth marketing strategist. "
            f"{self._voice_block(None)}"
            "Design a complete campaign. Return ONLY valid JSON with keys: objectives (list), "
            "target_audience (str), messaging (str), visual_assets (list), "
            'schedule (list of {"day": "...", "platform": "...", "post": "..."}), cta (str).',
            f"Campaign: {name}\nGoal: {goal}\nProduct: {product}\nChannels: {', '.join(channels)}",
            temperature=0.7)
        report = {"name": name, "goal": goal, "product": product, **j}
        prev = self.recent_reports("campaign")
        self.store_report(name, "campaign", report)
        self.notify_slack(f"Marketing campaign '{name}' is ready for human review.")
        images = []
        if os.getenv("HF_API_KEY") and j.get("visual_assets"):
            try:
                images.append(self.generate_image(f"{name}: {j['visual_assets'][0]}"))
            except Exception as e:
                images.append({"error": str(e)})
        return {"campaign": report, "images": images,
                "previous_campaigns": prev, "human_approval_required": True}

    # ---------- 6. brand voice management ----------

    def set_brand_voice(self, voice, example):
        if not voice or not example:
            raise ValueError("voice and example required")
        with self._db().cursor() as cur:
            cur.execute("""
                INSERT INTO brand_voice (voice, example) VALUES (%s, %s)
                ON CONFLICT (voice) DO UPDATE SET example = EXCLUDED.example
            """, (voice, example))
        return {"brand_voice": voice, "example": example}

    def get_brand_voice(self, voice=None):
        with self._db().cursor() as cur:
            if voice:
                cur.execute("SELECT voice, example FROM brand_voice WHERE voice = %s", (voice,))
            else:
                cur.execute("SELECT voice, example FROM brand_voice ORDER BY created_at")
            return [{"voice": r[0], "example": r[1]} for r in cur.fetchall()]

    def _voice_block(self, tone=None):
        try:
            with self._db().cursor() as cur:
                if tone:
                    cur.execute("SELECT voice, example FROM brand_voice WHERE voice = %s", (tone,))
                else:
                    cur.execute("SELECT voice, example FROM brand_voice ORDER BY created_at LIMIT 1")
                row = cur.fetchone()
        except Exception:
            row = None
        if row:
            return f"Brand voice: {row[0]}. Style example:\n{row[1]}\n"
        return ""

    # ---------- 7. social media planning ----------

    def social_plan(self, topic, days=7):
        j = self._chat_json(
            "You are a social media manager. Build a weekly posting plan across LinkedIn, "
            "X, Instagram, Facebook, Reddit, Medium and the Company Blog. Return ONLY valid "
            'JSON {"plan": [{"day": "Monday", "platform": "...", "content": "..."}]}.',
            f"Topic: {topic}\nDays: {days}", temperature=0.7)
        self.store_report(f"{topic} social plan", "social_plan", j)
        return {"plan": j.get("plan", []), "human_approval_required": True}

    # ---------- 8. analytics summary ----------

    def analytics_summary(self, metrics):
        text = self._chat(
            "Summarize this marketing analytics data. Cover: top-performing posts, engagement "
            "trends, reach summary, best posting time, CTR summary. Plain text with headings.",
            json.dumps(metrics or {}, indent=2)[:8000], temperature=0.4)
        self.store_report("analytics summary", "analytics", {"metrics": metrics, "summary": text})
        return {"summary": text, "human_approval_required": False}

    # ---------- 9. competitor analysis ----------

    def competitor_analysis(self, competitors):
        if not competitors:
            raise ValueError("competitors list required")
        data = {}
        for c in competitors[:5]:
            data[c] = [{"title": t["title"], "snippet": t["snippet"]}
                       for t in self.tavily_search(f"{c} social media marketing content", 3)]
        text = self._chat(
            "Compare these competitors' marketing: posting frequency, messaging, branding, "
            "social engagement, product launches, ad strategies. Plain text with headings.",
            json.dumps(data, indent=2)[:8000], temperature=0.5)
        self.store_report("competitor analysis", "competitor",
                          {"competitors": competitors, "summary": text})
        return {"analysis": text, "human_approval_required": False}

    # ---------- 11. asset management ----------

    def store_report(self, title, kind, content):
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO reports (title, kind, content) VALUES (%s, %s, %s) "
                        "RETURNING id", (title, kind, Json(content or {})))
            rid = cur.fetchone()[0]
        self._log(kind, {"title": title, "report_id": rid})
        return rid

    def store_asset(self, name, kind, url):
        with self._db().cursor() as cur:
            cur.execute("INSERT INTO files (name, kind, url) VALUES (%s, %s, %s) RETURNING id",
                        (name, kind, url))
            fid = cur.fetchone()[0]
        return {"asset_id": fid, "name": name, "kind": kind, "url": url}

    # ---------- 12. department memory ----------

    def recent_reports(self, kind=None, limit=5):
        with self._db().cursor() as cur:
            if kind:
                cur.execute("SELECT title, kind, created_at FROM reports WHERE kind = %s "
                            "ORDER BY id DESC LIMIT %s", (kind, limit))
            else:
                cur.execute("SELECT title, kind, created_at FROM reports "
                            "ORDER BY id DESC LIMIT %s", (limit,))
            return [{"title": r[0], "kind": r[1], "created_at": r[2].isoformat()}
                    for r in cur.fetchall()]

    # ---------- 14. structured output for the CEO Agent ----------

    def run(self, task):
        if (task or {}).get("action") not in ACTIONS:
            return {"status": "error", "action": (task or {}).get("action"),
                    "error": f"unknown action: {task.get('action')}"}
        return self.handle(task, {
            "trend_research": lambda: self.trend_research(task.get("topic"), task.get("n", 8)),
            "trend_summary": lambda: self.trend_summary(task.get("topic"), task.get("n", 8)),
            "content_strategy": lambda: self.content_strategy(
                task.get("topic"), task.get("days", 7), task.get("goal", "")),
            "write_copy": lambda: self.write_copy(
                task.get("platform", "linkedin"), task.get("topic"), task.get("tone"),
                task.get("audience", "")),
            "generate_image": lambda: self.generate_image(task.get("prompt")),
            "create_campaign": lambda: self.create_campaign(
                task.get("name"), task.get("goal"), task.get("product"), task.get("channels")),
            "set_brand_voice": lambda: self.set_brand_voice(
                task.get("voice"), task.get("example")),
            "get_brand_voice": lambda: self.get_brand_voice(task.get("voice")),
            "social_plan": lambda: self.social_plan(task.get("topic"), task.get("days", 7)),
            "analytics_summary": lambda: self.analytics_summary(task.get("metrics") or {}),
            "competitor_analysis": lambda: self.competitor_analysis(
                task.get("competitors") or []),
            "store_asset": lambda: self.store_asset(
                task.get("name"), task.get("kind", "asset"), task.get("url")),
            "memory": lambda: self.recent_reports(task.get("kind"), task.get("limit", 5)),
        })


def demo():
    a = MarketingAgent()
    expected = {"trend_research", "trend_summary", "content_strategy", "write_copy",
                "generate_image", "create_campaign", "set_brand_voice", "get_brand_voice",
                "social_plan", "analytics_summary", "competitor_analysis", "store_asset",
                "memory", "task_create", "task_list", "task_update", "task_close"}
    assert ACTIONS == expected, ACTIONS - expected
    assert a.run({"action": "nope"})["status"] == "error"
    assert a.run({"action": "write_copy", "platform": "tiktok", "topic": "x"})["status"] == "error"
    assert isinstance(a._voice_block(), str)
    print("MarketingAgent demo OK")


def _main():
    if "--demo" in sys.argv:
        demo()
        return
    if "--task" in sys.argv:
        i = sys.argv.index("--task")
        task = json.loads(sys.argv[i + 1])
        print(json.dumps(MarketingAgent().run(task), indent=2, default=str))
        return
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    _main()
