"""Shared plumbing for department agents: HTTP, LLM, PostgreSQL, Redis, Tavily.

Reused by sales_agent.py and marketing_agent.py so the services are wired once.
"""

import json
import os
import re
from urllib.parse import urlparse, urlencode
from urllib import request as urllib_request
from urllib import error as urllib_error

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    psycopg2 = None

try:
    import redis as redis_mod
except ImportError:
    redis_mod = None

DEPARTMENT_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS department_logs (
  id BIGSERIAL PRIMARY KEY,
  department TEXT NOT NULL DEFAULT 'unknown',
  action TEXT NOT NULL,
  detail JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _domain_of(url):
    if not url:
        return None
    u = url if "://" in url else "//" + url
    try:
        host = urlparse(u).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


class BaseAgent:
    DEPARTMENT = "unknown"
    SCHEMA_SQL = DEPARTMENT_LOGS_SQL

    def __init__(self):
        self.conn = None
        self._redis = None

    # ---------- HTTP ----------

    def _http_json(self, method, url, payload=None, headers=None, timeout=60, form=False):
        req = urllib_request.Request(url, method=method, headers=headers or {})
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        data = None
        if payload is not None:
            data = urlencode(payload).encode() if form else json.dumps(payload).encode()
            req.add_header("Content-Type",
                           "application/x-www-form-urlencoded" if form else "application/json")
        try:
            with urllib_request.urlopen(req, data=data, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw else None
        except urllib_error.HTTPError as e:
            raise RuntimeError(f"{url} -> HTTP {e.code}: "
                               f"{e.read().decode('utf-8', 'replace')[:500]}") from e
        except urllib_error.URLError as e:
            raise RuntimeError(f"{url} -> {e.reason}") from e

    # ---------- PostgreSQL ----------

    def _db(self):
        if self.conn is None:
            if psycopg2 is None:
                raise RuntimeError("psycopg2 not installed: pip install psycopg2-binary")
            dsn = os.getenv("DATABASE_URL")
            if not dsn:
                raise RuntimeError("DATABASE_URL not set")
            self.conn = psycopg2.connect(dsn)
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                cur.execute(self.SCHEMA_SQL)
        return self.conn

    def _log(self, action, detail):
        try:
            with self._db().cursor() as cur:
                cur.execute(
                    "INSERT INTO department_logs (department, action, detail) "
                    "VALUES (%s, %s, %s)",
                    (self.DEPARTMENT, action, Json(detail or {})))
        except Exception:
            pass

    # ---------- Redis ----------

    def _r(self):
        if self._redis is None and redis_mod is not None and os.getenv("REDIS_URL"):
            self._redis = redis_mod.from_url(os.getenv("REDIS_URL"))
        return self._redis

    def active_batch(self, batch_id):
        if not batch_id or self._r() is None:
            return None
        try:
            raw = self._r().get(f"{self.DEPARTMENT}:batch:{batch_id}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def mark_batch(self, batch_id, payload):
        if not batch_id or self._r() is None:
            return
        try:
            self._r().set(f"{self.DEPARTMENT}:batch:{batch_id}", json.dumps(payload), ex=86400)
        except Exception:
            pass

    # ---------- LLM (OpenAI-compatible, e.g. Groq) ----------

    def _chat(self, system, user, temperature=0.4):
        key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        data = self._http_json(
            "POST", f"{base}/chat/completions",
            {"model": model, "temperature": temperature, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}]},
            {"Authorization": f"Bearer {key}"})
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected LLM response: {data}") from None

    def _chat_json(self, system, user, temperature=0.4):
        text = self._chat(system, user, temperature=temperature)
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"text": text}

    # ---------- Tavily search ----------

    def tavily_search(self, query, n=10):
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY not set")
        data = self._http_json(
            "POST", "https://api.tavily.com/search",
            {"api_key": key, "query": query, "max_results": n, "search_depth": "basic"})
        out = []
        for r in (data or {}).get("results", []) or []:
            url = (r.get("url") or "").strip()
            if url:
                out.append({"title": r.get("title"), "url": url,
                            "domain": _domain_of(url),
                            "snippet": (r.get("content") or "")[:300]})
        return out

    # ---------- structured results ----------

    def handle(self, task, handlers):
        try:
            task = task or {}
            action = task.get("action")
            fn = handlers.get(action)
            if not fn:
                raise ValueError(f"unknown action: {action}")
            result = fn()
            out = {"status": "ok", "action": action}
            if isinstance(result, dict):
                out.update(result)
            else:
                out["result"] = result
            return out
        except Exception as e:
            return {"status": "error", "action": (task or {}).get("action"),
                    "error": str(e)}
