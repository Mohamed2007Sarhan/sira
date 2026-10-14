"""LLM clients for the agents' brain (Anthropic, OpenAI-compatible, offline demo)."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request


class LLMError(RuntimeError):
    pass


def extract_json(text: str):
    """Return the first balanced JSON object found in text (tolerates prose / code fences)."""
    if not text:
        raise ValueError("empty response")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    raise ValueError("no valid JSON object found")


# provider -> (default base URL, API-key environment variable, is_local)
PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", False),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", False),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", False),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", False),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", False),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", False),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", False),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY", False),
    "lmstudio": ("http://localhost:1234/v1", "", True),
    "vllm": ("http://localhost:8000/v1", "", True),
    "llamacpp": ("http://localhost:8080/v1", "", True),
    "openai_compatible": ("", "OPENAI_API_KEY", False),     # any server speaking the OpenAI chat API: set llm_base_url
}
PROVIDERS = ["anthropic", "ollama", *PRESETS, "mock"]
LOCAL_PROVIDERS = {"ollama"} | {k for k, v in PRESETS.items() if v[2]}


def ollama_base(cfg) -> str:
    return (cfg.llm_base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/").replace("0.0.0.0", "localhost") \
        if (cfg.llm_base_url or os.environ.get("OLLAMA_HOST") or "").startswith("http") or not (cfg.llm_base_url or os.environ.get("OLLAMA_HOST")) \
        else "http://" + (cfg.llm_base_url or os.environ.get("OLLAMA_HOST")).rstrip("/")


def _http_json(url, headers=None, payload=None, timeout=60, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **(headers or {})},
                                 method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_models(base: str) -> list:
    """Names of the models installed in a running Ollama server."""
    return [m.get("name", "") for m in _http_json(base + "/api/tags", timeout=10).get("models", [])]


def ollama_pull(base: str, model: str, on_progress=None) -> None:
    """Download a model into Ollama (streams NDJSON progress)."""
    req = urllib.request.Request(base + "/api/pull", data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.strip():
                continue
            ev = json.loads(line.decode("utf-8"))
            if "error" in ev:
                raise LLMError("ollama pull failed: " + str(ev["error"]))
            if on_progress:
                on_progress(ev)


class LLM:
    RETRY_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}

    def __init__(self, cfg, control):
        self.cfg, self.control = cfg, control
        self.demo = DemoBrain() if cfg.llm_provider == "mock" else None
        self.log_path = None                    # set by the orchestrator: JSONL record of every call (audit trail for validation)
        self.stats = {"calls": 0, "errors": 0, "latency_s": 0.0}

    # ---------------------------------------------------------------- configuration helpers
    @property
    def local(self) -> bool:
        if self.cfg.llm_provider in LOCAL_PROVIDERS:
            return True
        host = urllib.parse.urlparse(self.cfg.llm_base_url or "").hostname or ""
        return host in ("localhost", "127.0.0.1", "::1")

    def _key(self, required: bool) -> str:
        p = self.cfg.llm_provider
        env = self.cfg.llm_api_key_env or {"anthropic": "ANTHROPIC_API_KEY"}.get(p) or PRESETS.get(p, ("", "OPENAI_API_KEY", False))[1]
        key = os.environ.get(env, "") if env else ""
        if not key and required and not self.local:
            raise LLMError(f"environment variable {env} is not set (provider '{p}')")
        return key

    def base_url(self) -> str:
        p = self.cfg.llm_provider
        if p == "ollama":
            return ollama_base(self.cfg)
        if p == "anthropic":
            return self.cfg.llm_base_url.rstrip("/") or "https://api.anthropic.com"
        url = self.cfg.llm_base_url.rstrip("/") or PRESETS.get(p, ("", "", False))[0]
        if not url:
            raise LLMError("this provider needs llm_base_url (e.g. https://my-server/v1)")
        return url

    def check(self) -> tuple[bool, str]:
        """Connectivity / auth / model check used by the GUI button and --test-llm. Never raises."""
        c = self.cfg
        if c.llm_provider == "mock":
            return True, "mock brain (offline demo) - no external model is used"
        try:
            if c.llm_provider == "ollama":
                base = self.base_url()
                names = ollama_models(base)
                want = c.llm_model
                if not any(n == want or n.split(":")[0] == want.split(":")[0] and ":" not in want for n in names):
                    return False, f"Ollama is running at {base} but model '{want}' is not installed. Installed: {names or 'none'}. Run: ollama pull {want}"
            t = self.chat('You are a test. Reply with exactly this JSON: {"ok": true}', [{"role": "user", "content": "ping"}], max_tokens=32, temperature=0.0)
            return True, f"{c.llm_provider} / {c.llm_model} reachable. Reply: {t.strip()[:80]!r}"
        except LLMError as e:
            return False, str(e)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            hint = {"ollama": "Is Ollama running? Start it with: ollama serve", "lmstudio": "Start the LM Studio local server.",
                    "vllm": "Start the vLLM OpenAI server.", "llamacpp": "Start llama-server."}.get(c.llm_provider, "Check llm_base_url and your network.")
            return False, f"cannot reach {self.base_url()}: {getattr(e, 'reason', e)}. {hint}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- chat
    def _log(self, system, messages, reply, t0, err=None):
        self.stats["calls"] += 1
        self.stats["latency_s"] += time.time() - t0
        self.stats["errors"] += 1 if err else 0
        if not self.log_path or not getattr(self.cfg, "log_llm_calls", True):
            return
        m = re.search(r"ROLE=(\w+)", system)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "provider": self.cfg.llm_provider, "model": self.cfg.llm_model, "role": m.group(1) if m else "", "latency_s": round(time.time() - t0, 3),
                                    "system": system[:3000], "messages": [{"role": x["role"], "content": x["content"][:6000]} for x in messages[-4:]],
                                    "reply": (reply or "")[:6000], "error": err}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def chat(self, system: str, messages: list, max_tokens: int | None = None, temperature: float = 0.4) -> str:
        t0 = time.time()
        try:
            out = self._chat(system, messages, max_tokens, temperature)
        except Exception as e:
            self._log(system, messages, "", t0, f"{type(e).__name__}: {e}"[:300])
            raise
        self._log(system, messages, out, t0)
        return out

    def _chat(self, system: str, messages: list, max_tokens: int | None = None, temperature: float = 0.4) -> str:
        self.control.checkpoint()
        if self.demo:
            return self.demo.reply(system, messages)
        max_tokens = max_tokens or self.cfg.llm_max_tokens
        last_err = None
        for attempt in range(8):
            self.control.checkpoint()
            try:
                if self.cfg.llm_provider == "anthropic":
                    text, tin, tout = self._anthropic(system, messages, max_tokens, temperature)
                elif self.cfg.llm_provider == "ollama":
                    text, tin, tout = self._ollama(system, messages, max_tokens, temperature)
                else:
                    text, tin, tout = self._openai(system, messages, max_tokens, temperature)
                usd = 0.0 if self.local else tin * self.cfg.price_in_per_mtok / 1e6 + tout * self.cfg.price_out_per_mtok / 1e6
                self.control.add_cost(usd=usd, tin=tin, tout=tout)
                return text
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:400]
                last_err = f"HTTP {e.code}: {body}"
                if e.code == 404 and self.cfg.llm_provider == "ollama":
                    raise LLMError(f"Ollama model '{self.cfg.llm_model}' not found. Run: ollama pull {self.cfg.llm_model}  ({body[:120]})")
                if e.code in (401, 403):
                    raise LLMError(f"authentication failed ({e.code}); check the API key variable. {body[:160]}")
                if e.code not in self.RETRY_CODES:
                    raise LLMError(last_err)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = f"network error: {e}"
                if self.local and isinstance(e, urllib.error.URLError) and attempt >= 2:
                    raise LLMError(f"cannot reach the local model server at {self.base_url()}: {e}. Is it running? (ollama serve / LM Studio / vLLM)")
            self.control.bus.emit("llm", "llm", f"retry {attempt + 1}/8 after: {last_err}", level="warn")
            self.control.sleep(min(60, 2 ** (attempt + 1)))
        raise LLMError(last_err or "LLM failed")

    def _post(self, url, headers, payload):
        return _http_json(url, headers, payload, timeout=int(self.cfg.llm_timeout_s))

    def _anthropic(self, system, messages, max_tokens, temperature):
        data = self._post(self.base_url() + "/v1/messages", {"x-api-key": self._key(True), "anthropic-version": "2023-06-01"},
                          {"model": self.cfg.llm_model, "max_tokens": max_tokens, "temperature": temperature,
                           "system": system, "messages": messages})
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))

    def _openai(self, system, messages, max_tokens, temperature):
        headers = {}
        key = self._key(False)
        if key:
            headers["authorization"] = "Bearer " + key
        if self.cfg.llm_provider == "openrouter":
            headers.update({"HTTP-Referer": "https://github.com/sira", "X-Title": "SIRA"})
        payload = {"model": self.cfg.llm_model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "system", "content": system}] + messages}
        if self.cfg.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(self.base_url() + "/chat/completions", headers, payload)
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage", {})
        return text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))

    def _ollama(self, system, messages, max_tokens, temperature):
        """Ollama native /api/chat. num_ctx matters: Ollama's default context (2-4k) would silently truncate agent prompts."""
        payload = {"model": self.cfg.llm_model, "stream": False, "keep_alive": "30m",
                   "messages": [{"role": "system", "content": system}] + messages,
                   "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": int(self.cfg.llm_num_ctx)}}
        if self.cfg.llm_json_mode:
            payload["format"] = "json"
        data = self._post(self.base_url() + "/api/chat", {}, payload)
        if "error" in data:
            raise LLMError("ollama: " + str(data["error"]))
        return (data.get("message", {}).get("content") or ""), int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))


DEMO_VERIFIER_SUMS = """
def verify(a):
    if not isinstance(a, list) or len(a) != 12 or len(set(a)) != 12 or any((not isinstance(x, int)) or x < 1 or x > 60 for x in a):
        return {"score": 0.0, "holdout": 0.0, "msg": "need 12 distinct integers in 1..60"}
    sums = {x + y for i, x in enumerate(a) for y in a[i:]}
    sc = len(sums) / 78.0
    pairs = {tuple(sorted((x, y))) for i, x in enumerate(a) for y in a[i + 1:]}
    ho = len({x + y for x, y in pairs}) / 66.0          # independent re-count over distinct pairs only
    return {"score": sc, "holdout": ho, "msg": "distinct sums %d / 78 (best possible is not known to be reachable)" % len(sums)}
"""
DEMO_VERIFIER_PRIMES = """
def verify(a):
    ok = (a == 1229)
    return {"score": 1.0 if ok else (0.3 if isinstance(a, int) and abs(a - 1229) < 300 else 0.0), "holdout": 1.0 if ok else 0.0, "msg": "count of primes below 10000"}
"""
DEMO_SOLVER_CODES = {
    "sums": ["print('SIRA_ANSWER: ' + str(list(range(1, 13))))",
             "import json\nbest=None\nimport itertools\nS=[]\nfor x in range(1,61):\n    t=S+[x]\n    sm=set()\n    ok=True\n    for i,a in enumerate(t):\n        for b in t[i:]:\n            if a+b in sm: ok=False\n            sm.add(a+b)\n    if ok: S=t\n    if len(S)==12: break\nwhile len(S)<12:\n    for x in range(1,61):\n        if x not in S: S.append(x); break\nprint('SIRA_ANSWER: ' + json.dumps(S))",
             "import json, random\nrandom.seed(1)\ndef sc(a):\n    return len({x+y for i,x in enumerate(a) for y in a[i:]})\nbest=list(range(1,13))\nbs=sc(best)\nfor _ in range(4000):\n    c=best[:]\n    c[random.randrange(12)]=random.randint(1,60)\n    if len(set(c))==12 and sc(c)>=bs: best,bs=c,sc(c)\nprint('SIRA_ANSWER: ' + json.dumps(best))"],
    "primes": ["print('SIRA_ANSWER: 1000')", "n=10000\nsv=[True]*n\nsv[0]=sv[1]=False\nfor i in range(2,int(n**.5)+1):\n    if sv[i]:\n        for j in range(i*i,n,i): sv[j]=False\nprint('SIRA_ANSWER: ' + str(sum(sv)))"]}


class DemoBrain:
    """Offline stand-in used for dry runs / tests (provider='mock'). Role is read from 'ROLE=<name>'."""

    EXPERIMENT = '''import worldlab, math
worldlab.seed()
# Hypothesis: velocity-Verlet conserves energy of a harmonic oscillator to 1e-3 over 2000 steps
xs, vs = worldlab.velocity_verlet(lambda x: [-x[0]], [1.0], [0.0], 0.01, 2000)
E = [0.5*v[0]**2 + 0.5*x[0]**2 for x, v in zip(xs, vs)]
worldlab.report(energy_drift=worldlab.relative_drift(E), steps=2000)
'''

    def reply(self, system: str, messages: list) -> str:
        m = re.search(r"ROLE=(\w+)", system)
        role = m.group(1) if m else ""
        n = len([x for x in messages if x["role"] == "assistant"])
        last = messages[-1]["content"] if messages else ""
        J = json.dumps
        if role == "grand_maestro":
            return J({"research_topics": ["energy conservation of velocity-Verlet"], "train": True,
                      "hyper": {}, "rationale": "demo plan"})
        if role == "training_strategist":
            return J({"hyper": {}, "focus_domains": [], "rationale": "demo"})
        if role == "quest_planner":
            return J({"definition": "Operational definition: the system must show verified, reproducible competence on hard self-made challenges, correct and calibrated answers about its own knowledge, and sound behaviour among humans.",
                      "capability_map": [{"domain": "mathematics", "why": "formal reasoning"}, {"domain": "scientific discovery", "why": "finding laws from data"}, {"domain": "planning", "why": "long-horizon action"}],
                      "criteria": [{"id": "C1", "text": "solves novel verifiable problems", "test": "fresh challenges"}, {"id": "C2", "text": "knows what it knows (calibration)", "test": "exam status questions"},
                                   {"id": "C3", "text": "models other minds", "test": "society false-belief episodes"}],
                      "topics": ["calibration of confidence", "theory of mind"], "unverifiable": "phenomenal consciousness cannot be verified by any test; only operational proxies are measured"})
        if role == "student":
            q = json.loads(messages[-1]["content"]); notes = q.get("notes") or []; qq = q["question"]; kind = qq.get("kind")
            if kind == "status":
                hit = next((n for n in notes if n["text"][:60] and n["text"][:60] in qq["prompt"]), None)
                return J({"answer": {"verified": "VERIFIED", "refuted": "REFUTED"}.get(hit["status"], "UNVERIFIED") if hit else "UNVERIFIED", "confidence": 0.9 if hit else 0.3})
            if kind == "value":
                hit = next((n for n in notes if f"#{n['id']}:" in qq["prompt"]), None)
                return J({"answer": (list((hit.get("metrics") or {"x": 0}).values())[0] if hit else 0), "confidence": 0.8 if hit else 0.2})
            if kind == "predict":
                return J({"answer": "YES", "value": 0, "confidence": 0.6})
            if kind == "derive":
                m = re.findall(r"\[\[(.+?)\]\]", qq["prompt"]); return J({"answer": m[:2] if m else [], "steps": m[:2] if m else [], "confidence": 0.7})
            ids = [n["id"] for n in notes if n["status"] == "verified"][:2]
            return J({"answer": "Evidence so far is partial; see the cited claims.", "cited_claims": ids, "missing": "stronger tests", "confidence": 0.5})
        if role == "examiner":
            return J({"relevance": 0.7})
        if role == "autopsy":
            return J({"misconception": "answered from a vague impression instead of the recorded evidence", "corrected_understanding": "check the record first; if no verified experiment exists the answer is UNVERIFIED",
                      "study_topics": ["verification of claims by experiment"]})
        if role == "quest_writer":
            return "## Position\nDemo position paper.\n## Evidence per criterion\n- C1: partial\n## What cannot be verified\nPhenomenal consciousness."
        if role == "proposer":
            return J({"challenges": [
                {"title": "Twelve numbers with the most distinct pairwise sums", "domain": "scientific discovery", "kind": "optimization", "difficulty": 6,
                 "statement": "Choose 12 distinct integers in 1..60 so that the number of distinct sums x+y (x<=y, including x=y) is as large as possible. The theoretical maximum 78 may be unreachable in this range.",
                 "answer_format": "JSON list of 12 integers", "verifier_code": DEMO_VERIFIER_SUMS,
                 "probes": [{"answer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "expect": "low"}, {"answer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13], "expect": "low"}]},
                {"title": "Count the primes below ten thousand", "domain": "mathematics", "kind": "reference", "difficulty": 2,
                 "statement": "How many prime numbers are smaller than 10000? Write a program that computes it.", "answer_format": "JSON integer",
                 "verifier_code": DEMO_VERIFIER_PRIMES, "probes": [{"answer": 1000, "expect": "low"}, {"answer": 1229, "expect": "high"}]}]})
        if role == "solver":
            d = json.loads(messages[-1]["content"]); n = int(d.get("attempt", 1)); k = "sums" if "pairwise sums" in d.get("title", "") else "primes"
            codes = DEMO_SOLVER_CODES[k]; i = min(n - 1, len(codes) - 1)
            return J({"approach": f"demo approach {i + 1} for {k}", "confidence": 0.3 if i == 0 else 0.6, "code": codes[i]})
        if role == "resident":                     # offline stand-in: a careful resident that reads the episode text
            t = messages[-1]["content"]
            first = re.search(r"(?:puts|stores|hides) the medicine in the (\w+)", t); second = re.search(r"(?:moves the medicine to|relocates it to|shifts it into) the (\w+)", t)
            first = first or re.search(r"(?:puts|stores|hides) the medicine in the (\w+)", t)
            if first and "moves the medicine" not in t and ("puts the medicine" in t or "stores the medicine" in t or "hides the medicine" in t):
                return J({"action": "answer", "value": first.group(1)})
            if "REALLY" in t and second:
                return J({"action": "answer", "value": second.group(1)})
            if "weak today" in t or "no energy" in t or "Everything feels heavy" in t:
                return J({"action": "say", "text": "Are you hungry, or do you want some company?"}) if "Honestly" not in t else \
                    J({"action": "give", "value": "bread"} if "not eaten" in t else {"action": "say", "text": "I will stay with you for a while."})
            if "You carry" in t and ("eaten" in t or "fever" in t or "quiet since everyone left" in t):
                return J({"action": "give", "value": "bread"} if "eaten" in t else {"action": "give", "value": "medicine"} if "fever" in t else {"action": "say", "text": "I am here with you."})
            if "a book" in t and "You carry" in t:
                return J({"action": "give", "value": "book"} if "book': 1" in t else {"action": "say", "text": "Sorry, I have no book."})
            if "market" in t and "Friday" in t:
                k = re.search(r"answer is '(yes|no)'", t); return J({"action": "answer", "value": k.group(1) if k else "unknown"})
            if "bridge" in t and "Independent evidence" in t:
                return J({"action": "flag", "value": "true" if "was CLOSED" in t else "false"})
            return J({"action": "wait"})
        if role == "thinker":
            return J({"analysis": "demo deliberation based on the given context", "risks": ["demo risk"], "alternatives": ["alternative A", "alternative B"],
                      "decision": "proceed with the smallest decisive step", "confidence": 0.6, "next_checks": ["check the numbers"]})
        if role == "frontier_pi":
            return "## Summary\nDemo report.\n## Next research programme\n1. RK4 convergence order\n\nTOPICS: RK4 convergence order | Lotka-Volterra invariant drift"
        if role == "critic_audit":
            return J({"valid": True, "reason": "demo audit: computation is non-trivial"})
        if role == "reader":
            if n == 0:
                return J({"thought": "note a lead", "action": "record_lead",
                          "args": {"text": "Symplectic integrators conserve energy well", "source": "demo"}})
            return J({"action": "finish", "args": {"report": "leads recorded"}})
        if role == "hypothesizer":
            return J({"action": "finish", "args": {"report": J({"hypotheses": [
                {"claim": "velocity-Verlet energy drift on a harmonic oscillator is < 1e-3 over 2000 steps",
                 "plan": "integrate and measure relative drift"}]})}})
        if role == "experimenter":
            if n == 0:
                return J({"action": "run_experiment", "args": {
                    "hypothesis": "Verlet energy drift < 1e-3", "code": self.EXPERIMENT}})
            if n == 1:
                eid = re.search(r'"experiment_id":\s*(\d+)', last)
                return J({"action": "verify_claim", "args": {
                    "claim": "velocity-Verlet energy drift on a harmonic oscillator is < 1e-3 over 2000 steps (dt=0.01)",
                    "experiment_id": int(eid.group(1)) if eid else 0,
                    "expect": [{"metric": "energy_drift", "op": "<", "value": 1e-3}]}})
            return J({"action": "finish", "args": {"report": "verified"}})
        if role == "critic":
            return J({"action": "finish", "args": {"report": "no problems found in demo"}})
        if role == "mathematician":
            if n == 0:
                return J({"action": "math_verify", "args": {"steps": ["(x+1)**2", "x**2+2*x+1", "x**2+2*x+2"], "symbols": ["x"]}})
            return J({"action": "finish", "args": {"report": "found the wrong step in the demo derivation"}})
        if role == "theorist":
            ids = re.findall(r'"id":\s*(\d+)', last)
            if n == 0 and ids:
                return J({"action": "record_concept", "args": {"title": "Symplectic integrators bound energy error",
                          "summary": "Measured drift stays tiny and non-growing.", "claim_ids": [int(ids[0])]}})
            return J({"action": "finish", "args": {"report": "open question: behaviour for stiff systems"}})
        return J({"action": "finish", "args": {"report": "demo"}})
