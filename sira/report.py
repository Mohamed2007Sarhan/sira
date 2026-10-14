"""Run reports + read-only dashboard. Pure standard library. Reads <run>/sira.db strictly read-only; every value is HTML-escaped."""
from __future__ import annotations

import hmac
import html
import json
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _ro(db_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _q(c, sql, args=()):
    try:
        return [dict(r) for r in c.execute(sql, args).fetchall()]
    except sqlite3.Error:
        return []


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def summarize(run_dir) -> dict:
    run = Path(run_dir)
    db = run / "sira.db"
    if not db.exists():
        raise FileNotFoundError(f"no sira.db in {run}")
    cfg, fin = _read_json(run / "config.json"), _read_json(run / "final_report.json")
    c = _ro(db)
    try:
        gens = _q(c, "SELECT id,parent,method,score,accepted,created FROM generations ORDER BY id")
        exp = (_q(c, "SELECT COUNT(*) n, COALESCE(SUM(timed_out),0) t, COALESCE(SUM(CASE WHEN exit_code=0 THEN 1 ELSE 0 END),0) ok FROM experiments") or [{}])[0]
        rew = (_q(c, "SELECT COUNT(*) n, COALESCE(SUM(reward),0) r, COALESCE(SUM(penalty),0) p FROM rewards") or [{}])[0]
        exm = (_q(c, "SELECT COUNT(*) n, AVG(score) a FROM exams") or [{}])[0]
        counts = {t: (_q(c, f"SELECT COUNT(*) n FROM {t}") or [{"n": 0}])[0]["n"] for t in ("lessons", "mistakes", "thoughts", "tools", "worlds", "concepts", "attempts")}
        out = {
            "run": run.name, "path": str(run), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {k: cfg.get(k) for k in ("goal_text", "quest_goal", "mode", "base_model", "llm_provider", "llm_model", "budget_usd", "budget_gpu_hours", "budget_wall_hours", "target_score") if k in cfg},
            "stop_reason": fin.get("stop_reason"), "finished": bool(fin),
            "generations": gens,
            "best_score": max([g["score"] for g in gens if g["accepted"] and g["score"] is not None], default=None),
            "claims": {r["status"]: r["n"] for r in _q(c, "SELECT status, COUNT(*) n FROM claims GROUP BY status")},
            "experiments": {"total": exp.get("n", 0), "ok": exp.get("ok", 0), "timed_out": exp.get("t", 0)},
            "evals": _q(c, "SELECT id,label,model,total,n,created FROM evals ORDER BY id DESC LIMIT 15"),
            "rewards": {"events": rew.get("n", 0), "reward": rew.get("r", 0), "penalty": rew.get("p", 0)},
            "challenges": _q(c, "SELECT status, COUNT(*) n, ROUND(AVG(best_score),3) avg_best FROM challenges GROUP BY status"),
            "exams": {"n": exm.get("n", 0), "avg_score": exm.get("a")},
            "counts": counts,
        }
    finally:
        c.close()
    return out


def list_runs(root) -> list:
    root = Path(root)
    if (root / "sira.db").exists():
        dirs = [root]
    else:
        dirs = sorted((d for d in root.iterdir() if (d / "sira.db").exists()), key=lambda d: d.stat().st_mtime, reverse=True) if root.is_dir() else []
    rows = []
    for d in dirs:
        try:
            s = summarize(d)
            rows.append({"run": d.name, "state": ("finished: " + str(s["stop_reason"])) if s["finished"] else "running or interrupted", "mode": s["config"].get("mode"),
                         "best_score": s["best_score"], "generations": len(s["generations"]), "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "sira.db").stat().st_mtime))})
        except Exception as e:
            rows.append({"run": d.name, "state": f"unreadable: {type(e).__name__}", "mode": None, "best_score": None, "generations": 0, "updated": ""})
    return rows


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_markdown(s: dict) -> str:
    L = [f"# SIRA run report: {s['run']}", "", f"*generated {s['generated']}* - state: **{'finished: ' + str(s['stop_reason']) if s['finished'] else 'running or interrupted'}**", ""]
    L += [f"- {k}: {v}" for k, v in s["config"].items()]
    L += ["", f"**Best accepted score:** {_f(s['best_score'])}", "", "## Generations", "", "| id | parent | method | score | accepted |", "|---|---|---|---|---|"]
    L += [f"| {g['id']} | {g['parent']} | {g['method']} | {_f(g['score'])} | {'yes' if g['accepted'] else 'no'} |" for g in s["generations"]] or ["| - | - | - | - | - |"]
    e = s["experiments"]
    L += ["", "## Knowledge & experiments", "", f"- claims: {json.dumps(s['claims'])}", f"- experiments: {e['total']} total, {e['ok']} exit 0, {e['timed_out']} timed out",
          f"- rewards: {s['rewards']['events']} events, +{_f(s['rewards']['reward'], 2)} / -{_f(s['rewards']['penalty'], 2)}", f"- exams: {s['exams']['n']} (avg {_f(s['exams']['avg_score'])})",
          f"- challenges: {json.dumps(s['challenges'])}", f"- memory: {json.dumps(s['counts'])}", "", "## Latest evaluations", "", "| id | label | total | n |", "|---|---|---|---|"]
    L += [f"| {x['id']} | {x['label']} | {_f(x['total'])} | {x['n']} |" for x in s["evals"]] or ["| - | - | - | - |"]
    return "\n".join(L) + "\n"


_CSS = ("*{box-sizing:border-box}:root{--bg:#fff;--fg:#1b1f24;--mut:#6a737d;--card:#f6f8fa;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--ac:#0969da}"
        "@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--card:#161b22;--bd:#30363d;--ok:#3fb950;--bad:#f85149;--ac:#58a6ff}}"
        "body{margin:0;padding:24px;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1000px;margin-inline:auto}"
        "h1{font-size:22px}h2{font-size:17px;margin-top:28px}.cards{display:flex;flex-wrap:wrap;gap:10px}.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;min-width:130px}"
        ".card b{display:block;font-size:20px}.mut{color:var(--mut)}.w{overflow-x:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--bd);padding:5px 8px;text-align:left}a{color:var(--ac)}")


def _chart(gens) -> str:
    pts = [g for g in gens if g["score"] is not None]
    if not pts:
        return '<p class="mut">No scored generations yet.</p>'
    W, H, pad = 640, 220, 34
    top = max(1.0, max(g["score"] for g in pts))
    xs = lambda i: pad + (W - 2 * pad) * (i / max(1, len(pts) - 1))
    ys = lambda v: H - pad - (H - 2 * pad) * (v / top)
    line = " ".join(f"{xs(i):.1f},{ys(g['score']):.1f}" for i, g in enumerate(pts))
    dots = "".join(f'<circle cx="{xs(i):.1f}" cy="{ys(g["score"]):.1f}" r="4" fill="var({"--ok" if g["accepted"] else "--bad"})"><title>gen {g["id"]}: {g["score"]:.3f} ({"accepted" if g["accepted"] else "rejected"})</title></circle>' for i, g in enumerate(pts))
    return (f'<div class="w"><svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="score per generation"><line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="var(--bd)"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}" stroke="var(--bd)"/><text x="4" y="{pad}" fill="var(--mut)" font-size="11">{top:.2f}</text><text x="4" y="{H - pad}" fill="var(--mut)" font-size="11">0</text>'
            f'<polyline points="{line}" fill="none" stroke="var(--ac)" stroke-width="2"/>{dots}</svg></div><p class="mut">green = accepted, red = rejected</p>')


def _table(head, rows) -> str:
    e = html.escape
    return '<div class="w"><table><tr>' + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>"


def render_html(s: dict, refresh: int = 0, back: str = "") -> str:
    e, ex = html.escape, s["experiments"]
    cards = [("best score", _f(s["best_score"])), ("generations", len(s["generations"])), ("experiments", ex["total"]), ("claims", sum(s["claims"].values())),
             ("exams avg", _f(s["exams"]["avg_score"])), ("net reward", _f(s["rewards"]["reward"] - s["rewards"]["penalty"], 2))]
    state = "finished: " + str(s["stop_reason"]) if s["finished"] else "running or interrupted"
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            + (f"<meta http-equiv='refresh' content='{int(refresh)}'>" if refresh else "") + f"<title>SIRA - {e(s['run'])}</title><style>{_CSS}</style></head><body>"
            + (f"<p><a href='{e(back)}'>&larr; all runs</a></p>" if back else "") + f"<h1>SIRA run: {e(s['run'])}</h1><p class='mut'>{e(state)} &middot; generated {e(s['generated'])}</p>"
            + "<div class='cards'>" + "".join(f"<div class='card'><span class='mut'>{e(k)}</span><b>{e(str(v))}</b></div>" for k, v in cards) + "</div>"
            + "<h2>Configuration</h2>" + _table(["key", "value"], list(s["config"].items()))
            + "<h2>Score per generation</h2>" + _chart(s["generations"])
            + "<h2>Generations</h2>" + _table(["id", "parent", "method", "score", "accepted"], [(g["id"], g["parent"], g["method"], _f(g["score"]), "yes" if g["accepted"] else "no") for g in s["generations"]])
            + "<h2>Knowledge</h2>" + _table(["status", "claims"], list(s["claims"].items()) or [("-", 0)])
            + "<h2>Experiments, rewards, memory</h2>" + _table(["metric", "value"], [("experiments ok / total / timed out", f"{ex['ok']} / {ex['total']} / {ex['timed_out']}"), ("reward events", s["rewards"]["events"]),
                                                                              ("reward / penalty", f"+{_f(s['rewards']['reward'], 2)} / -{_f(s['rewards']['penalty'], 2)}")] + list(s["counts"].items()))
            + "<h2>Challenges</h2>" + _table(["status", "count", "avg best"], [(c["status"], c["n"], c["avg_best"]) for c in s["challenges"]] or [("-", 0, "-")])
            + "<h2>Latest evaluations</h2>" + _table(["id", "label", "total", "n"], [(x["id"], x["label"], _f(x["total"]), x["n"]) for x in s["evals"]] or [("-", "-", "-", "-")])
            + "</body></html>")


def _resolve(root) -> dict:
    root = Path(root)
    if (root / "sira.db").exists():
        return {root.name: root}
    return {d.name: d for d in root.iterdir() if (d / "sira.db").exists()} if root.is_dir() else {}


def make_server(root, host="127.0.0.1", port=8765, token=None) -> ThreadingHTTPServer:
    if host not in _LOOPBACK and not token:
        raise ValueError("refusing to listen on a non-loopback address without --token (the dashboard exposes run data)")
    root = Path(root)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            b = body.encode("utf-8")
            self.send_response(code)
            for k, v in (("Content-Type", ctype), ("Content-Length", str(len(b))), ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                         ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; img-src data:"), ("Referrer-Policy", "no-referrer")):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/healthz":
                return self._send(200, "ok", "text/plain")
            if token:
                given = (q.get("token") or [""])[0] or self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(given.encode(), token.encode()):
                    return self._send(401, "unauthorized", "text/plain")
            qs = f"?token={token}" if token else ""
            runs = _resolve(root)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            try:
                if u.path == "/api/runs":
                    return self._send(200, json.dumps(list_runs(root), default=str), "application/json")
                if len(parts) == 3 and parts[:2] == ["api", "run"] and parts[2] in runs:
                    return self._send(200, json.dumps(summarize(runs[parts[2]]), default=str), "application/json")
                if len(parts) == 2 and parts[0] == "run" and parts[1] in runs:
                    return self._send(200, render_html(summarize(runs[parts[1]]), refresh=10, back=("/" + qs) if len(runs) > 1 else ""))
                if not parts:
                    if len(runs) == 1:
                        return self._send(200, render_html(summarize(next(iter(runs.values()))), refresh=10))
                    rows = "".join(f"<tr><td><a href='/run/{html.escape(r['run'], quote=True)}{html.escape(qs)}'>{html.escape(r['run'])}</a></td><td>{html.escape(str(r['state']))}</td><td>{html.escape(str(r['mode']))}</td>"
                                   f"<td>{html.escape(_f(r['best_score']))}</td><td>{r['generations']}</td><td>{html.escape(r['updated'])}</td></tr>" for r in list_runs(root))
                    return self._send(200, f"<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='15'><title>SIRA runs</title><style>{_CSS}</style><h1>SIRA runs</h1>"
                                           f"<div class='w'><table><tr><th>run</th><th>state</th><th>mode</th><th>best</th><th>gens</th><th>updated</th></tr>{rows}</table></div>")
            except Exception as ex:
                return self._send(500, f"error: {type(ex).__name__}", "text/plain")
            return self._send(404, "not found", "text/plain")

    return ThreadingHTTPServer((host, port), H)


def serve(root, host="127.0.0.1", port=8765, token=None) -> None:
    srv = make_server(root, host, port, token)
    print(f"SIRA dashboard on http://{host}:{srv.server_address[1]}/" + ("?token=..." if token else "") + "  (read-only, Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
