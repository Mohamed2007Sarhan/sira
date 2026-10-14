"""SQLite store: claims (knowledge), experiments, generations, lessons, meta. Thread-safe."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS claims(
  id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
  status TEXT NOT NULL,            -- lead | verified | refuted | inconclusive | vetoed
  source TEXT, evidence TEXT, created REAL);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, hypothesis TEXT, code TEXT,
  stdout TEXT, stderr TEXT, exit_code INTEGER, timed_out INTEGER, result TEXT,
  duration REAL, backend TEXT, seed INTEGER, parent INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS generations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, parent INTEGER, path TEXT, method TEXT,
  score REAL, detail TEXT, accepted INTEGER, notes TEXT, created REAL);
CREATE TABLE IF NOT EXISTS lessons(
  id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, text TEXT, created REAL);
CREATE TABLE IF NOT EXISTS used_seeds(ns TEXT, idx INTEGER, PRIMARY KEY(ns, idx));
CREATE TABLE IF NOT EXISTS tools(
  name TEXT PRIMARY KEY, description TEXT, code TEXT, example_args TEXT, status TEXT,   -- active | pending | disabled
  created_by TEXT, uses INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS worlds(
  name TEXT PRIMARY KEY, domain TEXT, description TEXT, source TEXT, status TEXT,       -- active | pending | disabled
  created_by TEXT, created REAL);
CREATE TABLE IF NOT EXISTS concepts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, summary TEXT, claim_ids TEXT, created REAL);
CREATE TABLE IF NOT EXISTS mistakes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, area TEXT, description TEXT, fix TEXT, created REAL);
CREATE TABLE IF NOT EXISTS evals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, model TEXT, total REAL, n INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS quests(
  id INTEGER PRIMARY KEY AUTOINCREMENT, goal TEXT, seeds TEXT, definition TEXT, criteria TEXT, status TEXT, created REAL);
CREATE TABLE IF NOT EXISTS exams(
  id INTEGER PRIMARY KEY AUTOINCREMENT, quest_id INTEGER, round INTEGER, variant TEXT, kind TEXT, question TEXT, truth TEXT,
  answer TEXT, confidence REAL, score REAL, reason TEXT, created REAL);
CREATE TABLE IF NOT EXISTS society_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, quest_id INTEGER, round INTEGER, policy TEXT, metrics TEXT, log TEXT, created REAL);
CREATE TABLE IF NOT EXISTS challenges(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, domain TEXT, kind TEXT, statement TEXT, verifier_code TEXT, probes TEXT,
  difficulty INTEGER, status TEXT, best_score REAL DEFAULT 0, attempts INTEGER DEFAULT 0, created_by TEXT, notes TEXT, created REAL);
CREATE TABLE IF NOT EXISTS attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, challenge_id INTEGER, n INTEGER, approach TEXT, code TEXT, answer TEXT, score REAL, holdout REAL,
  reward REAL, penalty REAL, note TEXT, created REAL);
CREATE TABLE IF NOT EXISTS rewards(
  id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, item TEXT, reward REAL, penalty REAL, note TEXT, created REAL);
CREATE TABLE IF NOT EXISTS thoughts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, layer TEXT, question TEXT, result TEXT, created REAL);
CREATE TABLE IF NOT EXISTS eval_items(
  eval_id INTEGER, pid TEXT, domain TEXT, passed INTEGER, reason TEXT, prompt TEXT, answer TEXT);
"""


class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            for table in ("mistakes", "lessons"):                      # provenance of every memory item (migration-safe)
                for col, ddl in (("source", "TEXT DEFAULT 'llm'"), ("confidence", "REAL DEFAULT 0.5"), ("status", "TEXT DEFAULT 'active'")):
                    try:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                    except sqlite3.OperationalError:
                        pass

    # -- generic -------------------------------------------------------------
    def _x(self, sql, args=()):
        with self.lock:
            return self.conn.execute(sql, args)

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # -- meta ----------------------------------------------------------------
    def get_meta(self, k, default=None):
        r = self.rows("SELECT v FROM meta WHERE k=?", (k,))
        return json.loads(r[0]["v"]) if r else default

    def set_meta(self, k, v):
        self._x("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    # -- claims --------------------------------------------------------------
    def add_claim(self, text, status, source="", evidence=None):
        cur = self._x("INSERT INTO claims(text,status,source,evidence,created) VALUES(?,?,?,?,?)",
                      (text, status, source, json.dumps(evidence or {}, default=str), time.time()))
        return cur.lastrowid

    def set_claim_status(self, cid, status, evidence=None):
        if evidence is None:
            self._x("UPDATE claims SET status=? WHERE id=?", (status, cid))
        else:
            self._x("UPDATE claims SET status=?, evidence=? WHERE id=?", (status, json.dumps(evidence, default=str), cid))

    def claims(self, status=None, limit=500):
        if status:
            return self.rows("SELECT * FROM claims WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit))
        return self.rows("SELECT * FROM claims ORDER BY id DESC LIMIT ?", (limit,))

    def search_claims(self, query, limit=12):
        words = [w for w in query.lower().split() if len(w) > 2][:8]
        rows = self.rows("SELECT * FROM claims WHERE status!='vetoed' ORDER BY id DESC LIMIT 800")
        scored = []
        for r in rows:
            t = r["text"].lower()
            s = sum(1 for w in words if w in t)
            if s:
                scored.append((s, r))
        scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
        return [r for _, r in scored[:limit]]

    # -- experiments ---------------------------------------------------------
    def add_experiment(self, **kw):
        cols = ("agent", "hypothesis", "code", "stdout", "stderr", "exit_code", "timed_out",
                "result", "duration", "backend", "seed", "parent")
        vals = [kw.get(c) for c in cols]
        if not isinstance(vals[7], (str, type(None))):
            vals[7] = json.dumps(vals[7], default=str)
        cur = self._x(f"INSERT INTO experiments({','.join(cols)},created) VALUES({','.join('?' * len(cols))},?)",
                      (*vals, time.time()))
        return cur.lastrowid

    def experiment(self, eid):
        r = self.rows("SELECT * FROM experiments WHERE id=?", (eid,))
        return r[0] if r else None

    def experiments(self, limit=300):
        return self.rows("SELECT id,agent,hypothesis,exit_code,timed_out,duration,backend,seed,parent,created "
                         "FROM experiments ORDER BY id DESC LIMIT ?", (limit,))

    # -- generations ---------------------------------------------------------
    def add_generation(self, parent, path, method, score, detail, accepted, notes=""):
        cur = self._x("INSERT INTO generations(parent,path,method,score,detail,accepted,notes,created) "
                      "VALUES(?,?,?,?,?,?,?,?)",
                      (parent, path, method, score, json.dumps(detail, default=str), int(accepted), notes, time.time()))
        return cur.lastrowid

    def generations(self):
        rows = self.rows("SELECT * FROM generations ORDER BY id")
        for r in rows:
            r["detail"] = json.loads(r["detail"] or "{}")
        return rows

    # -- lessons -------------------------------------------------------------
    def add_lesson(self, topic, text, source="llm", confidence=0.5, status="active"):
        return self._x("INSERT INTO lessons(topic,text,created,source,confidence,status) VALUES(?,?,?,?,?,?)", (topic, text, time.time(), source, confidence, status)).lastrowid

    def lessons(self, limit=30, include_inactive=False):
        return self.rows("SELECT * FROM lessons " + ("" if include_inactive else "WHERE status='active' ") + "ORDER BY id DESC LIMIT ?", (limit,))

    def set_memory_status(self, table, item_id, status, confidence=None):
        assert table in ("mistakes", "lessons")
        self._x(f"UPDATE {table} SET status=? WHERE id=?", (status, item_id))
        if confidence is not None:
            self._x(f"UPDATE {table} SET confidence=? WHERE id=?", (confidence, item_id))

    # -- tools / worlds / concepts / mistakes ---------------------------------
    def put_tool(self, name, description, code, example_args, status, created_by):
        self._x("INSERT INTO tools(name,description,code,example_args,status,created_by,created) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, code=excluded.code, "
                "example_args=excluded.example_args, status=excluded.status",
                (name, description, code, json.dumps(example_args), status, created_by, time.time()))

    def tool(self, name):
        r = self.rows("SELECT * FROM tools WHERE name=?", (name,))
        return r[0] if r else None

    def tools(self, status=None):
        if status:
            return self.rows("SELECT * FROM tools WHERE status=? ORDER BY name", (status,))
        return self.rows("SELECT * FROM tools ORDER BY name")

    def set_tool_status(self, name, status):
        self._x("UPDATE tools SET status=? WHERE name=?", (status, name))

    def bump_tool(self, name):
        self._x("UPDATE tools SET uses=uses+1 WHERE name=?", (name,))

    def put_world(self, name, domain, description, source, status, created_by):
        self._x("INSERT INTO worlds(name,domain,description,source,status,created_by,created) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET domain=excluded.domain, description=excluded.description, "
                "source=excluded.source, status=excluded.status", (name, domain, description, source, status, created_by, time.time()))

    def world(self, name):
        r = self.rows("SELECT * FROM worlds WHERE name=?", (name,))
        return r[0] if r else None

    def worlds(self, status=None):
        if status:
            return self.rows("SELECT * FROM worlds WHERE status=? ORDER BY name", (status,))
        return self.rows("SELECT * FROM worlds ORDER BY name")

    def set_world_status(self, name, status):
        self._x("UPDATE worlds SET status=? WHERE name=?", (status, name))

    def add_concept(self, title, summary, claim_ids):
        return self._x("INSERT INTO concepts(title,summary,claim_ids,created) VALUES(?,?,?,?)",
                       (title, summary, json.dumps(claim_ids), time.time())).lastrowid

    def concepts(self, limit=200):
        return self.rows("SELECT * FROM concepts ORDER BY id DESC LIMIT ?", (limit,))

    def add_mistake(self, area, description, fix="", source="mechanical", confidence=0.9, status="active"):
        return self._x("INSERT INTO mistakes(area,description,fix,created,source,confidence,status) VALUES(?,?,?,?,?,?,?)",
                       (area, description, fix, time.time(), source, confidence, status)).lastrowid

    def mistakes(self, limit=40, include_inactive=False):
        return self.rows("SELECT * FROM mistakes " + ("" if include_inactive else "WHERE status='active' ") + "ORDER BY id DESC LIMIT ?", (limit,))

    def add_thought(self, layer, question, result):
        return self._x("INSERT INTO thoughts(layer,question,result,created) VALUES(?,?,?,?)",
                       (layer, question, json.dumps(result, default=str), time.time())).lastrowid

    def thoughts(self, limit=200, layer=None):
        rows = self.rows("SELECT * FROM thoughts " + ("WHERE layer=? " if layer else "") + "ORDER BY id DESC LIMIT ?",
                         (layer, limit) if layer else (limit,))
        for r in rows:
            r["result"] = json.loads(r["result"])
        return rows

    def add_exam(self, quest_id, rnd, variant, kind, question, truth, answer, confidence, score, reason):
        return self._x("INSERT INTO exams(quest_id,round,variant,kind,question,truth,answer,confidence,score,reason,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (quest_id, rnd, variant, kind, question[:3000], str(truth)[:1500], str(answer)[:3000], confidence, score, reason[:500], time.time())).lastrowid

    def exams(self, quest_id=None, rnd=None, limit=2000):
        q, a = "SELECT * FROM exams", []
        w = []
        if quest_id is not None:
            w.append("quest_id=?"); a.append(quest_id)
        if rnd is not None:
            w.append("round=?"); a.append(rnd)
        return self.rows(q + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY id DESC LIMIT ?", (*a, limit))

    def add_society_run(self, quest_id, rnd, policy, metrics, log):
        return self._x("INSERT INTO society_runs(quest_id,round,policy,metrics,log,created) VALUES(?,?,?,?,?,?)",
                       (quest_id, rnd, policy, json.dumps(metrics), json.dumps(log, default=str)[:60000], time.time())).lastrowid

    def society_runs(self, limit=50):
        rows = self.rows("SELECT * FROM society_runs ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["metrics"], r["log"] = json.loads(r["metrics"]), json.loads(r["log"])
        return rows

    def add_quest(self, goal, seeds):
        return self._x("INSERT INTO quests(goal,seeds,definition,criteria,status,created) VALUES(?,?,?,?,?,?)",
                       (goal, json.dumps(seeds), "", "[]", "studying", time.time())).lastrowid

    def update_quest(self, qid, **kw):
        for k, v in kw.items():
            if k in ("definition", "criteria", "status"):
                self._x(f"UPDATE quests SET {k}=? WHERE id=?", (v if isinstance(v, str) else json.dumps(v), qid))

    def quest(self, qid):
        r = self.rows("SELECT * FROM quests WHERE id=?", (qid,))
        if r:
            r[0]["criteria"], r[0]["seeds"] = json.loads(r[0]["criteria"] or "[]"), json.loads(r[0]["seeds"] or "[]")
        return r[0] if r else None

    # -- challenges / attempts / rewards ------------------------------------------
    def add_challenge(self, title, domain, kind, statement, verifier_code, probes, difficulty, created_by):
        return self._x("INSERT INTO challenges(title,domain,kind,statement,verifier_code,probes,difficulty,status,created_by,notes,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (title, domain, kind, statement, verifier_code, json.dumps(probes), difficulty, "active", created_by, "", time.time())).lastrowid

    def challenge(self, cid):
        r = self.rows("SELECT * FROM challenges WHERE id=?", (cid,))
        if r:
            r[0]["probes"] = json.loads(r[0]["probes"] or "[]")
        return r[0] if r else None

    def challenges(self, status=None, limit=500):
        rows = self.rows("SELECT id,title,domain,kind,difficulty,status,best_score,attempts,created_by,notes FROM challenges " +
                         ("WHERE status=? " if status else "") + "ORDER BY id DESC LIMIT ?", (status, limit) if status else (limit,))
        return rows

    def update_challenge(self, cid, **kw):
        for k, v in kw.items():
            if k in ("status", "best_score", "attempts", "notes"):
                self._x(f"UPDATE challenges SET {k}=? WHERE id=?", (v, cid))

    def add_attempt(self, cid, n, approach, code, answer, score, holdout, reward, penalty, note):
        return self._x("INSERT INTO attempts(challenge_id,n,approach,code,answer,score,holdout,reward,penalty,note,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (cid, n, approach[:600], code[:20000], str(answer)[:4000], score, holdout, reward, penalty, note[:500], time.time())).lastrowid

    def attempts(self, cid):
        return self.rows("SELECT * FROM attempts WHERE challenge_id=? ORDER BY n", (cid,))

    def add_reward(self, source, item, reward, penalty, note):
        return self._x("INSERT INTO rewards(source,item,reward,penalty,note,created) VALUES(?,?,?,?,?,?)", (source, item[:200], reward, penalty, note[:300], time.time())).lastrowid

    def rewards(self, limit=500):
        return self.rows("SELECT * FROM rewards ORDER BY id DESC LIMIT ?", (limit,))

    def reward_totals(self):
        r = self.rows("SELECT source, SUM(reward) AS r, SUM(penalty) AS p, COUNT(*) AS n FROM rewards GROUP BY source")
        return {x["source"]: {"reward": x["r"] or 0.0, "penalty": x["p"] or 0.0, "n": x["n"]} for x in r}

    # -- evaluations (full test results, visible in the GUI) ---------------------
    def add_eval(self, label, model, total, n, items):
        with self.lock:
            eid = self._x("INSERT INTO evals(label,model,total,n,created) VALUES(?,?,?,?,?)",
                          (label, str(model), total, n, time.time())).lastrowid
            self.conn.executemany("INSERT INTO eval_items VALUES(?,?,?,?,?,?,?)",
                                  [(eid, i["id"], i["domain"], int(i["passed"]), i["reason"][:200], i.get("prompt", "")[:3000],
                                    i.get("answer", "")[:3000]) for i in items])
        return eid

    def evals(self, limit=100):
        return self.rows("SELECT * FROM evals ORDER BY id DESC LIMIT ?", (limit,))

    def eval_items(self, eval_id):
        return self.rows("SELECT * FROM eval_items WHERE eval_id=?", (eval_id,))

    # -- seeds ---------------------------------------------------------------
    def next_index(self, ns):
        with self.lock:
            r = self.rows("SELECT COALESCE(MAX(idx),-1)+1 AS n FROM used_seeds WHERE ns=?", (ns,))
            n = r[0]["n"]
            self._x("INSERT INTO used_seeds(ns,idx) VALUES(?,?)", (ns, n))
            return n

    def close(self):
        with self.lock:
            self.conn.close()
