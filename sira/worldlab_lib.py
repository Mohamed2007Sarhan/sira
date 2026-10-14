"""worldlab - the 'Black Box' toolkit available inside the sandbox as `import worldlab`.

Pure standard library (numpy used automatically if present). It gives the agents a *reality
checking* kit: numerical integrators, conservation-law checks, a small molecular-dynamics engine,
mass-action chemical kinetics, and CODATA constants. Everything is deterministic given SIRA_SEED.

Contract with the Verifier: an experiment ends by calling  worldlab.report(metric=value, ...)
which prints one line  SIRA_RESULT: {json}.  The Verifier (outside the sandbox) checks the
metrics against the `expect` conditions - the experiment never judges itself.
"""
from __future__ import annotations

import json
import math
import os
import random

try:                     # optional
    import numpy as np   # noqa: F401
    HAS_NUMPY = True
except Exception:        # pragma: no cover
    HAS_NUMPY = False

# ---- CODATA 2018 constants (SI) ---------------------------------------------------------
K_B = 1.380649e-23
N_A = 6.02214076e23
R_GAS = 8.314462618
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
G_NEWTON = 6.67430e-11
E_CHARGE = 1.602176634e-19
G_ACCEL = 9.80665


def seed() -> int:
    """Seed Python's RNG (and numpy) from SIRA_SEED; returns the seed."""
    s = int(os.environ.get("SIRA_SEED", "0"))
    random.seed(s)
    if HAS_NUMPY:
        import numpy as _np
        _np.random.seed(s % (2 ** 32))
    return s


def report(**metrics) -> None:
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float)):
            clean[k] = float(v) if math.isfinite(v) else str(v)
        else:
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = str(v)
    print("SIRA_RESULT: " + json.dumps(clean, sort_keys=True), flush=True)


# ---- integrators ---------------------------------------------------------------------------
def rk4(f, y0, t0, t1, dt):
    """Classic RK4 for y' = f(t, y); y is a list of floats. Returns (ts, ys)."""
    n = max(1, int(round((t1 - t0) / dt)))
    h = (t1 - t0) / n
    t, y = t0, list(y0)
    ts, ys = [t], [list(y)]
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, [a + h / 2 * b for a, b in zip(y, k1)])
        k3 = f(t + h / 2, [a + h / 2 * b for a, b in zip(y, k2)])
        k4 = f(t + h, [a + h * b for a, b in zip(y, k3)])
        y = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(y, k1, k2, k3, k4)]
        t += h
        ts.append(t)
        ys.append(list(y))
    return ts, ys


def velocity_verlet(accel, x0, v0, dt, steps):
    """Velocity-Verlet for x'' = accel(x); x, v lists. Returns (xs, vs) trajectories."""
    x, v = list(x0), list(v0)
    a = accel(x)
    xs, vs = [list(x)], [list(v)]
    for _ in range(steps):
        x = [xi + vi * dt + 0.5 * ai * dt * dt for xi, vi, ai in zip(x, v, a)]
        a_new = accel(x)
        v = [vi + 0.5 * (ai + bi) * dt for vi, ai, bi in zip(v, a, a_new)]
        a = a_new
        xs.append(list(x))
        vs.append(list(v))
    return xs, vs


def relative_drift(values) -> float:
    """max |v_i - v_0| / max(|v_0|, tiny) - use for energy / mass / momentum conservation."""
    v0 = values[0]
    scale = max(abs(v0), 1e-300)
    return max(abs(v - v0) for v in values) / scale


# ---- molecular dynamics (2D Lennard-Jones, reduced units) --------------------------------------
def lj_md(n_side=4, steps=400, dt=0.004, temperature=0.8, density=0.7, cutoff=2.5):
    """Small 2D Lennard-Jones MD (velocity-Verlet, shifted potential, periodic box).
    Returns dict(total_energy=[...], kinetic=[...], potential=[...], n=int, box=float).
    Reduced units: epsilon = sigma = m = k_B = 1. Seeded through SIRA_SEED."""
    rng = random.Random(int(os.environ.get("SIRA_SEED", "0")))
    n = n_side * n_side
    box = math.sqrt(n / density)
    a = box / n_side
    pos = [[(i + 0.5) * a, (j + 0.5) * a] for i in range(n_side) for j in range(n_side)]
    vel = [[rng.gauss(0, math.sqrt(temperature)), rng.gauss(0, math.sqrt(temperature))] for _ in range(n)]
    mx = sum(v[0] for v in vel) / n
    my = sum(v[1] for v in vel) / n
    vel = [[v[0] - mx, v[1] - my] for v in vel]
    rc2 = cutoff * cutoff
    shift = 4 * (cutoff ** -12 - cutoff ** -6)

    def forces(p):
        f = [[0.0, 0.0] for _ in range(n)]
        pot = 0.0
        for i in range(n - 1):
            xi, yi = p[i]
            for j in range(i + 1, n):
                dx = xi - p[j][0]
                dy = yi - p[j][1]
                dx -= box * round(dx / box)
                dy -= box * round(dy / box)
                r2 = dx * dx + dy * dy
                if r2 < rc2:
                    inv2 = 1.0 / r2
                    inv6 = inv2 ** 3
                    pot += 4 * (inv6 * inv6 - inv6) - shift
                    ff = 24 * inv2 * (2 * inv6 * inv6 - inv6)
                    f[i][0] += ff * dx
                    f[i][1] += ff * dy
                    f[j][0] -= ff * dx
                    f[j][1] -= ff * dy
        return f, pot

    f, pot = forces(pos)
    E, K, U = [], [], []
    for _ in range(steps):
        for i in range(n):
            vel[i][0] += 0.5 * dt * f[i][0]
            vel[i][1] += 0.5 * dt * f[i][1]
            pos[i][0] = (pos[i][0] + dt * vel[i][0]) % box
            pos[i][1] = (pos[i][1] + dt * vel[i][1]) % box
        f, pot = forces(pos)
        for i in range(n):
            vel[i][0] += 0.5 * dt * f[i][0]
            vel[i][1] += 0.5 * dt * f[i][1]
        ke = 0.5 * sum(v[0] ** 2 + v[1] ** 2 for v in vel)
        K.append(ke)
        U.append(pot)
        E.append(ke + pot)
    return {"total_energy": E, "kinetic": K, "potential": U, "n": n, "box": box}


# ---- chemical kinetics (mass action) ----------------------------------------------------------
def kinetics(reactions, c0, t_end, dt=1e-3):
    """Mass-action kinetics. reactions: list of (reactants:dict, products:dict, k).
    c0: dict species->concentration. Returns (times, {species: [c(t)...]}) integrated with RK4."""
    species = sorted(set(c0) | {s for r, p, _ in reactions for s in list(r) + list(p)})
    idx = {s: i for i, s in enumerate(species)}

    def f(_t, y):
        dy = [0.0] * len(species)
        for reac, prod, k in reactions:
            rate = k
            for s, nu in reac.items():
                rate *= max(y[idx[s]], 0.0) ** nu
            for s, nu in reac.items():
                dy[idx[s]] -= nu * rate
            for s, nu in prod.items():
                dy[idx[s]] += nu * rate
        return dy

    y0 = [float(c0.get(s, 0.0)) for s in species]
    ts, ys = rk4(f, y0, 0.0, t_end, dt)
    return ts, {s: [row[idx[s]] for row in ys] for s in species}


def element_balance(formula_counts_by_species, conc_series, index=-1):
    """Total amount of each element at time index. formula_counts_by_species: {'H2O': {'H':2,'O':1}, ...}."""
    totals = {}
    for sp, counts in formula_counts_by_species.items():
        c = conc_series[sp][index]
        for el, n in counts.items():
            totals[el] = totals.get(el, 0.0) + n * c
    return totals


# ---- physics helpers -------------------------------------------------------------------------------
def ideal_gas_pressure(n_mol, T, V):
    return n_mol * R_GAS * T / V


def projectile_range(v0, angle_deg, g=G_ACCEL):
    return v0 ** 2 * math.sin(2 * math.radians(angle_deg)) / g


def data_path(name: str) -> str:
    """Path of a file downloaded by the agents' web tools (read-only data store)."""
    for base in ("/data", "data"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in the data store")


# =============================================================================================
#  WORLDS: a uniform interface for simulated realities (built-in + agent-defined)
# =============================================================================================
import importlib
import statistics


class World:
    """Base class. Override initial_state() and step(); optionally invariants().
    State is a dict of floats/lists. invariants() returns {name: value} quantities the laws must conserve."""
    name = "unnamed"
    domain = "unknown"
    description = ""

    def __init__(self, **params):
        self.params = params

    def initial_state(self) -> dict:
        raise NotImplementedError

    def step(self, state: dict, dt: float) -> dict:
        raise NotImplementedError

    def invariants(self, state: dict) -> dict:
        return {}

    def run(self, steps: int, dt: float, state: dict | None = None) -> dict:
        s = state if state is not None else self.initial_state()
        states, inv = [s], {k: [v] for k, v in self.invariants(s).items()}
        for _ in range(int(steps)):
            s = self.step(s, dt)
            states.append(s)
            for k, v in self.invariants(s).items():
                inv.setdefault(k, []).append(v)
        return {"states": states, "invariants": inv}

    def check(self, steps: int = 1000, dt: float = 0.01) -> dict:
        """Relative drift of every declared invariant over a run (the 'laws hold' test)."""
        tr = self.run(steps, dt)
        return {k: relative_drift(v) for k, v in tr["invariants"].items() if v}


def _rk4_dict(deriv, state: dict, dt: float, keys) -> dict:
    def vec(s):
        return [s[k] for k in keys]

    def f(_t, y):
        d = deriv({k: v for k, v in zip(keys, y)})
        return [float(d.get(k, 0.0)) for k in keys]

    _, ys = rk4(f, vec(state), 0.0, dt, dt)
    new = dict(state)
    new.update({k: v for k, v in zip(keys, ys[-1])})
    return new


class LawWorld(World):
    """TERRA INCOGNITA: a world with NO predefined laws. Define your own:
         w = LawWorld("my_world", domain="new science")
         w.set_state(x=1.0, v=0.0)
         w.add_law("spring", lambda s: {"x": s["v"], "v": -4.0 * s["x"]})   # derivatives dstate/dt
         w.add_invariant("energy", lambda s: 0.5 * s["v"]**2 + 2.0 * s["x"]**2)
         w.check()   # does the law you invented actually conserve what you claim?
       Laws return partial derivative dicts; they are summed and integrated with RK4."""

    def __init__(self, name="unknown", domain="unknown", description=""):
        super().__init__()
        self.name, self.domain, self.description = name, domain, description
        self._laws, self._invs, self._state = [], {}, {}

    def set_state(self, **kw):
        self._state = {k: float(v) for k, v in kw.items()}
        return self

    def add_law(self, name, fn, doc=""):
        self._laws.append((name, fn, doc))
        return self

    def add_invariant(self, name, fn):
        self._invs[name] = fn
        return self

    def initial_state(self):
        return dict(self._state)

    def step(self, state, dt):
        def deriv(s):
            tot = {}
            for _, fn, _ in self._laws:
                for k, v in fn(s).items():
                    tot[k] = tot.get(k, 0.0) + v
            return tot
        return _rk4_dict(deriv, state, dt, sorted(self._state))

    def invariants(self, state):
        return {k: fn(state) for k, fn in self._invs.items()}


class Pendulum(World):
    name, domain, description = "pendulum", "physics", "nonlinear pendulum; invariant: energy"

    def initial_state(self):
        return {"th": self.params.get("th0", 1.0), "w": self.params.get("w0", 0.0)}

    def step(self, s, dt):
        g, L = self.params.get("g", G_ACCEL), self.params.get("L", 1.0)
        return _rk4_dict(lambda q: {"th": q["w"], "w": -g / L * math.sin(q["th"])}, s, dt, ("th", "w"))

    def invariants(self, s):
        g, L = self.params.get("g", G_ACCEL), self.params.get("L", 1.0)
        return {"energy": 0.5 * (L * s["w"]) ** 2 + g * L * (1 - math.cos(s["th"]))}


class LotkaVolterra(World):
    name, domain, description = "lotka_volterra", "ecology", "predator-prey; invariant V(x,y)"

    def initial_state(self):
        return {"x": self.params.get("x0", 10.0), "y": self.params.get("y0", 5.0)}

    def _p(self):
        p = self.params
        return p.get("a", 1.1), p.get("b", 0.4), p.get("c", 0.4), p.get("d", 0.1)

    def step(self, s, dt):
        a, b, c, d = self._p()
        return _rk4_dict(lambda q: {"x": a * q["x"] - b * q["x"] * q["y"], "y": d * q["x"] * q["y"] - c * q["y"]}, s, dt, ("x", "y"))

    def invariants(self, s):
        a, b, c, d = self._p()
        return {"V": d * s["x"] - c * math.log(s["x"]) + b * s["y"] - a * math.log(s["y"])}


class SIR(World):
    name, domain, description = "sir", "epidemiology", "SIR epidemic; invariant S+I+R"

    def initial_state(self):
        return {"S": self.params.get("S0", 0.99), "I": self.params.get("I0", 0.01), "R": 0.0}

    def step(self, s, dt):
        b, g = self.params.get("beta", 0.5), self.params.get("gamma", 0.1)
        return _rk4_dict(lambda q: {"S": -b * q["S"] * q["I"], "I": b * q["S"] * q["I"] - g * q["I"], "R": g * q["I"]}, s, dt, ("S", "I", "R"))

    def invariants(self, s):
        return {"population": s["S"] + s["I"] + s["R"]}


class Heat1D(World):
    name, domain, description = "heat1d", "thermodynamics", "1-D diffusion, insulated ends; invariant total heat"

    def initial_state(self):
        n = int(self.params.get("n", 50))
        return {"u": [1.0 if n // 3 < i < 2 * n // 3 else 0.0 for i in range(n)]}

    def step(self, s, dt):
        k, dx = self.params.get("k", 1.0), self.params.get("dx", 1.0)
        u, n = s["u"], len(s["u"])
        r = k * dt / dx ** 2
        if r > 0.5:
            raise ValueError("unstable: k*dt/dx^2 must be <= 0.5")
        new = [u[i] + r * ((u[i - 1] if i > 0 else u[i]) - 2 * u[i] + (u[i + 1] if i < n - 1 else u[i])) for i in range(n)]
        return {"u": new}

    def invariants(self, s):
        return {"total_heat": sum(s["u"])}


class Life(World):
    name, domain, description = "life", "computation", "Conway's Game of Life on a torus; tracks population"

    def initial_state(self):
        n, rng = int(self.params.get("n", 24)), random.Random(int(os.environ.get("SIRA_SEED", "0")))
        return {"grid": [[1 if rng.random() < 0.3 else 0 for _ in range(n)] for _ in range(n)]}

    def step(self, s, dt):
        g, n = s["grid"], len(s["grid"])
        out = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                c = sum(g[(i + a) % n][(j + b) % n] for a in (-1, 0, 1) for b in (-1, 0, 1)) - g[i][j]
                out[i][j] = 1 if (c == 3 or (g[i][j] and c == 2)) else 0
        return {"grid": out}

    def invariants(self, s):
        return {}

    def population(self, s):
        return sum(map(sum, s["grid"]))


class GeneticAlgorithm(World):
    name, domain, description = "genetic", "evolution", "GA maximising ones in a bit-string; tracks best fitness"

    def initial_state(self):
        rng = random.Random(int(os.environ.get("SIRA_SEED", "0")))
        L, N = int(self.params.get("length", 40)), int(self.params.get("pop", 40))
        return {"pop": [[rng.randint(0, 1) for _ in range(L)] for _ in range(N)], "gen": 0}

    def step(self, s, dt):
        rng = random.Random(int(os.environ.get("SIRA_SEED", "0")) * 1000 + s["gen"])
        mu = self.params.get("mutation", 0.01)
        pop = s["pop"]
        nxt = []
        for _ in range(len(pop)):
            a, b = max(rng.sample(pop, 3), key=sum), max(rng.sample(pop, 3), key=sum)
            cut = rng.randrange(len(a))
            child = [(1 - x if rng.random() < mu else x) for x in a[:cut] + b[cut:]]
            nxt.append(child)
        return {"pop": nxt, "gen": s["gen"] + 1}

    def best(self, s):
        return max(sum(x) for x in s["pop"])


class IdealGasBox(World):
    name, domain, description = "gas_box", "thermodynamics", "elastic particles in a box; invariant kinetic energy; pressure from wall impulse"

    def initial_state(self):
        rng, n = random.Random(int(os.environ.get("SIRA_SEED", "0"))), int(self.params.get("n", 200))
        T = self.params.get("T", 1.0)
        return {"x": [rng.random() * self.params.get("L", 10.0) for _ in range(n)], "v": [rng.gauss(0, math.sqrt(T)) for _ in range(n)],
                "impulse": 0.0, "t": 0.0}

    def step(self, s, dt):
        L = self.params.get("L", 10.0)
        x, v, imp = [], [], s["impulse"]
        for xi, vi in zip(s["x"], s["v"]):
            xi += vi * dt
            if xi < 0 or xi > L:
                imp += 2 * abs(vi)
                xi = -xi if xi < 0 else 2 * L - xi
                vi = -vi
            x.append(xi)
            v.append(vi)
        return {"x": x, "v": v, "impulse": imp, "t": s["t"] + dt}

    def invariants(self, s):
        return {"kinetic_energy": 0.5 * sum(vi * vi for vi in s["v"])}

    def pressure(self, s):          # 1-D: force per "area" = impulse / (2 walls * time)
        return s["impulse"] / (2 * max(s["t"], 1e-12))


BUILTIN_WORLDS = {c.name: c for c in (Pendulum, LotkaVolterra, SIR, Heat1D, Life, GeneticAlgorithm, IdealGasBox)}
BUILTIN_WORLDS["terra_incognita"] = LawWorld


def list_builtin_worlds():
    return {k: {"domain": getattr(v, "domain", "unknown"), "description": getattr(v, "description", "")} for k, v in BUILTIN_WORLDS.items()}


def load_world(name: str, **params):
    """Built-in world by name, or an agent-defined world: module world_<name>.py exposing build(**params) -> World."""
    if name in BUILTIN_WORLDS and name != "terra_incognita":
        return BUILTIN_WORLDS[name](**params)
    if name == "terra_incognita":
        return LawWorld(**params)
    mod = importlib.import_module("world_" + name)
    return mod.build(**params)


# =============================================================================================
#  REALITY LAB: scenarios, sweeps, comparison with real data
# =============================================================================================
def summarize(values):
    v = [float(x) for x in values]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": statistics.fmean(v), "std": statistics.pstdev(v), "min": min(v), "max": max(v)}


def sweep(fn, grid: dict, seeds=(0,)):
    """Run fn(**params) for every combination in grid and every seed (SIRA_SEED is set). Failures are recorded,
    not hidden. Returns list of {params, seed, value | error}."""
    import itertools
    keys = list(grid)
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        for sd in seeds:
            os.environ["SIRA_SEED"] = str(sd)
            random.seed(sd)
            try:
                out.append({"params": params, "seed": sd, "value": fn(**params)})
            except Exception as e:
                out.append({"params": params, "seed": sd, "error": f"{type(e).__name__}: {e}"})
    return out


def compare_to_data(sim, real) -> dict:
    """Error metrics between a simulated series and a real measured series (same length)."""
    if len(sim) != len(real) or not real:
        raise ValueError("series must have equal, non-zero length")
    errs = [s - r for s, r in zip(sim, real)]
    mean_r = statistics.fmean(real)
    ss_res = sum(e * e for e in errs)
    ss_tot = sum((r - mean_r) ** 2 for r in real) or 1e-300
    return {"rmse": math.sqrt(ss_res / len(errs)), "mae": sum(abs(e) for e in errs) / len(errs),
            "max_abs_err": max(abs(e) for e in errs), "r2": 1 - ss_res / ss_tot}


def fit_linear(x, y) -> dict:
    n, mx, my = len(x), statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x) or 1e-300
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    icpt = my - slope * mx
    ss_res = sum((b - (slope * a + icpt)) ** 2 for a, b in zip(x, y))
    ss_tot = sum((b - my) ** 2 for b in y) or 1e-300
    return {"slope": slope, "intercept": icpt, "r2": 1 - ss_res / ss_tot, "n": n}


def load_csv_column(name: str, column: int | str = 0, skip_header: bool = True) -> list:
    """Read one numeric column of a downloaded CSV (from the data store)."""
    import csv
    rows = list(csv.reader(open(data_path(name), newline="")))
    head = rows[0] if skip_header else None
    idx = head.index(column) if isinstance(column, str) and head else int(column)
    return [float(r[idx]) for r in rows[1 if skip_header else 0:] if len(r) > idx and r[idx].strip() not in ("", "NA", "NaN")]


# =============================================================================================
#  MATH LAB: check derivations step by step (sympy if available, numeric otherwise)
# =============================================================================================
def _num_namespace():
    ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    ns.update(abs=abs, min=min, max=max, pow=pow, sum=sum)
    return ns


def _split_eq(expr: str):
    if "=" in expr and "==" not in expr and "<=" not in expr and ">=" not in expr:
        l, r = expr.split("=", 1)
        return l.strip(), r.strip()
    return expr.strip(), None


def math_verify(steps, symbols=("x",), mode="chain", samples=40, tol=1e-7, domain=(0.3, 2.7)) -> dict:
    """mode='chain': every step expression equals the previous one (a derivation).
       mode='identity': every step 'lhs = rhs' holds for all values of the symbols.
       Returns {ok, first_bad, details[{step, ok, counterexample?, method}]}. Symbols are sampled in `domain`."""
    rng = random.Random(1234)
    pts = [{s: rng.uniform(*domain) for s in symbols} for _ in range(samples)]
    ns = _num_namespace()

    def ev(e, pt):
        return eval(e.replace("^", "**"), {"__builtins__": {}}, {**ns, **pt})

    def equal(a, b):
        method = "numeric"
        try:
            import sympy
            syms = sympy.symbols(" ".join(symbols))
            if not isinstance(syms, (list, tuple)):
                syms = [syms]
            loc = dict(zip(symbols, syms))
            ea, eb = sympy.sympify(a.replace("^", "**"), locals=loc), sympy.sympify(b.replace("^", "**"), locals=loc)
            if sympy.simplify(ea - eb) == 0:
                return True, None, "symbolic"
            method = "symbolic+numeric"
        except Exception:
            pass
        for pt in pts:
            try:
                va, vb = ev(a, pt), ev(b, pt)
            except Exception as e:
                return False, {"error": f"{type(e).__name__}: {e}", "at": pt}, method
            if abs(va - vb) > tol * max(1.0, abs(va), abs(vb)):
                return False, {"at": {k: round(v, 6) for k, v in pt.items()}, "lhs": va, "rhs": vb}, method
        return True, None, method

    details, first_bad = [], None
    prev = None
    for i, st in enumerate(steps):
        if mode == "identity":
            l, r = _split_eq(st)
            if r is None:
                details.append({"step": i, "text": st, "ok": False, "counterexample": "not an equation 'lhs = rhs'"})
                first_bad = i if first_bad is None else first_bad
                continue
            ok, ce, m = equal(l, r)
        else:
            l, r = _split_eq(st)
            cur = l if r is None else r      # allow "= expr" style by taking rhs
            if prev is None:
                ok, ce, m = True, None, "start"
            else:
                ok, ce, m = equal(prev, cur)
            prev = cur
        details.append({"step": i, "text": st, "ok": ok, "method": m, **({"counterexample": ce} if ce else {})})
        if not ok and first_bad is None:
            first_bad = i
    return {"ok": first_bad is None, "first_bad": first_bad, "details": details}
