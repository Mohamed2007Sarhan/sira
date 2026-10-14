"""Evaluation Core - verifiable problems, fresh every time, answers never visible to any model.

* Problems are generated programmatically from HMAC(secret_key, namespace:set:index) seeds.
  The key lives in the run directory, outside every sandbox, so no model/agent can predict a seed.
* Namespaces: 'train' (used for rejection-sampling data), 'eval' and 'confirm' (fresh sets).
  Eval problems whose prompt equals a training prompt are re-rolled (no contamination).
* Text answers end with `ANSWER: <value>`; code answers are executed in the SANDBOX against hidden
  test inputs, and compared with expected outputs computed here on the host. The sandbox never sees
  the expected outputs.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

DOMAINS = ["math", "logic", "physics", "chemistry", "code"]       # default evaluation domains
ALL_DOMAINS = DOMAINS + ["arith"]                                  # arith = small-model ladder, opt-in via cfg.domains

TEXT_SUFFIX = ("\n\nThink step by step, then finish with a final line exactly of the form\n"
               "ANSWER: <your answer>")
CODE_SUFFIX = ("\n\nReturn ONLY the complete Python function(s) in one ```python code block. "
               "Do not read input or print anything.")


@dataclass
class Problem:
    id: str
    domain: str
    kind: str                  # text | code
    prompt: str
    difficulty: int
    answer: object = None      # hidden from models
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"id": self.id, "domain": self.domain, "kind": self.kind, "prompt": self.prompt}


# =============================================================================== generators
def _fmt(x, nd=4):
    return f"{x:.{nd}g}"


def g_math(rng, d):
    k = rng.randrange(7)
    if k == 0:
        x, a = rng.randint(-10 * d, 10 * d), rng.randint(2, 5 + 4 * d)
        b = rng.randint(-20 * d, 20 * d)
        return (f"Solve for x: {a}x + ({b}) = {a * x + b}. x is an integer.", x, {"cmp": "int"})
    if k == 1:
        a, b, m = rng.randint(2, 9 + 10 * d), rng.randint(5, 30 * d * d), rng.randint(7, 40 * d + 20)
        return (f"Compute {a}^{b} mod {m}.", pow(a, b, m), {"cmp": "int"})
    if k == 2:
        g = rng.randint(2, 12)
        p, q = rng.randint(2, 9 * d + 3), rng.randint(2, 9 * d + 3)
        n1, n2 = g * p, g * q
        if rng.random() < 0.5:
            return (f"What is the greatest common divisor of {n1} and {n2}?", math.gcd(n1, n2), {"cmp": "int"})
        return (f"What is the least common multiple of {n1} and {n2}?", n1 * n2 // math.gcd(n1, n2), {"cmp": "int"})
    if k == 3:
        a, s, n = rng.randint(-9, 20), rng.randint(2, 6 + d), rng.randint(8, 15 + 10 * d)
        return (f"An arithmetic series has first term {a} and common difference {s}. "
                f"What is the sum of its first {n} terms?", n * (2 * a + (n - 1) * s) // 2, {"cmp": "int"})
    if k == 4:
        n = rng.randint(6, 9 + 3 * d)
        kk = rng.randint(2, n // 2)
        return (f"How many ways are there to choose {kk} items from {n} distinct items?", math.comb(n, kk), {"cmp": "int"})
    if k == 5:
        n = rng.randint(20, 60 * d * d + 40)
        return (f"Write the decimal number {n} in binary (digits only, no prefix).", bin(n)[2:], {"cmp": "str"})
    p, q = rng.randint(-8 * d, 8 * d), rng.randint(-8 * d, 8 * d)
    return (f"What is the larger root of x^2 - ({p + q})x + ({p * q}) = 0 ? (an integer)", max(p, q), {"cmp": "int"})


_OPS = ["and", "or", "xor"]


def _rand_formula(rng, vars_, depth):
    if depth == 0 or rng.random() < 0.25:
        v = rng.choice(vars_)
        return ("not", v) if rng.random() < 0.35 else v
    op = rng.choice(_OPS)
    return (op, _rand_formula(rng, vars_, depth - 1), _rand_formula(rng, vars_, depth - 1))


def _eval_formula(f, env):
    if isinstance(f, str):
        return env[f]
    if f[0] == "not":
        return not _eval_formula(f[1], env)
    a, b = _eval_formula(f[1], env), _eval_formula(f[2], env)
    return (a and b) if f[0] == "and" else (a or b) if f[0] == "or" else (a != b)


def _show_formula(f):
    if isinstance(f, str):
        return f
    if f[0] == "not":
        return f"NOT {_show_formula(f[1])}" if isinstance(f[1], str) else f"NOT ({_show_formula(f[1])})"
    return f"({_show_formula(f[1])} {f[0].upper()} {_show_formula(f[2])})"


def g_logic(rng, d):
    k = rng.randrange(3)
    if k == 0:
        vs = ["P", "Q", "R", "S"][: 2 + d]
        f = _rand_formula(rng, vs, 1 + d)
        cnt = 0
        for bits in range(2 ** len(vs)):
            env = {v: bool(bits >> i & 1) for i, v in enumerate(vs)}
            cnt += _eval_formula(f, env)
        return (f"Variables {', '.join(vs)} are each True or False. For how many of the {2 ** len(vs)} possible "
                f"assignments is the formula {_show_formula(f)} True? (XOR is exclusive or.)", cnt, {"cmp": "int"})
    if k == 1:
        kind = rng.choice(["arith", "geom", "quad"])
        if kind == "arith":
            a, s = rng.randint(-5, 20), rng.randint(2, 9)
            seq = [a + s * i for i in range(5 + d)]
            nxt = a + s * len(seq)
        elif kind == "geom":
            a, r = rng.randint(1, 5), rng.randint(2, 3)
            seq = [a * r ** i for i in range(4 + d)]
            nxt = a * r ** len(seq)
        else:
            a, b, c = rng.randint(0, 3), rng.randint(1, 4), rng.randint(0, 5)
            seq = [a * i * i + b * i + c for i in range(5 + d)]
            nxt = a * len(seq) ** 2 + b * len(seq) + c
        return (f"What is the next number in the sequence {', '.join(map(str, seq))}, ... ?", nxt, {"cmp": "int"})
    names = rng.sample(["Ann", "Bob", "Cy", "Dee", "Eli", "Fay", "Gus"], 3 + d)
    order = names[:]                       # tallest -> shortest
    stm = [f"{order[i]} is taller than {order[i + 1]}." for i in range(len(order) - 1)]
    rng.shuffle(stm)
    pos = rng.randint(1, len(order))
    ordn = {1: "tallest", 2: "second tallest", 3: "third tallest", 4: "fourth tallest", 5: "fifth tallest"}.get(pos, f"#{pos}")
    return (" ".join(stm) + f" Who is the {ordn} of them all?", order[pos - 1], {"cmp": "str"})


def g_physics(rng, d):
    k = rng.randrange(5)
    if k == 0:
        v, th = rng.randint(8, 25 + 10 * d), rng.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        r = v * v * math.sin(math.radians(2 * th)) / 9.81
        return (f"A projectile is launched from level ground at {v} m/s at {th} degrees above the horizontal. "
                f"Ignoring air resistance and using g = 9.81 m/s^2, what is its horizontal range in metres? "
                f"(3 significant figures)", r, {"cmp": "float", "rel": 0.01})
    if k == 1:
        n, T, V = round(rng.uniform(0.5, 5), 1), rng.randint(250, 450), round(rng.uniform(5, 60), 1)
        p = n * 8.314 * T / (V / 1000) / 1000
        return (f"{n} mol of an ideal gas is at {T} K in a {V} litre container. Using R = 8.314 J/(mol K), "
                f"what is the pressure in kPa?", p, {"cmp": "float", "rel": 0.01})
    if k == 2:
        u, a, t = rng.randint(0, 20), round(rng.uniform(0.5, 6), 1), rng.randint(2, 6 + 3 * d)
        return (f"A car moving at {u} m/s accelerates uniformly at {a} m/s^2 for {t} s. "
                f"How far does it travel in metres?", u * t + 0.5 * a * t * t, {"cmp": "float", "rel": 0.01})
    if k == 3:
        rs = [rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20]) for _ in range(2 + (d > 1))]
        req = 1 / sum(1 / r for r in rs)
        return (f"Resistors of {', '.join(str(r) for r in rs)} ohms are connected in parallel. "
                f"What is the equivalent resistance in ohms?", req, {"cmp": "float", "rel": 0.01})
    L = round(rng.uniform(0.2, 3.0), 2)
    return (f"What is the small-angle period in seconds of a simple pendulum of length {L} m? "
            f"(g = 9.81 m/s^2)", 2 * math.pi * math.sqrt(L / 9.81), {"cmp": "float", "rel": 0.01})


ATOMIC = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
          "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546}
FORMULAS = [("H2O", {"H": 2, "O": 1}), ("CO2", {"C": 1, "O": 2}), ("NaCl", {"Na": 1, "Cl": 1}),
            ("H2SO4", {"H": 2, "S": 1, "O": 4}), ("CaCO3", {"Ca": 1, "C": 1, "O": 3}),
            ("Mg(OH)2", {"Mg": 1, "O": 2, "H": 2}), ("Al2O3", {"Al": 2, "O": 3}), ("C6H12O6", {"C": 6, "H": 12, "O": 6}),
            ("NH3", {"N": 1, "H": 3}), ("Fe2O3", {"Fe": 2, "O": 3}), ("CuSO4", {"Cu": 1, "S": 1, "O": 4}),
            ("KNO3", {"K": 1, "N": 1, "O": 3}), ("C2H5OH", {"C": 2, "H": 6, "O": 1}), ("HCl", {"H": 1, "Cl": 1})]


def g_chem(rng, d):
    k = rng.randrange(5)
    atoms_txt = ", ".join(f"{e}={m}" for e, m in ATOMIC.items())
    if k == 0:
        f, counts = rng.choice(FORMULAS)
        mm = sum(ATOMIC[e] * n for e, n in counts.items())
        n = round(rng.uniform(0.2, 9.9), 1)
        return (f"Using atomic masses ({atoms_txt}) in g/mol, what is the mass in grams of {n} mol of {f}?",
                n * mm, {"cmp": "float", "rel": 0.003})
    if k == 1:
        m, e = round(rng.uniform(1.0, 9.9), 1), rng.randint(1, 5)
        c = m * 10 ** -e
        return (f"What is the pH of a {c:.1e} M solution of a strong monoprotic acid at 25 C? (2 decimal places)",
                -math.log10(c), {"cmp": "float", "abs": 0.02})
    if k == 2:
        g = round(rng.uniform(2, 60 * d), 1)
        return (f"For 2 H2 + O2 -> 2 H2O, how many grams of water form from {g} g of H2 with excess O2? "
                f"(H=1.008, O=15.999)", g / (2 * 1.008) * (2 * 1.008 + 15.999), {"cmp": "float", "rel": 0.01})
    if k == 3:
        c1, v1, c2 = round(rng.uniform(2, 12), 1), rng.randint(10, 100), round(rng.uniform(0.1, 1.9), 2)
        return (f"A {c1} M stock solution (V1 = {v1} mL) is diluted to {c2} M. What is the final volume in mL?",
                c1 * v1 / c2, {"cmp": "float", "rel": 0.01})
    f, counts = rng.choice(FORMULAS)
    mm = sum(ATOMIC[e] * n for e, n in counts.items())
    g = rng.randint(5, 200)
    return (f"Using atomic masses ({atoms_txt}), how many moles are in {g} g of {f}?", g / mm,
            {"cmp": "float", "rel": 0.01})


# ---- code tasks: spec, reference (trusted, host-executed), hidden-input generator ------------------------
def _ints(rng, n, lo=-20, hi=20):
    return [rng.randint(lo, hi) for _ in range(rng.randint(0, n))]


def _ints_gen(r, n, lo=-20, hi=20):
    return [r.randint(lo, hi) for _ in range(r.randint(0, n))]


def _f_filtered_sum(r):
    op, opsrc = r.choice([("squares", "x*x"), ("cubes", "x*x*x"), ("absolute values", "abs(x)"), ("values", "x")])
    kind = r.randrange(4)
    if kind == 0:
        cond, csrc = r.choice([("even", "x % 2 == 0"), ("odd", "x % 2 != 0"), ("negative", "x < 0")])
    elif kind == 1:
        k = r.randint(3, 9)
        cond, csrc = f"divisible by {k}", f"x % {k} == 0"
    elif kind == 2:
        t = r.randint(-5, 10)
        cond, csrc = f"greater than {t}", f"x > {t}"
    else:
        t = r.randint(2, 12)
        cond, csrc = f"with absolute value at most {t}", f"abs(x) <= {t}"
    return dict(name="filtered_sum", spec=f"Write `filtered_sum(xs)` returning the sum of the {op} of those integers in the list xs that are {cond} (0 if none).",
                ref=f"def filtered_sum(xs):\n    return sum({opsrc} for x in xs if {csrc})\n",
                gen=lambda rr: [[_ints_gen(rr, 12)] for _ in range(12)] + [[[]], [[0]]])


def _f_kth(r):
    k = r.randint(1, 4)
    big = r.random() < 0.5
    word = "largest" if big else "smallest"
    return dict(name="kth_value", spec=f"Write `kth_value(xs)` returning the {k}-th {word} DISTINCT value of the integer list xs, or None if there are fewer than {k} distinct values.",
                ref=f"def kth_value(xs):\n    s = sorted(set(xs), reverse={big})\n    return s[{k - 1}] if len(s) >= {k} else None\n",
                gen=lambda rr: [[_ints_gen(rr, 10, -9, 9)] for _ in range(12)] + [[[]], [[5]]])


def _f_rle(r):
    order = r.choice(["char_first", "count_first"])
    skip1 = r.random() < 0.5
    desc = ("<char><count>" if order == "char_first" else "<count><char>") + (", omitting the count when it is 1" if skip1 else "")
    ref = ("def rle(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
           f"        n = j - i\n        c = '' if ({skip1} and n == 1) else str(n)\n        out.append(" + ("s[i] + c" if order == "char_first" else "c + s[i]") + ")\n        i = j\n    return ''.join(out)\n")
    return dict(name="rle", spec=f"Write `rle(s)` that run-length encodes the string s as {desc} for each run of equal characters (e.g. 'aaab' with <char><count> gives 'a3b1'). The empty string gives ''.",
                ref=ref, gen=lambda rr: [["".join(rr.choice("abc") * rr.randint(1, 4) for _ in range(rr.randint(0, 6)))] for _ in range(12)] + [[""], ["a"]])


def _f_brackets(r):
    kinds = r.choice([("()", "(", {")": "("}), ("()[]", "([", {")": "(", "]": "["}), ("()[]{}", "([{", {")": "(", "]": "[", "}": "{"})])
    alpha, opens, pairs = kinds
    names = alpha
    return dict(name="is_balanced", spec=f"Write `is_balanced(s)` returning True if the brackets {names} in string s are correctly balanced and nested (all other characters ignored), else False.",
                ref=f"def is_balanced(s):\n    pairs = {pairs!r}\n    st = []\n    for ch in s:\n        if ch in {opens!r}:\n            st.append(ch)\n        elif ch in pairs:\n            if not st or st.pop() != pairs[ch]:\n                return False\n    return not st\n",
                gen=lambda rr: [["".join(rr.choice(alpha + "a") for _ in range(rr.randint(0, 10)))] for _ in range(14)] + [[""], [alpha]])


def _f_affix(r):
    suffix = r.random() < 0.5
    word = "suffix" if suffix else "prefix"
    ref = ("def common_affix(strs):\n    if not strs:\n        return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s." +
           ("endswith" if suffix else "startswith") + "(p):\n            p = " + ("p[1:]" if suffix else "p[:-1]") + "\n    return p\n")
    return dict(name="common_affix", spec=f"Write `common_affix(strs)` returning the longest common {word} of a list of strings ('' if the list is empty or there is none).",
                ref=ref, gen=lambda rr: [[[("abab" * 3)[: rr.randint(0, 6)] + "".join(rr.choice("xyz") for _ in range(rr.randint(0, 3))) for _ in range(rr.randint(0, 4))]] for _ in range(12)] + [[[]], [[ "flower", "flow", "flight"]]])


def _f_recurrence(r):
    a, b = r.randint(0, 6), r.randint(1, 6)
    return dict(name="rec_mod", spec=f"Write `rec_mod(n, m)` returning the n-th term modulo m of the sequence T(0)={a}, T(1)={b}, T(k)=T(k-1)+T(k-2) for k>=2. n may be as large as 10**6, so use an iterative method.",
                ref=f"def rec_mod(n, m):\n    x, y = {a}, {b}\n    for _ in range(n):\n        x, y = y, x + y\n    return x % m\n",
                gen=lambda rr: [[rr.randint(0, 150), rr.randint(2, 1000)] for _ in range(10)] + [[0, 5], [1, 5]])


def _f_dedupe(r):
    last = r.random() < 0.5
    which = "LAST" if last else "FIRST"
    ref = ("def dedupe(xs):\n    seen = set()\n    out = []\n    for x in " + ("reversed(xs)" if last else "xs") +
           ":\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return " + ("out[::-1]" if last else "out") + "\n")
    return dict(name="dedupe", spec=f"Write `dedupe(xs)` returning xs with duplicates removed, keeping the {which} occurrence of each element and the relative order of the kept elements.",
                ref=ref, gen=lambda rr: [[_ints_gen(rr, 14, 0, 6)] for _ in range(12)] + [[[]]])


def _f_window(r):
    k = r.randint(2, 4)
    mx = r.random() < 0.5
    return dict(name="window_best", spec=f"Write `window_best(xs)` returning the {'maximum' if mx else 'minimum'} sum over all contiguous windows of EXACTLY {k} elements of the list xs, or None if len(xs) < {k}.",
                ref=f"def window_best(xs):\n    if len(xs) < {k}:\n        return None\n    return {'max' if mx else 'min'}(sum(xs[i:i+{k}]) for i in range(len(xs) - {k} + 1))\n",
                gen=lambda rr: [[[rr.randint(-9, 9) for _ in range(rr.randint(0, 10))]] for _ in range(14)])


def _f_roman(r):
    if r.random() < 0.5:
        return dict(name="roman_to_int", spec="Write `roman_to_int(s)` converting a valid Roman numeral string (1..3999) to an integer.",
                    ref="def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t = 0\n    for i, ch in enumerate(s):\n        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:\n            t -= v[ch]\n        else:\n            t += v[ch]\n    return t\n",
                    gen=lambda rr: [[_to_roman(rr.randint(1, 3999))] for _ in range(14)] + [["IV"], ["MCMXCIV"]])
    return dict(name="int_to_roman", spec="Write `int_to_roman(n)` converting an integer 1..3999 to its standard Roman numeral string (subtractive notation).",
                ref="def int_to_roman(n):\n    out = ''\n    for v, s in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:\n        while n >= v:\n            out += s\n            n -= v\n    return out\n",
                gen=lambda rr: [[rr.randint(1, 3999)] for _ in range(14)] + [[4], [3999]])


def _f_matrix(r):
    op = r.choice(["transposed", "rotated 90 degrees clockwise", "rotated 90 degrees counter-clockwise", "flipped upside down (rows reversed)", "flipped left-right"])
    body = {"transposed": "[list(r) for r in zip(*m)]", "rotated 90 degrees clockwise": "[list(r) for r in zip(*m[::-1])]",
            "rotated 90 degrees counter-clockwise": "[list(r) for r in zip(*m)][::-1]",
            "flipped upside down (rows reversed)": "[list(r) for r in m[::-1]]", "flipped left-right": "[list(r[::-1]) for r in m]"}[op]
    return dict(name="reshape", spec=f"Write `reshape(m)` returning the non-empty rectangular matrix m (list of row lists) {op}, as a new list of lists.",
                ref=f"def reshape(m):\n    return {body}\n",
                gen=lambda rr: [[[[rr.randint(0, 9) for _ in range(c)] for _ in range(rw)]] for rw, c in [(rr.randint(1, 4), rr.randint(1, 4)) for _ in range(10)]])


def _f_pairs(r):
    t = r.randint(-3, 12)
    return dict(name="count_pairs", spec=f"Write `count_pairs(xs)` returning the number of index pairs i < j with xs[i] + xs[j] == {t}.",
                ref=f"def count_pairs(xs):\n    return sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[i] + xs[j] == {t})\n",
                gen=lambda rr: [[_ints_gen(rr, 12, -3, 9)] for _ in range(14)] + [[[]]])


def _f_caesar(r):
    k = r.randint(1, 25)
    return dict(name="shift", spec=f"Write `shift(s)` applying a Caesar shift of +{k} to every ASCII letter of s (preserving case, wrapping around the alphabet) and leaving all other characters unchanged.",
                ref=f"def shift(s):\n    out = []\n    for ch in s:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - 97 + {k}) % 26 + 97))\n        elif 'A' <= ch <= 'Z':\n            out.append(chr((ord(ch) - 65 + {k}) % 26 + 65))\n        else:\n            out.append(ch)\n    return ''.join(out)\n",
                gen=lambda rr: [["".join(rr.choice("abcxyzABCXYZ 1!") for _ in range(rr.randint(0, 12)))] for _ in range(12)] + [[""]])


CODE_FACTORIES = [_f_filtered_sum, _f_kth, _f_rle, _f_brackets, _f_affix, _f_recurrence, _f_dedupe, _f_window,
                  _f_roman, _f_matrix, _f_pairs, _f_caesar]


def _to_roman(n):
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                 (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out += s
            n -= v
    return out


def _run_ref(src: str, name: str, inputs: list):
    ns: dict = {}
    exec(compile(src, "<reference>", "exec"), ns)       # trusted code written in this file
    fn = ns[name]
    return json.loads(json.dumps([fn(*a) for a in inputs]))


def g_code(rng, d):
    t = CODE_FACTORIES[rng.randrange(len(CODE_FACTORIES))](rng)
    inputs = t["gen"](rng)
    expected = _run_ref(t["ref"], t["name"], inputs)
    return (t["spec"], expected, {"cmp": "code", "fn": t["name"], "inputs": inputs, "ref_src": t["ref"]})


def g_arith(rng, d):
    """Learnable ladder for small models: level 1 two-digit sums < 100, level 2 full two-digit sums, level 3 three-digit sums (a capability cliff)."""
    lo, hi = {1: (10, 49), 2: (10, 99), 3: (100, 999)}[min(3, max(1, d))]
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    return (f"What is {a} + {b}?", a + b, {"cmp": "int"})


GENERATORS = {"arith": g_arith, "math": g_math, "logic": g_logic, "physics": g_physics, "chemistry": g_chem, "code": g_code}


# =============================================================================== answer checking
_ANS = re.compile(r"ANSWER\s*:\s*(.+)", re.I)


def extract_answer(text: str):
    m = _ANS.findall(text or "")
    if not m:
        return None
    a = m[-1].strip().strip("`*").strip()
    return re.sub(r"\s*\.$", "", a)


def _num(s: str):
    s = s.replace(",", "").replace("−", "-")
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None


def extract_code(text: str):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return blocks[-1]
    return text if "def " in (text or "") else None


def check_text(p: Problem, answer_text: str) -> tuple[bool, str]:
    a = extract_answer(answer_text)
    if a is None:
        return False, "no ANSWER line"
    cmp = p.meta["cmp"]
    if cmp == "int":
        v = _num(a)
        return (v is not None and abs(v - round(v)) < 1e-9 and int(round(v)) == int(p.answer)), f"got {a!r}"
    if cmp == "float":
        v = _num(a)
        if v is None:
            return False, f"not a number: {a!r}"
        if "abs" in p.meta:
            return abs(v - p.answer) <= p.meta["abs"], f"got {v}"
        return abs(v - p.answer) <= p.meta["rel"] * abs(p.answer), f"got {v}"
    return re.sub(r"[^a-z0-9]", "", a.lower()) == re.sub(r"[^a-z0-9]", "", str(p.answer).lower()), f"got {a!r}"


HARNESS = '''

import json as __json
import signal as __signal
def __alarm(*_):
    raise TimeoutError("test timeout")
__signal.signal(__signal.SIGALRM, __alarm)
__inputs = __json.load(open("inputs.json"))
__out = []
for __a in __inputs:
    __signal.alarm(3)
    try:
        __r = {fn}(*__a)
        __json.dumps(__r)
        __out.append({{"ok": True, "v": __r}})
    except BaseException as __e:
        __out.append({{"ok": False, "e": type(__e).__name__}})
    finally:
        __signal.alarm(0)
print("SIRA_CODE_OUT:" + __json.dumps(__out))
'''


def check_code(p: Problem, answer_text: str, sandbox) -> tuple[bool, str]:
    code = extract_code(answer_text)
    if not code:
        return False, "no code block"
    fn = p.meta["fn"]
    src = code + HARNESS.format(fn=fn)
    r = sandbox.run(src, files={"inputs.json": json.dumps(p.meta["inputs"])}, timeout=25, network="none")
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("SIRA_CODE_OUT:")), None)
    if line is None:
        return False, f"no output (rc={r.exit_code}, timeout={r.timed_out}) {r.stderr[-120:]!r}"
    try:
        out = json.loads(line[len("SIRA_CODE_OUT:"):])
    except Exception:
        return False, "unparseable output"
    exp = p.answer
    if len(out) != len(exp):
        return False, "wrong number of outputs"
    bad = sum(1 for o, e in zip(out, exp) if not (o.get("ok") and o.get("v") == e))
    return bad == 0, f"{len(exp) - bad}/{len(exp)} hidden tests passed"


def oracle_response(p: Problem) -> str:
    """The programmatic solver's answer in the required format (used ONLY for training-namespace problems)."""
    if p.kind == "code":
        return "```python\n" + p.meta["ref_src"] + "```"
    a = p.answer
    return f"ANSWER: {a:.4g}" if isinstance(a, float) else f"ANSWER: {a}"


# =============================================================================== EvalCore
@dataclass
class EvalResult:
    total: float
    n: int
    by_domain: dict
    details: list
    by_level: dict = field(default_factory=dict)       # "domain:level" -> {passed, n, rate}: finer capabilities for the retention guard


class EvalCore:
    def __init__(self, run_dir, db, domains=None):
        self.run_dir, self.db = Path(run_dir), db
        self.domains = [d for d in (domains or DOMAINS) if d in GENERATORS] or list(DOMAINS)
        key_file = self.run_dir / "evaluator.key"
        if not key_file.exists():
            key_file.write_text(secrets.token_hex(32))
            os.chmod(key_file, 0o600)
        self._key = key_file.read_text().strip().encode()
        self.train_hashes: set = set(db.get_meta("train_hashes", []))

    def _seed(self, ns: str, set_idx: int, i: int, attempt: int = 0) -> int:
        msg = f"{ns}:{set_idx}:{i}:{attempt}".encode()
        return int.from_bytes(hmac.new(self._key, msg, hashlib.sha256).digest()[:8], "big")

    def make_problem(self, ns, set_idx, i, domain, difficulty, attempt=0) -> Problem:
        seed = self._seed(ns, set_idx, i, attempt)
        rng = random.Random(seed)
        text, ans, meta = GENERATORS[domain](rng, difficulty)
        kind = "code" if domain == "code" else "text"
        prompt = text + (CODE_SUFFIX if kind == "code" else TEXT_SUFFIX)
        return Problem(id=f"{ns}-{set_idx}-{i}", domain=domain, kind=kind, prompt=prompt,
                       difficulty=difficulty, answer=ans, meta=meta)

    def make_set(self, ns: str, n: int, domains=None, weights: dict | None = None) -> list:
        """Fresh set. `weights` biases domains (e.g. towards weak ones) for TRAIN sets only."""
        domains = list(domains or self.domains)
        set_idx = self.db.next_index(f"set:{ns}")
        if weights:
            pool = []
            for dom in domains:
                pool += [dom] * max(1, int(round(weights.get(dom, 1.0) * 4)))
        else:
            pool = domains
        probs, seen = [], set()
        for i in range(n):
            dom = pool[i % len(pool)]
            diff = 1 + (i // len(pool)) % 3
            for attempt in range(300):
                p = self.make_problem(ns, set_idx, i, dom, diff, attempt)
                h = hashlib.sha1(p.prompt.encode()).hexdigest()[:12]
                if h in seen or (ns != "train" and h in self.train_hashes):
                    continue
                seen.add(h)
                break
            probs.append(p)
        if ns == "train":
            self.train_hashes |= seen
            self.db.set_meta("train_hashes", sorted(self.train_hashes))
        return probs

    def public_examples(self, k=5) -> list:
        """What the agents may know: the *kind* of problems (separate namespace, not used for scoring)."""
        out = []
        for i in range(k):
            dom = self.domains[i % len(self.domains)]
            p = self.make_problem("public", 0, i, dom, 1 + i % 3)
            out.append({"domain": p.domain, "example_prompt": p.prompt.split("\n\n")[0][:300]})
        return out

    def verify(self, p: Problem, answer: str, sandbox) -> tuple[bool, str]:
        return check_code(p, answer, sandbox) if p.kind == "code" else check_text(p, answer)

    def score(self, problems: list, answers: dict, sandbox, workers: int = 3) -> EvalResult:
        def one(p):
            ans = answers.get(p.id, "")
            try:
                ok, why = self.verify(p, ans, sandbox)
            except Exception as e:           # a verifier crash is a failed answer, never a pass
                if type(e).__name__ == "Stopped":
                    raise
                ok, why = False, f"verifier error: {type(e).__name__}"
            return {"id": p.id, "domain": p.domain, "level": p.difficulty, "passed": bool(ok), "reason": why}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            details = list(ex.map(one, problems))
        by, lv = {}, {}
        for d in details:
            for table, key in ((by, d["domain"]), (lv, f"{d['domain']}:{d['level']}")):
                a = table.setdefault(key, [0, 0])
                a[0] += d["passed"]
                a[1] += 1
        total = sum(d["passed"] for d in details) / max(1, len(details))
        fmt = lambda t: {k: {"passed": v[0], "n": v[1], "rate": v[0] / v[1]} for k, v in t.items()}
        return EvalResult(total, len(details), fmt(by), details, fmt(lv))

    # chat-format prompt for the model under improvement
    @staticmethod
    def prompt_for(p: Problem) -> str:
        return p.prompt
