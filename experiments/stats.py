"""Statistics used in the report: mean, sample SD, exact t-based 95% CI, paired t-test, Wilcoxon signed-rank, Cohen's d (paired)."""
import math
from scipy import stats as st


def summary(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    half = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "ci95": (m - half, m + half), "min": min(xs), "max": max(xs)}


def paired(a, b):
    """a, b: matched samples (same seeds). Returns mean difference with 95% CI, t-test p, Wilcoxon p, Cohen's d."""
    d = [x - y for x, y in zip(a, b)]
    s = summary(d)
    out = {"mean_diff": s["mean"], "sd_diff": s["sd"], "ci95": s["ci95"], "n": len(d)}
    if len(d) > 1 and s["sd"] > 0:
        out["t_p"] = float(st.ttest_1samp(d, 0.0).pvalue)
        out["cohen_d"] = s["mean"] / s["sd"]
        try:
            out["wilcoxon_p"] = float(st.wilcoxon(d).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    return out


def fmt(s, pct=True):
    k = 100 if pct else 1
    return f"{s['mean'] * k:.1f} +- {s['sd'] * k:.1f} (95% CI {s['ci95'][0] * k:.1f} to {s['ci95'][1] * k:.1f}, n={s['n']})"
