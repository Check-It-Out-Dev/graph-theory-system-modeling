"""The statistics of the prompt-under-test pipeline (eval/put/METRICS.md holds the definitions in prose).

    wilson(k, n, z)                    Wilson score interval of a proportion
    obligatory(k, n, bound, z)         True when the lower Wilson bound reaches the bound (default 0.90)
    mad(pairs)                         mean absolute difference of paired values (judge test-retest)
    pooled_sd(groups)                  pooled standard deviation of replicate groups
    delta(judge_mad, pooled, T, k, z)  the noise floor: max(judge MAD, z * pooled SD * sqrt(2 / (T k)))
    paired_bootstrap(a, b, n, seed)    CI of the mean per-task difference b - a, resampling tasks then replicates
    sign_test(diffs)                   one-sided exact sign test P(X >= x | n, 1/2) over non-zero differences
    kappa(x, y), ac1(x, y)             Cohen's kappa and Gwet's AC1 for two binary raters, with agreement and prevalence
    spearman(x, y)                     rank correlation with average ranks for ties

Stdlib only, deterministic (seeded), so a published figure can be recomputed from the committed artifacts.
"""

import math
import random
from statistics import mean


def wilson(k, n, z=1.96):
    """-> (lo, hi) of the Wilson score interval; (0, 1) for n == 0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def obligatory(k, n, bound=0.90, z=1.96):
    return n > 0 and wilson(k, n, z)[0] >= bound


def min_n_for_bound(bound=0.90, z=1.96):
    """Smallest n at which n passes out of n reach the bound (0 failures): n / (n + z^2) >= bound."""
    n = 1
    while wilson(n, n, z)[0] < bound:
        n += 1
    return n


def mad(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return mean(abs(a - b) for a, b in pairs) if pairs else 0.0


def pooled_sd(groups):
    """Pooled SD over replicate groups (each a list of values of one task); groups of one value add nothing."""
    num = den = 0.0
    for g in groups:
        g = [v for v in g if v is not None]
        if len(g) < 2:
            continue
        m = mean(g)
        num += sum((v - m) ** 2 for v in g)
        den += len(g) - 1
    return math.sqrt(num / den) if den else 0.0


def delta(judge_mad, pooled, tasks, k, z=1.96):
    agent = z * pooled * math.sqrt(2.0 / (tasks * k)) if tasks and k else 0.0
    return max(judge_mad, agent), agent


def paired_bootstrap(a, b, resamples=10000, seed=20260923, alpha=0.05):
    """a, b: {task: [replicate scores]} for two arms on the same tasks. -> (mean diff, lo, hi).
    Each resample draws tasks with replacement, then within each drawn task draws each arm's replicates with
    replacement, and averages the per-task differences."""
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return 0.0, 0.0, 0.0
    point = mean(mean(b[t]) - mean(a[t]) for t in tasks)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        diffs = []
        for _ in tasks:
            t = tasks[rng.randrange(len(tasks))]
            ra = [a[t][rng.randrange(len(a[t]))] for _ in a[t]]
            rb = [b[t][rng.randrange(len(b[t]))] for _ in b[t]]
            diffs.append(mean(rb) - mean(ra))
        stats.append(mean(diffs))
    stats.sort()
    lo = stats[int(math.floor(alpha / 2 * resamples))]
    hi = stats[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]
    return point, lo, hi


def sign_test(diffs):
    """-> (positives, n, one-sided p) over the non-zero differences."""
    nz = [d for d in diffs if d != 0]
    n, x = len(nz), sum(1 for d in nz if d > 0)
    p = sum(math.comb(n, i) for i in range(x, n + 1)) / (2 ** n) if n else 1.0
    return x, n, p


def _binary_table(x, y):
    n = len(x)
    p_o = sum(1 for a, b in zip(x, y) if a == b) / n
    p1 = sum(x) / n
    p2 = sum(y) / n
    return n, p_o, p1, p2


def kappa(x, y):
    """Cohen's kappa for binary ratings (lists of 0/1). -> dict with kappa, agreement, prevalence."""
    n, p_o, p1, p2 = _binary_table(x, y)
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    k = (p_o - p_e) / (1 - p_e) if p_e < 1 else (1.0 if p_o == 1 else 0.0)
    return {"kappa": k, "agreement": p_o, "prevalence": (p1 + p2) / 2, "n": n}


def ac1(x, y):
    """Gwet's AC1 for binary ratings: chance agreement 2 pi (1 - pi), pi the mean yes-rate of both raters."""
    n, p_o, p1, p2 = _binary_table(x, y)
    pi = (p1 + p2) / 2
    p_e = 2 * pi * (1 - pi)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0
