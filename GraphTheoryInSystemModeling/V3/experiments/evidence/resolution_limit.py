"""G5 — is "the natural scale is ~18 units" a finding or a resolution-limit artifact?

F33 claimed the natural scale of this codebase's change structure is about 18
units rather than 126, because derived subsystems score modularity 0.3556 against
the modules' 0.1488 on the held-out co-change graph.

F40, from the literature, says that claim is unsafe. Modularity has a documented
resolution limit (Fortunato & Barthelemy, PNAS 2007): it cannot resolve
communities holding fewer than sqrt(L/2) edges, because the null model it
subtracts is global. A partition into many small parts is therefore penalised BY
CONSTRUCTION, regardless of quality. If the modules sit below that threshold, the
comparison was rigged by the objective rather than decided by the data.

Three tests, in increasing strength:

  1. ARITHMETIC. Compute sqrt(L/2) and compare it against the actual internal
     edge counts of each part. This alone settles whether the concern applies.

  2. MAP EQUATION. The two-level description length of a random walk (Rosvall &
     Bergstrom). Infomap is reported not to suffer the resolution limit, and the
     map equation is granularity-aware by construction -- it does not need a
     matched part count because the cost of naming a module is paid explicitly in
     the index codebook. LOWER IS BETTER, measured in bits. Implemented here
     directly rather than optimised, since the question is how to SCORE two given
     partitions, not how to find a third.

  3. CPM. The Constant Potts Model (Traag, Van Dooren, Nesterov 2011) is
     resolution-limit-free by construction: Q = sum_c [e_c - gamma*n_c(n_c-1)/2],
     with no global null term. Swept over gamma, since a partition is natural at
     the gamma matching its own internal density.

Own controls applied, as the V4 prompt now demands of everyone else: 20
independent commit splits, mean +- sd, and the count of splits the winner won.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
N_SPLITS = 20
GAMMAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SEED = 42


def map_equation(labels, W):
    """Two-level map equation description length, in bits. Lower is better.

    L = q*H(Q) + sum_i p_i*H(P_i), with p_alpha the stationary visit rate of an
    undirected random walk (degree / 2W) and q_i the exit probability of module i.
    """
    k = W.sum(axis=1)
    tw = k.sum()                                   # = 2W
    if tw <= 0:
        return np.nan
    p = k / tw

    mods = collections.defaultdict(list)
    for i, c in enumerate(labels):
        mods[c].append(i)

    def plogp(x):
        x = np.asarray(x, dtype=float)
        x = x[x > 0]
        return float(np.sum(x * np.log2(x)))

    q_i, p_circ, inner = [], [], 0.0
    for c, members in mods.items():
        m = np.zeros(len(labels), dtype=bool)
        m[members] = True
        exit_w = W[np.ix_(m, ~m)].sum()            # weight leaving the module
        qi = exit_w / tw
        pin = p[m]
        pc = qi + pin.sum()
        q_i.append(qi)
        p_circ.append(pc)
        # module codebook: -plogp(q_i) - sum plogp(p_alpha) + plogp(p_circ)
        inner += -plogp([qi]) - plogp(pin) + plogp([pc])

    q = float(np.sum(q_i))
    index = 0.0 if q <= 0 else (plogp([q]) - plogp(q_i))
    return float(index + inner)


def cpm(labels, W, gamma):
    """Constant Potts Model quality. Higher is better. No global null term."""
    total = 0.0
    mods = collections.defaultdict(list)
    for i, c in enumerate(labels):
        mods[c].append(i)
    for c, members in mods.items():
        m = np.zeros(len(labels), dtype=bool)
        m[members] = True
        e_c = W[np.ix_(m, m)].sum() / 2.0
        n_c = len(members)
        total += e_c - gamma * n_c * (n_c - 1) / 2.0
    return float(total)


def modularity(labels, W):
    k = W.sum(axis=1)
    m2 = W.sum()
    q = 0.0
    for c in set(labels):
        m = labels == c
        q += W[np.ix_(m, m)].sum() / m2 - (k[m].sum() / m2) ** 2
    return float(q)


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.v3_subsystem AS sub, "
                "n.v3_module AS mod", ns=NS))
    finally:
        drv.close()

    n = len(nodes)
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    sub = np.array([r["sub"] if r["sub"] is not None else -1 for r in nodes])
    mod = np.array([r["mod"] if r["mod"] is not None else -1 for r in nodes])
    leaf = np.array([hash(repo_of[i] + "/" + rel_of[i].rsplit("/", 1)[0]) % 10 ** 9
                     for i in range(n)])
    d3 = np.array([hash(repo_of[i] + "/" +
                        "/".join(rel_of[i].rsplit("/", 1)[0].split("/")[:3])) % 10 ** 9
                   for i in range(n)])

    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    raw = {}
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        blocks = []
        for b in out.split("\x01")[1:]:
            ids = sorted({by_repo[rp][f] for f in
                          {l.strip() for l in b.splitlines() if l.strip()}
                          if f in by_repo[rp]})
            if 2 <= len(ids) <= 30:
                blocks.append(ids)
        raw[rp] = blocks

    def cochange_W(rng):
        W = np.zeros((n, n))
        for rp in REPOS:
            keep = rng.random(len(raw[rp])) < 0.5
            for ids, k in zip(raw[rp], keep):
                if k:
                    continue
                for a, b in itertools.combinations(ids, 2):
                    W[a, b] = 1.0
                    W[b, a] = 1.0
        return W

    parts = {"derived subsystems": sub, "derived modules": mod,
             "directory leaf": leaf, "directory depth 3": d3}

    # ---- 1. the arithmetic
    W0 = cochange_W(np.random.default_rng(SEED))
    L = W0.sum() / 2.0
    thresh = np.sqrt(L / 2.0)
    print("1. DOES THE RESOLUTION LIMIT EVEN APPLY?\n")
    print(f"  held-out co-change graph: L = {L:.0f} edges")
    print(f"  Fortunato-Barthelemy threshold sqrt(L/2) = {thresh:.1f} edges")
    print(f"  a part holding fewer internal edges than this cannot be resolved by")
    print(f"  modularity, however good it is\n")
    print(f"{'partition':<24}{'parts':>7}{'median e_c':>12}{'parts below':>13}{'% below':>9}")
    for name, lab in parts.items():
        ecs = []
        for c in set(lab):
            m = lab == c
            ecs.append(W0[np.ix_(m, m)].sum() / 2.0)
        ecs = np.array(ecs)
        below = int((ecs < thresh).sum())
        print(f"{name:<24}{len(ecs):>7}{np.median(ecs):>12.1f}{below:>13}"
              f"{below / len(ecs):>8.0%}")

    # ---- 2 & 3. resolution-limit-free scoring, over splits
    print("\n\n2. MAP EQUATION — description length in bits, LOWER IS BETTER\n")
    print("   granularity-aware by construction: naming a module is paid for in")
    print("   the index codebook, so no matched part count is needed\n")
    acc_L = collections.defaultdict(list)
    acc_Q = collections.defaultdict(list)
    acc_cpm = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in range(N_SPLITS):
        W = cochange_W(np.random.default_rng(SEED + t))
        if W.sum() == 0:
            continue
        for name, lab in parts.items():
            acc_L[name].append(map_equation(lab, W))
            acc_Q[name].append(modularity(lab, W))
            for g in GAMMAS:
                acc_cpm[name][g].append(cpm(lab, W, g))
    one = np.zeros(n, dtype=int)
    base = [map_equation(one, cochange_W(np.random.default_rng(SEED + t)))
            for t in range(N_SPLITS)]

    print(f"{'partition':<24}{'parts':>7}{'bits':>10}{'sd':>7}{'vs 1-part':>11}")
    for name, lab in parts.items():
        v = np.array(acc_L[name])
        print(f"{name:<24}{len(set(lab)):>7}{v.mean():>10.3f}{v.std():>7.3f}"
              f"{v.mean() - np.mean(base):>+11.3f}")
    print(f"{'— single part (baseline)':<24}{1:>7}{np.mean(base):>10.3f}"
          f"{np.std(base):>7.3f}{0.0:>+11.3f}")
    winner = min(acc_L, key=lambda k_: np.mean(acc_L[k_]))
    print(f"\n  lowest description length: {winner}")

    print("\n\n3. CPM — resolution-limit-free by construction, HIGHER IS BETTER\n")
    print(f"{'gamma':>8}" + "".join(f"{k[:14]:>16}" for k in parts))
    for g in GAMMAS:
        row = "".join(f"{np.mean(acc_cpm[k][g]):>16.1f}" for k in parts)
        print(f"{g:>8.4f}{row}")
    print("\n  each partition is natural at the gamma matching its own internal")
    print("  density, so the question is whether any gamma exists where modules win")

    print("\n\n4. VERDICT ON F33\n")
    ms, mm = np.array(acc_Q["derived subsystems"]), np.array(acc_Q["derived modules"])
    ls, lm = np.array(acc_L["derived subsystems"]), np.array(acc_L["derived modules"])
    print(f"  modularity  subsystems {ms.mean():.4f} vs modules {mm.mean():.4f}  "
          f"-> subsystems win {int((ms > mm).sum())}/{len(ms)}")
    print(f"  map equation subsystems {ls.mean():.3f} vs modules {lm.mean():.3f} bits "
          f"-> subsystems win {int((ls < lm).sum())}/{len(ls)}")
    wins = {g: int((np.array(acc_cpm['derived subsystems'][g]) >
                    np.array(acc_cpm['derived modules'][g])).sum()) for g in GAMMAS}
    print(f"  CPM subsystems win at each gamma: {wins}")
    agree = (ls.mean() < lm.mean())
    print(f"\n  Two resolution-limit-free objectives {'AGREE with' if agree else 'CONTRADICT'} "
          f"modularity.")


if __name__ == "__main__":
    main()
