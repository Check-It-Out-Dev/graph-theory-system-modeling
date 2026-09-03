"""#54 — the full picture: observe the whole algebraic topology at once.

Every result in this programme so far has been a table. This renders the object
itself, in one figure, so the structure the numbers describe can be SEEN:

  A  the base space — content-kNN spring layout, nodes coloured by the promoted
     v4_subsystem, shaped by entity type (the layers), with the 92 meta-path
     fibers drawn through their hubs. The topology, the partition and the fibers
     in one frame.
  B  the free layered view — subsystem x layer occupancy. The owner's point made
     visible: a subsystem is a vertical slice, so every column should MIX layers;
     a pure column would be a layer masquerading as a subsystem.
  C  the meet — contingency of the two parents (content partition x cohort-fiber
     partition). The blocks are the ~136 lattice cells the promoted operator
     works on: dark block = agreement, scattered mass = the disagreements the
     quotient resolves.
  D  the fiber dual — the 92 meta-path fibers as nodes, shared-member edges,
     Louvain communities, size = arity, colour = community. "Clusters of fibers"
     as an actual object.

Output: figures/v3_full_topology.png (300 dpi) + printed observations.
"""
import collections
import itertools
import math
import os
import subprocess

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
KNN, SEED, COHORT_RES = 12, 42, 2.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
LAYERS = ["Actor", "Process", "Resource", "Rule", "Context", "Event"]
MARKS = {"Actor": "^", "Process": "o", "Resource": "s",
         "Rule": "v", "Context": "D", "Event": "*"}


def main():
    os.makedirs(OUT, exist_ok=True)
    drv = GraphDatabase.driver(URI, auth=AUTH)
    with drv.session() as s:
        nodes = list(s.run(
            "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
            "RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
            "n.embedding AS emb, n.v4_subsystem AS v4", ns=NS))
        hyper = list(s.run(
            "MATCH (h:HyperedgeCandidate {namespace:$ns}) "
            "RETURN h.metapath AS mp, h.member_ids AS members, h.hub_id AS hub, "
            "h.idf_weight AS idf", ns=NS))
    drv.close()

    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    et = np.array([r["et"] for r in nodes])
    v4 = np.array([r["v4"] if r["v4"] is not None else -1 for r in nodes])
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    # content kNN graph + champion A (parent 1)
    Sq = X @ X.T
    np.fill_diagonal(Sq, -np.inf)
    W0 = np.zeros((n, n))
    knn_lists = {}
    for i in range(n):
        row = Sq[i].copy()
        row[[j for j in range(n) if repo_of[j] != repo_of[i]]] = -np.inf
        top = [int(j) for j in np.argpartition(-row, KNN)[:KNN]
               if np.isfinite(row[j]) and row[j] > 0]
        knn_lists[i] = top
        for j in top:
            W0[i, j] = W0[j, i] = max(W0[i, j], row[j])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b in np.argwhere(np.triu(W0, 1) > 0):
        G.add_edge(int(a), int(b), weight=float(W0[a, b]))
    cm = nx.community.louvain_communities(G, weight="weight", resolution=1.18, seed=SEED)
    A = np.full(n, -1)
    for c, grp in enumerate(cm):
        for v in grp:
            A[v] = c

    # cohort-fiber B (parent 2), full data
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    commits = []
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        for b in out.split("\x01")[1:]:
            ids = sorted({by_repo[rp][f] for f in
                          {l.strip() for l in b.splitlines() if l.strip()}
                          if f in by_repo[rp]})
            if 2 <= len(ids) <= 30:
                commits.append(ids)
    fibers = []
    meta_span = []
    for h in hyper:
        mem = [idx[m] for m in h["members"] if m in idx]
        if h["hub"] in idx:
            mem.append(idx[h["hub"]])
        mem = sorted(set(mem))
        if len(mem) >= 2:
            fibers.append((mem, max(h["idf"], 0.05)))
            meta_span.append((h["mp"], mem, idx.get(h["hub"]), h["idf"]))
    n_meta = len(fibers)
    for ids in commits:
        fibers.append((ids, 1.0 / math.log(2 + len(ids))))
    node_f = collections.defaultdict(list)
    for fi, (mem, w) in enumerate(fibers):
        for m in mem:
            node_f[m].append(fi)
    FE = collections.Counter()
    for m, fl in node_f.items():
        if len(fl) > 1:
            for a, b in itertools.combinations(fl, 2):
                FE[(a, b)] += 1
    FG = nx.Graph()
    FG.add_nodes_from(range(len(fibers)))
    for (a, b), sh in FE.items():
        FG.add_edge(a, b, weight=sh * (fibers[a][1] + fibers[b][1]) / 2.0)
    comms = nx.community.louvain_communities(FG, weight="weight",
                                             resolution=COHORT_RES, seed=SEED)
    fcl = {f: c for c, grp in enumerate(comms) for f in grp}
    score = collections.defaultdict(lambda: collections.defaultdict(float))
    for fi, (mem, w) in enumerate(fibers):
        for m in mem:
            score[m][fcl[fi]] += w
    seedlab = {m: max(sc, key=sc.get) for m, sc in score.items()}
    B = np.full(n, -1)
    for m, c in seedlab.items():
        B[m] = c
    for _ in range(30):
        changed = 0
        for i in np.random.default_rng(SEED).permutation(n):
            if i in seedlab:
                continue
            votes = collections.Counter()
            for j in knn_lists[i]:
                if B[j] >= 0:
                    votes[B[j]] += W0[i, j]
            if votes:
                best = max(votes, key=votes.get)
                if best != B[i]:
                    B[i] = best
                    changed += 1
        if not changed:
            break
    if (B < 0).any():
        B[B < 0] = collections.Counter(B[B >= 0]).most_common(1)[0][0]

    # layout: PCA init, spring refine on the content graph
    Xc = X - X.mean(axis=0)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    p0 = {i: (U[i, 0] * S_[0], U[i, 1] * S_[1]) for i in range(n)}
    pos = nx.spring_layout(G, pos=p0, iterations=60, seed=SEED, weight="weight")

    fig, axes = plt.subplots(2, 2, figsize=(22, 19))
    fig.suptitle(
        "CheckItOutV3 — the whole algebraic topology in one frame\n"
        "promoted partition MEET-QUOTIENT v2 (0.3738 held-out modularity, 18/20 vs both parents)",
        fontsize=15)

    # A — base space
    ax = axes[0, 0]
    cmap = plt.get_cmap("tab20")
    for a, b in G.edges():
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="0.85", lw=0.3, zorder=1)
    mp_col = {"P_R_P": "#d62728", "A_P_A": "#1f77b4", "A_P_R": "#2ca02c"}
    for mp, mem, hub, idf in meta_span:
        if hub is None:
            continue
        for m in mem:
            if m != hub:
                ax.plot([pos[hub][0], pos[m][0]], [pos[hub][1], pos[m][1]],
                        color=mp_col.get(mp, "k"), lw=0.7,
                        alpha=min(0.85, 0.25 + 0.2 * idf), zorder=2)
    for t in LAYERS:
        m = et == t
        if not m.sum():
            continue
        ax.scatter([pos[i][0] for i in np.where(m)[0]],
                   [pos[i][1] for i in np.where(m)[0]],
                   c=[cmap(v4[i] % 20) for i in np.where(m)[0]],
                   marker=MARKS[t], s=26, linewidths=0.2,
                   edgecolors="black", zorder=3)
    ax.set_title("A · base space: content-kNN layout · colour = v4 subsystem · "
                 "shape = layer · fibers through hubs (red P-R-P, blue A-P-A, green A-P-R)")
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], marker=MARKS[t], color="w",
                              markerfacecolor="0.6", markeredgecolor="k",
                              markersize=8, label=t) for t in LAYERS],
              loc="lower left", fontsize=8, ncol=2)

    # B — subsystem x layer occupancy
    ax = axes[0, 1]
    subs = sorted(set(v4))
    M = np.zeros((len(LAYERS), len(subs)))
    for li, t in enumerate(LAYERS):
        for si, sview in enumerate(subs):
            M[li, si] = int(((et == t) & (v4 == sview)).sum())
    im = ax.imshow(np.log1p(M), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(LAYERS)), LAYERS)
    ax.set_xticks(range(len(subs)), [str(sx) for sx in subs], fontsize=7)
    for li in range(len(LAYERS)):
        for si in range(len(subs)):
            if M[li, si] > 0:
                ax.text(si, li, int(M[li, si]), ha="center", va="center",
                        fontsize=6.5,
                        color="white" if np.log1p(M[li, si]) < np.log1p(M).max() * 0.6
                        else "black")
    ax.set_title("B · the free layered view: subsystem × layer occupancy\n"
                 "(a vertical slice MIXES layers; a pure column would be a layer in disguise)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="log(1+files)")

    # C — the meet of the two parents
    ax = axes[1, 0]
    Alab, Blab = sorted(set(A)), sorted(set(B))
    CT = np.zeros((len(Alab), len(Blab)))
    for i in range(n):
        CT[Alab.index(A[i]), Blab.index(B[i])] += 1
    cells = int((CT > 0).sum())
    im = ax.imshow(np.log1p(CT), aspect="auto", cmap="magma")
    ax.set_xlabel(f"parent B: cohort-fiber partition ({len(Blab)} parts)")
    ax.set_ylabel(f"parent A: content partition ({len(Alab)} parts)")
    ax.set_title(f"C · the meet A ∧ B: {cells} lattice cells\n"
                 "(blocks = agreements; scatter = the disagreements the quotient resolves)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="log(1+files)")

    # D — fiber dual graph (meta-path fibers only, for readability)
    ax = axes[1, 1]
    FGm = FG.subgraph(range(n_meta))
    posf = nx.spring_layout(FGm, iterations=80, seed=SEED, weight="weight")
    for a, b in FGm.edges():
        ax.plot([posf[a][0], posf[b][0]], [posf[a][1], posf[b][1]],
                color="0.8", lw=0.5, zorder=1)
    for fi in FGm.nodes():
        mp = meta_span[fi][0]
        ax.scatter(*posf[fi], s=30 + 14 * len(fibers[fi][0]),
                   c=[plt.get_cmap("tab10")(fcl[fi] % 10)],
                   edgecolors=mp_col.get(mp, "k"), linewidths=1.4, zorder=2)
    ax.set_title("D · the fiber dual: 92 meta-path fibers · size = arity · "
                 "fill = fiber community · rim = meta-path type")
    ax.axis("off")

    out = os.path.join(OUT, "v3_full_topology.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=300)
    print(f"figure written: {out}")

    # observations
    print("\nOBSERVATIONS (panel B, the layered view per subsystem):")
    pure = 0
    for si, sx in enumerate(subs):
        col = M[:, si]
        k = int(col.sum())
        mix = sorted(((int(c), LAYERS[li]) for li, c in enumerate(col) if c > 0),
                     reverse=True)
        share = mix[0][0] / max(k, 1)
        tag = "LAYER-LIKE" if share > 0.85 and k >= 10 else "slice"
        if tag == "LAYER-LIKE":
            pure += 1
        top = ", ".join(f"{c} {t}" for c, t in mix[:3])
        print(f"  sub {sx:>3} n={k:<4} [{tag:<10}] {top}")
    print(f"\n  {pure} of {len(subs)} subsystems are layer-like (>85% one type);")
    print("  the rest mix layers — vertical slices, which is what a subsystem should be.")


if __name__ == "__main__":
    main()
