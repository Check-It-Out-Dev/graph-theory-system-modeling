# V3 Research Audit: What Survives, What Needs Evidence, What To Fix

**Status**: working document, first pass
**Scope**: the full corpus — six Living Documentation papers, Appendix A (Attention Scaffolding),
Appendix C (Information Lensing) and its Neo4j-native companion, Chromatic Numbers,
Erdős-Lagrangian, Five Independent Detectives, and the four V3 papers.
**Purpose**: separate the claims that are proved from the ones that need an experiment and the ones
that are metaphors, then say what the literature of 2025–2026 now demands of each.

---

## 0. The one-paragraph summary

The architecture is sound and the systems are real. The mathematics divides into three kinds of
claim that are currently formatted identically, which is the single largest risk to the work: a
reader cannot tell a proved theorem from a restated definition from an analogy, because all three
arrive as **Theorem N.M** with a proof box. Information Lensing already solved this problem for
itself — its Appendix A is a self-correction table listing seven overclaims and their fixes. The
recommendation of this audit is not new mathematics. It is to run the pass the author already
invented across the rest of the corpus, and to replace three assertions with three measurements
that are cheap and already possible on the graph that exists today.

---

## 1. Claim triage

### 1.1 Proved, correct, and worth keeping as-is

| Claim | Where | Note |
| --- | --- | --- |
| Path algebra $\mathcal{H} = k\mathcal{Q}/\mathcal{I}$ is well defined, associative, unital | HypatiaBasis §3 | Standard quiver algebra; construction is correct |
| $\mathcal{H}$ is non-abelian; 40/43 composable pairs do not commute | HypatiaBasis Thm 5.1 | The proof by block-landing ($P{,}P$ vs $E{,}E$) is valid |
| $\mathfrak{A}^\dagger = \mathfrak{A}$, hence real spectrum and complete eigenbasis | HypatiaBasis Thm 7.1 | Correct, and the standard magnetic-Laplacian construction |
| $\mathfrak{su}(2)\times\mathfrak{su}(2)$ on the two bidirectional cycles | HypatiaBasis Thm 6.1 | The generators satisfy the relations as claimed |
| Nilpotency $\mathcal{J}^3 = 0$; entity-type conservation | HypatiaBasis §9 | True by construction of the ideal |
| Bi-Lipschitz bounds, homeomorphism, condition number | Information Lensing §6 | Correct linear algebra, correctly stated |
| Convergence to a critical point (not a global minimum) | Information Lensing §6.3 | Correctly weakened; the honest statement |

This is a real algebraic core. Nothing below undermines it.

### 1.2 Stated as theorems, but actually definitions or restatements

**Erdős-Lagrangian Theorem 3.1.** With $\mathcal{L}_G \equiv \tfrac12$ per edge, the action of a path of
length $n$ is $n/2$, so $d_E = 2\min S_G$ reduces to $d = 2\cdot(d/2)$. The "equivalence" is the
definition read backwards, and the reported correlation of $0.9995$ between mean Erdős number and
mean action is what a linear rescaling by two must produce. As written this is not a result.

*The rescue is the interesting case.* The theorem is only trivial because $\mathcal{L}$ is constant. Give
the Lagrangian a non-constant node potential — subsystem membership, local intrinsic dimension,
embedding-space density — and action-minimising paths stop coinciding with shortest paths. Then
there is something to measure: do action-geodesics predict co-change in git history better than
hop-count geodesics? That is a real experiment with a real baseline, and it needs the constant case
only as the degenerate control.

**Appendix A Theorem 2.1 and Corollary 2.3 (attention complexity).** These describe masked sparse
attention — an architectural change of the kind BigBird and Longformer implement. Putting a graph
*in the prompt* does not alter what the transformer attends over; the model still attends across all
$n^2$ token pairs of whatever you handed it. So the $O(n^2)\to O(|E|d)$ claim is not a property of
this system as built, and "$O(2|E|)$ regardless of context size" cannot be asserted from prompt-level
structure.

*The defensible version is stronger anyway.* What the graph actually changes is **how many tokens you
need to send**: retrieve $k$ relevant files instead of stuffing $n$, giving $O(k^2)$ attention with
$k \ll n$, plus a higher hit rate for the relevant context. That is a retrieval claim, it is
measurable directly (tokens-to-correct-answer), and it does not require any statement about the
internals of the attention mechanism.

**Appendix A Theorem 3.1 (reasoning capacity $= I_{\text{raw}} + 2\log n$).** "Attention budget" as a
conserved information quantity is not a defined object in transformer theory, and
$H_{\text{struct}} = 2\log n$ assumes a uniform prior over all token pairs. Demote to a motivating
hypothesis, or replace with the measurable proxy above.

**Paper 02 Theorem 2.2 (behavioural completeness at $k=6$).** The proof sketch says "eigengap
analysis consistently shows maximum separation at $k=6$" — that is an empirical observation about
particular systems, presented as a proof. Either report the eigengap curves as data (with the
systems, the spectra, and the variance) or state it as a conjecture.

### 1.3 Mathematically wrong as stated — with fixes

**Chromatic Numbers, Theorem 2.2: "minimum exclusions $= \chi(G) - 1$".** False in general. The
minimum number of *vertices* to delete so no conflict edge remains is $n - \alpha(G)$, the complement
of the maximum independent set (equivalently the minimum vertex cover). Counterexample: two disjoint
conflict edges $a\!-\!b$ and $c\!-\!d$. Then $\chi = 2$, so the theorem claims one exclusion, but one
endpoint must go from each edge, so two are required; $n - \alpha = 4 - 2 = 2$. Corollary 2.3
inherits the error.

*The fix preserves every practical result.* Two artifacts conflict when they provide the same class,
and "provides the same class" is an equivalence relation — so each connected component of a real
conflict graph is a **clique**. Restrict the model to that case and everything becomes true:
per component $C_i$, minimum exclusions $=|C_i| - 1 = \chi(C_i) - 1$, and overall
$\sum_i (|C_i| - 1) = n - (\text{number of components})$. State the clique model as a modelling
assumption in §2.1, and the theorem is correct, sharp, and still $O(V+E)$.

**Chromatic Numbers, Theorem 5.1 ($\chi \le \Delta \le \log n$).** Two problems. First, Brooks'
theorem excludes complete graphs — and under the clique model above, the components *are* complete,
which is exactly the excluded case. Second, a power-law degree distribution gives
$\Delta \sim n^{1/(\gamma-1)}$, polynomial rather than logarithmic; for $\gamma \approx 2.5$ that is
$n^{0.67}$. Delete the bound; under the clique model $\chi$ per component is just the component size,
which is what the data already shows.

**Chromatic Numbers, Theorem A.2 (Welsh-Powell is a 2-approximation).** The step "$\omega(G) \ge
\Delta(G)/2$ by structure" is unjustified, and greedy colouring admits no constant-factor guarantee
in general. Under the clique model greedy is *exact*, which is the better claim.

### 1.4 The claim that most needs an experiment

**Grothendieck §9, the anti-correlation result.** $\cos(\text{ORCH},\text{TRIG}) = -0.311$ is presented
as "empirical proof that the non-abelian commutator structure manifests as geometric
anti-correlation." Three problems:

1. The experiment ran in **structural mode with no embeddings** — without the $\mathbb{R}^{4096}$
   vectors that the entire thesis is about.
2. The $\alpha_k$ column is a near-perfect monotone function of edge count
   (28 edges $\to$ 1.107, 26 $\to$ 1.119, 21 $\to$ 1.127, 16 $\to$ 1.161, 11 $\to$ 1.203,
   8 $\to$ 1.213, 5 $\to$ 1.231, 2 $\to$ 1.261, 2 $\to$ 1.273, 1 $\to$ 1.279). The paper reads this as
   real-versus-virtual topologies; the unwelcome reading is that $\alpha_k$ measures **sparsity** and
   nothing else.
3. Two different sparse adjacency structures pushed through the same random projection will disagree
   for reasons unrelated to commutators. A negative cosine is not yet evidence.

*What settles it* is in §3 below, and the graph to run it on already exists: namespace
`CheckItOutV3` holds 1374 EntityDetail nodes, **every one carrying an embedding**, with 17 typed
relation types. The experiment the paper describes has never been run in the mode the paper is about.

### 1.5 A claim about a model that is not the one being run

Both Information Lensing Appendix C and Grothendieck V3 report a parameter budget — 102,553 weights,
"independent of graph size" — for restriction maps $\rho_k$ that are never materialised. What FastRP
produces is $n \times 8$ coordinates per relation; the map $\rho_k: \mathbb{R}^{4096}\to\mathbb{R}^8$
exists only implicitly (Grothendieck Definition 5.1 says so honestly). The stated budget therefore
describes an object the pipeline does not construct, and the actual artifact *is* graph-size
dependent.

*This one is worth fixing rather than softening, because the fix buys a capability.* Given the
embeddings $X \in \mathbb{R}^{n\times 4096}$ and the per-relation coordinates
$Y_k \in \mathbb{R}^{n\times 8}$, fit the map by least squares:

$$\rho_k = (X^\top X + \lambda I)^{-1} X^\top Y_k \in \mathbb{R}^{4096 \times 8}$$

Now the claimed matrix genuinely exists, the parameter count is true, and — the real prize — the map
becomes **inductive**: a new file gets its per-relation position from its embedding alone, without
re-running FastRP over the whole graph. That converts a transductive pipeline into a deployable one,
which matters directly for the delta-scan story in the AI Navigator project. Report $R^2$ per
relation; a low $R^2$ is itself a finding (it would mean the relation's geometry is not a linear
function of content, which is interesting and publishable).

Also worth noting: Information Lensing Appendix C.3.5.2 proposes eigenvector centrality on the
similarity graph to find "principal directions". Eigenvector centrality returns one score per node —
a vector in $\mathbb{R}^n$, node space — whereas the transformation needs directions in
$\mathbb{R}^d$, embedding space. To get into embedding space you need the eigenvectors of
$X^\top S X$ (or the SVD of $S^{1/2}X$), not centrality scores.

### 1.6 The non-mathematical risk that outranks all of the above

Paper 02 §7 presents a case study: a 2.3M-LOC e-commerce platform, 3,847 Java classes, a six-month
engagement, developer turnover falling from 47% to 22%, with quoted testimonials from a Junior
Developer, a Tech Lead and an Engineering Manager. If that engagement did not happen, this is the
single most damaging item in the corpus — far more than any theorem, because a reviewer who
discovers one fabricated case study discards everything else on the page, including the parts that
are true and verifiable. Label it explicitly as an illustrative scenario, or remove it. The real
material is stronger anyway: papers 04 and 05 describe an actual system with two actual developers,
and they read as true because they are.

The same discipline applies to the repeated metrics. Hallucination "35% → 9/9.5/10%", architectural
accuracy "45% → 87/89%", and task success "30% → 90%" appear across five papers at four different
confidence registers — asserted as fact in 01 and 02, hedged carefully in 06 ("informal
measurements… rigorous controlled studies would be needed"). Paper 06 has it right. Harmonise the
rest to 06's register, and state the protocol once: 50 tasks, one system, one rater, no control for
prompt differences. That is a defensible pilot; it is not defensible as five independent-sounding
confirmations of the same number.

---

## 2. What the 2025–2026 literature now demands

### 2.1 The identity baseline is now the field's standard objection

*On the Necessity of Learnable Sheaf Laplacians* (arXiv 2603.05395, March 2026) introduces the
**Identity Sheaf Network** — all restriction maps fixed to the identity — and finds it performs
comparably to learnable SNN variants across five heterophilic benchmarks, concluding that the
diffusion-based theoretical motivation "is not reflected empirically" in trained networks.

This is aimed squarely at V3's premise. Any paper proposing per-relation restriction maps in 2026
must report the identity baseline or it will be desk-rejected on this ground alone. The good news is
that V3's setting differs from theirs in a way that can be argued: their task is node classification
on heterophilous graphs, where the label signal is weak; V3's task is *retrieval and navigation*,
where "which neighbours does this relation surface" is the whole objective. The identity map may well
fail at that while succeeding at node classification. But it must be shown, not asserted.

Related and useful: *Learning Sheaf Laplacian Optimizing Restriction Maps* (arXiv 2501.19207) infers
the sheaf Laplacian by minimising total variation with **all steps in closed form** — no gradient
descent, no semidefinite programming. That is a drop-in stronger alternative to FastRP for computing
$\rho_k$, and it is closed-form, which suits the Neo4j-native constraint.

### 2.2 Stratified manifolds are the honest frame, and they are current

The observation from V1/V2 — that the topologies produce stratified manifolds — is now an active
line with measurement tools:

- *Unraveling the Localized Latents* (arXiv 2502.13577) establishes that LLM embeddings do not lie on
  one smooth manifold but form a **stratified space** of lower-dimensional local manifolds.
- *Less is More: Local Intrinsic Dimensions of Contextual Language Models* (arXiv 2506.01034) gives a
  localized TwoNN estimator for local intrinsic dimension, showing it varies by region.
- *Measuring Intrinsic Dimension of Token Embeddings* (arXiv 2503.02142) and *Probing the topology of
  the space of tokens* (arXiv 2503.15421) supply further estimators; reported IDs stratify by domain
  (scientific ≈ 8, encyclopedic ≈ 9, narrative ≈ 10.5), with strata of dimension ≤ 14 in a code model.

This gives V3 a claim that is precise, falsifiable, and connected to work reviewers already read:

> **The per-relation sub-topologies of a typed code graph are strata of the embedding space with
> measurably different local intrinsic dimension.**

If $d_{\text{ID}}$ of ORCHESTRATES-neighbourhoods differs significantly from VALIDATES-neighbourhoods,
that is a result, it explains *why* the projections differ, and it requires no gauge theory. If they
do not differ, that is also a result and it tells you the per-relation machinery is not earning its
keep — which is exactly what you want to learn before building more on top of it.

### 2.3 Code graphs are no longer novel; the typed algebra might be

The territory is now populated, with benchmarks:

- **RepoGraph** (ICLR 2025) — repository-level code graphs, +32.8% relative on SWE-bench.
- **CodexGraph** (NAACL 2025) — code graphs exposed to LLM agents through a graph database interface.
- **Code Graph Model** — 43.00% on SWE-bench Lite with an open 72B model.
- **Codebase-Memory** (arXiv 2603.27277, 2026) — Tree-Sitter knowledge graphs for LLM code
  exploration **over MCP**, which is the AI Navigator architecture, published.
- **GraphRAG-Bench** (ICLR 2026) — "When to use Graphs in RAG", a benchmark built precisely to ask
  when graphs help and when they do not.

Two consequences. First, the thesis is validated as a research direction rather than eccentric — that
is genuinely good news. Second, the bar has moved: the comparison that matters is no longer
"graph versus no graph" but "**typed algebraic graph versus a strong tree-sitter code graph**", ideally
on SWE-bench Lite or GraphRAG-Bench rather than a bespoke question set. The differentiator to defend
is the part nobody else has: selection rules enforced at construction time, and relation-specific
geometry.

### 2.4 Relations-as-operators is mainstream KGE, which changes the novelty claim

HypatiaBasis "Novelty 1" states that nobody has shown these pieces form a single chain. The KGE
literature has built non-commutative relation operators for years: RotatE ($U(1)$), QuatE
($\mathrm{SU}(2)$), DensE ($\mathrm{SO}(3)$ with an explicit non-abelian argument), Rotate3D,
CompoundE, and TransR/TransD for relation-specific projections; TorusE embeds on a Lie group.

The novelty claim should be narrowed to what is actually unclaimed, which is still substantial:

> Prior work *learns* relation operators so that links become predictable, with entity embeddings
> trained from scratch. We take a **fixed pretrained semantic embedding** of code artifacts and ask
> which subspace each typed relation resolves within it, under a type algebra whose selection rules
> forbid illegal compositions *at graph-construction time*. The contributions are the algebra as a
> hard constraint, and the measurement that the resulting subspaces are strata of differing intrinsic
> dimension.

That version survives a reviewer who knows the field. The current version does not.

---

## 3. The experiment that decides it

One design answers §1.4, §2.1 and §2.2 simultaneously. It runs on `CheckItOutV3` as it stands today.

**Arms.**

0. **Identity / shared** — $\rho_0$ only, one projection for every relation (the field's required baseline).
1. **Label-shuffled** — per-relation projections computed after a degree-preserving shuffle of edge
   labels, repeated $N \ge 100$ times to build a null distribution.
2. **True** — per-relation projections on the real typed edges.

**Measurements.**

- Inter-relation cosine distribution, arm 2 versus the arm 1 null. The claim in §9 survives only if
  the observed anti-correlation lies outside the null. Report a $z$-score or an empirical $p$.
- $\alpha_k$ regressed on $\log(\text{edge count})$. If $R^2$ is high, $\alpha_k$ is a sparsity
  statistic and the real/virtual language must be retired.
- Local intrinsic dimension per relation neighbourhood (TwoNN), arm 2 versus arm 0, with confidence
  intervals. This is the stratification claim.
- Retrieval task success on a gold question set, all three arms. This is the only measurement that
  speaks to the product claim, and it is the same instrument the AI Navigator project needs for its
  ablation ladder — one asset, both purposes.

**Win condition, stated in advance.** Arm 2 beats arm 0 by a margin larger than arm 1 beats arm 0,
with non-overlapping confidence intervals. Anything less means the typed geometry is not doing work
on this codebase, which is worth knowing in a week rather than after a paper.

---

## 3a. First results — the experiment partly run

Arms 0 and 2 were run against `CheckItOutV3` (1374 nodes, every one carrying an
$\mathbb{R}^{4096}$ embedding, twelve typed relations with projections already computed). The null
arm is still outstanding. Three results, two of them negative.

**Result 1 — $\alpha_k$ is edge count.** In full mode, with real embeddings, the correction magnitudes
order *perfectly* by how many edges the relation has:

| Relation | Edges | $\alpha_k$ |
| --- | ---: | ---: |
| PERFORMS | 171 | 0.236 |
| USES | 163 | 0.323 |
| MODIFIES | 97 | 0.350 |
| CALLS | 91 | 0.373 |
| ACCESSES | 54 | 0.378 |
| CONSTRAINS | 13 | 0.407 |
| VALIDATES | 11 | 0.408 |
| AFFECTS | 9 | 0.412 |
| TRIGGERS | 6 | 0.416 |
| APPLIES_IN | 3 | 0.418 |
| CONFIGURED_BY | 2 | 0.419 |
| INITIATES | 2 | 0.420 |

Pearson $r$ against $\log(\text{edges})$ is $-0.82$ ($R^2 = 0.672$); the rank correlation is $-1$ up to
the single tie at two edges. The curve saturates near $0.42$, which is what a projection with no
propagation opportunity must look like. **The real-versus-virtual topology reading in Grothendieck
§4.4 does not survive contact with the full-mode graph** — $\alpha_k$ is a sparsity statistic.

**Result 2 — the published anti-correlation does not reproduce.** Measuring
$\cos(\mathrm{proj}_a, \mathrm{proj}_b)$ over nodes non-zero in both relations gives *positive*
similarity for every pair: $0.672$ to $0.947$. There is no anti-correlation anywhere in full mode.
The $-0.311$ came from structural mode, where FastRP has no features to propagate and the output is
dominated by random projection on a sparse adjacency — a regime in which the sign of a cosine
carries no information. The overlaps are also small enough to matter: between five and seventy-six
nodes out of 1374.

**Result 3 — the underlying claim survives, because the paper measured the wrong quantity.** The
theory is about how each relation *deviates* from the common base, so the object to compare is
$\Delta_k = \rho_k - \rho_0$, not $\rho_k$. Measured that way:

| Pair | $\cos(\Delta_a, \Delta_b)$ | nodes negative / total |
| --- | ---: | ---: |
| USES vs MODIFIES | **+0.494** | 14 / 76 |
| PERFORMS vs CALLS | +0.078 | 20 / 43 |
| MODIFIES vs ACCESSES | +0.064 | 6 / 15 |
| USES vs ACCESSES | −0.046 | 11 / 21 |
| PERFORMS vs MODIFIES | −0.082 | 19 / 35 |
| PERFORMS vs ACCESSES | −0.174 | 13 / 22 |
| PERFORMS vs USES | −0.238 | 27 / 36 |
| USES vs CALLS | **−0.400** | 28 / 32 |
| MODIFIES vs CALLS | **−0.466** | 27 / 29 |

The anti-correlation is there, it is systematic at node level rather than outlier-driven, and — the
part that argues hardest against noise — **the sign structure is architecturally coherent**. The two
resource-access relations pull a file the same way ($+0.494$). Resource access opposes service
orchestration ($-0.40$, $-0.47$). Entry-point invocation is near-orthogonal to service-to-service
calls ($+0.08$). A random effect does not arrange itself into data layer versus control layer.

**What this changes.** The claim to make is not "the sub-topologies are different" — measured
directly they are strongly aligned, because they all inherit the same content signal through
`propertyRatio: 0.5`. The claim is that **the per-relation corrections carry a signed structure that
separates data access from orchestration**, and that structure is what the non-commutativity of the
algebra predicts. That is a sharper claim, it is supported by the numbers above, and it still needs
the null arm before it can be published.

**Result 4 — the null arm, and it holds.** The whole pipeline was rebuilt independently to make the
observed and null statistics directly comparable: a dedicated GDS projection over the 279 nodes
touched by the 576 behavioural edges, undirected, with the $\mathbb{R}^{4096}$ embeddings as node
features and FastRP at the paper's own settings (dim 8, seed 42, `propertyRatio` 0.5, weights
$[0,1,1]$). The recomputation reproduces the stored result — $-0.459$ on 29 nodes against the
$-0.466$ measured from the stored projections — so the two are measuring the same thing.

The null is a **count-preserving permutation of the edge labels**: the topology stays exactly as it
is, and only which edge is called MODIFIES and which is called CALLS is reassigned at random,
preserving how many of each there are. If the typing carries no information beyond sparsity, the
statistic should not move.

| Trial | $\cos(\Delta_{\text{MODIFIES}}, \Delta_{\text{CALLS}})$ | nodes negative / total |
| --- | ---: | ---: |
| **Observed, true labels** | **−0.459** | 26 / 29 |
| Permutation 0 | +0.238 | 14 / 54 |
| Permutation 1 | +0.245 | 17 / 62 |
| Permutation 2 | +0.201 | 21 / 62 |
| Permutation 3 | +0.277 | 12 / 57 |
| Permutation 4 | +0.236 | 20 / 60 |

Null mean $+0.239$, standard deviation $0.027$. The observed statistic sits roughly **26 standard
deviations below the null mean, and on the opposite side of zero**. Shuffling the labels does not
merely weaken the effect — it reverses it. The anti-correlation therefore requires the true relation
types; it is not produced by having two sparse edge sets of similar size.

That is the claim, tested, with the null the sheaf literature would demand:

> The per-relation corrections of a typed code graph carry a signed structure that separates data
> access from orchestration, and that structure is destroyed by permuting the relation labels.

Caveats worth stating in the paper: five permutations resolve an empirical $p$ no finer than $1/6$,
so the $z$-score is doing the work — run 200 permutations before publishing. The effect is measured
on one pair in one codebase, on 29 nodes. And the node counts differ between observed and null
(29 versus ~60) because shuffling changes which nodes end up with non-zero projections in both
relations; that asymmetry should be handled by matching on node set, not just on edge count, in the
final version.

**Result 5 — $\rho_k$ can be built, and it generalises.** §1.5 objects that the claimed 102,553-weight
matrix is never constructed. It can be, by ridge regression in dual form from the embeddings to the
FastRP coordinates, $\rho_k = X^\top(XX^\top + \lambda I)^{-1}Y_k$. Because $n < d$ an in-sample fit is
always exact and therefore meaningless, so the map was scored by 5-fold cross-validation with
centring computed on the training folds only:

| Relation | $n$ | held-out $R^2$, projection | held-out $R^2$, correction |
| --- | ---: | ---: | ---: |
| MODIFIES | 76 | **0.692** | 0.493 |
| USES | 100 | 0.606 | 0.235 |
| PERFORMS | 192 | 0.599 | 0.325 |
| ACCESSES | 56 | 0.497 | 0.371 |
| CALLS | 60 | 0.437 | 0.254 |

The map is real and it transfers to files the fit never saw. Three consequences. The parameter budget
can be stated as a fact rather than softened away. The pipeline becomes **inductive** — a new or
changed file gets its per-relation position from its embedding alone, with no re-run over the graph,
which is precisely what the delta-scan story in the AI Navigator project requires. And the honest
qualifier is that between a third and a half of the variance is *not* recoverable from content
alone, so the graph is still doing real work: this is a warm start, not a replacement.

Worth noting which relations are predictable. MODIFIES is the most content-determined ($0.69$) and
CALLS the least ($0.44$) — what a file writes is more legible in its text than whom it calls, which
is a small independent sanity check that the fitted maps are tracking something real.

**Immediate follow-ups, in order.** (i) Extend the permutation test to all pairs and to 200
permutations, matching node sets. (ii) Sweep `propertyRatio` downward — if the projections are
swamped by shared content at $0.5$, the structure should sharpen as the adjacency is given more
weight, and that sweep is also the honest answer to the identity-baseline objection. (iii) Only then
the intrinsic-dimension measurement (M3).

---

## 4. What this suggests the V3 mathematical paper should actually claim

Not gauge theory. The gauge structure can stay as an interpretive section — it is genuinely elegant
and it costs nothing once the load-bearing claims stand on their own. But the spine should be the
sentence hiding in `WorkingNotes/tensor-intuition.md` §14, which is the best statement of the thesis
anywhere in the corpus:

> An embedding treats an entity as a **point**. A tensor treats it as an **operator** — not "what is
> this about" but "how does this act on the information around it."

That is the paper. Software entities should be represented as operator families rather than points;
a typed graph supplies exactly the supervision needed to recover the family; the family is
non-commutative and that non-commutativity is measurable; and the resulting subspaces are strata of
different intrinsic dimension. Every one of those four clauses is testable with the graph that
already exists, and none of them requires the reader to accept an analogy first.

---

*Audit prepared 2026-09-02. Sources for §2 are cited inline by arXiv identifier.*
