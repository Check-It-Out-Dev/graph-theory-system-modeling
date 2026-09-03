# Relations as Operators: Typed Restriction Maps on Code Graphs

## With a permutation null, an inductive fit, and an account of what does not hold

**Version**: 2.0.0 | **Date**: 2026-09-02
**Authors**: Norbert Marchewka (architecture), Claude Opus 5 (synthesis and experiments)
**Substrate**: `CheckItOutV3` — 1374 files of a production Spring Boot platform, each carrying an
$\mathbb{R}^{4096}$ embedding, connected by 17 typed relation types under the Hypatia algebra.

---

## §0 Map of results — read this before anything else

This document grew as a lab notebook (§6a–§6s were appended as the experiments ran) and is
organised by *discovery order*, not by topic. This map is the topical index. Every claim below is
backed by a finding node in the Neo4j `V3Lab` namespace (F1–F107) and an experiment script under
`experiments/`; refuted claims are kept in the text with their refutations, never silently removed.

**The promoted system** (what survived everything):

| component | method | measured | where |
| --- | --- | --- | --- |
| pairwise ranker | RRF(content embedding, lexical TF-IDF) | AUC **0.8432**, 20/20 | §6o |
| partition | **MEET-QUOTIENT v2** — meet of content and cohort-fiber partitions, two-mechanism cell quotient | modularity **0.3738**, 18/20 vs both parents | §6s |
| n-ary evidence | meta-path hyperedges, IDF centre-weighted | within-cohort co-change **21.7×** base | §6f, F95 |
| human loop | disagreement shortlist | 568,003 pairs → **332** questions | §6m |
| full-picture | `figures/v3_full_topology.png` | 18/19 subsystems are vertical slices | F107 |

**The retired constructions** (each died by measurement; the reason travels with the corpse):

| construction | verdict | where |
| --- | --- | --- |
| α_k "real vs virtual" topologies | tracks edge sparsity | §4 |
| −0.311 anti-correlation as non-abelian proof | does not reproduce | §4 |
| orientation obstruction (ℤ/2 from lstsq) | conditioning artifact | §6a.5→§6b.4 |
| rotation as the edge model class | loses to a 1-param scalar on 34/34 signatures | §6d |
| Berry phase / holonomy / curvature | functions of the retired rotation fits | §6d, §6q |
| magnetic Laplacian (fitted *and* trophic phase) | shatters both ways — double seal | §6d, §6q |
| ℝ¹³⁶ composite as clustering input | worse than its own base at every width | §6h |
| triple-lens (S/B/T) fusion for ranking | 0/20 under two fusion rules | §6o |
| fine-grained co-association consensus | resolution knife-edge | §6r |

**The transferable lessons** (the part most likely to outlive this codebase): constrain estimators to
the group the object lives in (§6b.4); reliability *and* significance, never one alone (§6c);
diversify the mechanism, not the input slice (§6o.2); never propagate a ranker into a partition base
(§6p.1); granularity, balance, split-variance, model class and degenerate objectives are the five
nuisance controls that each reversed at least one conclusion here (Grothendieck v4.1 prompt, §3).

Reading paths: *the mathematics* → §1–§5, §6b–§6d; *the system* → §6e, §6o–§6s; *the method* →
§6c, §6j, the controls. Prompts implementing this document: `Promts/V5/HypatiaV5_agent.md` (indexer)
and `Promts/V4/GrothendieckGraphOrganizer_V4.xml` v4.1 (organizer).

---

## Abstract

An embedding places a code artifact at a *point*. We argue it should be treated as an *operator* —
not "what is this file about" but "how does this file act on the files around it" — and that a typed
dependency graph supplies exactly the supervision needed to recover the operator family. We define
per-relation restriction maps $\rho_k$ on a fixed pretrained embedding space, show that the family is
non-commutative in a sense that is measurable rather than merely definable, and test the measurement
against a count-preserving permutation of the relation labels. On a production codebase the
per-relation *corrections* $\Delta_k = \rho_k - \rho_0$ carry a signed structure that separates data
access from service orchestration ($\cos(\Delta_{\text{MODIFIES}}, \Delta_{\text{CALLS}}) = -0.459$),
and permuting the labels does not weaken that structure but reverses it (null mean $+0.239$,
$\mathrm{sd}\ 0.027$). We further show that $\rho_k$ can be materialised by ridge regression and
transfers to files never seen during fitting (held-out $R^2$ up to $0.69$), which makes the pipeline
inductive. We are explicit about three earlier claims that do not survive contact with the full-mode
graph, including one of our own headline results.

---

## 1. Motivation: points versus operators

A vector embedding answers *where* an artifact sits in semantic space. It cannot answer *how* that
artifact acts, because a point has no action. Two files may embed within $0.05$ cosine of each other
and play opposite architectural roles — one writing to a resource, one orchestrating three services.

The operator view is not new in knowledge-graph embedding: RotatE represents relations as $U(1)$
rotations, QuatE as $\mathrm{SU}(2)$, DensE as $\mathrm{SO}(3)$ with an explicit non-abelian argument,
and TransR/TransD project entities into relation-specific spaces. What is different here is the
input and the constraint:

- Those methods **learn entity representations from scratch** so that links become predictable. We
  take a **fixed pretrained semantic embedding** of real file contents and ask which subspace each
  typed relation resolves within it. The embedding is not ours to move.
- The relation types are not free labels. They come from a typed quiver with **selection rules** that
  forbid illegal compositions, enforced at graph-construction time, so no edge exists that the
  algebra disallows.

The question this paper answers is therefore narrow and empirical: **given a fixed semantic space and
a legally-constructed typed graph, do the typed relations resolve genuinely different subspaces, and
how would we know we were not fooling ourselves?**

---

## 2. The algebra (summary of established results)

We restate without proof the results established in *The Hypatia Basis*; §7 of that paper contains
the arguments and they are unaffected by anything below.

Let $\mathcal{Q}$ be the typed quiver on six entity vertices $\{A, P, Ru, E, C, R\}$ with 17 typed
arrows, and let $\mathcal{H} = k\mathcal{Q}/(\mathcal{I}_{\text{select}} + \mathcal{J}^3)$ be the path
algebra modulo the selection ideal and depth-3 nilpotency.

- **(H1)** $\mathcal{H}$ is associative, unital, and finite-dimensional with $\dim \mathcal{H} = 66$.
- **(H2)** $\mathcal{H}$ is non-abelian: of 43 legal depth-2 compositions, 40 fail to commute. The
  witness is that $\tau\iota$ lands in block $(P,P)$ while $\iota\tau$ lands in $(E,E)$.
- **(H3)** The algebra matrix $\mathfrak{A} \in \mathbb{C}^{6\times6}$, built as a magnetic Laplacian
  with phase $g$, satisfies $\mathfrak{A}^\dagger = \mathfrak{A}$; its spectrum is real and its
  eigenvectors form a complete basis.
- **(H4)** Restricted to the two bidirectional cycles, the commutator subalgebra contains
  $\mathfrak{su}(2)_{EP} \times \mathfrak{su}(2)_{RuC}$.

These are statements about the *algebra*. They say nothing on their own about geometry: a graph can
satisfy all four while every relation resolves the identical subspace. Bridging that gap is the
work of §3–§5, and it is where our earlier drafts overreached.

---

## 3. Restriction maps and corrections

**Definition 3.1 (restriction map).** For relation $R_k$, a restriction map is a linear
$\rho_k : \mathbb{R}^{d} \to \mathbb{R}^{r}$ with $d = 4096$, $r = 8$. It induces the metric space
$\mathcal{T}_k = (\{\rho_k(e(v))\}_{v \in V},\ \|\cdot\|_2)$ — the *sub-topology* of $R_k$.

**Definition 3.2 (common base and correction).** $\rho_0$ is the map obtained from the union of all
typed edges. The correction of relation $k$ is $\Delta_k = \rho_k - \rho_0$, and its magnitude is
$\alpha_k = \frac1n\sum_v \|\rho_k(e(v)) - \rho_0(e(v))\|_2$.

**Computation.** In this work $\rho_k$ is realised by FastRP with the node embeddings as features,
identical hyperparameters across relations (dimension 8, seed 42, `propertyRatio` 0.5, iteration
weights $[0,1,1]$), so that the *only* difference between $\rho_0$ and $\rho_k$ is which edges are
present. FastRP is a randomised projection, not a trained model — it optimises no loss — and the
Johnson–Lindenstrauss guarantee at $r=8$ for $n \approx 10^3$ is far below the distortion-free regime.
Both facts are limitations, stated here rather than in a footnote, and §6 shows the consequence.

---

## 4. What does not hold

We report three negative results before the positive one. Two of them retract claims from earlier
drafts of this work.

**4.1 $\alpha_k$ measures sparsity, not distinctiveness.** Earlier work read small $\alpha_k$ as a
"real" topology and large $\alpha_k$ as "virtual". On the full-mode graph the correction magnitudes
order *perfectly* by edge count across all twelve relations measured, from PERFORMS (171 edges,
$\alpha = 0.236$) monotonically to INITIATES (2 edges, $\alpha = 0.420$), saturating near $0.42$ —
which is the value a projection must approach when it has no propagation opportunity at all. Against
$\log(\text{edges})$, $r = -0.82$ and $R^2 = 0.672$; the rank correlation is $-1$ up to a single tie.

> **The real/virtual distinction should be withdrawn.** $\alpha_k$ is a statistic about how many
> edges a relation has.

**4.2 The sub-topologies are not directly anti-correlated.** Our previous headline result reported
$\cos(\rho_{\text{ORCH}}, \rho_{\text{TRIG}}) = -0.311$ as evidence that non-commutativity manifests
geometrically. It does not reproduce. Measured over nodes non-degenerate in both relations, every
pair is strongly *positive*: $0.672$ to $0.947$. The earlier figure came from a structural-mode run
with no embeddings, where FastRP has nothing to propagate and the output is dominated by random
projection on a sparse adjacency — a regime in which the sign of a cosine carries no information.

> **The $-0.311$ result should be withdrawn.** Measured directly, the sub-topologies are aligned,
> because at `propertyRatio` $0.5$ they all inherit the same content signal.

**4.3 The parameter budget described an object that was never built.** The reported 102,553 weights
characterise matrices $\rho_k$ that the pipeline does not construct; what it produces is $n \times 8$
coordinates per relation, which is graph-size dependent. §5.3 repairs this rather than softening it.

---

## 5. What does hold

**5.1 The corrections carry signed structure.** The theory concerns how each relation *deviates* from
the common base, so the object to compare is $\Delta_k$, not $\rho_k$. Measured that way, over nodes
non-degenerate in both relations:

| Pair | $\cos(\Delta_a, \Delta_b)$ | negative / total |
| --- | ---: | ---: |
| USES vs MODIFIES | $+0.494$ | 14 / 76 |
| PERFORMS vs CALLS | $+0.078$ | 20 / 43 |
| PERFORMS vs USES | $-0.238$ | 27 / 36 |
| USES vs CALLS | $-0.400$ | 28 / 32 |
| MODIFIES vs CALLS | $-0.466$ | 27 / 29 |

The effect is systematic at node level rather than driven by outliers, and its **sign structure is
architecturally coherent**: the two resource-access relations pull a file the same way, resource
access opposes service orchestration, and entry-point invocation is near-orthogonal to
service-to-service calls. This coherence is the first reason to doubt that it is noise. The second is
§5.2.

**5.2 A permutation null.** We rebuilt the computation independently — a dedicated projection over
the 279 nodes touched by the 576 behavioural edges, undirected, same hyperparameters — and it
reproduces the stored measurement ($-0.459$ against $-0.466$). We then permuted the relation labels
while preserving how many edges carry each label. The topology is untouched; only *which* edge is
called MODIFIES and which CALLS changes.

| Trial | $\cos(\Delta_{\text{MODIFIES}}, \Delta_{\text{CALLS}})$ |
| --- | ---: |
| **Observed** | $\mathbf{-0.459}$ |
| Permutations 0–4 | $+0.238,\ +0.245,\ +0.201,\ +0.277,\ +0.236$ |

Null mean $+0.239$, $\mathrm{sd}\ 0.027$; the observed statistic lies roughly 26 standard deviations
below it and on the opposite side of zero. **Shuffling the labels does not attenuate the effect, it
reverses it.** The anti-correlation therefore requires the true typing and is not a consequence of
comparing two sparse edge sets of similar size.

**5.3 $\rho_k$ exists and is inductive.** Fitting $\rho_k = X^\top(XX^\top + \lambda I)^{-1}Y_k$ by
ridge regression, and scoring by 5-fold cross-validation with training-fold centring (in-sample fit
is exact and meaningless at $n<d$), gives held-out $R^2$ of $0.44$–$0.69$ for the projections and
$0.24$–$0.49$ for the corrections. The matrix therefore genuinely exists at $4096\times8$ per
relation, the parameter budget is a fact, and a file absent from the graph can be positioned from its
content alone. Between a third and a half of the variance is *not* recoverable that way, so the graph
continues to do real work: this is a warm start, not a replacement.

---

## 6. Interpretation, one prediction that failed, and one honest worry

**6.1 A hyperparameter sweep, and a corrected mechanism.** Our first reading was that the shared
content signal *swamps* the relation-specific one, predicting that the structure should sharpen as
`propertyRatio` is lowered and the adjacency is weighted more heavily. We tested it across the range,
with three label permutations at each setting:

| `propertyRatio` | observed | null mean | separation |
| ---: | ---: | ---: | ---: |
| 0.000 | $-0.170$ | $+0.283$ | $-0.452$ |
| 0.125 | $-0.139$ | $+0.298$ | $-0.436$ |
| 0.250 | $-0.081$ | $+0.267$ | $-0.349$ |
| **0.500** | $-0.459$ | $+0.186$ | $\mathbf{-0.645}$ |
| 0.750 | $-0.507$ | $+0.094$ | $-0.601$ |
| 1.000 | $-0.546$ | $+0.076$ | $-0.622$ |

**The prediction was wrong, and the direction of the error is informative.** The anti-correlation
does not weaken with more content — it strengthens monotonically, from $-0.170$ at pure structure to
$-0.546$ at full content weight, while the null mean falls from $+0.283$ toward zero.

The mechanism is therefore the opposite of what we proposed. Content is not noise that dilutes the
structural signal; content is **what makes relation-specific propagation meaningful**. When the
initial vectors are content-derived, propagating them through two different adjacencies produces
systematically different results. When the initial vectors are random — the `propertyRatio` $\to 0$
regime, which is exactly the structural mode of our earlier work — the difference between two
relations is dominated by projection noise, the effect is weakest, and the null is largest. That
independently explains why the retracted $-0.311$ measurement of §4.2 was unreliable: it was taken in
the worst available regime.

Two practical consequences. The effect is **robust to the hyperparameter**: the observed statistic is
negative and the null positive at every setting tested, with separation never better than $-0.35$.
And the separation is maximised at $0.5$, which is the value the pipeline already uses — a choice
made for other reasons that turns out to be near-optimal for this measurement.

**6.2 The worry that remains.** The projections are mutually aligned at $0.67$–$0.95$, so the
relation-specific component is small in norm relative to the shared one. A reader is entitled to ask
whether something that small matters for any downstream task.

The worry we cannot yet dismiss is that the perturbation is small in norm relative to the shared
component, so a reader is entitled to ask whether it matters for any downstream task. **We do not yet
have that evidence.** The measurement in §5 is geometric; it shows the structure is real, not that it
is useful. §8 states the experiment that would close this, and we regard the claim as incomplete
until it is run.

This is also the honest answer to the current objection in the sheaf literature. *On the Necessity of
Learnable Sheaf Laplacians* (arXiv 2603.05395) introduces an identity-map baseline and finds it
competitive with learnable restriction maps across five heterophilic benchmarks. Our setting differs —
their task is node classification, ours is retrieval and navigation, where "which neighbours does
this relation surface" is the entire objective — but the difference is an argument, not a result.
Any future version of this work must report the identity baseline on a downstream task.

---

## 6a. The geometry of the sub-topologies

Four further measurements, which together turn the loose talk of "sub-topologies" into something
with a shape, an overlap structure, and a curvature.

**6a.1 Bubbles — which parts of the graph each relation touches.** Of 1374 files, only 279 carry any
behavioural edge at all; the sub-topologies live on that subset. Their supports are strikingly
unequal, and two containment facts fall straight out:

| Relation | support | exclusive to it |
| --- | ---: | ---: |
| PERFORMS | 192 | 116 |
| USES | 100 | 14 |
| MODIFIES | 76 | **0** |
| CALLS | 60 | 10 |
| ACCESSES | 56 | 13 |

MODIFIES has **no** exclusive files: its support is contained entirely in that of USES (Jaccard
$0.76 = 76/100$). **Nothing in this codebase writes to a resource without also reading one.** And at
the other extreme, CALLS $\cap$ ACCESSES $= \varnothing$ — **no file both orchestrates another service
and touches a resource directly**. Those are architectural invariants, recovered from geometry rather
than asserted in a style guide, and either would be worth a lint rule.

153 files sit in exactly one bubble, 72 in two, 29 in three, 25 in four. Every one of the four-bubble
files is a `*Service` — `PartnershipOpportunityService`, `NotificationService`, `LegalConsentService`,
`OAuthCallbackService` and so on. The multi-role files are the orchestrators, which is what one would
hope, and it means "how many bubbles does this file live in" is a usable measure of architectural
load.

**6a.2 Back in the original space, the relations are nearly orthogonal.** The fitted $\rho_k$ is a
$4096\times8$ matrix, so its column space is an 8-dimensional subspace of the original embedding
space: the directions relation $k$ actually reads. Principal angles between those subspaces:

| Pair | directions shared (angle $<30°$) | min angle | mean angle |
| --- | ---: | ---: | ---: |
| USES & MODIFIES | 1 | $25.1°$ | $48.1°$ |
| every other pair | **0** | $53°$–$64°$ | $72°$–$78°$ |

This resolves the apparent contradiction in §4.2. Measured in $\mathbb{R}^8$ the sub-topologies look
strongly aligned ($0.67$–$0.95$), because every projection inherits the same dominant content
direction. Measured where they come from, in $\mathbb{R}^{4096}$, they are **nearly orthogonal** —
each relation reads its own directions of the embedding space. The alignment is an artifact of the
shadow; the orthogonality is the structure. And the single exception is exactly the pair with the
containment relation, USES and MODIFIES, which share one direction.

**6a.3 Shape.** Each cloud is examined by the spectrum of its covariance and by a TwoNN estimate of
local intrinsic dimension. The clouds are not round: effective rank sits well below the ambient
eight, so the sub-topologies occupy lower-dimensional sheets inside their own projection space — which
is the stratification hypothesis of §7 appearing one level down, and an argument for measuring it
properly.

**6a.4 Holonomy, and what it detects.** Fit the transition $M_{i\to j}$ carrying positions in topology
$i$ to positions in topology $j$ over their shared support, compose around a closed loop of three
relations, and ask whether you return to where you started. No loop does: $\|H - I\|_F/\sqrt8$ runs
from $0.70$ to $1.05$. But the *rotation* carried by the loop is highly structured:

| Loop | rotation |
| --- | ---: |
| USES $\to$ MODIFIES $\to$ ACCESSES $\to$ USES | $0.8°$ |
| PERFORMS $\to$ USES $\to$ MODIFIES $\to$ PERFORMS | $2.8°$ |
| USES $\to$ MODIFIES $\to$ CALLS $\to$ USES | $22.5°$ |
| PERFORMS $\to$ MODIFIES $\to$ ACCESSES $\to$ PERFORMS | $42.2°$ |
| PERFORMS $\to$ USES $\to$ CALLS $\to$ PERFORMS | $77.8°$ |
| PERFORMS $\to$ USES $\to$ ACCESSES $\to$ PERFORMS | $80.5°$ |

Loops that stay among the data-touching relations are almost holonomy-free; loops that cross into
service orchestration or direct resource access rotate by tens of degrees. **The holonomy is
measuring whether the loop crossed an architectural boundary.** That is the Berry-phase claim of the
earlier papers in a form that was actually computed, and it is the first version of it that could be
used for anything: a boundary detector with a continuous score.

**6a.5 Homotopy, stated exactly.** $\mathrm{GL}(8,\mathbb{R})$ has exactly two connected components,
distinguished by the sign of the determinant. A transition map with $\det > 0$ can be deformed
continuously to the identity through invertible maps — it is, in the only sense that matters here,
*solid*. One with $\det < 0$ cannot: it reverses orientation, and no continuous path of invertible
maps joins it to doing nothing. This is a genuine topological invariant, not an analogy, and it is one
line of code.

Of the eighteen transitions with enough shared support, eleven came out orientation-preserving and
seven reversed, and the reversals appeared to concentrate on exactly the crossings that also carry
high holonomy. **That result does not survive, and it is withdrawn here.** The transitions were fitted
by unconstrained least squares, which returns a general $8\times8$ matrix rather than an element of
$\mathrm{O}(8)$; conditioning ran as high as $442$, meaning those maps were collapsing directions
rather than rotating them, so the sign of the determinant was reporting numerical noise. §6b refits
the same question under an orthogonality constraint and bootstraps the sign: **every relation whose
determinant is stable under resampling is orientation-preserving**, and every apparent reversal has a
determinant that is a coin flip (ALGEBRA_VIOLATION $56\%$, IMPLEMENTS $48\%$, EXTENDS $59\%$ agreement
across 200 bootstraps). There is no evidence for an orientation obstruction in this codebase.

The lesson generalises past this paper. A $\mathbb{Z}/2$ invariant computed from an unconstrained fit
is not a topological statement about the data; it is a statement about the fit. If the object being
measured is supposed to live in a group, the estimator has to be constrained to that group.

**6a.6 The interaction tensor.** These are not separate results; they are slices of one object indexed
by pairs and triples of relations. Collecting them gives $T_2[i,j,\cdot] \in \mathbb{R}^5$ — support
Jaccard, subspace overlap, correction cosine, orientation, rigidity — and $T_3[i,j,k]$, the holonomy
rotation around the loop.

The use of assembling it is that the relation set stops being a design decision. Two relations should
be *one* lens when they share the graph, share the subspace, pull the same way, preserve orientation
and transition rigidly; they must stay separate when they do not. Scoring the pairs on that product
(so that any single disqualifying property vetoes a merge rather than being averaged away) gives one
candidate and one only:

| Pair | Jaccard | subspace | $\Delta$-cos | orientation | merge score |
| --- | ---: | ---: | ---: | :---: | ---: |
| USES & MODIFIES | 0.76 | 0.64 | $+0.49$ | $+$ | **0.276** |
| MODIFIES & ACCESSES | 0.13 | 0.20 | $+0.06$ | $+$ | 0.002 |
| all others | — | — | — | — | $0.000$ |

USES and MODIFIES are separated from every other pair by two orders of magnitude. Whether $0.276$ is
"enough" is not something this paper can settle — the threshold has to be calibrated on a downstream
task, not chosen by the person hoping for a merge — but the *ranking* is unambiguous, and it says the
17-relation vocabulary should be audited this way rather than trusted.

---

## 6b. The connection sheaf: putting the phase back on the graph

§6a.4 composed transition maps around loops in *relation* space — PERFORMS $\to$ USES $\to$ MODIFIES
$\to$ PERFORMS. That is a real measurement, but it discards the graph: the answer is one number per
triple of relation types and it cannot say *where* in the codebase the phase lives. This section puts
the connection on the graph itself.

**6b.1 Construction.** Each node carries a stalk $\mathbb{R}^8$, its position in the base topology.
Each relation type $k$ carries one orthogonal map $O_k \in \mathrm{O}(8)$ — "to compare the two ends
of a $k$-edge, first rotate by $O_k$" — fitted in closed form by orthogonal Procrustes (Kabsch) over
that relation's edges. The connection Laplacian is then

$$L = \bigoplus_v \deg(v)\,I_8 \;-\; \bigl(O_k \text{ in block } (v,u) \text{ for each edge } (u,v,k)\bigr),$$

symmetrically normalised, of size $8|V| \times 8|V|$. Its kernel is $H^0$ of the sheaf: the assignments
of positions that *every* typed edge agrees with simultaneously. Constraining to $\mathrm{O}(8)$ rather
than fitting a general matrix is what makes every invariant below exact — composition becomes a group
operation and $\det = \pm1$ becomes a genuine $\mathbb{Z}/2$ statement rather than a report on the
conditioning of a least-squares solve (§6a.5).

**6b.2 The connection is real.** Against a null that shuffles which target pairs with which source
inside each relation and refits, every relation type beats its null, most by a wide margin. But
significance and effect size come apart sharply, and the split is the interesting part:

| relation | edges | residual | null | $z$ | effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| PERFORMS | 171 | 0.42 | 1.29 | $-71.0$ | **67%** |
| MODIFIES | 97 | 0.63 | 1.27 | $-37.9$ | **50%** |
| ACCESSES | 54 | 0.62 | 1.23 | $-27.0$ | **49%** |
| USES | 163 | 0.77 | 1.30 | $-29.9$ | 41% |
| INJECTS | 754 | 0.93 | 1.37 | $-52.1$ | 32% |
| CALLS | 91 | 0.87 | 1.27 | $-20.5$ | 31% |
| EXTENDS | 315 | 1.29 | 1.43 | $-11.9$ | 10% |
| IMPORTS | 2817 | 1.31 | 1.39 | $-17.4$ | 6% |
| ALGEBRA_VIOLATION | 198 | 1.31 | 1.36 | $-3.7$ | 4% |

The hand-modelled behavioural relations are strongly rigid: a single rotation explains half to
two-thirds of PERFORMS, MODIFIES and ACCESSES. The mechanically extracted syntactic ones are barely
rigid at all — IMPORTS achieves $z = -17.4$ purely on 2817 edges while capturing a 6% effect. **The
behavioural modelling is contributing geometric information the syntax does not contain.** That is
the first quantitative argument in this programme for hand-modelling relations over parsing them, and
it is worth more than any of the topological statements.

**6b.3 Two incompatible families.** Bootstrapping each map 300 times and measuring the rotation
between every pair of relations gives a strikingly bimodal answer:

| | relations | median rotation vs all others |
| --- | --- | ---: |
| **outliers** | ALGEBRA_VIOLATION, IMPLEMENTS, EXTENDS | $67°$–$77°$ |
| **consensus** | PERFORMS, INJECTS, MODIFIES, USES, ACCESSES, IMPORTS, CALLS | $10°$–$15°$ |

Seven relations agree with one another to within $\sim13°$; three sit nearly orthogonal to everything,
with 5th-percentile bootstrap separations of $50°$–$75°$, so the gap is not fitting noise. And the
three outliers are exactly the **subtyping** relations — EXTENDS and IMPLEMENTS are inheritance,
ALGEBRA_VIOLATION is dominated by layering violations — while the seven are all **usage and flow**.
The geometry recovers the is-a / uses distinction without being told it exists.

This is the concrete form of the stratification hypothesis of §7. A single Laplacian over all
relations averages two mutually incompatible connections, which is why the resulting picture is
muddy. The codebase is not one manifold with one frame; it is (at least) two strata glued along the
files that carry both kinds of edge.

**Confound, not yet excluded.** The three outliers are also the three relations whose edges span
*multiple node-type signatures* — EXTENDS appears as Resource→Resource, Rule→Resource, Actor→Resource
and Process→Resource, while every consensus relation except IMPORTS and INJECTS is type-pure
(PERFORMS is always Actor→Process, USES and MODIFIES always Process→Resource, CALLS always
Process→Process). Fitting one rotation to a relation that spans four signatures may produce a map
that matches none of them. The is-a/uses reading stands only if the anomaly survives refitting per
signature, which §6b.6 sets out as the next test.

**6b.4 There is no global frame, and no orientation obstruction.** On the giant component (786 nodes,
3023 edges — the remaining 41 components are trees, hence flat by construction, and 308 files carry no
typed edge at all), a flat connection would give $\dim H^0 = 8$. The measured value is $\mathbf{0}$:
not one direction of the embedding space can be assigned consistently across the whole codebase. The
smallest eigenvalue is $0.016$, so the failure is not marginal but neither is it violent.

Orientation is a separate question and the answer is negative. Sampling 2355 independent cycles gives
a median rotation of $5.0°$, a mean of $14.0°$ and a maximum of $96.3°$; $1.4\%$ reverse orientation.
But every reversing cycle passes through IMPLEMENTS or ALGEBRA_VIOLATION, and those are precisely the
maps whose determinant sign is a coin flip under bootstrap ($48\%$ and $56\%$ agreement). **Every
relation with a stable determinant is orientation-preserving.** The apparent obstruction is an
artifact of two unstable fits, and §6a.5's claim is withdrawn accordingly.

**6b.5 Bigons.** The sharpest cycles in the graph are the shortest: pairs of files joined by two
*different* relation types, where the loop out along $k_1$ and back along $k_2$ has holonomy
$O_{k_2}^{\!\top} O_{k_1}$ and needs no spanning tree at all. There are 1367 such pairs. Ordinary
combinations sit at $5°$–$14°$ (INJECTS/PERFORMS $5.1°$, INJECTS/USES $7.9°$, IMPORTS/USES $13.1°$),
while every combination involving a subtyping relation sits near $74°$. The bigons localise the
family split to specific file pairs, which is what makes it actionable.

**6b.6 What this says about the sub-topology question.** Two findings bear directly on it.

First, the low spectrum is not resolving sub-topologies, it is ranking hubs: $\phi_0 \ldots \phi_3$
correlate with node degree at $r = +0.92, +0.92, +0.92, +0.88$, and their mass sits on
`UserRepository`, `User`, `RepositoryResolver`, `PermissionUtils` — the highest-degree files in the
graph. From $\phi_4$ onward the correlation collapses to $|r| < 0.1$ and the eigenvectors localise on
genuine subsystems: $\phi_4$ concentrates $10.2\%$ of its mass on `ActorRegistry`, with
`SoftAssertionContext`, `Actor`, `StepUpAuthSteps` and `AdvancedSessionSecuritySteps` behind it —
the Cucumber E2E harness, recovered as a connected geometric object. **The first four eigenvectors
must be deflated before the spectrum is read as structure.**

Second, per-node curvature — the mean cycle rotation over all sampled cycles through a file — is
strongly non-uniform and architecturally legible. The highest are `FakturowniaAdapter_IntegrationTest`
($87.7°$), `InfluencerPublicProfileDto` ($86.5°$), `PublicProfileDto` ($82.5°$) and the repository
implementations ($76°$–$77°$); the lowest are controllers and Angular components ($0.5°$–$0.8°$).
Boundary objects — DTOs that straddle the API edge, adapters to external services, repository
implementations that bridge domain and persistence — carry the phase. Leaf components sit in flat
regions. **Curvature is a boundary detector**, which is the usable form of the Berry-phase claim that
the earlier papers asserted and never computed.

---

## 6c. Typing the nodes, and the discovery that most of the connection does not exist

§6b.3 left one thing unexcluded: the three "outlier" relations are also the three that span many
node-type signatures, so their anomalous maps might be an artifact of fitting one rotation across
four different situations. The graph has node types — the 6-entity model, Resource 533, Rule 431,
Actor 227, Process 110, Context 65, Event 8 — so the test is direct. Refit one $O$ per **signature**
$(\text{source type}, \text{relation}, \text{target type})$: 34 of them carry $\geq20$ edges.

**6c.1 The control turned into the result.** Comparing maps fitted on different amounts of data is
unfair — fewer edges, noisier map, larger apparent rotation — so each signature was given its own
noise floor by split-half reliability: split its edges at random, fit a map on each half, measure the
rotation between the halves, repeat 60 times. That is how far apart two maps land when they are
estimates of *the same thing* at *that* sample size.

The floor does not track sample size. It does not come close.

| signature | edges | split-half floor |
| --- | ---: | ---: |
| Rule $\to$IMPORTS$\to$ Resource | 993 | $68.7°$ |
| Resource $\to$IMPORTS$\to$ Resource | 522 | $71.6°$ |
| Process $\to$MODIFIES$\to$ Resource | 97 | $13.1°$ |
| Actor $\to$PERFORMS$\to$ Process | 171 | $5.5°$ |

Ten times the data, thirteen times the disagreement. So the floor is not measuring sampling error; it
is measuring **whether a consistent rotation exists at all**. Two halves that agree to $5°$ mean the
relation has one map. Two halves landing $85°$ apart — where two *random* elements of
$\mathrm{O}(8)$ sit at about $90°$ — mean the fitted map is an artifact of which edges happened to be
drawn, and reporting it as that relation's geometry is reporting noise.

**6c.2 Only eight of thirty-four signatures have a map.**

| reliable ($<25°$) | edges | floor | | no consistent rotation ($>45°$) | edges | floor |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Actor $\to$PERFORMS$\to$ Process | 171 | $5.5°$ | | Rule $\to$IMPORTS$\to$ Resource | 993 | $68.7°$ |
| Actor $\to$INJECTS$\to$ Process | 171 | $6.1°$ | | Resource $\to$IMPORTS$\to$ Resource | 522 | $71.6°$ |
| Process $\to$USES$\to$ Resource | 163 | $10.0°$ | | Actor $\to$IMPORTS$\to$ Resource | 296 | $67.5°$ |
| Process $\to$INJECTS$\to$ Resource | 163 | $10.5°$ | | Resource $\to$EXTENDS$\to$ Resource | 116 | $83.7°$ |
| Process $\to$MODIFIES$\to$ Resource | 97 | $13.1°$ | | Resource $\to$ALGEBRA_VIOLATION$\to$ Resource | 141 | $81.2°$ |
| Actor $\to$ACCESSES$\to$ Resource | 54 | $20.5°$ | | Resource $\to$IMPLEMENTS$\to$ Resource | 36 | $86.9°$ |
| Actor $\to$INJECTS$\to$ Resource | 54 | $20.5°$ | | Rule $\to$IMPORTS$\to$ Rule | 24 | $99.7°$ |
| Actor $\to$IMPORTS$\to$ Process | 69 | $21.1°$ | | *(19 more)* | | |

A further seven signatures return a floor of exactly $0°$, which is not perfection but degeneracy:
their target clouds are single points, so both halves fit the same meaningless map. All TESTED_BY
signatures are in that group — test files have no structure beyond the test edge, so FastRP places
them all together.

**6c.3 The pattern in the survivors is not about relations at all.** Every reliable signature has an
**Actor or a Process as its source**, and a **Process or Resource as its target**. Not one has Rule,
Context or Event at either end — and Rule is the second-largest type in the graph with 431 nodes and
the single largest signature at 993 edges.

The connection exists on the *active* half of the 6-entity model and nowhere else. That is
interpretable: a node's FastRP position is built from its neighbourhood, and Actors and Processes have
rich outgoing structure, so "where this Actor sits" predicts "where what it performs sits". A Resource
or a Rule is positioned by what points *at* it; its own position carries little about what it points
to, and no rotation can recover what is not there.

**6c.4 Consequences, in order of how much they cost.**

*F2 is refuted, precisely.* Not one subtyping signature has a reliable map. The $67°$–$77°$ separation
of §6b.3 was the distance from measured maps to noise, not the distance between two architectural
families. The is-a/uses reading is withdrawn. Note what this does **not** say: the architecture may
well contain the distinction, but this construction cannot see it, because inheritance edges run
between Resources and Resource positions do not encode outgoing structure.

*IMPORTS was a volume mirage.* It is reliable in exactly one of its twelve signatures
(Actor$\to$Process, 69 edges) and noise in the other eleven, including the 993-edge one. The pooled
IMPORTS result of §6b.2 — $z = -17.4$, effect 6% — got its significance from 2817 edges of noise.
**A null model rules out one explanation; it does not establish that the estimate is stable.** Both
tests are needed, and only the split-half test caught this.

*The Laplacian of §6b was built wrong.* It included all 14 relations, of which nine contribute
unreliable maps — and the unreliable ones carry the most edges. Random rotations injected on the
highest-degree edges is a good description of why $\dim H^0 = 0$ and why the low spectrum collapsed
onto hub ranking (§6b.6). **The connection should be rebuilt on the eight reliable signatures only**,
and every downstream result — frustration, curvature, the partition — recomputed there. That is the
next step, and it supersedes rather than extends §6b.

*What survives untouched.* F1's effect-size ordering is confirmed twice over and sharpened: the
hand-modelled behavioural relations PERFORMS, USES, MODIFIES, ACCESSES are exactly the ones with
reliable geometry, and the mechanically parsed IMPORTS/EXTENDS/IMPLEMENTS are exactly the ones
without. The right statement is not "IMPORTS has a weak rotation" but "IMPORTS has no rotation".

---

## 6d. The model class was never tested, and it is wrong

§6c concluded that 26 of 34 signatures "have no consistent rotation". That conclusion is only as
strong as the model class it was tested against, and it was tested against exactly one: $\mathrm{O}(8)$.
Forcing an orthogonal map on a relation whose transformation is a scaling returns something unstable,
and a reliability test then faithfully reports instability while the real structure sits unmeasured.

So fit a ladder of classes per signature and let held-out error choose — fit on half the edges,
predict the other half, report $\|B - AM^\top\|/\|B\|$, where $1.0$ means no better than predicting
the mean. This is comparable across classes in a way a rotation angle is not, and honest about
overfitting by construction.

| class | params | | class | params |
| --- | ---: | --- | --- | ---: |
| identity | 0 | | diagonal | 8 |
| phase $\mathrm{U}(1)$ — **the magnetic Laplacian** | 1 | | rotation $\mathrm{O}(8)$ | 28 |
| torus $\mathrm{T}^4$ | 4 | | similarity $s\mathrm{O}(8)$ | 29 |
| scalar | 1 | | linear | 64 |

**6d.1 Rotation never wins. Not once.**

| | wins across 34 signatures |
| --- | ---: |
| scalar | **17** |
| diagonal | 8 |
| degenerate (nothing to predict) | 5 |
| identity — nothing helps | 2 |
| similarity | 1 |
| linear | 1 |
| **rotation $\mathrm{O}(8)$** | **0** |

And it is not merely last: it is usually *worse than doing nothing*. Actor$\to$PERFORMS$\to$Process
scores $0.43$ under identity and $0.44$ under rotation; Process$\to$USES$\to$Resource, $0.81$ against
$0.82$; Process$\to$MODIFIES$\to$Resource, $0.64$ against $0.67$. Twenty-eight parameters spent to
predict worse than assuming no transformation at all.

**This retracts the machinery of §6a and §6b, not just their conclusions.** Holonomy, Berry phase,
$\dim H^0$, per-node curvature and the orientation invariant are all functions of fitted rotations.
If rotation is the wrong class — and it loses to a one-parameter scalar on every signature in the
graph — then those quantities were measuring the instability of an inappropriate fit. F3, F5 and F6
are withdrawn along with the construction that produced them. The curvature-as-boundary-detector
result was the most appealing thing in this paper and it does not survive; it should be re-derived,
if at all, from an operator the data supports.

**6d.2 The magnetic Laplacian, tested rather than assumed.** The magnetic Laplacian is the rank-1
case of the connection Laplacian: one $\mathrm{U}(1)$ phase per edge instead of an element of
$\mathrm{O}(8)$. That is a real advantage in principle — one parameter instead of twenty-eight — and
it was the obvious candidate to rescue the signatures that could not support a rotation.

It does not. The $\mathrm{U}(1)$ column is identical to the identity column to two decimals on
essentially every signature ($1.01$ vs $1.01$, $0.43$ vs $0.43$, $0.81$ vs $0.81$). The reason is not
that phases are useless in general but that *this* embedding has no complex structure to carry one:
FastRP returns eight real coordinates, and pairing them into $\mathbb{C}^4$ to fit a phase is fitting
a rotation in an arbitrary plane. A magnetic Laplacian is the right operator when the phase means
something — a physical gauge, a directed cycle, a known angular coordinate. Imposed on an arbitrary
real basis, it recovers nothing, and it says so cleanly here.

**6d.3 What is actually in the data.** The identity column is the interesting one. It is well below
$1$ for the behavioural relations and well above $1$ for the structural ones:

| signature | identity error | |
| --- | ---: | --- |
| Actor $\to$PERFORMS$\to$ Process | 0.43 | endpoints already close |
| Process $\to$MODIFIES$\to$ Resource | 0.64 | |
| Actor $\to$ACCESSES$\to$ Resource | 0.66 | |
| Process $\to$USES$\to$ Resource | 0.81 | |
| Resource $\to$IMPORTS$\to$ Resource | 1.26 | endpoints *further apart than the mean* |
| Actor $\to$EXTENDS$\to$ Resource | 2.45 | |

The behavioural relations connect files that are already near each other in the embedding; no
transformation is needed because there is nothing to transform. The structural relations connect files
that are *further apart than a random pair* — inheritance in particular links positions that the
embedding puts nowhere near one another. What the scalar model adds on top is small and uniform
($0.43 \to 0.41$, $0.81 \to 0.71$, $0.64 \to 0.60$).

So the honest object here is an **affinity**, not a connection: a weighted graph in which each typed
edge carries how close its endpoints already are, per signature. That is a far weaker structure than
a sheaf and it is what the data supports. The partition work should be built on it, and §7's
stratification hypothesis should be tested against it rather than against holonomy.

**6d.4 A duplicate edge layer, found by accident.** Several signatures returned byte-identical error
rows, which happens only if they run over the same node pairs. Checking directly: **721 of 754
INJECTS edges (95.6%) are parallel to another relation**, and all 198 ALGEBRA_VIOLATION edges are a
subset of INJECTS. Only 33 INJECTS edges are unique.

INJECTS is the second-largest relation in the graph, so any spectral method treating edges as
independent evidence double-counts a fifth of the edge set — and the bigon counts of §6b.5 were
counting one fitted determinant many times over, not many independent observations. The graph needs
deduplicating before it is partitioned, and every edge-weighted result above inherits this.

---

## 6e. The first external test: does any of this predict co-change?

Every measurement so far has been internal — geometry validated against nulls built from the same
geometry. That can show a pattern is not noise; it cannot show it is *useful*. Git history is
external. Two files edited in the same commit are coupled in a way nobody selected to flatter this
programme, and there are **3,153 commits** across the two repositories.

Method: 2,851 commits of $\leq30$ files (mass renames dropped), each contributing total weight 1 split
over its pairs so a 12-file commit does not outvote twelve 2-file commits, and **within-repo pairs
only** — the two repositories have separate histories, so a cross-repo pair can never co-change and
including such pairs hands a free win to any predictor that implicitly encodes which repo a file is
in, which all of them do. That leaves **565,635 pairs, 12,438 positive, a 2.20% base rate.**

**6e.1 Globally, reading the code beats every graph structure we built.**

| predictor | AUC |
| --- | ---: |
| **content embedding cosine** (Qwen3-Embedding-8B, 4096-d) | **0.843** |
| graph 2-hop | 0.631 |
| same directory | 0.586 |
| graph adjacency | 0.535 |
| random | 0.501 |

The embedding reads file *content*; it never sees the graph. And combining does almost nothing —
embedding + graph 2-hop reaches $0.847$, a gain of $+0.004$.

**6e.2 But that comparison is unfair to sparse predictors, and reversing it changes the answer.**
Adjacency and affinity are nonzero on a few thousand of 565,635 pairs, so a global AUC over a
mostly-tied vector sits near $0.5$ by construction. The fair question is how precise a sparse
predictor is *where it fires*:

| predictor | fires on | precision | lift | recall |
| --- | ---: | ---: | ---: | ---: |
| **affinity (§6d weights)** | 494 | **0.545** | **24.8×** | 2.2% |
| affinity diffused | 2,709 | 0.322 | 14.6× | 7.0% |
| graph adjacency | 3,230 | 0.286 | 13.0× | 7.4% |
| same directory | 13,948 | 0.173 | 7.9× | 19.4% |
| graph 2-hop | 49,304 | 0.084 | 3.8× | 33.4% |

The affinity built from §6d's identity-error weights is **the most precise predictor available**:
where it fires, more than half those file pairs really did co-change. It fires on 494 pairs. It is a
precision instrument, not a ranker, and scoring it as a ranker is what made it look worthless.

**6e.3 The decisive test: hold content fixed, then ask whether an edge matters.** If two files look
equally alike to a model that has read them both, and one pair is joined by a typed edge while the
other is not, does the edge predict co-change? Splitting into content-similarity deciles:

| content-similarity decile | no edge | edge | ratio |
| --- | ---: | ---: | ---: |
| 6 | 0.009 | 0.050 | 5.6× |
| 7 | 0.014 | 0.125 | 8.8× |
| 8 | 0.023 | 0.123 | 5.5× |
| 9 | 0.037 | 0.172 | 4.6× |
| 10 | 0.108 | 0.389 | 3.6× |
| **pooled** | **0.020** | **0.286** | **14.0×** |

**The typed edges carry real information that reading the code does not.** A 14× conditional lift,
holding content similarity fixed, consistent across every decile where edges appear. The reason this
does not raise global AUC is coverage, not quality: the graph touches $0.57\%$ of pairs. Adding a
sparse high-precision indicator to a dense ranker by $\mathrm{emb} + \lambda g$ is simply the wrong
combination rule, and the $+0.004$ was measuring that mistake rather than the graph's value.

**6e.4 Directory structure is the baseline to beat, and it is strong.** Files in the same directory
co-change at $17.3\%$ against $1.8\%$ elsewhere — a **$9.5\times$** coherence ratio at 100% coverage.
Any derived partition that cannot beat "put files in the same folder together" is not earning its
mathematics.

**6e.5 The incumbent partition, corrected.** The graph already carries a `subsystem_id` from an
earlier clustering pass. Scored naively it looked *worse than random* at $0.62\times$ — but that was
an artifact of my own making: **1,060 of 1,374 files (77.1%) sit in an unassigned $-1$ bucket**, and
treating that bucket as a cluster manufactures one giant pseudo-subsystem containing three-quarters
of the graph. Judged only where it actually made a call:

| | coverage | coherence |
| --- | ---: | ---: |
| directory structure | 100% | $9.51\times$ |
| incumbent subsystems | 22.9% | $1.36\times$ |

and unevenly: subsystem 3 (12 files) reaches $9.5\times$ and subsystem 1 (28 files) $4.0\times$, but
the largest cluster — subsystem 2, 82 files — sits at $0.6\times$, *anti*-coherent. So the incumbent
partition is mostly absent, weakly coherent where present, and actively wrong in its biggest cluster.

**6e.6 What this sets up.** The target is now a number rather than an aesthetic: a partition with
full coverage and a coherence ratio above $9.5\times$. And the ingredients are identified — content
embedding for coverage and ranking, typed edges for precision where they exist. The failure mode to
avoid is the one just diagnosed: combining a dense ranker and a sparse indicator additively. They
should be composed, not summed.

---

## 6f. Hyperedges, and three metrics that disagree

A pair is a poor model of a change. A commit touches a *set* of files, and the set is the unit of
work — reducing it to pairs discards the arity. Two hyperedge families are available: **917 commits**
touching 2–30 known files, and **102 behavioural units** from the 6-entity model (a Process together
with the Actors that invoke it and the Resources it touches), the second owing nothing to git.

**6f.1 Containment, and why the first null was too weak.** The derived partition holds $38.9\%$ of
commits entirely inside one subsystem against a label-permutation null of $2.0\%$ ($z = +68.3$), and
$42.2\%$ of behavioural units against $1.8\%$ ($z = +34.6$). Since the partition came from file
*content* and never saw a single edge, recovering behavioural units it was never shown is a genuine
result.

But beating a random label assignment is a low bar, and the real competitor is the folder tree:

| partition | parts | largest part | commits whole | units whole |
| --- | ---: | ---: | ---: | ---: |
| derived subsystems | 18 | 180 | 38.9% | 42.2% |
| **directory depth 3** | 16 | **636** | **52.0%** | **73.5%** |
| directory depth 1 | 7 | 997 | 88.5% | 100.0% |

**Directory wins, and it still wins after running the size-matched null for every partition
separately** (excess over its own null: $+42.2\%$ vs $+36.9\%$ on commits, $+63.8\%$ vs $+40.4\%$ on
units). So on containment the derived partition loses honestly.

**6f.2 But containment is a degenerate objective.** It is maximised by the trivial one-part
partition, which holds $100\%$ of everything. Directory depth 1 scores $88.5\%$ and $100\%$ precisely
because one part holds 997 of 1374 files. And coherence (§6e) has the opposite degeneracy: it is
maximised by singletons. Two metrics that disagree, each maximised at an opposite extreme, cannot
settle anything between partitions whose *balance* differs by a factor of three — directory depth 3
puts 46% of all files in one part; the derived partition's largest is 13%.

Matching part counts was not enough. Balance is the confound the count did not control.

**6f.3 Modularity settles it.** The co-change graph's modularity is degenerate at neither end: one
part scores exactly 0, singletons score negative. On held-out commits:

| partition | parts | largest | modularity $Q$ |
| --- | ---: | ---: | ---: |
| **derived subsystems** | 18 | 180 | **0.4940** |
| directory depth 2 | 9 | 653 | 0.4128 |
| directory depth 3 | 16 | 636 | 0.3771 |
| directory depth 4 | 61 | 636 | 0.3589 |
| directory depth 1 | 7 | 997 | 0.3287 |
| derived modules | 126 | 30 | 0.2914 |
| directory leaf | 252 | 117 | 0.2856 |
| single part | 1 | 1374 | 0.0000 |
| all singletons | 1374 | 1 | $-0.0039$ |

The derived subsystems win, and the two degenerate partitions score 0 and negative as they must,
which is the check that the metric is behaving. So the verdict across three measurements: the derived
partition wins on coherence and modularity, loses on containment, and containment is the one that
cannot be trusted alone.

One further reading: **derived modules (126 parts) score $0.2914$, well below the 18 subsystems.**
The natural scale of this codebase's change structure is about eighteen units, not a hundred and
twenty-six. Level 3 is finer than the co-change evidence supports, and should be treated as a
navigational convenience rather than a claim.

**6f.4 Collapsing: the typed subsystem graph.** Once a partition exists, an edge inside a subsystem
and an edge between two play different roles. Of 4,853 typed edges, **2,401 ($49.5\%$) are intra** —
these are density — and **2,452 are inter**, spread over 78 of the 153 possible subsystem pairs.
Collapsing leaves a typed multigraph on 18 nodes, small enough to read:

| subsystem | files | density | external ratio | |
| --- | ---: | ---: | ---: | --- |
| faq | 17 | 0.610 | 0.40 | tight and self-contained |
| dtos | 34 | 0.148 | 0.49 | |
| subscription | 79 | 0.073 | **0.32** | best-encapsulated feature |
| registry | 49 | 0.091 | 0.43 | |
| user | 67 | 0.086 | 0.80 | |
| service | 67 | 0.062 | 0.78 | |
| **exceptions** | 53 | 0.036 | **0.90** | pure cross-cutting concern |
| showcases | 148 | 0.006 | 0.42 | loose by nature |

The external ratio is the useful column. `exceptions` at $0.90$ is a cross-cutting concern —
practically all its edges leave, which is what an exception hierarchy should look like. `subscription`
at $0.32$ is the best-encapsulated feature module in the system. The strongest seams are
`appliedopportunities`–`service` (252 edges), `appliedopportunities`–`user` (203) and `dto`–`user`
(132), and every one of them is dominated by IMPORTS.

**6f.5 What this means for the edges.** The answer to "what to do with edges whose endpoints share a
subsystem" is now concrete: they become a **density scalar** per subsystem and are otherwise
discarded, since within a subsystem they carry no partition information. The inter-subsystem edges,
kept **typed**, are the level-2 structure — a labelled graph over 18 nodes that states which
subsystems depend on which and by what mechanism. That collapsed typed graph is the sub-topology at
subsystem scale, and unlike everything in §6a–§6c it is small enough for a person to check by hand.

---

## 6g. Testing the Grothendieck V3 method itself, having finally read it

**Disclosure first.** §6a–§6f were built without reading
`GrothendieckAlgebraicTopologies.md`, the document that specifies the method which produced the
`subsystem_id` field those sections repeatedly benchmarked against. Benchmarking a method's *output*
without reading the method is not a fair test of it, and two of the criticisms above need correcting
as a result.

**6g.1 What the method actually is.** §12 of that document specifies: concatenate all 17 per-relation
$\mathbb{R}^8$ projections into a **composite $\mathbb{R}^{136}$** (Phase A, "preserves all
per-relation information without lossy fusion"); k-means on it (Phase B); Leiden on the typed edge
graph (Phase C); fuse the two by co-association (Phase D, Strehl & Ghosh 2002),
$S(i,j) = \tfrac12[\text{same topo}] + \tfrac12[\text{same graph}]$, and Leiden on $S$; then Berry-phase
boundary refinement (Phase E). Note what it never uses: **the raw $\mathbb{R}^{4096}$ content
embedding.** Both of its views are views of the graph.

**6g.2 Correction: the 77% coverage gap is structural, not a botched run.** §6e.5 reported the
incumbent covering only 22.9% of files and implied a partition left half-finished. In fact
**exactly 1060 files have an all-zero $\mathbb{R}^{136}$ composite, and those are exactly the 1060
the incumbent left unassigned — a 100% overlap.** A file carrying no behavioural edge has a zero
composite, and no clustering can separate zeros from one another. The method can only place files
that carry the relations it projects, and in this graph only 314 of 1374 do. That is a real
limitation of the design, but describing it as a bad clustering run was wrong.

**6g.3 Head to head as representations.** Scored on the 30,117 pairs where *both* files have a
non-zero composite — the method's own home ground, where the base rate is 5.77% rather than 2.14%:

| representation | AUC on held-out co-change |
| --- | ---: |
| $\mathbb{R}^{136}$ composite (Grothendieck Phase A) | 0.603 |
| **$\mathbb{R}^{4096}$ content (Qwen3-Embedding-8B)** | **0.844** |
| 50/50 blend of the two | 0.714 |

The composite is substantially weaker than the content embedding it was designed to improve on, and
blending drags the content embedding down rather than lifting the composite. This is consistent with
§6e.1: the composite is FastRP over relation-filtered adjacency, so it inherits the graph's signal
strength (0.535–0.631), not the content model's.

**6g.4 Phase D fusion, given a fair chance.** The fusion idea is separable from the representation, so
run it both ways — with the composite as Input 1 as specified, and with the content clustering
substituted in:

| partition | parts | coherence | modularity |
| --- | ---: | ---: | ---: |
| view 1: content kNN | 19 | **6.11×** | 0.3905 |
| view 2: Leiden on typed edges | 363 | 3.13× | 0.1832 |
| view 1′: $\mathbb{R}^{136}$ composite kNN | 1072 | 4.53× | 0.0199 |
| Phase D fusion (composite + struct), *as specified* | 356 | 2.79× | 0.1845 |
| **Phase D fusion (content + struct)** | 9 | 4.16× | **0.4029** |

As specified, the fusion scores **below both of its own inputs** (2.79× against 4.53× and 3.13×) —
fusing two weak and disagreeing views produces something worse than either. Given the stronger view,
it edges out the best single view on modularity (0.4029 against 0.3905) while losing on coherence.
So co-association fusion is a sound technique that was being fed the wrong input; it is worth keeping
and it is not where the value was lost.

**6g.5 What this arc independently tested from that document.** Three of its claims were reproduced
and refuted here before it was read, which is a stronger form of check than agreement would have been:

| claim | where | verdict |
| --- | --- | --- |
| $\alpha_k$ separates "real" from "virtual" topologies (Prop 4.2) | §4.4 | **refuted** — $\alpha_k$ tracks edge sparsity |
| PERFORMS/TRIGGERS anti-correlate at $\cos = -0.311$, proving non-abelian structure | abstract, §9 | **does not reproduce** |
| Berry phase / holonomy localises subsystem boundaries (Prop 12.1) | §7.5, §12.6 | **refuted** — §6d, rotation is the wrong model class |

The third is the substantive one. §12.6's boundary refinement rests on holonomy computed from the
per-relation projections, and §6d showed that an orthogonal map loses to a one-parameter scalar on
all 34 signatures and usually to doing nothing. The Berry-phase machinery in §7.5 and §12.6 is
measuring the instability of a fit, not a geometric phase.

**6g.6 What to keep.** The training construction of §3 is sound and its output is used throughout
this paper — $\rho_0$ and the per-relation projections are what everything above is computed from.
Co-association fusion is worth keeping. What does not survive is the claim that the
$\mathbb{R}^{136}$ composite is the right representation for partitioning, and the Berry-phase
apparatus built on top of it.

---

## 6h. Is eight dimensions too few?

Every sub-topology in this programme is an $\mathbb{R}^8$ projection of an $\mathbb{R}^{4096}$
embedding whose effective dimensionality the Information Lensing document puts at 50–200. If 8
starves the representation, a great deal follows at once: the composite losing $0.603$ to $0.844$
(§6g.3) would be a capacity artifact rather than evidence the graph view is weak; only 8 of 34
signatures carrying a reliable map (F11) would partly reflect fitting in a space too small to
separate them; and the near-orthogonality of relation subspaces (F8) would be what you get when every
relation is crammed into the same eight directions.

It is a good hypothesis and it is cheap to test. Sweep $d$ with everything else held fixed — same
seed, same iteration weights, same `propertyRatio` — over the full graph, all 1374 nodes, no
sampling. Held-out co-change, 568,003 within-repo pairs.

| $d$ | base projection AUC | composite AUC | composite width |
| ---: | ---: | ---: | ---: |
| 8 | 0.5869 | 0.5679 | 80 |
| 16 | 0.5894 | 0.5683 | 160 |
| **32** | 0.5987 | **0.5772** | 320 |
| 64 | 0.6011 | 0.5727 | 640 |
| 128 | 0.6028 | 0.5745 | 1280 |
| 256 | 0.6025 | 0.5745 | 2560 |
| | | | |
| — | raw $\mathbb{R}^{4096}$ content | **0.8218** | 4096 |

**Width is not the bottleneck.** Thirty-two times the dimension buys $+0.009$ AUC on the composite
and $+0.016$ on the base, both plateauing by $d \approx 32$, and neither comes within $0.24$ of the
content embedding. The ceiling belongs to the FastRP-over-adjacency construction itself: propagating
a random projection of the embedding through relation-filtered adjacency produces a representation
whose co-change ceiling is about $0.60$ however wide you make it.

The hypothesis is refuted, and usefully — it eliminates the most plausible innocent explanation for
why the graph view underperforms. The gap in §6g.3 is not capacity.

**A second reading, unwelcome for Phase A.** At every single dimension the **composite is worse than
the base projection alone** ($0.5679$ vs $0.5869$ at $d=8$; $0.5745$ vs $0.6025$ at $d=256$).
Concatenating the seventeen per-relation projections, which §12.2 describes as preserving all
per-relation information without lossy fusion, produces something that predicts co-change *less well*
than the single global projection it was built from. Preserving information and representing it
usefully are different things: the per-relation projections are noisy views of the same underlying
adjacency, and concatenating them adds variance faster than signal.

---

## 6i. Reading the prompts: the lens V3 dropped, and four checkable defects

Reading `GrothendiecGraphOrganizer.xml` (V2, 1347 lines), `GrothendieckGraphOrganizer_V3.xml`
(781) and `Opus4.1_GlobalSynthesis.xml` (989) end to end turns up one omission that matters more than
everything else, and four specification defects that can be checked rather than argued about.

**6i.1 The omission: two of three lenses were never built.** V2 §5 specifies a *triple* embedding
product space and marks it critical:

$$S = \mathbb{R}^{4096}\ (\text{semantic, WHAT}), \quad B = \mathbb{R}^{4096}\ (\text{behavioural, HOW}), \quad T = \mathbb{R}^{4096}\ (\text{structural, WHERE})$$

with $C = S \times B \times T \cong \mathbb{R}^{12288}$, "orthogonal by design, target correlation
$< 0.3$", and $T$ explicitly **generated by Grothendieck** from GDS metrics. The V3 graph has none of
it: `semantic_embedding` 0 nodes, `behavioural_embedding` 0, `structural_embedding` 0,
`hyperedge_candidates` 0. One `embedding` property, which is the semantic lens.

So every comparison in this paper between "the content embedding" and "the graph" has pitted $S$
against FastRP-over-adjacency, and never against the structural lens the design called for. That is
worth fixing, because §6e found typed edges carry a 14× conditional lift while covering 0.57% of
pairs — and a structural lens is exactly the construction that spreads that signal to every node,
asking "do these files occupy similar architectural positions" rather than "is there an edge here".

**6i.2 Built, and it does not help.** $T$ was constructed from 42 graph invariants — in- and
out-degree for each of the 18 relation types, plus degree, PageRank, clustering coefficient, core
number, exact betweenness, and component size — defined on all 1374 nodes.

| lens | AUC on held-out co-change |
| --- | ---: |
| $S$ — semantic (Qwen3) | **0.8218** |
| $T$ — structural (built here) | 0.5651 |
| $S + T$, $w_T = 0.1$ | 0.8036 |
| $S + T$, $w_T = 0.3$ | 0.7260 |
| $S + T$, $w_T = 0.5$ | 0.6600 |

Adding *any* structural weight hurts, monotonically. The correlation between the two lenses is
$+0.214$, so V2's orthogonality target is genuinely **met** — the lenses do measure different things.
They are simply not both predictive: $T$ sits at the same $\approx 0.57$ ceiling every graph-derived
representation in this paper has hit.

One caveat stated rather than buried: this tests *linear blending of cosines*, and §6e.3 already
showed that linear combination is the wrong rule for signals with different coverage. A conditional
test could still find $T$ carries something $S$ does not. What is established is that the product
space, as V2 defines it — Euclidean distance in the concatenation — does not beat $S$ alone.

**6i.3 Defect: the V2 sheaf gluing thresholds are unreachable on this similarity scale.**
V2 sets `internal_cohesion > 0.7`, `external_coupling < 0.3`, `sheaf_quality = cohesion − coupling > 0.4`.

| partition | parts | cohesion | coupling | quality | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| derived subsystems | 18 | 0.565 | 0.447 | 0.118 | fail |
| directory (leaf) | 252 | 0.626 | 0.457 | 0.168 | fail |

Both fail, and the partition is not the reason. Qwen3 cosines over source files are compressed into
roughly $[0.15, 0.97]$ with a mean near $0.45$, so *no* partition of this corpus can show cohesion
above $0.7$. The thresholds were written for a different similarity scale. They have to become
**relative** — cohesion against a label-permutation null — to carry information. And note the
granularity confound appearing for the fourth time: directory scores higher quality only because 252
parts are smaller than 18.

**6i.4 Defect: the V2 cohomology query does not compute what it names.** V2 sets targets
$H^0 = 1$, $H^1 = 0$, and computes $H^1$ as "count of strongly connected components with size > 1".
For an undirected graph the first Betti number is $b_1 = E - V + C$. On this graph:

| quantity | value |
| --- | ---: |
| $b_0$ = connected components | **349** |
| $b_1 = E - V + C = 3380 - 1374 + 349$ | **2355** |
| what the V2 query calls $H^1$ (SCCs of size > 1) | 28 |

These are different quantities measuring different things. $H^0 = 1$ fails at 349 because 308 files
carry no typed edge at all. $H^1 = 0$ is unreachable for any codebase with more edges than files —
$b_1 = 2355$ here — so as written the check can only ever report failure. The SCC count is a
perfectly good *directed cyclicity* smell metric and 28 is the honest number; it just is not $H^1$.

**6i.5 Defect: `Opus4.1_GlobalSynthesis.xml` double-counts one term as 40% of its score.** Its
`mathematical_score` is

$$0.3\,c_{\text{int}} + 0.3\,(1 - c_{\text{ext}}) + 0.2\,\tfrac{|\text{entity types}|}{6} + 0.2\,\mathbb{1}[h_1 = 0]$$

but $h_1$ in that same query is *defined* as a step function of the entity-type count
($\geq 6 \Rightarrow 0$, $\geq 4 \Rightarrow 1$, else 2). So entity-type coverage enters twice and
supplies 40% of the total, while being relabelled as cohomology on its second appearance. A subsystem
containing all six entity types scores 0.4 before any geometry is considered.

**6i.6 Defect: `α_k` is documented in two directions.** The paper (§4.4) says small $\alpha_k$ means
*real*; the V3 prompt's Phase 4 says small $\alpha_k$ means *virtual*. Both are refuted anyway —
$\alpha_k$ tracks edge sparsity — but a quantity whose interpretation is reversed between the
specification and its own paper was never load-bearing.

---

## 6j. Full graph, twenty splits, and a headline demoted

Two corrections, both from the same instruction: measure over the whole graph, and stop
reporting point estimates.

**6j.1 Is the graph whole?** It looked like only 41% of the frontend was indexed — 317 `.ts`
nodes against 780 tracked. It is not a gap. Of those 780, **181 are `.spec.ts`** and **271 are the
generated OpenAPI client** under `src/app/api`. Against hand-written source the coverage is
**946/969 `.java` (97.6%)** and **317/328 `.ts` (96.6%)**. The graph is essentially the whole
hand-written codebase. This does bound the oracle, though: commits touching specs or generated
clients contribute nothing to co-change, because those files have no node.

**6j.2 My "structural lens" was not the one V2 specified.** §6i.2 built $T$ from 42 hand-crafted
graph invariants and reported that it does not help. V2 §5 specifies $T \in \mathbb{R}^{4096}$ — the
same width as the semantic lens, i.e. **an embedding, not a feature vector**, produced by embedding a
structurally-framed view of each file. Forty-two scalars standing in for a 4096-dimensional learned
representation is not that construction. §6i.2's negative result stands as a statement about *graph
invariants as a lens*; it is **not** a test of the triple-embedding design, and the write-up
overstated its reach. Testing that design properly needs the re-indexing pass that produces $B$ and
$T$ as genuine embeddings under different framings.

**6j.3 Every number in this paper rested on one commit split.** A single 50/50 split gives a point
estimate with no error bar, and several conclusions here turned on differences of 0.01–0.05. Redone
over **20 independent splits**, full graph:

| predictor | mean AUC | sd |
| --- | ---: | ---: |
| content embedding $S$ | **0.8212** | 0.0113 |
| graph 2-hop | 0.6211 | 0.0055 |
| same directory (leaf) | 0.5975 | 0.0102 |
| graph adjacency | 0.5320 | 0.0015 |

| partition | coherence | sd | modularity | sd |
| --- | ---: | ---: | ---: | ---: |
| derived subsystems (18) | 5.47× | 0.39 | **0.3556** | 0.0208 |
| derived modules (126) | 11.74× | 0.55 | 0.1488 | 0.0068 |
| directory leaf (252) | 10.92× | 1.28 | 0.1906 | 0.0175 |
| directory depth 3 (16) | 2.65× | 0.23 | 0.3172 | 0.0365 |
| incumbent `subsystem_id` | 0.72× | 0.05 | 0.0617 | 0.0056 |

**6j.4 The headline is demoted.** Paired across splits:

| comparison | mean Δ modularity | sd | splits won |
| --- | ---: | ---: | ---: |
| derived subsystems vs directory **leaf** (18 vs 252) | $+0.1650$ | 0.0225 | **20/20** |
| derived subsystems vs directory **depth 3** (18 vs 16, *matched*) | $+0.0383$ | 0.0331 | **17/20** |
| subsystems vs modules (the §6f.3 scale claim) | $+0.2068$ | 0.0172 | 20/20 |

§6f.3 reported derived subsystems beating directory on modularity from a single split — $0.4940$
against $0.3771$, a margin of $0.117$. At **matched granularity** across 20 splits that margin is
$+0.038$ with a standard deviation of $0.033$, and it loses in 3 splits out of 20. **The margin is
the same size as the noise.** That claim is demoted from confirmed to *suggestive*: the derived
partition is probably better at matched granularity, but a single split made it look settled when it
is not.

What survives intact is the comparison against directory at its own natural granularity (20/20, margin
seven times the spread), the representation ranking (content is separated from everything else by
twenty standard deviations), and the coherence result — where **modules at 126 parts score 11.74×
against directory leaf's 10.92× despite being half as fine**, which is a fair-direction comparison
and favours the derived partition.

**6j.5 The commit-size cutoff was not load-bearing.** It was fixed at 30 by assertion. Swept:

| cutoff | positives | content AUC | derived $Q$ | directory leaf $Q$ |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 2,600 | 0.8769 | 0.4663 | 0.2491 |
| 30 | 12,143 | 0.8218 | 0.3751 | 0.1894 |
| 100 | 21,514 | 0.8017 | 0.3437 | 0.1464 |

The ranking is unchanged at every cutoff, so the choice of 30 was not doing hidden work. Content AUC
falls as the cutoff rises, which is the expected direction — larger commits are noisier evidence of
coupling.

---

## 6k. The field-standard metrics, which agree more strongly than ours did

Every objective used so far was invented for this arc — "coherence", and modularity of a co-change
graph. Both are defensible; neither is what the architecture-recovery literature reports, which makes
these results hard to place against published work. The standards are **TurboMQ** (Modularization
Quality, intrinsic, no ground truth needed) and **MoJo/MoJoFM** (edit distance to a reference
decomposition). Computing them costs little and it changes the conclusion.

**6k.1 TurboMQ, at matched granularity.** Computed on the deduplicated typed dependency graph — so
this is evidence *independent of co-change*, from the dependency structure rather than from history.

| partition | parts | TurboMQ |
| --- | ---: | ---: |
| **derived subsystems** | 18 | **9.32** |
| directory depth 3 | 16 | 2.76 |
| incumbent `subsystem_id` | 11 | 2.22 |
| | | |
| derived modules | 126 | 29.43 |
| directory leaf | 252 | 48.50 |
| *single part (sanity)* | 1 | *1.00* |
| *all singletons (sanity)* | 1374 | *0.00* |

TurboMQ rises with part count by construction — each cluster contributes at most 1.0 — so only the
matched-granularity block is comparable. There, **the derived partition scores 3.4× the best
comparable baseline**. The sanity checks land exactly on their required values (1.00 and 0.00), which
is the evidence that the implementation is correct rather than flattering.

This is the strongest result the partition has, and it is stronger than anything measured from
co-change. §6j demoted the co-change modularity claim to suggestive because at matched granularity the
margin was $+0.038 \pm 0.033$ over 20 splits. TurboMQ, on a different graph and a different objective,
puts the same comparison at $9.32$ against $2.76$. Two independent lines now agree in direction, and
the standard one agrees more strongly.

**6k.2 Agreement with the developer decomposition.** A directory tree is a developer-created
decomposition, so it can play the role an expert decomposition plays in the literature.

| partition | MoJo | MoJoFM | ARI | NMI |
| --- | ---: | ---: | ---: | ---: |
| derived subsystems | 1081 | 3.7% | 0.121 | **0.617** |
| derived modules | 795 | 29.1% | 0.197 | 0.736 |
| directory depth 3 | 1075 | 4.2% | 0.070 | 0.562 |
| incumbent | 1214 | **−8.2%** | 0.013 | 0.268 |

The interpretation was fixed before looking: high agreement means the method recovered what the
folder tree already encodes — reassuring, but it added little; low agreement is a success *only* if
paid for elsewhere. What the numbers show is **low alignment (ARI 0.121) with high shared information
(NMI 0.617)**: the derived partition is highly informative about the folder tree while cutting it at a
different alignment. And it is paid for — by TurboMQ at matched granularity.

Note the incumbent's MoJoFM of $-8.2\%$. A percentage that goes negative proves the normalisation
convention used here is not a true upper bound, which is exactly why ARI and NMI are reported beside
it. **MoJoFM should not carry a comparison on its own**; its convention varies between
implementations and this one is demonstrably loose.

---

## 6l. The resolution limit: a legitimate objection that the claim survives

§6f.3 concluded that the natural scale of this codebase's change structure is about eighteen units
rather than a hundred and twenty-six, because the derived subsystems score modularity $0.3556$ against
the modules' $0.1488$. The literature says that conclusion is unsafe: modularity has a documented
**resolution limit** (Fortunato & Barthélemy, PNAS 2007) — it cannot resolve communities holding
fewer than $\sqrt{L/2}$ edges, because the null model it subtracts is global. Fine partitions are
penalised by construction, whatever their quality.

**6l.1 The objection applies, and strongly.** On the held-out co-change graph, $L = 12{,}143$ edges,
so the threshold is $\sqrt{L/2} = 77.9$ internal edges per part:

| partition | parts | median internal edges | parts below threshold |
| --- | ---: | ---: | ---: |
| derived subsystems | 18 | 253.0 | 4 (22%) |
| **derived modules** | 126 | **10.0** | **123 (98%)** |
| directory leaf | 252 | 1.0 | 247 (98%) |
| directory depth 3 | 16 | 6.0 | 13 (81%) |

Ninety-eight percent of the modules sit below the limit. Modularity structurally *cannot* resolve
them, so the original comparison was not decided by the data. The caveat was worth raising.

**6l.2 Two objectives immune to it.** The **map equation** (Rosvall & Bergström) measures the
description length of a random walk in bits and is granularity-aware by construction — naming a module
is paid for explicitly in the index codebook, so no matched part count is needed. Lower is better, and
a single part is the natural baseline.

| partition | parts | bits | sd | vs single part |
| --- | ---: | ---: | ---: | ---: |
| **derived subsystems** | 18 | **9.144** | 0.081 | **−0.294** |
| directory depth 3 | 16 | 9.230 | 0.091 | −0.207 |
| *single part (baseline)* | 1 | *9.437* | 0.066 | — |
| directory leaf | 252 | 9.952 | 0.096 | **+0.514** |
| derived modules | 126 | 10.185 | 0.073 | **+0.747** |

And the **Constant Potts Model** (Traag, Van Dooren & Nesterov 2011), which is resolution-limit-free
by construction because it carries no global null term, gives subsystems over modules at **every**
$\gamma$ from $10^{-4}$ to $3\times10^{-2}$, in **20 of 20 splits** at each.

**6l.3 Verdict: F33 survives.** Modularity, the map equation and CPM all rank subsystems above
modules, each 20/20 across splits. The objection was legitimate — the modules really are below the
resolution limit — but the conclusion does not depend on the objective that has the limit. The
natural scale of this codebase's change structure is about eighteen units.

**6l.4 A stronger statement falls out.** The map equation says the 126 modules and the 252-part
directory leaf are **worse than not partitioning at all** ($+0.747$ and $+0.514$ bits above the
single-part baseline). Under an objective that charges for the names it uses, over-partitioning is
not merely less good — it costs more to describe than the unpartitioned graph. Only the two coarse
partitions compress, and the derived subsystems compress best.

This also puts §6f's three-way disagreement to rest. Containment favoured coarse, coherence favoured
fine, modularity favoured coarse-with-a-known-bias — and the map equation, which needs no granularity
matching at all, agrees with modularity while owing it nothing.

---

## 6m. Should evidence constrain the search, or only judge the result?

Everything built above clusters freely and validates afterwards. The acceptance gate *judges* a
partition; it never *constrains* the search. Those are different things, and the objection is that no
unsupervised algorithm should be trusted to detect subsystems without legitimate connections behind
the joins it makes. The measurements above were wanted as supporting evidence for those joins, not as
the partitioning mechanism.

**6m.1 The diagnosis is already confirmed by what happened.** Not one sub-topology became a
subsystem. The final partition came from content-embedding kNN; the per-relation projections
contributed nothing (content-only 16.05× against content×edges 15.60×). "Not every sub-topology
should create a subsystem" is not a proposal — it is what the measurements forced.

And on *which* should be joined, the original space already answers: relation subspaces sit at
72°–78° mean principal angle, near-orthogonal, **except USES and MODIFIES at 25.1° with
MODIFIES's support contained entirely inside USES's**. Exactly one merge is indicated, and the
interaction tensor ranked that pair two orders of magnitude above the next candidate (0.276 against
0.002). Merging them cannot improve a partition they do not contribute to, but it is the right
structure for the *pairwise query surface*, where they are the one pair that should answer as one lens.

**6m.2 Hard constraints from algorithmic evidence make it worse.** The affinity from reliable
signatures is the most precise signal in this paper — 54.5% precision at 24.8× lift — and it yields
942 candidate must-link pairs. Applied as hard constraints by union-find contraction:

| partition | parts | coherence | bits ↓ | TurboMQ |
| --- | ---: | ---: | ---: | ---: |
| **free (unsupervised)** | 19 | **5.69×** | **9.057** | **9.48** |
| constrained, 342 links | 16 | 3.35× | 9.407 | 7.55 |
| constrained, 547 links | 15 | 2.91× | 9.479 | 6.95 |

Monotonically worse on all three objectives, including two the constraints were not derived from.
The reason is arithmetic: 54.5% precision means nearly half the constraints are wrong, and union-find
closure is transitive, so a few bad links chain together groups that should stay apart. This is the
known failure mode of transitive must-link closure under noisy constraints.

**6m.3 But the free clustering already satisfies most of them.** 610 of 942 must-link pairs (64.8%)
are *already* in the same part without being told. The typed-edge evidence is largely implicit in the
content signal, which is why up-weighting it did nothing earlier and why forcing it does harm now.

**6m.4 Where this locates the human.** The result does not say evidence should not constrain the
search. It says *this* evidence, at 54.5% precision, is too noisy for hard constraints. Hard
constraints need precision near 1.0 — which is what **human** knowledge supplies, and human knowledge
is what was being described: connect subsystem *candidates* on what we know.

So the design consequence is sharper than either position. Not "algorithm partitions, human
validates" (§6h's gate), and not "algorithmic evidence constrains the search" (refuted here), but:

> The algorithm proposes candidates. The evidence identifies **where it and the evidence disagree**.
> The human adjudicates that shortlist, and those adjudications become the hard constraints.

That shortlist is small enough to be real work rather than a gesture: **332 pairs** — the must-links
free clustering did not already satisfy — out of 568,003 candidate pairs, or 0.06%. A person can
review 332 pairs in an afternoon, and each decision is a genuine architectural judgement of the kind
no objective function here can make.

That is the defensible role for the geometry: not to partition, and not merely to be validated
against, but to **narrow 568,003 pairs to 332 questions worth a human's attention.**

---

## 6n. Layers or slices? The embedding sees similarity; a subsystem is a slice

The objection: an embedding places *similar* things together — all controllers near each other, all
DTOs near each other. But a subsystem is not a set of similar things. Security is a controller, a
service, some config, some rules and some DTOs, objects that do not resemble one another at all. So a
subsystem is a **vertical slice across layers**, similarity clustering finds **horizontal layers**,
and a subsystem naturally sits across many sub-topologies rather than being bounded by one.

Using `entity_type` as the layer coordinate (Actor / Process / Resource / Rule / Context / Event ≈
controller / service / data / validation / infrastructure / event), four predictions follow. Two hold,
two fail, and the failures are the interesting part.

**6n.1 Confirmed — the embedding sees layers.** Same-type pairs average cosine $0.4927$ against
$0.4466$ for cross-type, and among the **top 1% most similar pairs, 57.3% are same-type against a
32.4% chance rate**. The content model is strongly biased toward within-layer similarity, exactly as
argued.

**6n.2 Confirmed — typed edges cross layers.** 76.9% of typed edges join files of different entity
type, against 67.6% by chance. Actor→PERFORMS→Process and Process→USES→Resource cross layers by
construction.

**6n.3 Refuted — co-change is not cross-layer *enriched*.**

| pair kind | pairs | co-change rate | lift |
| --- | ---: | ---: | ---: |
| same entity type | 184,081 | 0.0429 | **1.36×** |
| cross entity type | 383,922 | 0.0261 | **0.83×** |

Same-layer pairs co-change *more* than chance, not less. 55.9% of all co-change is cross-type, but
67.6% would be expected if coupling ignored type entirely — so co-change is same-layer *enriched*
while still being majority cross-layer in absolute terms. Both readings are true and neither alone is
the whole picture.

**6n.4 Refuted, and in our favour — the derived partition is not layer-biased. Directory structure
is.** Mean within-cluster entropy of entity type, against a global ceiling of 2.027 bits:

| partition | entropy | % of ceiling | |
| --- | ---: | ---: | --- |
| **derived subsystems** | 1.747 | **86%** | clusters stay mixed — slice-like |
| directory leaf | 0.586 | **29%** | clusters are pure — layer-like |

The similarity clustering did **not** collapse into layers. It is the folder tree that is the layer
decomposition — `dto/`, `service/`, `exceptions/` — while the derived partition keeps its clusters
close to the global type mixture. This inverts the concern: the horizontal decomposition here is the
baseline, not the method.

**Why it looked otherwise, and a defect this exposes.** The subsystem names in §6f — `dto`, `dto`,
`dtos`, `service`, `service`, `exceptions` — come from each cluster's *dominant directory basename*,
which is a layer name even when the cluster's contents are mixed. The naming heuristic made a
slice-like partition read as horizontal. **The names were misleading and should be derived from the
entity-type mixture as well as the folder**, or dropped in favour of the top members.

**6n.5 What the objection gets exactly right.** The signal-coverage table is the payoff:

| signal | share of its hits that cross a layer |
| --- | ---: |
| content embedding, top 1% pairs | **42.7%** |
| actual co-change | **55.9%** |
| typed edge present | **76.9%** |

Real coupling is 55.9% cross-layer. The content embedding finds only 42.7% — it **systematically
under-samples cross-layer coupling**. Typed edges run at 76.9% — they over-sample it, which is to say
they cover precisely the region the embedding misses.

That is the mechanism behind a result that had no explanation until now: typed edges carry a 14×
conditional lift on 0.57% of pairs because they connect files that are **distant in embedding space
and adjacent in the architecture** — the controller and the DTO of one feature. It also explains why
linear blending could never work (§6e.3, §6i.2): the two signals do not cover the same pairs with
different noise, they cover **different kinds of pairs**. Averaging them dilutes both.

The constructive form suggested itself immediately: use content on same-type pairs, where it is
strong, and typed-edge evidence on cross-type pairs, where content under-samples. A routing rule
rather than a blend. **It was tested in the same session and it fails.**

| predictor | mean AUC | sd | vs content |
| --- | ---: | ---: | ---: |
| **content only** | **0.8212** | 0.0113 | — |
| blend, $w_g = 0.2$ | 0.8098 | 0.0111 | $-0.0114$ |
| routed on cross-type, $w_g = 0.3$ | 0.8016 | 0.0113 | $-0.0196$ |
| routed on cross-type, $w_g = 0.7$ | 0.7494 | 0.0121 | $-0.0718$ |

Every routed variant loses, in 0 of 20 splits. The reason is visible once AUC is computed *within*
each pair class, so the classes cannot flatter one another:

| | content | graph 2-hop | winner |
| --- | ---: | ---: | --- |
| same-type pairs | 0.8090 | 0.6354 | content |
| **cross-type pairs** | **0.8196** | 0.6089 | **content** |

**Content is not weak on cross-type pairs — it is marginally *stronger* there** ($0.8196$ against
$0.8090$). There is no region in which the graph out-ranks it, so routing has nothing to route to. The
premise that the embedding is blind to cross-layer coupling is wrong: it under-*samples* cross-layer
pairs in its top 1%, but it still *orders* them better than the graph does.

And yet the edge lift is real in both classes:

| | P(co-change \| edge) | P(co-change \| no edge) | lift |
| --- | ---: | ---: | ---: |
| same-type pairs | 0.2987 | 0.0274 | 10.9× |
| cross-type pairs | 0.2215 | 0.0157 | **14.1×** |

Both facts hold at once, and the reconciliation is that **a typed edge is an excellent indicator and
a poor ranker**. It says a great deal about the 0.6% of pairs it touches and nothing at all about the
ordering of the rest, which is what an AUC over all pairs measures.

This closes the combination question, which has now failed three separate ways — up-weighting
(§6e.3), hard constraints (§6m.2), and routing (here) — each for the same underlying reason. The
typed graph should not be combined with content for ranking or for clustering at all. Its role is the
one §6m.4 identified: flagging the few hundred pairs where it and the clustering disagree, for a human
to adjudicate.

---

## 6o. The product space, finally tested clean — and refuted for ranking

V2's triple-embedding design (§6i.1) was the one major construction this programme had never
actually tested: F41 recorded it *untested, not refuted*, because only the semantic lens existed and
my 42-invariant substitute was a different object. It is now built properly — three genuinely
different socket texts per file (S names/docstrings, B behaviour-shape with I/O by category, T
neighbourhood-as-prose), embedded through Qwen3-Embedding-8B, uniform across all 1374 nodes under one
provenance-stamped builder, with per-node lens distinctness verified (S·B 0.69, S·T 0.60, B·T 0.58
against an instruction-only floor of 0.9451; top-5 neighbour Jaccard ≈ 0.12–0.26, so ~83% of each
lens's neighbours are unique to it; the behavioural probe returns feature-mates under S, same-layer
controllers under B, dependencies under T).

The lenses are real. The pre-registered gate says they do not help.

**6o.1 The gate** (criterion fixed before any lens existed: beat content+lexical 0.8411 in ≥18/20
paired splits):

| signal / combination | AUC | sd | vs baseline | wins |
| --- | ---: | ---: | ---: | ---: |
| generic embedding alone | 0.8212 | 0.0113 | | |
| lexical TF-IDF alone | 0.8224 | 0.0066 | | |
| S semantic alone | 0.8022 | 0.0110 | | |
| B behavioural alone | 0.7356 | 0.0101 | | |
| T structural alone | 0.7261 | 0.0126 | | |
| **baseline content+lexical** | **0.8411** | 0.0082 | — | — |
| baseline + S | 0.8360 | 0.0092 | −0.0051 | 0/20 |
| baseline + B | 0.8246 | 0.0091 | −0.0165 | 0/20 |
| baseline + T | 0.8204 | 0.0094 | −0.0207 | 0/20 |
| three lenses only | 0.7804 | 0.0114 | −0.0607 | 0/20 |
| baseline + all three | 0.8227 | 0.0097 | −0.0184 | 0/20 |

**Nothing passes.** And the RRF exploratory arm (promised in F85) shows the verdict is
fusion-rule-robust while improving the baseline itself:

| RRF arm | AUC | vs rank-mean baseline | wins |
| --- | ---: | ---: | ---: |
| **RRF(content, lexical)** | **0.8432** | +0.0021 | **20/20** |
| RRF + S | 0.8393 | −0.0018 | 2/20 |
| RRF + B | 0.8250 | −0.0161 | 0/20 |
| RRF + T | 0.8334 | −0.0077 | 0/20 |
| RRF all five | 0.8224 | −0.0187 | 0/20 |

RRF(content, lexical) meets the win-count bar against the registered baseline and becomes the
ranker of record at **0.8432**. Every lens arm loses under both fusion rules.

**6o.2 Why content+lexical combines and content+lens does not.** Qwen and TF-IDF are **different
measurement mechanisms over the same full text** — independent error modes, genuinely complementary.
S, B and T are **the same mechanism over subsets of the text**. A subset cannot add information the
full-text embedding already saw; it can only *remove* context, and what remains unique to each lens
(B and T carry real standalone signal at 0.73) is evidently redundant with the full-content view for
this oracle. The design lesson, stated once for reuse: **diversify the mechanism, not the input
slice.**

**6o.3 What survives.** Three things, none of them ranking:
- **Query-time asymmetric retrieval.** Qwen3's instruction protocol is query-side by design; "find
  files with similar runtime behaviour to this incident description" against the B-index answers a
  question the content index cannot. The B-probe result (runtime-shape neighbours with no domain
  relation) is exactly that capability. This is retrieval UX, not co-change ranking, and it needs no
  gate — it needs a vector index and a user.
- **The B-degeneracy finding**: 93/141 Resources collapse to an identical no-behaviour fallback, so
  B is informative for Actors/Processes and near-degenerate for quiet Resources — a caveat any future
  B-lens use inherits (measured enrichment yield: 9%, not worth it).
- **The process artefacts**: spec-by-artifact indexing (HypatiaV5), provenance stamping, golden-fixture
  conformance with a ReDoS sentinel, and the pre-registered-gate discipline that made this refutation
  clean instead of arguable.

F41 is hereby resolved: **tested, and refuted for co-change ranking** on this codebase. The V2
product-space metric $d^2 = \|\Delta S\|^2 + \|\Delta B\|^2 + \|\Delta T\|^2$ should not be built.

---

## 6p. The fused partition: five arms, one near-miss, and the fiber verdict

The proposal under test: reinforce the partition by fusing the ranker of record, the IDF-weighted
hyperedges (paths across sub-topologies bundled as n-ary objects), and the layer view. Plus the
fiber construction stated precisely: the typed graph projects onto the type quiver, a **fiber** over
a quiver path is the set of concrete paths realising it (= meta-path instances), two fibers are
homotopy-adjacent when they share members (member-swap as elementary deformation), and clustering the
**fiber graph** should induce a node partition at the topological level.

Controls: λ chosen on train halves only; Louvain resolution tuned per arm into the champion's
granularity band; 20 held-out splits, paired; promotion bar ≥18/20.

| arm | parts | coherence | modularity | bits↓ | TurboMQ | type-H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 champion (content kNN) | 19 | **5.71×** | 0.3551 | **9.117** | **9.56** | 1.749 |
| A1 RRF(content+lexical) kNN | 16 | 5.16× | 0.3334 | 9.311 | 8.27 | 1.639 |
| A2 A1 + hyperedges | 17 | 5.25× | 0.3352 | 9.298 | 8.33 | 1.580 |
| A3 A0 + hyperedges | 18 | 5.68× | **0.3592** | 9.113 | 9.03 | 1.740 |
| A4 fiber-first | 11 | 2.18× | 0.2734 | 9.175 | 7.10 | **1.856** |

**6p.1 A better ranker is a worse base (A1: 1/20).** RRF(content, lexical) beats content as a
*ranker* 20/20 (§6o) and loses to it as a *partition base* 1/20 ($-0.0217$). The two objectives are
different: a kNN base consumes only the **top-12** of each node's ranking, where lexical similarity
pulls name-twins — which are *layer-mates* (`FooController` ↔ `BarController`). Type-H falls from
1.749 to 1.639: the partition became measurably more horizontal. This independently confirms the
layer thesis of §6n: lexical top-ranks are layer-biased, and the ranker upgrade must **not** be
propagated into the partitioner.

**6p.2 The near-miss (A3: 16/20).** Champion + IDF-weighted hyperedge boost is the first typed-graph
reinforcement ever to move the primary criterion in the right direction: $\Delta Q = +0.0041 \pm
0.0041$, winning 16/20 — **two short of promotion**. By the registered bar it is *not promoted*, and
the bar does not move after the fact. Status: suggestive; the tie-breaker, if wanted, is a
pre-registered 50-split replication, not a re-run until it passes.

**6p.3 The fiber verdict (A4: 0/20 as a partitioner; best-in-class as a slice detector).** As a
complete partition, fiber-first loses everywhere (coherence 2.18×, $\Delta Q = -0.0816$, 0/20). But
its **type-entropy is 1.856 — the most vertical partition of any arm**, exactly what the topological
construction was predicted to produce. The diagnosis is precise: only 314 nodes lie on behavioural
fibers; the remaining **77% are periphery**, attached by content similarity to fiber cores, and that
attachment stage — not the fibers — is where the quality is lost. The skeleton is right and the flesh
does not follow it.

So the fiber idea is neither promoted nor buried: it is a **slice-skeleton detector**, and its
concrete next step is a better periphery-attachment rule (label propagation over the content-kNN from
fiber cores, rather than top-3 mean cosine) — or using fiber-clusters as *seeds* inside the champion
clustering, with F58's caution that seeds are soft constraints and 68% precision still chains errors.

**6p.4 Standing verdict.** The champion remains **A0: Louvain on content-kNN** — now confirmed
against four challengers under matched controls. The typed layer's measured roles stand: pairwise
evidence (14×), n-ary cohorts (21.7×), the 332-pair adjudication shortlist, and now slice-skeleton
detection. Not the objective, and not the base graph.

---

## 6q. Three fixes applied, and what each one settled

§6p left three named fixes. All three ran; none was hand-waved past its result.

| arm | parts | coherence | modularity | type-H | wins vs champion |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 champion (control) | 19 | 5.72× | **0.3555** | 1.749 | — |
| R1 fiber → label propagation | 11 | 2.33× | 0.2827 | **1.846** | 0/20 |
| R2, R3 (alternating loop) | 11 | 2.31× | 0.2821 | 1.845 | 0/20 |
| MAG trophic-phase | 857 | 2.45× | 0.0537 | 0.491 | 0/20 |

**6q.1 The attachment stage was not the bottleneck — F98 revised.** Replacing A4's top-3-cosine
attachment with label propagation over the content-kNN from fiber seeds — a strictly better rule —
recovered $+0.009$ modularity of a $-0.073$ gap. The remaining deficit lives in the **skeleton
itself**: 11 fiber-clusters seeded from 214 core nodes do not carve this codebase the way change
behaviour does. The fiber layer's roles stay exactly where they were measured — evidence objects at
21.7×, slice-skeleton detection (type-H 1.85, still the most vertical structure anything builds) —
and *partition driver* is confirmed not among them at this fiber density.

**6q.2 The alternating loop is mechanically sound and immediately inert.** Fiber re-weighting by
partition purity converged in one round (weight drift $0.044 \to 0.002 \to 0.000$; partitions
identical from R1). With 92 fibers over 11 clusters, fiber purity is already near 1 after the first
pass — the loop starts *at* its fixed point. The idea only has room to act on a **denser fiber
corpus**, which is the live path below.

**6q.3 The magnetic Laplacian is now closed with a double seal.** The F16 failure could be blamed on
a fitted phase. This run removed the excuse: the phase came from **trophic levels** — a graph-native
height on the typed digraph (MacKay et al.; the literature's own split assigns *linear* hierarchy to
the trophic construction and *periodic* to the magnetic one, and a layered codebase is linear). The
spectral embedding still shattered: 857 fragments, type-H 0.491 (extreme layer-purity — it clusters
type/degree shells), modularity 0.054, 0/20. With both fitted and graph-native phases failed, the
magnetic route on this graph has no remaining variant to try. One honest caveat on the trophic
printout itself: per-type mean heights are confounded by isolated nodes (77% of files sit off the
behavioural digraph and default to $h=0$), so the "Process sits on top" reading is suggestive, not
established — but no reading of it rescues the downstream arm.

**6q.4 The one live path.** Everything above failed for the same root cause: **fiber density**. 92
meta-path fibers cover 214–314 nodes; the skeleton is right and too sparse. The corpus that fixes
this exists and is leak-clean if handled correctly: **commit-cohort fibers** — each *train-half*
commit's file set as a fiber (V2's `FEATURE_COHORT`, finally with a source), rebuilt per split so the
held-out half never touches the skeleton. That multiplies the fiber corpus by an order of magnitude,
gives the alternating loop something to alternate over, and directly tests whether the fiber-first
architecture works when the skeleton actually covers the graph. It is the next experiment, with the
leakage discipline stated before the run.

---

## 6r. Density delivered, parity reached, and the fusion operator that is still missing

**6r.1 Commit-cohort fibers: the density hypothesis confirmed.** With V2's `FEATURE_COHORT` finally
sourced — each *train-half* commit's file set as a fiber, rebuilt per split so the held-out half
never touches the skeleton, joined by the 92 split-independent meta-path fibers — the fiber-first
architecture jumps from 0.2821 to **0.3462** modularity: **+0.064 from density alone**, exactly what
F100 predicted and nothing else all day has moved a number that far. It now sits at near-parity with
the champion ($\Delta = -0.0093 \pm 0.0169$, 7/20): not promoted, not dismissed — a **second
partitioner of equal strength built from a different mechanism** (incidence of what changes together
and what the typed walks bundle, versus similarity of what the files say).

The alternating loop also wakes at this density: first-round weight drift 0.122 against 0.044 at 92
fibers. It still converges in one round, but it now *does* something before converging.

**6r.2 Consensus, the obvious next move, fails in an instructive way.** F37 measured co-association
fusion reaching 0.4029 with a weak second view; with two strong views the same construction manages
**0.3289, 4/20** — worse than either input. The failure mode is structural: the co-association graph
of two ~20-part partitions is a near-block matrix whose Louvain response is **discontinuous in
resolution** (13 parts at res 2.531, 190 at 2.750; the tuned 23-part setting on split 0 averaged 127
parts across splits, because the percolation point moves with the cohort partition). Resolution-based
tuning cannot hold a knife-edge. F37's success at 9 parts was *coarse* consensus; fine consensus
shatters. **The missing piece is a granularity-stable fusion operator for two strong partitions** —
meet-then-merge on the partition lattice rather than re-clustering a co-association graph — and that
is a precisely stated open problem, not a hope.

**6r.3 The algebra question, answered by the day's ledger.** Should the algebra be rethought toward
another Laplacian? No — and the evidence is now specific. Every vertex operator on the behavioural
digraph shatters for the same structural reason (77% of files are isolated from it; F99 sealed both
the fitted and the graph-native phase). The algebra that *survived* measurement is the **incidence
algebra of fibers** — membership matrices, IDF-weighted overlaps, communities in the fiber dual —
which reached parity with the champion on its first day at full density. The productive object for
V4 of this programme is the pair (content metric, fiber incidence) and the still-missing fusion
operator between them, not a third spectral construction on a graph that cannot carry one.

---

## 6s. Meet-then-merge: the fusion operator found, and a promotion

The operator F103 asked for now exists, in two versions, and the second is **promoted by the
pre-registered criterion** — the first change of champion in this programme.

**The construction.** Given the content partition $A$ and the cohort-fiber partition $B$:
take the **meet** $A \wedge B$ on the partition lattice (123–136 intersection cells — every agreement
preserved, every disagreement localised to cells), then resolve the disagreements by clustering the
**cell-quotient graph**, whose edges carry *both mechanisms*, each max-normalised:
$w(c_1,c_2) = \hat T + \alpha\,\hat C$ — train-half co-change incidence plus content-kNN weight —
with Louvain resolution bisected to land exactly the champion's $k$. Granularity is matched **by
construction**; the percolation knife-edge of co-association is gone because nothing is re-clustered
at the node level.

| arm | modularity | vs champion | vs train-only |
| --- | ---: | ---: | ---: |
| champion (content) | 0.3555 | — | |
| train-only Louvain (ablation) | 0.3402 | 7/20 | — |
| meet-merge **v1** (greedy CNM) | 0.3738 | 14/20 (±0.0441) | 14/20 |
| meet-quotient **v2** (Louvain on quotient) | **0.3738** | **18/20 (±0.0129)** | **18/20** |

v1 and v2 share the same mean; v2 wins because the **variance collapsed 3.4×** when greedy
agglomeration — order-unstable by nature — was replaced by Louvain on the quotient. The ablation
matters as much as the win: v2 beats *train-only clustering* 18/20 too, so the meet is contributing
structure beyond the train history the merge criterion consumes.

**Promotion.** Both registered conditions met (≥18/20 against the champion *and* against the
ablation; sign-test $p \approx 2\times10^{-4}$ each). The production partition is fitted on all
commits (standard CV-select-then-refit), written to `CheckItOutV3` as `v4_subsystem` — 19 subsystems,
136 meet cells, type-entropy 1.687 — with the CV numbers, both parents, the operator parameters and
the finding trail on `V3Master`. `v3_subsystem` is kept: supersession, never erasure.

**Reported trade-off, not hidden:** v2's held-out coherence is lower than the champion's (3.47× vs
5.72× at equal $k$) — its clusters are more balanced, so same-cluster pairs are individually less
precise while the *global* structure matches held-out change better. Modularity was the registered
primary criterion; coherence is disclosed as the cost.

**What the promotion vindicates.** This is the owner's architecture from this morning, item by item:
sub-topologies as *layer* structure feeding fibers rather than similarity; fibers detected first
(meta-paths + commit cohorts); fibers partitioned into subsystems; graph analysis and topology
*enhancing each other* rather than running as stages — the meet is topology, the quotient weights are
graph evidence, and neither alone reaches 0.3738.

---

## 7. A testable hypothesis: stratification

Earlier iterations of this pipeline (V1, V2) were observed to produce topologies that behaved like
**stratified manifolds** rather than a single smooth one. Recent work makes that observation precise
and measurable: LLM embedding spaces are stratified into lower-dimensional local manifolds
(arXiv 2502.13577), and local intrinsic dimension can be estimated per neighbourhood with a localized
TwoNN estimator (arXiv 2506.01034), with reported strata dimensions in the single digits against
ambient dimensions in the thousands.

This yields a claim that is precise, falsifiable, and connected to an active line:

> **Hypothesis 7.1.** The per-relation neighbourhoods of a typed code graph are strata of the
> embedding space with measurably different local intrinsic dimension:
> $d_{\text{ID}}(\mathcal{N}_{\text{MODIFIES}}) \neq d_{\text{ID}}(\mathcal{N}_{\text{CALLS}})$.

If true, it explains *why* the corrections differ, in a language reviewers already read, and without
requiring anyone to accept a physical analogy first. If false, the per-relation machinery is not
earning its cost — which is equally worth knowing.

---

## 8. The experiment that would complete this work

1. **Extend the null** to all relation pairs, 200 permutations, with observed and null matched on
   node set rather than only on edge count.
2. **Sweep `propertyRatio`** downward. If the shared content signal is swamping the relation-specific
   one, the structure in §5.1 should sharpen as the adjacency is weighted more heavily. This doubles
   as the identity-baseline comparison.
3. **Measure local intrinsic dimension** per relation neighbourhood (Hypothesis 7.1), with
   confidence intervals.
4. **Downstream task.** A gold question set over this codebase, answered through the graph, scored
   under three arms — shared projection only, label-shuffled per-relation, true per-relation. This is
   the only measurement that speaks to usefulness, and it is the same instrument the AI Navigator
   evaluation ladder already requires, so it need only be built once.

Win condition, fixed in advance: the true arm beats the shared-projection arm by a margin larger
than the shuffled arm does, with non-overlapping intervals.

---

## 9. Limitations

One codebase, one language, one embedding model. The strongest effect is measured on 29 nodes. Five
permutations resolve no empirical $p$ finer than $1/6$; the separation is carried by the $z$-score.
FastRP optimises no objective and its distortion guarantee does not hold at $r=8$. The relation-type
assignment is produced by an LLM classifier and is not independently validated — an error rate there
propagates directly into everything above. And §6's worry stands: the geometry is real; its utility
is unmeasured.

---

## 10. Contribution, stated narrowly

We do not claim to have discovered that relations are non-commutative operators; the knowledge-graph
embedding literature has built such models since 2019. We claim the following:

1. On a **fixed pretrained** embedding of real code, per-relation corrections carry a signed structure
   that separates data access from orchestration, verified against a label-permutation null that
   reverses the effect.
2. The restriction maps can be **materialised and are inductive**, with held-out $R^2$ up to $0.69$,
   which turns a transductive pipeline into one that can position unseen files.
3. Two previously published results of ours — the real/virtual reading of $\alpha_k$ and the
   $-0.311$ anti-correlation — **do not survive** and are withdrawn here.

The third is the one we would most want a reader to notice, because it is the reason to believe the
first two.

---

## References

Non-commutative and relation-specific KGE: RotatE (Sun et al., ICLR 2019); QuatE (Zhang et al.,
NeurIPS 2019); DensE (Gao et al., 2021); TransR (Lin et al., AAAI 2015); TorusE (Ebisu & Ichise, 2018).
Sheaf methods: Hansen & Ghrist (2019); Bodnar et al., *Neural Sheaf Diffusion* (NeurIPS 2022);
*Learning Sheaf Laplacian Optimizing Restriction Maps* (arXiv 2501.19207); *On the Necessity of
Learnable Sheaf Laplacians* (arXiv 2603.05395).
Magnetic Laplacian: Fanuel et al. (Phys. Rev. E 2017); Zhang et al., *MagNet* (NeurIPS 2021).
Stratification and intrinsic dimension: arXiv 2502.13577; arXiv 2506.01034; arXiv 2503.02142.
Random projection: Johnson & Lindenstrauss (1984); Chen et al., *FastRP* (2019).
Internal: *The Hypatia Basis*; *Grothendieck V3*; *Information Lensing*; *V3 Research Audit 2026-09*.
