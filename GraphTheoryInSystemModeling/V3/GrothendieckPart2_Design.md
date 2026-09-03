# Grothendieck Part 2 — Curation and Navigation: the design

**Date**: 2026-09-02 | **Status**: DESIGN (pre-implementation)
**Inputs**: the cemented unsupervised layer — v4 partition (F106), hyperedge layer (F95),
layered view (F107), disagreement shortlist (F60), lens indexes (F108).
**Question**: after the machine proposes, how do an agent and a human *judge*, and how is the
graph organised inside each subsystem for the least cognitive effort per answer?

---

## 1. The research verdict on the 3-level NavigationMaster

The proposed architecture — NavigationMaster → SubsystemNavigator → EntityDetail, with AI clues
at every node — turns out to be **independently convergent with the strongest schemas in the
field**, on both the human side and the machine side. The sweep (C4/GraphRAG anchors + an arXiv
pass over the 2025–2026 graph-RAG and code-agent literature):

| method | what it is | what part 2 takes from it |
| --- | --- | --- |
| **C4 model** (Brown) | 4 zoom levels for humans | one-question-per-level; **5–9 items per view** (Miller bound). Our L1/L2/L3 = C4 minus deployment |
| **GraphRAG** (Microsoft, 2404.16130) | hierarchical Leiden + LLM community reports | bottom-up reports as AI clues; *global* (map-reduce over reports) vs *local* (neighbourhood) query modes; root summaries at **97% fewer tokens** |
| **ArchRAG** (2502.09891) | *attributed* communities (structure + embedding-similarity links), cross-layer C-HNSW index, adaptive filtering | (a) clustering on structure+content beats structure-only — their ablation's biggest term; our meet-quotient (T̂+α·Ĉ) is the same design family, already measured on our own oracle. (b) **embed the L1/L2 summaries in the same vector space as files** so one query matches at every level and descends — trivial at our scale, adopted. Their token saving vs GraphRAG: up to 250× |
| **LocAgent** (ACL 2025, 2503.09089) | heterogeneous code graph + agent tools `SearchEntity` / `TraverseGraph` / `RetrieveEntity` | the navigation *tool contract*: lexical entity index (BM25 over IDs and content) beside vector kNN; type-aware bounded-hop traversal; **results rendered as trees, not edge lists** (measured to improve LLM graph reasoning). Ablation: typed multi-hop traversal is what carries function-level accuracy |
| **CodexGraph / RepoGraph** (NAACL/ICLR 2025) | graph-DB interface for agents; repo graph plugin (+32.8% SWE-bench) | validates Cypher-as-navigation — our stored entry queries ARE the interface |
| **CodeCompass** (2602.20048) | the **Navigation Paradox**: on hidden-dependency tasks, graph navigation 99.4% coverage vs 78.2% for BM25/vanilla — big context ≠ discovery | (a) the argument for the whole layer: retrieval fails exactly where dependencies are structural, not lexical. (b) **adoption is the bottleneck**: 58% of trials never called the graph tool; when called, 99.5% coverage; checklist-style mandated protocol → 100% adoption. (c) the G1/G2/G3 task taxonomy for our benchmark |
| also ranked | Youtu-GraphRAG (schema-guided vertical agents), Deep GraphRAG (Ant; global/local balance), E²GraphRAG (efficiency), core-based hierarchies (k-core alternative to Leiden), GAM (hierarchical agentic memory), SHERLOC (agents burn ~half their budget on localization — the cost C5 measures) | context; no design change |

**Verdict: keep the skeleton — it is C4-for-graphs plus GraphRAG-reports avant la lettre — and
adopt five disciplines from the field:** one-question-per-level; the 5–9 fan-out budget;
bottom-up reports with global/local modes; summaries embedded in the file vector space
(ArchRAG); and a **mandated entry protocol** — the clue architecture is worthless if agents
skip it, and skipping is the measured default (CodeCompass). No replacement schema found that
does more. SEI "Views & Beyond" adds one idea we take (module view ≠ connector view — the
layered view and the typed seam graph are *two lenses on the same L2*, kept separate).

## 2. What part 2 inherits (all measured, nothing aspirational)

| asset | measured | role in part 2 |
| --- | --- | --- |
| v4 partition, 19 subsystems | 0.3738, 18/20 (F106) | the **candidates** |
| layered view (subsystem × entity type) | 18/19 vertical slices (F107) | curation lens + internal axis |
| per-subsystem external ratio + typed seams | §6f.4 | contracts on L2 |
| hyperedges, IDF-weighted | 21.7× (F95) | internal spines + curation evidence |
| disagreement shortlist | 332 pairs (F60) | the curation **work queue** |
| per-subsystem trophic height (local) | §6q caveat-free on connected members | internal flow axis |
| lens vector indexes | ONLINE (F108) | local search mode |
| B-lens degeneracy on quiet Resources | F88 | a caveat every report must carry |

## 3. The pipeline: C1–C5

### C1 — Dossiers (deterministic, no judgement)
Per candidate subsystem, auto-compile: size; layer profile (its panel-B column); external ratio
and top 3 typed seams; top-10 TF-IDF terms; 3 medoid files (content-space); internal hyperedges
(count, top 3 by IDF); disagreement pairs touching it; **stability** — its per-subsystem ARI
across the 20 evaluation splits (a subsystem that dissolves under resampling should not be
curated as if solid); trophic span of its members. *Algorithms: all already implemented in the
experiment scripts; this phase is extraction, not invention.*

### C2 — Curation (the agent phase; judgement with grounded triggers)
For each candidate: **keep / merge / split / retype / rename**, where the triggers are measured,
the decision is the agent's, and every decision is written as a node:

- **layer-purity flag** (>85% one entity type, n≥10): probably a *layer*, not a subsystem —
  e.g. sub 13 (44/49 Rule = the validation/test cluster). Decision space: keep but **retype** as
  `role:'LAYER'` (cross-cutting), or dissolve into the slices it serves. Never silently keep a
  layer masquerading as a slice.
- **mega flag** (>20% of corpus): sub 17 (375 files, the whole FE). Split proposals come from
  the *other parent* — the cohort-fiber partition's cuts inside it — plus folder features. The
  agent picks cuts that make architectural sense; the fan-out budget (§5) forces the issue.
- **singleton/micro flag** (n<3): absorb into the neighbour with the strongest hyperedge/seam
  evidence, or mark as genuine outlier with a reason.
- **naming**: from dossier (dominant folder + entity mix + top terms + medoids) — never the
  folder basename alone (F65's misleading-names lesson).
- Decisions land as `(:CurationDecision {action, rationale, evidence, decided_by, at})` linked
  to the subsystem — **auditable, and reusable as precision≈1 constraints in future delta runs**
  (the F58→F60 chain: hard constraints are safe exactly when a judge with near-1 precision made
  them).
- The **332-pair shortlist is the work queue**: each pair is a localised disagreement between
  evidence and clustering; merge/split decisions live precisely there.

### C3 — Internal organisation (the layered-clues answer)
Inside each curated subsystem, the graph is organised on **two measured axes plus two anchors**:

- **Layer axis** = entity type (Actor / Process / Resource / Rule / Context / Event) — the free
  layered view, now *within* the slice. This is the primary grouping for display and traversal.
- **Flow axis** = trophic height computed **on the subsystem's own behavioural subgraph**
  (connected members only — computing locally dissolves the F99 isolated-node confound; members
  with no behavioural edge inherit their layer's median height). Within a subsystem, height
  orders controller → service → data *as measured*, not as assumed.
- **Entry points**: files with external in-edges (callers outside the subsystem) or Actors with
  no internal callers. These are "start here" — the first thing an agent or human reads.
- **Spines**: the subsystem's *internal* A→P→R fibers, longest/highest-IDF first — "the
  subsystem in one walk". A spine is the minimal path that touches every layer; reading one
  spine ≈ understanding the slice's shape.
- Within a layer, order members by internal degree (hub first). If a layer inside one subsystem
  exceeds the fan-out budget, group it into **navigation modules** — and label them honestly:
  navigation modules are a *display artifact for the 5–9 rule*, not a coupling claim (F33/F56
  measured that fine-grained modules are below the co-change evidence's resolution; we do not
  re-litigate that, we just fold long lists).

### C4 — The navigation contract (least cognitive effort, both species)
- **L1 NavigationMaster** answers: *what exists and where do I go?* Fields: system summary, the
  subsystem list (name + one-line + size + role), the six stored entry queries, global caveats.
- **L2 SubsystemNavigator** answers: *what does this do and where do I enter?* Fields (the
  GraphRAG-style community report, generated bottom-up from the dossier): `ai_summary`,
  `responsibilities`, `layer_profile`, `entry_points`, `spines`, `contracts` (typed seams with
  external ratio), `caveats` (e.g. B-lens degeneracy), `curation_history`.
  **Two lenses, kept separate** (SEI): the *module view* (layers/members) and the *connector
  view* (typed seams) are different questions; merging them is what makes diagrams unreadable.
- **L3 EntityDetail** answers: *what does this file do?* — existing fields plus `layer`,
  `local_height`, `spine_membership`, `entry_point` flag.
- **Traversal rules**: every question type has a ≤3-hop route from L1; fan-out ≤9 at every
  level (fold into navigation modules when exceeded); **scent rule** (foraging): every child
  label must predict its content — name + one-line + 3 exemplars, because a wrong scent costs a
  whole wasted traversal.
- **Tool contract** (LocAgent's measured trio, in our terms): a *lexical entity lookup*
  (name/term → nodes; the RRF ranker's lexical arm) beside the *vector kNN* (lens indexes,
  F108), plus *typed bounded-hop traversal* (Cypher over the behavioural edges, hop ≤3, typed
  filters). Traversal results are **rendered as indented trees, not edge lists** — measured to
  improve LLM graph reasoning.
- **Embedded summaries** (ArchRAG): L1/L2 `ai_summary` texts get lens embeddings from the same
  Qwen3 service and enter the vector index — one query matches at every level, then descends.
  At 1374+20 nodes this is a for-loop, not an index problem.
- **Two query modes**, mirroring GraphRAG: *global* (match/map-reduce over L2 reports — "what
  handles payments?") and *local* (kNN from a seed + neighbourhood — "what behaves like
  this?").
- **The protocol is mandated, not offered.** CodeCompass measured a 58% silent-skip rate for an
  optional graph tool, and skipping collapses coverage to baseline. So L1's `ai_instruction` is
  a **checklist the consuming agent must walk** (the CLAUDE.md entry protocol already works this
  way — steps 1–4 are mandatory queries), and agent prompts that consume this graph mandate the
  first call. A clue architecture with optional adoption is decoration.

### C5 — Acceptance (pre-registered, like everything else)
Curation is judgement, so the gate measures the *product*, not the decisions:
1. **Tokens-to-answer benchmark, stratified by CodeCompass's task taxonomy** — ~30 questions in
   three strata: **G1 semantic** (answer is keyword-findable: "where is X handled"), **G2
   structural** (2–4 typed hops from a named start: "what does Y's flow touch"), **G3 hidden**
   (no lexical overlap between question and answer files — reachable only through the graph).
   Each answered two ways: navigation-first (L1→L2→L3 + tool contract) vs flat vector-RAG
   baseline. Gate, per stratum: **G3 is where navigation must win** (CodeCompass: 99.4% vs
   78.2% coverage); **G1 is where it must not lose** (BM25 wins G1 there — parity is the bar);
   overall **≥50% fewer tokens at ≥ equal accuracy** (GraphRAG 97%, ArchRAG up to 250× say the
   headroom is real). An unstratified benchmark would let nav-shaped questions flatter the
   system — the strata keep it honest.
2. Structure: 100% of files reachable in ≤3 hops; fan-out ≤9 everywhere; every L2 report
   grounded (each claim traceable to a dossier field).
3. Audit: every curation decision carries rationale + evidence; the layered-view and seam
   numbers on L2 match a fresh recomputation (no stale clues).
4. **Adoption probe**: run one consuming-agent task with the protocol mandated and one with it
   optional; log tool-call counts. If optional ≈ zero calls, that is expected (CodeCompass) and
   is the argument for keeping the mandate — record it, don't fight it.

## 4. What part 2 must NOT do (standing refutations)
No new vertex Laplacians (F99/F104). No lens fusion into ranking (F90). No fine-grained
co-association (F103). No folder-only names (F65). No treating navigation modules as coupling
claims (F33/F56). No trusting a write without a read-back (F93). And curation never edits the
unsupervised layer's outputs — it *annotates and supersedes*, so the measured artifact stays
reproducible underneath the judged one.

## 5. Delivery plan
C1 dossier extractor (script, one run) → C2 curation session (agent + owner on the 332-pair
queue and the flags) → C3 internal-organisation writer (script) → C4 clue generation (bottom-up,
L3→L2→L1) → C5 benchmark. C1/C3/C4 are mechanical; C2 is the judged phase; C5 is the gate.

## 6. Part 3 (recorded 2026-09-02, out of part-2 scope): the routed answering model

Owner's architecture, fixed here so part 2 builds toward it. **Understanding is precomputed
and stored; the runtime model is a navigator of stored understanding, not an understander.**

- **Three answering tiers, cheapest first**: (1) an **answered-questions cache** (~100 MFAQ +
  question-space index; the model's only job is the translation judgment "is this that?");
  (2) on miss, **mandated traversal** of the hierarchy — NavigationMaster → subsystem → node,
  consuming the clues curation deposited, so subsystems arrive pre-understood; (3) raw code as
  last resort.
- **Placement of intelligence**: schema + tools in a custom **MCP server**; navigation policy
  in a **small fine-tuned model**; codebase knowledge in the graph; instance answers in the
  cache. The model is trained on the *method* (routing, traversal, tool use over the schema),
  never on the codebase — so **model and graph are independently replaceable**; the contract
  is the schema (the LeanAlgebra-vs-instance split, applied to answering).
- **Feasibility anchors (measured, not hoped)**: LocAgent — fine-tuned Qwen2.5-32B ≈ Claude-3.5
  at 86% lower cost, 7B ≈ GPT-4o pipelines, trained on ~770 successful trajectories,
  generalizing to unseen repos because tools/schema stayed constant (the replaceability claim);
  Know-Before-Fix 2607.11111 — precomputed repo-QA as a knowledge tier; CodeCompass — server-
  side intelligence works iff the protocol is mandated.
- **Cache safety**: every cached answer records the subsystems/nodes it depended on; delta
  runs bi-temporally invalidate dependents; the router's confidence gate defaults to
  "traverse anyway" on doubt. The near-miss (close embedding, different intent) is the
  failure mode to instrument first.
- **Ordering is forced**: trajectories over our schema are the training data, and they can
  only be generated once C4's clue layer exists. Part 2 is the construction of part 3's
  training environment. The C5 trajectory run (frontier model, mandated protocol, stratified
  questions, full logs) triple-counts: benchmark number, first training corpus, and the spec
  of what the navigator must imitate.

## References
- Edge et al., *GraphRAG* — arXiv:2404.16130
- Wang et al., *ArchRAG: Attributed Community-based Hierarchical RAG* — arXiv:2502.09891
- Chen, Tang, Deng et al., *LocAgent: Graph-Guided LLM Agents for Code Localization* (ACL 2025)
  — arXiv:2503.09089
- Paipuru, *The Navigation Paradox in Large-Context Agentic Coding* — arXiv:2602.20048
- Liu et al., *CodexGraph* (NAACL 2025); Ouyang et al., *RepoGraph* (ICLR 2025)
- Brown, *The C4 model*; Pirolli & Card, information foraging (scent); SEI *Views & Beyond*
- Ranked but not load-bearing: Youtu-GraphRAG 2508.19855, Deep GraphRAG 2601.11144,
  E²GraphRAG 2505.24226, core-based hierarchies 2603.05207, GAM 2604.12285, SHERLOC 2606.24820
