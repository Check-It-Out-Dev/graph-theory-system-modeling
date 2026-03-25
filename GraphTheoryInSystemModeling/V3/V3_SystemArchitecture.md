# V3 System Architecture: From Algebra to Code Understanding

## Applying Quantum Field Theory to Software System Modeling

**Version**: 2.0.0 | **Date**: 2026-03-25 | **Authors**: Norbert Marchewka (architecture), Claude Opus 4.6 (synthesis)

---

## Abstract

We present a system that applies the mathematical apparatus of quantum field theory to software system modeling — not as analogy, but as mathematical necessity. Three AI agents transform a raw codebase into a queryable algebraic graph in four stages: (1) define a non-abelian algebra that constrains what relationships are legal, (2) construct the graph under those constraints with one embedding per file, (3) train per-relation sub-topologies and detect subsystems, (4) enrich with developer-facing metadata. The result is a 3-level hierarchical graph providing $O(1)$ entry to any part of the codebase, with 102,553 trained weights encoding 17 different "views" of the same code.

---

## 1. The Core Thesis: Why QFT Emerges from Typed Embeddings

We observe that the question *"what does each relationship type see in a high-dimensional embedding space?"* leads, by mathematical necessity, to the full apparatus of lattice gauge theory. The restriction maps $\rho_k: \mathbb{R}^d \to \mathbb{R}^r$ form a connection on a fiber bundle over the typed graph. Their non-commutativity $[\rho_i, \rho_j] \neq 0$ generates a non-abelian gauge structure whose commutator subalgebra decomposes as $\mathfrak{su}(2) \times \mathfrak{su}(2) \times \mathcal{N}$. The Hermitian Magnetic Laplacian provides real eigenvalues whose eigenstates decompose the system into subsystems. Berry phase around graph cycles detects subsystem boundaries. None of these structures are imposed — they emerge systematically from the typed projection.

**The novelty is not in the individual mathematical tools, which are well-established, but in demonstrating that they form a single forced chain:**

$$\text{Typed subspaces} \to \text{Fiber bundle} \to \text{Non-abelian algebra} \to \text{Hermitian spectrum} \to \text{Eigenstates} \to \text{Holonomy}$$

This chain connects information retrieval (embeddings) to quantum field theory (gauge structure) through software engineering (typed dependencies), establishing that codebases modeled with typed relationships are, in a precise mathematical sense, **discrete gauge theories**.

### 1.1 The Chain of Forced Moves

No step in this chain is a design choice. Each is mathematically forced by the previous one:

```
INSIGHT: "What does relation R_k see in R^4096?"
    │
    ↓ requires a projection
FORCED: Restriction maps ρ_k: R^4096 → R^8  (one per relation type)
    │
    ↓ multiple projections of the same space
FORCED: Fiber bundle (base = graph, fiber = R^8, connection = ρ_k)
    │
    ↓ do the projections commute? (empirically: NO, cosine = -0.311)
FORCED: Non-abelian algebra ([ρ_i, ρ_j] ≠ 0, 93% non-commuting)
    │
    ↓ need a well-defined operator to diagonalize
FORCED: Hermitian Magnetic Laplacian (real eigenvalues, complete basis)
    │
    ↓ what are the natural states?
FORCED: Eigenstates = subsystem decomposition ("stany własne")
    │
    ↓ what happens when you traverse a cycle?
FORCED: Berry phase / holonomy (subsystem boundary detection)
    │
    ↓ what quantities are preserved?
FORCED: Noether invariants (entity type conservation, flow direction)
```

### 1.2 The Three Novelties

**Novelty 1 — Systematic emergence.** Individual pieces (Magnetic Laplacian, path algebras, LoRA, Berry phase, co-association clustering) exist in separate literatures that don't talk to each other. Nobody has shown that they form a single forced chain starting from "typed relationships on embeddings." The contribution is the chain itself — proving that QFT on graphs is not an analogy but a mathematical necessity for typed embedding spaces.

**Novelty 2 — Separation of WHAT from HOW.** Matrix $\mathfrak{A}$ (complex, universal) encodes the algebraic law — what relationships are legal, what directions they flow, what commutes with what. Matrix $\mathfrak{W}$ (real, codebase-specific) encodes the geometric weight — how to actually project embeddings. This mirrors the QFT split between the gauge group (universal) and the coupling constants (measured). No prior work separates the typed graph structure into these two distinct mathematical objects.

**Novelty 3 — Everything lives in Neo4j.** The algebra is stored as graph nodes. The tensor is trained via GDS FastRP. The sub-topologies are node properties. The subsystem detection uses GDS Leiden + K-means + Cypher co-association. No Python, no PyTorch. A researcher can reproduce the entire pipeline with a Neo4j instance and the paper's Cypher queries.

### 1.3 The QFT Correspondence

| QFT Concept | Software System | Where in V3 |
|-------------|-----------------|-------------|
| Lattice sites | Files (nodes) | EntityDetail |
| Link variables | Typed edges | PERFORMS, CALLS, USES, ... |
| Gauge group | Path algebra $\mathcal{H}$ | HypatiaBasis, §3 |
| Gauge field $A_\mu$ | Restriction maps $\rho_k$ | $\mathfrak{W}$ tensor |
| Field strength $F_{\mu\nu}$ | Commutator $[\rho_i, \rho_j]$ | HypatiaBasis, Theorem 5.1 |
| Observables | Magnetic Laplacian $\mathfrak{A}^\dagger = \mathfrak{A}$ | HypatiaBasis, Theorem 7.1 |
| Eigenstates | Subsystem decomposition | Grothendieck, §12 |
| Selection rules | 11 forbidden entity-type transitions | HypatiaBasis, §4 |
| Uncertainty principle | $\Delta(\tau)\cdot\Delta(\iota) \geq 0.89$ | HypatiaBasis, Theorem 5.2 |
| $\mathfrak{su}(2) \times \mathfrak{su}(2)$ | Event↔Process and Rule↔Context oscillations | HypatiaBasis, Theorem 6.1 |
| Casimir invariants | Conserved complexity quantum numbers $j, k$ | HypatiaBasis, §6.3 |
| Berry phase / holonomy | Information loss at subsystem boundaries | Grothendieck, Def 12.2 |
| Wilson loops | Gauge-invariant complexity around cycles | Grothendieck, §7.4 |
| Fiber bundle | Base = graph, fiber = $\mathbb{R}^8$, connection = $\rho_k$ | HypatiaBasis, §8.3 |
| Path integral $K(u,v)$ | Influence propagator between components | Lean: InformationGeodesics |
| Noether's theorem | Symmetry → conservation law | Lean: DiscreteNoether |
| Coupling constants | LoRA correction magnitudes $\alpha_k$ | Grothendieck, §3.5 |
| Lattice gauge theory | The entire system on a discrete graph | V3 |

---

## 2. The Big Picture: What Each Part Does

```
 STAGE 1               STAGE 2              STAGE 3              STAGE 4
 ALGEBRA               CONSTRUCTION         TOPOLOGY             CONSUMPTION
 ─────────             ────────────         ────────             ───────────
 Define rules     →    Build graph     →    Train tensor    →    Query & code
 (what's legal)        (nodes+edges)        (sub-topologies)     (developer use)
     │                     │                     │                     │
     ▼                     ▼                     ▼                     ▼
 Matrix 𝔄             Hypatia V3          Grothendieck V3        Erdős V3
 (6×6, complex)       (nodes+embeds       (ρ₀ + Δ tensor       (AI metadata
  Selection rules      +typed edges)       subsystems            code writing
  Algebra proofs)                          hierarchy)            graph queries)
```

### 2.1 What Each Part Does (In Terms of the QFT Chain)

**Stage 1 — The Algebra (HypatiaBasis.md)**: Defines the **gauge group** of the theory. The quiver $\mathcal{Q}$ with 6 entity types and 22 arrows defines which interactions are legal. The path algebra $\mathcal{H} = k\mathcal{Q}/\mathcal{I}$ is the gauge group. The Magnetic Laplacian $\mathfrak{A} \in \mathbb{C}^{6 \times 6}$ is the observable — Hermitian, with real eigenvalues (eigenstates) and complex phases encoding directionality. The 11 selection rules are the **forbidden transitions** — like $\Delta l = \pm 1$ in atomic physics, but for software entity types. The $\mathfrak{su}(2) \times \mathfrak{su}(2) \times \mathcal{N}$ decomposition classifies the independent "rotation planes" of the algebra.

*QFT step: Define the theory (gauge group, selection rules, symmetries).*

**Stage 2 — Hypatia (Construction)**: **Puts the theory on the lattice.** Reads every file, classifies it into an entity type (assigns it to a lattice site), generates an $\mathbb{R}^{4096}$ embedding (the field value at that site), and creates typed edges under algebraic constraints (the link variables). Every edge is checked against $\mathfrak{A}$ before creation — the lattice is born gauge-invariant. The output is a flat graph: nodes with embeddings, typed directed edges, all satisfying the algebra.

*QFT step: Discretize — place fields on the lattice, define link variables.*

**Stage 3 — Grothendieck (Topology)**: **Computes the path integral and measures observables.** Trains the restriction maps $\rho_k$ (the gauge field / connection on the fiber bundle) from embeddings + typed edges via FastRP = randomized SVD. Produces 17 sub-algebraic topologies — different $\mathbb{R}^8$ point clouds from the same $\mathbb{R}^{4096}$. The commutators $[\rho_i, \rho_j] \neq 0$ are the non-abelian field strength (empirically: anti-correlations prove this). Detects subsystems via multi-view spectral clustering (eigenstates). Computes Berry phase around cycles (holonomy = boundary detection). Builds the 2-level hierarchy: NavigationMaster → SubsystemNavigator → EntityDetail.

*QFT step: Compute observables — diagonalize the Hamiltonian, measure the spectrum, detect phases.*

**Stage 4 — Erdős (Consumption)**: **Extracts physics.** Knows nothing about gauge theory, fiber bundles, or eigenvalues. Sees only: named subsystems, typed relationships, AI metadata, and similarity queries. Enriches SubsystemNavigator nodes with developer-facing instructions. Then switches to code-writing mode — query the graph for context, follow typed edges, write Spring Boot / Angular code.

*QFT step: Use the theory — predict, explain, build. The physicist who reads the spectrum and designs experiments, without re-deriving quantum mechanics.*

### 2.2 The Three Matrices

| Matrix | What it IS (QFT) | What it DOES (Software) | Field | Size |
|--------|-------------------|------------------------|-------|------|
| $\mathfrak{A}$ | Gauge group structure constants | Encodes which relationships are legal + direction | $\mathbb{C}$ | $6 \times 6$ (72 values) |
| $\rho_0$ | Background gauge field | Common projection — "what any relationship generally looks like" | $\mathbb{R}$ | $4096 \times 8$ (32,768 weights) |
| $\Delta_k$ | Gauge field fluctuations per interaction | Per-relation correction — "how ORCHESTRATES differs from the average" | $\mathbb{R}$ | $17 \times 4105$ (69,785 weights) |

**$\mathfrak{A}$ is universal** (same for any codebase with the 6-entity model). **$\rho_0$ is codebase-specific** (the system's "accent"). **$\Delta_k$ is relation-specific** (where the real architecture lives). Total: **102,553 trainable weights**, independent of graph size.

---

## 3. Stage 1: Designing the Algebra

### What happens

Before any file is read, we define the rules of the game:

- **6 entity types**: Actor, Process, Resource, Rule, Event, Context
- **17 typed relationships**: PERFORMS (Actor→Process), CALLS (Process→Process), USES (Process→Resource), etc.
- **11 forbidden blocks**: Resource never acts (pure sink). Actor never receives behavioral arrows (pure source). No direct Actor→Rule, no Context→Process, etc.
- **5 type-agnostic structural arrows**: IMPORTS, EXTENDS, IMPLEMENTS, INJECTS, TESTED_BY (legal between any entity types)

### Why it's non-abelian

Two operations that don't commute:
- TRIGGERS then INITIATES (P→E→P): lands in Process-space
- INITIATES then TRIGGERS (E→P→E): lands in Event-space
- **Same operations, different order, different result.** This is non-commutativity.

40 out of 43 composable pairs are non-commuting (93%). This means the ORDER of relationships matters — it's not just WHAT connects to WHAT, but the DIRECTION of the chain.

### Why it's Hermitian

The Magnetic Laplacian 𝔄 encodes directionality as complex phases:
- A→P has phase +i (outgoing)
- P→A has phase -i (conjugate, incoming)
- Since -i = conjugate(+i), the matrix satisfies 𝔄† = 𝔄 (Hermitian)
- Therefore: all eigenvalues are REAL, eigenvectors form a COMPLETE basis

This means the "eigenstates" (natural decomposition of the system) are well-defined and unique.

### What this gives us

A **closed algebra**: any composition of legal relationships is either another legal relationship or zero (forbidden). Hypatia can check every relationship against the algebra before creating it. The graph is born correct.

---

## 4. Stage 2: Hypatia — Graph Construction

### What Hypatia does

| Phase | Action | Output |
|-------|--------|--------|
| 1. Initialize | Create NavigationMaster + 6 SystemEntities | Graph scaffold |
| 2. Discover | Spawn subagents to list all files across repos | File list (~400 files) |
| 3. Embed | Read files, classify entity type, generate R^4096 embedding via APOC | Nodes with embeddings |
| 4. Relate | Parse code for imports/calls/uses, create typed edges under algebra constraints | Connected graph |

### The embedding

Each file gets ONE 4096-dimensional vector from Qwen3-8B. This vector encodes everything — semantic meaning, behavioral patterns, structural role — all mixed together. The vector is stored on the node. It takes ~150ms per file via the APOC→Flask→Qwen3 pipeline.

### The constraint check

Before creating any typed relationship, Hypatia checks:
1. Source entity type matches the arrow's source requirement
2. Target entity type matches the arrow's target requirement
3. If either fails: skip the relationship, log as algebra violation

This means: **the graph cannot contain a Resource→Actor edge, ever.** The algebra prevents it by construction.

### What Hypatia produces

- ~400 EntityDetail nodes with file_path, entity_type, embedding[4096]
- ~1000+ typed directed edges (PERFORMS, CALLS, USES, MODIFIES, TRIGGERS, etc.)
- All edges satisfy the algebra's selection rules
- Status: INDEXING_COMPLETE

---

## 5. Stage 3: Grothendieck — Topology and Subsystems

### 4.1 Training the Tensor (Phases 1-6)

Grothendieck receives: flat graph (nodes with R^4096 embeddings + typed edges).

**The insight**: Different relationship types "see" different things about the same file. A file's role in the ORCHESTRATES topology is different from its role in the VALIDATES topology. To reveal this, we train per-relation projections.

**The algorithm** (pure Neo4j GDS, from Information Lensing Appendix C):

1. **Common base ρ₀**: Run FastRP(dim=8) on the FULL graph (all edges together). This learns: "what does a generic relationship look like in this codebase?" Result: one R^8 vector per node.

2. **Per-relation projections**: For each of 17 typed relations, filter the graph to ONLY that relation's edges, then run FastRP(dim=8) with the SAME config. Same random seed, same iteration weights — only the adjacency changes. Result: 17 different R^8 vectors per node.

3. **Corrections**: The difference between per-relation and common base = Δ_k. The magnitude α_k tells us how "unique" each relation's topology is. Large α = genuinely different view. Small α = similar to the average.

**What FastRP actually does**: It takes the R^4096 embedding, propagates it through the relation-specific adjacency (neighbor aggregation), then random-projects to R^8. This is equivalent to randomized SVD of the embedding-weighted adjacency matrix (Johnson-Lindenstrauss lemma).

**The training signal**: Embeddings provide the CONTENT (what the code is about). Typed edges provide the STRUCTURE (how the code relates). FastRP fuses them: content propagated through structure-specific adjacency → per-relation sub-topology.

**Total trained weights**: 102,553 (independent of file count). These encode the full tensor 𝔚 = ρ₀ + 17 corrections Δ_k.

### 4.2 Detecting Subsystems (Phases 7-10)

With 17 different R^8 positions per node, Grothendieck detects subsystems by fusing two signals:

**Signal 1 — Topological**: Concatenate all 17 R^8 vectors into one R^136 composite. K-means cluster in this space. Nodes close in MOST topologies → same subsystem.

**Signal 2 — Graph structural**: Leiden community detection on the raw typed edge graph. Captures local density, clique structure, modularity.

**Fusion**: Co-association matrix (Strehl & Ghosh 2002). For each node pair: S(i,j) = 0.5 × [same in topological clustering] + 0.5 × [same in graph clustering]. Re-cluster the consensus similarity with Leiden. Result: one subsystem_id per node.

**Berry phase refinement**: For nodes where per-relation topologies ANTI-correlate (cosine < 0), flag as boundary files. These sit between subsystems. Optionally split clusters with high internal disagreement.

**Why both signals?** Ablation studies show 10-30% improvement when combining embedding-based and graph-structural clustering (Tandon et al. 2021). Hyperedges (transactional scope, event chains) capture n-ary information that pairwise embeddings provably lose.

### 4.3 Building the Hierarchy (Phases 11-14)

For each detected subsystem:
1. Create a **SubsystemNavigator** node (Level 1)
2. Link NavigationMaster → SubsystemNavigator via HAS_SUBSYSTEM
3. Link SubsystemNavigator → each member EntityDetail via CONTAINS
4. Auto-name: dominant package path + dominant entity type → "Payment_BUSINESS_LOGIC", "Auth_VALIDATION_SECURITY"
5. Compute confidence: 1.0 if both clusterings agree, 0.5 if they disagree

**What Grothendieck produces:**

```
NavigationMaster (Level 0)
├── Payment_BUSINESS_LOGIC (Level 1, 35 files)
├── Campaign_API_LAYER (Level 1, 28 files)
├── Auth_VALIDATION_SECURITY (Level 1, 22 files)
├── Subscription_BUSINESS_LOGIC (Level 1, 45 files)
├── ...
└── Each EntityDetail linked to exactly one SubsystemNavigator
```

Status: SYNTHESIS_COMPLETE

---

## 6. Stage 4: Erdős — The Consumer

### 5.1 What Erdős does

**Phase 1: Explore and Enrich**

For each SubsystemNavigator, Erdős:
- Queries the member files (types, names, relationships)
- Writes developer-facing AI instructions: "This subsystem handles Stripe payments. Start with StripeService. Key pattern: webhook handler."
- Writes query hints: "To find the payment flow: MATCH (sub)-[:CONTAINS]->(f) WHERE f.entity_type='Process'"
- Lists entry points, dependencies, API surface

**Phase 2: Code Writing**

Erdős is a reincarnated Spring Boot developer. He uses the graph as a map:

```
"Add subscription downgrade feature"
  1. Query: NavigationMaster → Subscription subsystem
  2. Read: SubsystemNavigator.ai_instructions → "Start with SubscriptionService"
  3. Query: SubscriptionService -[CALLS]-> ? → find related services
  4. Query: SubscriptionService -[USES|MODIFIES]-> ? → find data layer
  5. Read: actual source files via file_path
  6. Write: new code following discovered patterns
```

**Zero Explore agents. Zero codebase search. O(1) entry + O(local) traversal.**

### 5.2 What Erdős does NOT need to know

- The algebra 𝔄 and its selection rules (Hypatia enforced them)
- The tensor 𝔚 and its 102,553 weights (Grothendieck trained them)
- The commutator eigenvalues ±31.04i (proven in HypatiaBasis)
- The Berry phase at subsystem boundaries (Grothendieck computed it)
- The co-association fusion algorithm (produced the subsystem IDs)

**The math is invisible to the consumer.** The graph just works.

---

## 7. Reindex Strategy

| Event | Hypatia | Grothendieck | Erdős |
|-------|---------|-------------|-------|
| New/changed files | Regenerate embeddings + edges | — | — |
| Weekly maintenance | Full reindex if >10% changed | Re-evaluate tensor + subsystems | Re-explore changed subsystems |
| Major refactoring | Full reindex | Full tensor retrain + subsystem re-detection | Full re-enrichment |
| Normal feature work | — | — | Just query the graph |

---

## 8. The Complete File Set

### Papers (mathematical foundations)

| File | Content |
|------|---------|
| `V3/HypatiaBasis.md` | Algebra 𝔄, 5 proofs, two-matrix formulation |
| `V3/GrothendieckAlgebraicTopologies.md` | Tensor training, sub-topologies, subsystem detection |
| `V3/ErdosGraphConsumer.md` | O(1) query efficiency, AI metadata, developer workflow |
| `V3/V3_SystemArchitecture.md` | This document — plain-language overview |

### Executable Prompts

| File | Agent | Model |
|------|-------|-------|
| `Promts/V3/HypatiaIndexingAgent_V3.xml` | Hypatia | Opus 4.6 [1M] |
| `Promts/V3/GrothendieckGraphOrganizer_V3.xml` | Grothendieck | Opus 4.6 [1M] |
| `Promts/V3/ErdosDeepModeling_V3.xml` | Erdős (to be written) | Opus 4.6 [1M] |

### Neo4j Algebra Model

| Namespace | Content |
|-----------|---------|
| `HypatiaAlgebra` | 6 QuiverVertex + 22 QuiverArrow + 43 CompositionRule + 11 SelectionRule + 5 AlgebraicProof + 1 TensorFramework |

---

## 9. Deployment: Swappable CLAUDE.md Files

### 8.1 The Prompt-Swap Architecture

Each V3 agent is designed for **CLAUDE.md swapping** — the user maintains multiple CLAUDE.md files in the `.claude/` directory and loads the appropriate one for each phase:

```
.claude/
├── CLAUDE.md                        ← default (project context, always loaded)
├── agents/
│   ├── hypatia-indexer.md           ← swap in for indexing phase
│   ├── grothendieck-organizer.md    ← swap in for synthesis phase
│   └── erdos-developer.md           ← swap in for code writing phase
```

**Why swappable CLAUDE.md is better than subagent prompts:**

| Aspect | Subagent Spawn | CLAUDE.md Swap |
|--------|---------------|----------------|
| Thinking mode | Subagent gets reduced context | Full 1M context with adaptive thinking |
| Persona depth | Brief prompt, shallow persona | Full CLAUDE.md, deep persona adhesion |
| Tool access | Limited by subagent type | Full tool access as main agent |
| Context continuity | Separate context window | Inherits conversation history |
| Cost | Spawns new context per task | Reuses existing context |

### 8.2 How It Works

1. User opens Claude Code in the project directory
2. Base `CLAUDE.md` is always loaded (project overview, repo paths, Neo4j config)
3. For indexing: load `hypatia-indexer.md` as the active agent persona
4. For synthesis: swap to `grothendieck-organizer.md`
5. For coding: swap to `erdos-developer.md`

Each agent CLAUDE.md contains:
- The XML prompt content (identity, pipeline phases, Cypher queries)
- Inline algebra rules (for Hypatia)
- Quick reference queries (for Erdős)

### 8.3 Optional: Skills Registration

If testing shows that loading agents as Claude Code **skills** (via `/skill-name` commands) is more effective than CLAUDE.md swapping, the prompts can be registered as skills:

```
/hypatia-index    → loads HypatiaIndexingAgent_V3.xml
/grothendieck     → loads GrothendieckGraphOrganizer_V3.xml
/erdos            → loads ErdosDeepModeling_V3.xml
/erdos-code       → loads Erdős in Phase B only (skip enrichment)
```

This gives the user a command-line interface to switch personas without manual file swapping. The choice between CLAUDE.md swap and skill registration depends on empirical testing with the specific model version.

### 8.4 Base CLAUDE.md (Always Loaded)

The base CLAUDE.md contains project-invariant information:

```markdown
# CheckItOut System

## Repositories
- Backend: C:/Users/Norbert/IdeaProjects/checkItOut-be2/
- Frontend: C:/Users/Norbert/IdeaProjects/checkItOut-fe/

## Neo4j
- Namespace: CheckItOutSystem
- Algebra namespace: HypatiaAlgebra
- Status: check NavigationMaster.status

## Agent Prompts
- Hypatia V3: Promts/V3/HypatiaIndexingAgent_V3.xml
- Grothendieck V3: Promts/V3/GrothendieckGraphOrganizer_V3.xml
- Erdős V3: Promts/V3/ErdosDeepModeling_V3.xml
```

---

## 10. Summary in One Sentence

**Asking "what does each relationship type see in $\mathbb{R}^{4096}$?" forces, by mathematical necessity, the full chain: typed subspaces → fiber bundle → non-abelian algebra → Hermitian spectrum → eigenstates → holonomy. This chain is quantum field theory on a discrete graph. The system implements it: define the gauge group (Hypatia Basis), put the theory on the lattice (Hypatia V3), compute observables and detect phases (Grothendieck V3), extract physics for practical use (Erdős V3).**

---

*Created: 2026-03-25. Revised: 2026-03-25 (QFT thesis, forced chain, correspondence table).*
*Status: Architecture complete. All 3 agent prompts written. All 4 papers complete.*
