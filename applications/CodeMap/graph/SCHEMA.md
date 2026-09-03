# CodeMap graph schema v1 (draft for freeze at C4)

The contract between the three replaceable parts: the **graph** (facts), the **MCP server**
(deterministic intelligence), the **model** (behaviour). A model trained against
`schema_version: 1.x` must work with ANY codebase packed at `1.x`. This file is the spec;
prose elsewhere describes it but this file wins conflicts.

Two physical sides, one logical schema:
- **AUTHORING** (Neo4j, research-grade, big embeddings, full provenance) — where agents write.
- **RUNTIME** (`.codemappack`: LadybugDB single file + manifest) — what the app ships. Produced
  by export, which re-embeds text with the runtime embedder (never ship 4096-d authoring
  vectors) and strips authoring-only properties.

## 1. Node types

### L1 `:NavigationMaster` (exactly 1)
| prop | type | side | note |
|---|---|---|---|
| `schema_version` | string | both | semver; the compatibility key |
| `codebase_name`, `scanned_at`, `scan_commit` | string/datetime | both | provenance |
| `ai_summary` | string ≤120w | both | what the system IS |
| `ai_instruction` | string | both | **mandatory checklist** (protocol, not hint) |
| `subsystem_index` | string (rendered tree) | both | name + one-line + size + role, fan-out ≤9 (fold into labelled groups) |
| `global_caveats` | string[] | both | e.g. B-lens quiet-Resource caveat |
| `clue_version`, `generated_by`, `dossier_fingerprint` | string | authoring | ClueWriter provenance |

### L2 `:SubsystemNavigator` (one per curated subsystem, ~12–19)
`sub_id` (stable int), `name` (curated, F65 law), `role` (`SLICE|LAYER|OUTLIER`),
`ai_summary` (≤80w), `responsibilities` (string[] 3–5), `layer_profile` (json string),
`entry_points` (json: name + one-line, ≤9), `spines` (json: ordered file lists, highest-IDF
first), `contracts` (json: typed seams + external_ratio), `caveats` (string[]),
`curation_history` (string[]), embeddings per §4.
Module view (`layer_profile`/members) and connector view (`contracts`) stay separate fields.

### L3 `:EntityDetail` (one per file; EXISTS today, 1374)
Existing measured props stay untouched (`entity_type`, `file_path`, `v4_subsystem`,
`boundary_score`, lens embeddings, …). C3 adds: `layer` (=entity_type alias for the runtime),
`local_height` (float, per-subsystem trophic), `entry_point` (bool), `spine_membership`
(string[]), optional `ai_hint` (ONLY where the name misleads).

### `:Recipe` (the D5 layer — parameterized answers)
| prop | note |
|---|---|
| `recipe_id`, `intent` | intent = one-line "answers questions like…" |
| `params_schema` | JSON Schema of slots (entity name, subsystem, hop count…) |
| `cypher_neo4j`, `cypher_ladybug` | both dialects or `ladybug=null` marked TODO — the runtime NEEDS ladybug |
| `render` | `tree|table|list|path` — MCP renders results this shape (trees measured better for LLM reasoning) |
| `archetype` | see §5 — which question family it serves |
| `precision_note` | honest caveat string |

### `:MFQ` (evaluation + cache seed; questions first, gold later)
`q`, `aliases` (string[], incl. PL paraphrases — router training data), `role`
(`onboarder|dev|architect|security|pm|qa|em`), `stratum` (`G1|G2|G3`), `topic_bucket`,
`worth` (1–3), `gold_cypher` (nullable until Q1-gold), `gold_fingerprint` (hash of gold result
set), `gold_answer` (nullable), `depends_on_subsystems` (int[] — the cache-invalidation stamp).

### `:CurationDecision` (audit trail; authoring side, exported as history strings)
`subsystem`, `action` (`KEEP|MERGE|SPLIT|RETYPE|RENAME`), `target`, `rationale` (must cite
dossier fields), `evidence`, `decided_by`, `at`; revisions via `[:SUPERSEDED_BY {reason, at}]`.

### `:HyperedgeCandidate` (EXISTS, 92)
Unchanged: `key`, `metapath`, `hub_name`, `arity`, `idf_weight`, `precision_prior`, `member_ids`.

## 2. Edge types

**Navigation (new):** `(:NavigationMaster)-[:GUIDES]->(:SubsystemNavigator)-[:CONTAINS_MEMBER]->(:EntityDetail)`.
Distinct names on purpose — `CONTAINS` (folder tree) and `HAS_SUBSYSTEM` (legacy v3) already
exist and are NOT overloaded.

**Behavioural (EXIST, measured, never rewritten by phase 2):** `IMPORTS, INJECTS, EXTENDS,
CALLS, USES, PERFORMS, ACCESSES, IMPLEMENTS, MODIFIES, TRIGGERS, VALIDATES, AFFECTS,
TESTED_BY, CONSTRAINS, APPLIES_IN, CONFIGURED_BY, INITIATES` (+ plumbing: `CONTAINS`,
`HAS_DETAIL`, `IN_HYPEREDGE`, `ALGEBRA_VIOLATION`).

**Answering layer (new):** `(:MFQ)-[:ANSWERED_BY]->(:Recipe)`,
`(:MFQ)-[:DEPENDS_ON]->(:SubsystemNavigator)`, `(:Recipe)-[:ABOUT]->(:SubsystemNavigator)`
(optional scoping), `(:V3Master)-[:HAS_DECISION]->(:CurationDecision)` (authoring).

## 3. The mandated entry protocol (stored on L1, enforced by MCP)

```
1. Read NavigationMaster fully. 2. Try MFQ match (aliases index) — on confident hit, answer
from cache and STOP. 3. Else match a Recipe by intent; fill slots; execute. 4. Else descend:
subsystem_index → matched L2 report → entry_points/spines. 5. Open raw files only when clues
are insufficient — and SAY SO in the answer.
```
The MCP exposes this as the only entry tool-chain; skipping is not offered (CodeCompass: 58%
silent-skip when optional, 99.5% coverage when used).

## 4. Embeddings — two spaces, never mixed

| property | model | dim | side |
|---|---|---|---|
| `semantic_embedding` / `behavioral_` / `structural_` | Qwen3-Embedding-8B (Modal) | 4096 | authoring only |
| `embedding_rt` (on MFQ.q + aliases, Recipe.intent, L1/L2 ai_summary) | bundled runtime embedder (0.6B-class GGUF) | per manifest | runtime only, written at export |

Manifest records `embedder_id` + `embedder_dim`; query-side and doc-side MUST be the same
model. Cross-space cosine is a bug, not a feature (measured lesson, V3Lab).

## 5. Recipe archetypes (question family → recipe family)

| archetype | answers | typical stratum |
|---|---|---|
| `locate` | "where is X / which file handles X" | G1 |
| `flow` | "what happens when Y / trace this action" | G2 (spine walk) |
| `impact` | "what breaks / is affected if I change Z" | G2/G3 (reverse deps + cohorts) |
| `boundary` | "who talks to subsystem S, over what" | G2 (contracts) |
| `cohort` | "what changes together with F" | G3 (hyperedges) |
| `overview` | "what does subsystem S do / what exists" | G1 (L2/L1 reports) |
| `onboarding_path` | "minimal reading path to understand S" | G2 (entry→spine order) |
| `health` | "coupling hotspots, boundary nodes, layer purity" | G3 (dossier stats) |
| `rationale` | "why is it done this way" | answered ONLY from stored docs/decisions/comments — the honesty rule: if no source exists, say "not recorded" (LaToza: intent is the #1 hard question precisely because code alone can't answer it) |

## 6. `.codemappack` manifest (runtime artifact)

```json
{ "schema_version": "1.0", "dialect": "ladybug-0.17",
  "codebase": "checkItOut", "scan_commit": "...", "packed_at": "...",
  "embedder_id": "Qwen3-Embedding-0.6B-GGUF", "embedder_dim": 1024,
  "prompt_version_compat": ">=1.0 <2.0", "clue_version": "...",
  "counts": {"files": 1374, "subsystems": 19, "recipes": 0, "mfq": 0},
  "provenance": {"dossier_fingerprints": "...", "authoring_graph": "CheckItOutV3"} }
```
Swap validation: schema_version compatible ⇒ graph loads; prompt_version compatible ⇒ model
loads; then the 5-question smoke test. Mismatch = refusal with message, never silent degrade.

## 6b. Diagnostic layer — measurable state IN the graph (owner design, 02.09)

The pipeline's quality functionals are graph citizens, not report prose. Every run reads the
last state, measures fresh, compares, writes the new state — drift detection as a query, so
no gate ever depends on a constant frozen in a script (the incident that earned this: a
hardcoded conformance median went stale the first time a delta legitimately moved it).

`:DiagnosticState` — one node per observation:
| prop | note |
|---|---|
| `functional` | name from the catalogue below |
| `scope` | `global` \| `sub:<id>` |
| `value` | float, or JSON string for composites |
| `run_id`, `computed_by`, `computed_at` | provenance |

Last state = latest `computed_at` per (functional, scope). Append-only; never deleted.

**Functional catalogue v1**: `node_count`, `edge_count`, `hyperedge_census` (by source),
`lens_presence` (S=B=T counts), `l2_census` (by clue_version), `snapshot_count`,
`reachability` (files within 3 hops of L1), `subsystem_size`, `external_ratio`,
`trophic_service_median` (the ex-constant, now an observation), `coverage_census`
(never-indexed / stale / out-of-scope / collapsed — the L7 decomposition),
`gold_acceptance` (fingerprint of a reference gold re-run).

Agent contract (all three + Conductor): READ last state at run start; WRITE new state at run
end; report drift with a verdict (expected-from-my-changes | unexpected-investigate). The
runtime pack excludes this layer (authoring-side only).

## 7. Versioning law

`schema_version` bumps: PATCH = new optional props; MINOR = new node/edge/archetype types
(old models still work); MAJOR = anything a trained model would misread. The frozen master
prompt pins a `prompt_version`; distillation pairs record both versions; an eval run is only
comparable within (schema_version, prompt_version).
