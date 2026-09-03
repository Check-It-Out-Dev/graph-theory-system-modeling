# CodeMap — Stage 0: dividing the intelligence, variants, measurement

**Application codename: CodeMap** (owner's decision, 2 September 2026; repository name: `codemap`).
Parent document: `docs/00-ai-navigator-project.md` (approved 2 September). This plan puts that
document into operation at the start and connects it to the existing V3 research assets
(`graph-theory-system-modeling`, the Neo4j `CheckItOutV3` graph, the `V3Lab` journal F1–F111).

---

## 1. Division of labour — what lives where

| layer | contents | who produces it | how often |
|---|---|---|---|
| **GRAPH** (LadybugDB, the artifact) | facts about THIS code: partition v4 → curated subsystems, layers (entity_type), trophic height (a layering score borrowed from food-web analysis), entry points, spines (reading paths), seams (subsystem boundaries), hyperedges, navigation clues at levels L1/L2/L3, **recipes** (ready-made parameterised queries), 100+ most-frequent questions (MFQ) with gold, runtime embeddings | the authoring layer (Claude agents: Curator, ClueWriter, plus the C1/C3 scripts) | one full scan; a DELTA every month |
| **MCP** (TS server, stdio) | deterministic intelligence: `run_recipe`, `search` (full-text + HNSW), `explain`, Cypher execution, rendering results as TREES, enforcement of the entry protocol, MFQ cache lookup with dependency invalidation, telemetry | us (code, written once) | versioned with the schema |
| **MODEL** (small Qwen, GGUF) | behaviour: routing the question (cache → recipe → traversal), recipe classification and slot filling, answer decomposition, fluency with the MCP tools | LoRA on distillation pairs (D11) | rarely; the weights survive a graph swap |
| **MASTER PROMPT** (frozen, D2) | the interface contract: the mandatory protocol (checklist), the answer format, a slot for the project header | us; frozen BEFORE pair generation | versioned; any change means a new training run |
| **MFQ CACHE** | instance answers plus a dependency stamp (which subsystems); bi-temporal invalidation on delta | authoring layer plus runtime (which adds to it) | continuously |
| **HUMAN (owner)** | curation of the decisions the evidence leaves open (the Curator raises two-option questions with numbers); approval of names | session C2 | once per scan |

Decision D1 stands: **facts stored in weights go stale, so facts live in the graph; the weights learn only the method.**

## 2. Modelling variants at this stage, and the selection rule

Axis A — **answering mode**: (A1) 100% recipes (D5: classification plus slots; no free Cypher);
(A2) recipes plus emergency free Cypher under a GBNF grammar. **Start: A1.** A2 enters only when
the "no recipe" bucket exceeds 10% of failures (per document 00, section 11).

Axis B — **weights**: (B1) bare model + graph + MCP; (B2) LoRA_k (learning curve; stop rule: the
gain from doubling the data is smaller than the confidence-interval width).

Axis C — **graph richness**: (C1) lean (structure only, no clues) versus (C2) clue-rich (full
L1/L2 clues from phase 2). This is the ablation ladder of document 00 section 6, rungs (0)–(3), plus:

**H-COMP (the compensation hypothesis, to be pre-registered):** quality(model, graph) has
downward-sloping isoquants: a smaller model with a richer graph ≈ a larger model with a poorer
graph. A 2×2 grid {0.8/2B, 4B} × {lean, clue-rich} on the same Q; if (2B, rich) ≥ (4B, lean)
significantly (paired bootstrap), the product gets **artifact tiers**: weaker hardware → smaller
model plus a richer pack. This is exactly "the level of precomputed intelligence matched to the
chosen LLM".

**Variant selection rule**: the same Q set, paired bootstrap confidence intervals or McNemar; a
variant wins only on a significant difference; failures are classified into the three buckets
(COVERAGE / RETRIEVAL / USE). Bucket (i) is the authoring layer's backlog, not the training backlog.

## 3. Choosing the small LLM (as of 2 September 2026)

**Default workhorse: Qwen3-4B-Instruct-2507 Q4_K_M** (2.5 GB; D9 unchanged, the safe choice).
New since document 00: **Qwen3.5 Small now has stable GGUF files and Unsloth support**
(0.8B/2B/4B/9B; transformers v5 handles it automatically; QLoRA discouraged, but we do bf16 LoRA
anyway per D10). Conclusion: **Qwen3.5-2B and 4B enter the bake-off as full candidates**, not
merely as models to watch.

| candidate | RAM (Q4) | role in the bake-off |
|---|---|---|
| Qwen3-4B-Instruct-2507 | ~4 GB | default; reference benchmark |
| Qwen3.5-4B | ~4 GB | contender for the main tier |
| Qwen3.5-2B | ~2 GB | the H-COMP test: does a rich graph lift 2B to 4B level |
| Qwen3.5-0.8B | ~1 GB | minimal tier (weak hardware); router-only fallback |
| Gemma 4 E4B / Phi-4-mini | ~4 GB | only if the 4B models fail on format fidelity or latency |

**Selection metric** (lexicographic order): 1) execution accuracy on Q (recipe top-1 plus
result-set agreement); 2) tool-call format fidelity under GBNF (share of well-formed calls);
3) CPU latency (target under 5 s per answer, document 00 section 10); 4) RAM. Research finding
from document 00 section 5: after fine-tuning on about 600 pairs the leading models converge above
95%, so **the pair set matters more than the choice of base model**. Do not burn time on the model
axis before Q is built.

## 4. Measurement, including "cognitive ease" made operational

The Q set: 100 MFQ, growing to 300 (power calculation in document 00 section 6), stratified into
**G1 semantic / G2 structural (2–4 hops) / G3 hidden (zero lexical overlap)** — the CodeCompass
taxonomy. Navigation MUST win G3 and MUST NOT lose G1. Topic buckets on top.

**Cognitive ease — machine measures** (per trajectory, from MCP telemetry):
tokens-to-answer · tool-calls-to-answer · backtrack rate (returns to already-seen nodes) ·
wall-clock CPU time · router risk–coverage (does it know when it does not know). The ceiling is
always reported as a pair: teacher (Claude through the same MCP) versus small model, same Q.

**Cognitive ease — human measures** (the phase-2 gate, C5): fan-out ≤ 9 at every level · a
"scent" audit (the label predicts the content: name + one-liner + 3 examples) · 100% of files
reachable within 3 hops from L1 · time-to-first-correct-file (n = 1, the owner, informal —
reported honestly as an anecdote).

Harness: promptfoo with a custom *execution accuracy on LadybugDB* assert, plus pytest; assertions
in YAML (portability, given promptfoo's governance). The teacher's trajectory output is immediately
the distillation corpus (filter: execution verifier D11 + deduplication + decontamination against Q).

## 5. The swap contract (the UI's "replace the graph / replace the model")

**`.codemappack`** (the graph artifact): a LadybugDB file plus a manifest
`{schema_version, dialect, embedder_id, embedder_dim, prompt_version_compat, clue_version,
dossier_fingerprints, stats, provenance}`.
**Model**: GGUF plus a manifest `{prompt_version, chat_template_hash}`.
The swap UI validates: matching schema_version ⇒ the graph is accepted; matching prompt_version ⇒
the model is accepted; after every swap an automatic "five questions" smoke test runs (the chat
template / EOS trap, document 00 section 9). A mismatch is refused with a message, never silently
degraded.

**The embedder trap (important, from the V3 assets):** the research embeddings in Neo4j come from
Qwen3-Embedding-8B (4096 dimensions, on Modal). The runtime bundles a 0.6B-class embedder (1024
dimensions). Query and document MUST come from the same model ⇒ **the pack export re-embeds the
clues and the MFQ with the runtime embedder**; the properties are stored under separate names and
never mixed.

## 6. Sequence — what happens when (mapped onto the weeks of document 00)

| step | artifact | status |
|---|---|---|
| C1 dossiers (script `graph/scripts/c1_dossiers.py`) | 19 dossiers + INDEX | **DONE 2 Sept** — flags: sub 17 MEGA, sub 13 LAYER_PURITY, sub 18 MICRO, subs 16/2 ext ≈ 1.0 (merge candidates) |
| MFQ-20 → 100 (gold: query + result set + answer) | `eval/q/*.jsonl` | **next step** — unblocks everything (document 00 section 8, item 2) |
| Ladybug spike: migrate a slice, catalogue the dialect differences | `graph/ladybug/` | week 1 |
| C2 curation (prompt `graph/prompts/GrothendieckPart2_CuratorV5.md`) | :CurationDecision + report | after MFQ-20 |
| C3 internal organisation (script to be written per section C3) | trophic heights / entry points / spines at L3 | after C2 |
| C4 clues (prompt `graph/prompts/ClueWriterV5.md`) | L1/L2/L3 clues | after C3; **freeze the master prompt** |
| Ladder rungs (0) + (1): teacher + bare 4B on the lean graph | baseline | week 2, in parallel with the runtime |
| H-COMP grid + model choice (bake-off, section 3) | tier decision | week 3 |

Phase 2 (curation plus clues) builds the training environment of phase 3; the teacher trajectories
from C5 are at once the benchmark, the corpus and the imitation specification (F110).

## 7. Risks in joining the V3 assets to CodeMap

- **Cypher dialect**: recipes and entry queries are currently in the Neo4j dialect. Rewrite them
  for Ladybug during the spike; the recipe format carries both dialects (ClueWriter form rules).
- **Two embedders** (8B for authoring versus 0.6B at runtime) — section 5; never mix the spaces.
- **Train–serve consistency** (D2): the master prompt is frozen BEFORE pair generation; the prompt
  design is a C4 deliverable.
- **Kuzu is dead** — LadybugDB only (document 00 section 5, item 1); do not touch old Kuzu examples.
- **332 unpersisted pairs** (see `disagreement_note` in the dossiers): regenerate them by script
  during C2; never reconstruct them from memory.
