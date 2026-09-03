# AI Navigator — project design document

**Date:** 2 September 2026 · **Status:** approved to start · **Horizon:** about 30 days to publication
**Deliverables:** an article/tutorial with the mathematical derivation, a repository (application, training pipeline, evaluation harness), a downloadable application, and a demo video

---

## 1. Thesis

**We move intelligence out of the model weights and into the data structure.** A typed, algebraic code graph, carrying recipes, ready-made queries and navigation metadata (the "AI Navigator" pattern), is the brain. A small local model (0.6–4B class) is only the mouth: it picks a recipe, fills its slots and explains the result step by step. Nothing that can be computed once a month is computed at question time.

Product thesis: developers ask the same system the same questions hundreds of times. Scan the repository, build the graph, ship an offline application. Zero cost per token, no code leaves the machine, and new people get onboarded to the repository quickly.

## 2. Architecture — three layers

```
┌─────────────────────────────────────────────────────────────┐
│ AUTHORING LAYER (runs rarely, costs once)                    │
│ Claude SDK agents scan the repo → build/update the graph    │
│ One full scan + a monthly DELTA (fingerprint per file)      │
│ Also distributed as a plugin (the agent prompts)            │
└───────────────────────┬─────────────────────────────────────┘
                        ▼ artifact
┌─────────────────────────────────────────────────────────────┐
│ ARTIFACT: LadybugDB graph (embedded, Cypher, MIT)           │
│ typed nodes/edges + an algebra of permitted combinations    │
│ recipes, ready-made queries, 100+ FAQ, navigator hierarchy  │
│ + vector index (HNSW) and full-text search in the same DB   │
└───────────────────────┬─────────────────────────────────────┘
                        ▼ serving
┌─────────────────────────────────────────────────────────────┐
│ RUNTIME (on the user's machine, offline)                    │
│ Tauri 2 app (React/TS) · llama-server as a sidecar          │
│ small Qwen (GGUF, CPU) · local MCP server (stdio, TS)       │
│ Claude Desktop/Code consume the same graph through MCP      │
└─────────────────────────────────────────────────────────────┘
```

## 3. Decision log (with rationale)

**D1. Facts go in the graph; behaviour goes in the weights.** The LoRA learns *how to be a navigator* (Cypher fluency, recipe selection, answer decomposition, MCP tool use); the graph knows *what is true this month*. Reason: facts stored in weights go stale at every rescan. The graph gets replaced; the weights stay.

**D2. The small model's master prompt is FIXED and versioned like code.** (a) Train–serve consistency: distillation pairs generated under prompt P must be served under P, otherwise the distribution drifts. (b) Comparable evaluations: a fixed prompt is a controlled variable, so differences in results are attributable to the weights or to the graph. (c) The prompt is the interface contract. The one exception is a parameterised project header, injected as data into a slot rather than edited into the prompt.

**D3. Prompt evaluation is not a tool for improving prompts; it is diagnostics for the model and the graph.** The test set (100+ most-frequent questions with gold answers) measures system performance. Failures are classified into the three buckets of section 6 and become the backlog for the next authoring run.

**D4. The model does not compute topology.** Subtopologies, clusters (Leiden), rankings — all precomputed in the authoring layer and stored in the graph as metadata. The typed algebra (permitted and forbidden edge combinations) gives discrete resolution where embeddings give only fuzzy resolution.

**D5. Recipe selection instead of query generation.** The graph contains ready-made, parameterised queries plus explanation recipes. The small model classifies and fills slots. Turning a generation problem into a classification problem is a jump in quality and gives hard measurability (top-1 / top-3 accuracy).

**D6. Embedded database: LadybugDB, not Neo4j and not Kuzu.** Neo4j runs on the JVM and cannot be bundled. **Kuzu is dead** (archived 10 October 2025; the team was hired by Apple), so nothing is built on it. LadybugDB is the main community fork: MIT, active (v0.17.1, June 2026), Cypher, native vector index (HNSW) plus full-text search, a wheel under 9 MB, bindings for Python/Node/Rust/Swift/WASM, and an official `NEO4J_MIGRATE` migration path. One database file, read natively by the application and served over MCP to large models.

**D7. Shell: Tauri 2 (v2.11).** Binaries of about 3–9 MB (versus about 244 MB for Electron), roughly 2.4× less RAM, an official sidecar pattern, auto-update with mandatory signing. The React/TS frontend ports without changes. **Mobile caveat:** sidecar/spawn does NOT exist on iOS/Android; there llama.cpp is linked as a library (XCFramework) instead of running as a process. This is the only serious desktop-versus-mobile difference.

**D8. Model runtime: llama.cpp / llama-server as a sidecar.** OpenAI-compatible API on localhost; native function calling (`--jinja`); output format enforced with GBNF grammars or JSON Schema, which is critical with small models. Ollama is for development only, never for shipping.

**D9. Default model: Qwen3-4B-Instruct-2507 Q4_K_M** (2.5 GB, about 4 GB RAM) with a 1.7B/0.6B fallback for weaker hardware. Watch **Qwen3.5 Small (0.8B/2B/4B, March 2026, Apache 2.0)** as the successor; Gemma 4 E4B if tool-call format fidelity becomes the priority; Phi-4-mini if latency does. Research finding: fine-tuning on about 600 examples of one's own tools brings the leading small models to over 95%, so the base model's advantage matters less than our pair set.

**D10. Training: Unsloth on a single GPU.** LoRA (bf16) for models up to 4B needs 10 GB VRAM; a typical run costs under $5 (a 4090 at about $0.55/h) or is free on a Colab T4 for Qwen3 up to 4B. Merged-to-GGUF export is built into Unsloth. Trap number one: the chat template and EOS token after export. Pin versions and test the GGUF before publishing (lm-eval-harness as a sanity check).

**D11. Distillation with an execution verifier.** Teacher-to-student pairs are filtered by EXECUTING the generated Cypher against the database and comparing the result with the gold set. This is our advantage: a cheap, deterministic verifier. Deduplication (MinHash), train/test decontamination (n-gram plus embedding overlap), and a share of real pairs from our own prompt history.

**D12. Distribute the authoring agents as a plugin (prompts), not as SaaS.** The Claude SDK is shown in the demo video. Public chat hosting: NO (sessions, cost, abuse). Compute stays on the user's machine.

## 4. Stack — decision table

| Component | Choice | Version (Sept 2026) | Licence |
|---|---|---|---|
| Desktop/mobile shell | Tauri 2 | 2.11.x | MIT/Apache |
| Frontend | React + TS | — | — |
| LLM runtime | llama.cpp (llama-server, sidecar) | current | MIT |
| Model | Qwen3-4B-Instruct-2507 GGUF Q4_K_M | 2507 | Apache 2.0 |
| Model — watch | Qwen3.5 Small 2B/4B | March 2026 | Apache 2.0 |
| Graph database | **LadybugDB** (Kuzu fork) | 0.17.1 | MIT |
| MCP server | TS: `@modelcontextprotocol/sdk` or FastMCP-TS; stdio transport | spec 2025-11-25 (RC 2026-09-02) | MIT |
| Training | Unsloth (LoRA) → merged → GGUF | current | Apache 2.0 |
| Embedder | Qwen3-Embedding-0.6B (GGUF) or EmbeddingGemma-300M | — | Apache 2.0 / Gemma |
| Reranker | bge-reranker-v2-m3 or Qwen3-Reranker-0.6B | — | Apache 2.0 |
| Evaluation harness | promptfoo (custom assert: execution accuracy on LadybugDB) + pytest | 0.121.x | MIT |
| Sanity benchmark | lm-eval-harness (llama.cpp backend) | 0.4.12 | MIT |
| Teacher prompt optimisation | DSPy (GEPA/MIPROv2) — optional | 3.2.x | MIT |
| Model distribution | HuggingFace Hub (download on first start, SHA-256) + R2/S3 mirror | — | — |
| Signing/updates | Developer ID + notarization (macOS), Azure Trusted Signing (Windows), tauri-plugin-updater | — | — |

**Avoid:** Kuzu (dead), jina-reranker v2 / jina-embeddings-v4 (CC-BY-NC), CozoDB (dormant), QLoRA for Qwen3.5 (discouraged by Unsloth), public chat hosting.

## 5. Key research findings (things that changed the plan)

1. **Kuzu archived on 10 October 2025** (team hired by Apple). The successor is LadybugDB, "MIT forever", actively developed. Decision D6 was updated relative to the original recommendation.
2. **The sidecar does not work on mobile.** Mobile plans need FFI/XCFramework instead of a process. Desktop first; mobile is chapter two.
3. **Qwen3.5 Small series** (March 2026): 0.8B/2B/4B/9B, Apache 2.0, 262K context. A candidate for an upgrade during the project; beware the verbose thinking mode and the immature GGUF/Unsloth support (bug #4534).
4. **promptfoo acquired by OpenAI (March 2026).** Still MIT and still the standard, but a governance risk. Keep the assertions in portable YAML so they can be migrated.
5. **MCP RC 2026-09-02.** The spec is stabilising right now (stateless core). Write the server thin, with no dependence on session ids.
6. Realistic RAM: 0.6B ≈ 1 GB, 1.7B ≈ 2 GB, 4B ≈ 3.5–4 GB (Q4, 4–8K context). An 8 GB machine can run the 4B variant.
7. Existing text-to-Cypher resources: the Neo4j Text2Cypher dataset, SynthCypher, PIPE-Cypher. Review them before building our own pair generator.

## 6. Evaluation as mathematics (an article chapter)

**The set:** Q = {(question qᵢ, gold gᵢ)}, n ≥ 200–400 (power calculation: distinguishing 85% from 90% at α = 0.05 needs about 300 pairs). Sources: 100+ most-frequent questions plus our own prompt history. Gold = the set of nodes/values returned by executing the gold query, plus a reference answer.

**Metrics per layer:** router — cost-weighted accuracy; recipe selection — top-1/top-3; query — execution accuracy (agreement of result sets: exact / Jaccard); answer — faithfulness to context (judge/NLI) plus correctness against gold; end to end — task success rate.

**Comparisons:** paired statistics — the same Q through variants A/B; paired bootstrap confidence intervals (10k resamples) or McNemar for binary outcomes; a variant wins only on a significant difference. Wilson intervals for single proportions.

**The ablation ladder (the headline chart):** (0) teacher + graph + MCP = ceiling; (1) small bare model + graph + MCP; (2) small model + LoRA_k for increasing k (learning curve); (3) small model + LoRA with a crippled graph (control). **Training stop rule:** stop when the gain from doubling the data is smaller than the width of the confidence interval (learning-curve saturation).

**Failure diagnostics — three buckets:** (i) COVERAGE — the graph does not contain the answer (measure: judge on gold versus graph content); (ii) RETRIEVAL — the graph contains it but it was not found (recall@k of the gold recipe; this is where the embedder and reranker work); (iii) USE — it was found but the model did not use it (faithfulness). Bucket (i) is the authoring layer's backlog. **Bonus:** the router's risk–coverage curve (selective prediction): how well the model knows when it does not know, and escalates.

**The headline number:** about 90% task success on Q, ALWAYS reported next to the teacher ceiling on the same Q.

## 7. The 30-day plan

**Week 1 — data and foundations.** The Q set (100 most-frequent questions with gold; JSONL format: question, gold query, gold result set, reference answer, topic bucket). Migration of a graph sample from Neo4j to LadybugDB (`NEO4J_MIGRATE` or CSV → `COPY FROM`); Cypher coverage test (list of differences from the docs). Delta indexer: fingerprint per file → change set → re-model the delta → reconcile (update nodes, remove orphaned edges). Draft of the mathematics chapter.

**Week 2 — runtime.** Tauri 2 skeleton plus signed llama-server sidecar; chat with cards; a Cypher panel (second screen); MCP server (stdio, TS) over LadybugDB with the tools `run_recipe`, `search`, `explain`; the same server plugged into Claude Desktop as the proof of "one graph, two brains". Ladder baseline: rungs (0) and (1).

**Week 3 — training and evaluation.** Generation of distillation pairs (teacher through MCP; filtered by the execution verifier; deduplicated; decontaminated against Q). LoRA in Unsloth (Colab/4090), GGUF export, lm-eval-harness sanity check. promptfoo harness with the custom execution-accuracy assert; the full ladder plus the learning curve and the stop rule; failures classified into buckets.

**Week 4 — packaging and publication.** Frontend components that explain the methods (ladder visualisation, risk–coverage, buckets); installers (signing, notarization, updater); model download on first start (SHA-256, progress); demo video (including a 90-second segment showing the Messages API, SDK agents and MCP working together); editing of the article; publication of the repository and the agent plugin.

## 8. Tomorrow — Day 1 checklist

1. Create the `ai-navigator` repository (monorepo: `app/`, `graph/`, `training/`, `eval/`, `article/`).
2. Write the first 20 most-frequent questions with gold (the week's target: 100). This unblocks EVERYTHING else.
3. `pip install ladybug` → migrate a slice of the graph → rewrite three favourite queries into the Ladybug dialect; note the differences (material for the article).
4. Download Qwen3-4B-Instruct-2507-GGUF (Q4_K_M) plus `llama-server --jinja`; ask it five questions from Q by hand, without the graph; record the result (this is the "bare model" rung of the ladder).
5. In the evening: read the Ladybug-versus-Neo4j Cypher differences and the Tauri sidecar docs (links in section 12).

## 9. Risks and traps

- **Train–serve consistency** (D2): one change to the master prompt after the pairs are generated throws the training away. Freeze the prompt BEFORE week 3.
- **Chat template / EOS after GGUF export** — the number-one Unsloth trap; run the "five questions" test after every export.
- **Cypher dialect drift** (Ladybug ≠ Neo4j): write recipes in the Ladybug dialect from the start; the teacher must get a cheat sheet of the differences in the authoring prompt.
- **Scope:** advertising features are on hold; mobile is chapter two; public hosting never happens in this project.
- **promptfoo governance** (OpenAI) — keep the assertions portable.
- **Qwen3.5 is young** — do not tie the project to a model with export bugs; 4B-2507 is the safe workhorse.

## 10. Project success metrics

- Task success ≥ 90% on Q with a teacher ceiling ≥ 95% (reported as a pair).
- An ablation-ladder chart with at least four rungs and confidence intervals.
- Repository delta scan under 5 minutes when fewer than 5% of files changed.
- Application: installer under 50 MB (without the model), cold start under 10 s, answer under 5 s on CPU (4B Q4).
- Article published, plus repository, plugin and video.

## 11. Open questions (to settle along the way)

- 4B versus 2B (Qwen3.5) as the default model — the ladder decides.
- How many recipes versus how much free Cypher — start at 100% recipes and relax only if the "no recipe" bucket grows.
- Embedder as GGUF through llama-server, or sentence-transformers in a Python sidecar — decide in week 2 (preference: llama-server, one runtime).
- Whether the Cypher panel gets a read-only mode for ordinary users.

## 12. Key links

Tauri sidecar: v2.tauri.app/develop/sidecar · llama-server function calling: github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md · LadybugDB: ladybugdb.com, docs.ladybugdb.com/cypher/difference · migration: docs.ladybugdb.com/import/graph-databases · Unsloth Qwen: unsloth.ai/docs/models · Qwen3-4B-2507 GGUF: huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF · promptfoo CI: promptfoo.dev/docs/integrations/ci-cd · lm-eval-harness: github.com/EleutherAI/lm-evaluation-harness · MCP TS SDK: github.com/modelcontextprotocol/typescript-sdk · Neo4j Text2Cypher dataset: neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset · Kuzu post-mortem: theregister.com/2025/10/14/kuzudb_abandoned · Tauri signing: v2.tauri.app/distribute/sign

---
*Prepared on 2 September 2026 from the decisions taken in discussion and from the research of two agents (runtime; data and training). All versions and facts are from September 2026 sources.*
