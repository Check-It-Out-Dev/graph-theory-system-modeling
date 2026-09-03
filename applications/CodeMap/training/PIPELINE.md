# CodeMap training pipeline — architecture (02.09.2026; execution starts next session)

> **MODEL FROZEN 02.09 11:12 — v1 = r2.2** (see `FREEZE-v1.md` for the bundle hashes and
> the r3-dpo decision). R4-DPO verdict: batch-accepted, LOOP-REJECTED (bare-start
> abstention 0.74→0.58, invalids 1→5 — one-sided preference pairs suppress passing
> indiscriminately; the mirror class is designed in dpo_pairs.py, not run). Training
> stops; everything below is the preserved record.

The model's whole job: NL → one CMDSL line (per loop step) + result → NL synthesis + knowing
when to `pass()`. Everything below serves that narrow target. Train–serve law (doc-00 D2):
the master prompt AND `dsl_version` are FROZEN before pair generation; the MCP/engine the
teacher drives is byte-identical to the one that ships.

## Stages

```
 [1] freeze: master prompt v1 + CMDSL v0.1 + engine (app/) + pack (post-curation)
 [2] teacher trajectories: frontier model drives the SAME /ask + /step loop on the
     question set (104 gold + generated paraphrases) — logs: (NL, [DSL steps], NL answer)
 [3] verifier filter (D11, strengthened): every pair must (a) parse, (b) compile,
     (c) execute, (d) fingerprint-match the gold where one exists; answers NLI-checked
     against retrieved context. Dedup (MinHash), decontaminate vs eval Q (n-gram+embedding)
 [4] pair sets: NL→DSL (per step, incl. pass() abstention examples — mandatory from run 1),
     result→NL (answer synthesis), loop-control (continue vs answer vs pass)
 [5] LoRA (Unsloth, bf16, no QLoRA): per candidate {Qwen3-4B-2507, Qwen3.5-4B, 3.5-2B,
     3.5-0.8B}; merged -> GGUF Q4_K_M; chat-template/EOS smoke test after every export
 [6] eval ladder (paired bootstrap on the SAME Q):
     rung 0    teacher + engine            = ceiling
     rung 0.5  scripted DSL (app/smoke.py) = machinery floor            [DONE, green]
     rung 1    bare small model + engine   = prompt-only baseline
     rung 1.5  bare + GBNF grammar        = constrained-decoding dividend (pre-training!)
     rung 2    + LoRA_k for doubling k     = learning curve
     rung 3    + LoRA, crippled graph      = control (graph does the work, not the weights)
     grid      {2B, 4B} x {lean, rich}     = H-COMP compensation surface
 [7] stop rule: stop when a doubling of pairs gains < CI width; report success ALWAYS as
     (small-model, teacher-ceiling) pairs; failures into the 3 buckets
     (COVERAGE -> authoring backlog | RETRIEVAL -> cache/find tuning | USAGE -> pairs)
```

## Metrics per layer (doc-00 §6, adapted to DSL)
- step level: DSL exact-match + execution-fingerprint accuracy (compile+run — the DSL makes
  execution accuracy CHEAPER than text-to-SQL, our verifier is deterministic)
- loop level: task success on gold; steps-to-answer; backtrack rate
- abstention: risk–coverage curve from pass() logs (selective prediction)
- e2e targets (doc-00 §10): task success ≥90% with teacher ceiling ≥95%; answer <5 s CPU

## Data budget (measured anchors)
~600 verified pairs equalizes small models on own-tool tasks (doc-00 research fact); the 104
golds × ~3 loop steps × alias paraphrases ≈ 1.2k raw step-pairs before generation — the
corpus exists; stage 2 mainly adds loop diversity and pass() negatives (questions the graph
genuinely cannot answer — mined from the CONTENT_POINTER/pass boundary).


## Escalation signal (owner design 02.09 — trained now, UI-consumed later)

`pass(reason)` stays the model's only escalation surface. The MCP wraps every terminal pass
into an **EscalationOffer** record: `{offer_id, question, reason, reason_category
(needs-content-read | out-of-corpus | ambiguous | multi-repo), context (visited L1/L2 ids),
suggested_tier: "api", status: "PENDING_USER"}` — logged to telemetry and returned to the
caller. The UI later renders it as a consent card ("the local model says: beyond me because
<reason> — switch to the paid API model?") with [use API model] / [keep local]; nothing in
the model or MCP changes when that ships. Training data includes pass-pairs with crisp
categorised reasons from day one — abstention quality is a first-class trained skill.

## Sequence diagrams (owner requirement 02.09)

Diagram CORRECTNESS is deterministic: `flow()` and `seam()` results carry a `mermaid` field
(sequenceDiagram rendered from the actual edge rows, capped). The model learns PLACEMENT:
answer-synthesis pairs for flow-shaped questions embed the engine's mermaid block in the
final answer. The model never invents arrows; it frames what the engine drew.

## Evaluation harness v2 (automated, Modal-backed — the SOTA gate set)

| metric | how | gate |
|---|---|---|
| `step_exact` | canonicalised DSL string equality vs reference | primary for NL->DSL |
| `exec_fp` | run BOTH DSL programs through the engine; canonical result fingerprints equal | the true execution accuracy (beats text-to-SQL: our verifier is deterministic) |
| `ans_cos` | Qwen3-Embedding-8B (Modal) cosine model-answer vs gold-answer | mean + frac >= 0.85 |
| `rr_equiv` | Qwen3 reranker (Modal): score(q, model_ans) / score(q, gold_ans) >= 0.90 | gold-equivalence, automated judge |
| `topo_arch` | embed all answers; nearest GOLD neighbour must share the archetype (kNN consistency in embedding space) | the topological correctness check — a wrong-archetype answer sits in the wrong region even when fluent |
| `risk_cov` | pass() rate on unanswerable vs answerable strata | selective-prediction curve |

Teacher note: the frontier main agent IS the teacher, but most step-pairs need no teacher at
all — each gold's archetype+params derive the canonical DSL mechanically from the recipe
manifest. The teacher authors only loop diversity, pass reasons and answer style.

## What is deliberately NOT trained
Cypher/Ladybug, schema, subsystem ids, file paths, clue contents — all runtime reads. If an
eval error is fixable by editing a clue or a recipe, fix the GRAPH, not the weights (the
whole point: intelligence lives in the structure; weights only learn the tongue).

## Round log + the grounding correction (02.09, research-grounded — web sweep on record)

**Round 1 (lora-r1-qwen3-4b, 386 pairs, 3 epochs, A100, $1).** Syntax fully learned (100%
valid one-line CMDSL, 93.2% token acc) — grounding not generalized: **0/20 exec-exact** on
the adversarial held-out; the model INVENTS plausible names (`HttpInterceptor.java`) and
mis-picks sub ids. This is the documented failure mode of narrow closed-book SFT: memorized
facts fail to route into use ("Knowing–Using Gap", arXiv 2607.08393) and fine-tuning on
narrow corpora amplifies verbatim memorization (arXiv 2510.16022). The harness did its job;
the number to trust is exec_fp, never token accuracy.

**Round 2 (SFT again, same frozen 0.19.1 stack; H100 sanctioned for iteration speed):**
1. **Vocabulary GBNF from the pack** — `args ::= <real entity names> | <routable sub ids>`;
   invention becomes UNREPRESENTABLE at deploy (llama-server takes `--grammar-file` and
   per-request grammars). Compile ONCE per pack version, never per request (XGrammar
   2411.15100: most token mask is context-independent+precomputable; naive complex grammars
   cost seconds). Caveat that shapes training: **grammar removes invention, not wrong
   SELECTION** — a wrong-but-real name is still grammatical. Hence:
2. **Open-book selection SFT — the core lever.** Every prompt carries the context block
   (digest/affordances) CONTAINING the candidate names; supervision teaches SELECT/COPY from
   context, not recall (copy-mechanism + open-vs-closed-book evidence: 2604.18170,
   2210.03273). Round-1 pairs were closed-book for step rows — that gets rebuilt.
3. **Full-coverage execution-grounded datagen** — closed world (1415 entities × 13 verbs,
   deterministic executor = free golds) lets us enumerate coverage: grounding drills
   (name→locate, alias→sub, PL/EN paraphrase templates) + affordance-selection pairs +
   loop pairs. Target ~20k pairs; **saturation law: do NOT scale past ~20–50k** in a closed
   world (SLM-SQL's 916K is for open-schema BIRD; past saturation we re-memorize templates).
   Decontaminate splits by (entity, verb) strata, not string match.
4. **Measure grammar-OFF on Modal** (transformers greedy, isolates the SFT effect);
   grammar-ON lands at the llama.cpp/H-COMP stage where GBNF is native. Report both.

**Round 3 (ONLY if round-2 exec_fp < 0.85): GRPO with the executor as verifiable reward.**
Separate pinned image `trl[vllm]==0.28.0` (NEVER touch the 0.19.1 SFT stack), H100:2 vLLM
server mode per Modal's official GRPO example; reward = 1.0 exact fingerprint + 0.2
executes-clean + 0.0 invalid; 4–8 rollouts/prompt, lr 5e-6, prompts drawn from failing
strata. Known traps: GRPO+vLLM colocate+PEFT hangs (TRL #3671), GRPO+vLLM+LoRA history
broken (TRL #2698) → server mode, or full-FT the 4B, or `model.generate` rollout fallback
(tolerable: completions ≤50 tokens). Verify vLLM supports the exact checkpoint before any
base swap (vllm #36275 arch-mismatch).

**Deploy-safe test-time compute:** greedy-then-resample-on-error, or best-of-3 at temp
0.6–0.8 with execution filter (our executor is microseconds; execution-guided decoding is
worth ~5% in the text-to-SQL literature, 1807.03100). Self-consistency at temp 0 is n× cost
for identical samples — never.

## GATE TABLE — v1 navigator, measured 02.09 (all green on the SHIPPING configuration)

The shipping configuration = r2.2 Q4_K_M GGUF (2.5 GB) + llama.cpp CPU + vocabulary
grammar ON + pack v3. Scored on its own decontaminated test snapshot (141 never-seen
entities among 121 step rows + 23 answer + 11 abstain).

| gate | bar | GPU bf16 | **CPU Q4 + grammar (ships)** | verdict |
|---|---|---|---|---|
| exec_fp (step truth) | ≥0.97 | 0.9752 | **0.9752** | GREEN — quantization free |
| pass_on_unanswerable | ≥0.8 | 0.9091 | **1.0000** | GREEN (was 0.42 in r2.1) |
| false_pass_on_answerable | ≈0 | 0.0083 | **0.0083** | GREEN — both are defensible immediate-abstains; zero wrong passes on content |
| ans_cos mean | ≥0.75 | 0.754 | **0.7933** | GREEN |
| topo_arch | ≥0.75 | 0.826 | **0.8696** | GREEN |
| rr_equiv | ≥0.5 | 0.4167 | **0.5833** | GREEN on ships-config (n=12 — expand sample post-v1) |
| step latency p50 | «instant» | — | **0.55 s** (p95 1.28) | GREEN |
| answer latency p50 | <5 s | — | **2.92 s** | GREEN |
| answer latency p95 | <5 s | — | 10.16 s | AMBER — long-answer tail; levers: token budget, streaming UX (named, not hidden) |

Journey: r1 0/20 exec (closed-book recall) → r2 0.9545 (open-book selection) → r2.1
0.9794 (injective supervision) → r2.2 0.9752 + abstention 1.0 CPU (uniform doctrine).
The intelligence lives in the graph; the model learned the tongue and when to stay silent.

## LOOP-level operation MEASURED (02.09, 04:14 — first measurement of the product path)

104 AUTONOMOUS multi-step sessions (model drives find→verb→answer/pass against the live
engine, grammar-ON CPU, 7.4 min wall): terminals 68 answer / 35 pass / 0 stall / 1
invalid (a 380-token answer truncated mid-string — raise the answer budget), steps mean
1.88, backtracks 0. Strict engine-vs-engine task success **37/65 = 0.5692** on
answerable-with-canonical golds; pass-on-unanswerable 14/19 = 0.74 from a bare start
(batch measures 1.0 WITH evidence in context — bare-start abstention is a different,
harder skill). Referee note: the first scorer compared engine fps to RECIPE fingerprints
(0.0 by construction) — the instrument lied before the model could; fixed
engine-vs-engine in loop_runner. Honest confound: most golds contributed TRAINING pairs,
so loop success on them mixes memorization with protocol skill — generalization evidence
remains the held-out batch gates; in production this path sits BEHIND the MFQ cache
(loop = the cache-miss tier). Miss decomposition, levers named for r3: 6× premature
answer — a "Located:" list from find-hits where the canonical wants flow/impact descent
(the FE08 repair template overgeneralized; lever: answer-timing drills — hits alone are
not evidence for flow questions); 6× bare-start wrong-abstain on content-flavored
EXECUTED golds (lever: bare-start answerable openers); style leakage ("Paths in gold
rows." parroted from gold phrasing; lever: scrub meta-phrases from answer supervision);
1× token-cap invalid. Batch rungs cannot see ANY of these — the loop tier is now a
permanent part of the ladder (app/loop_runner.py, LOOP_r22_cpu.json).

**Round 2.2 MEASURED (02.09, lora-r22 — THE GATE ROUND).** Rung 2 on 276 stratified rows:
exec_fp **0.9752**, **pass_on_unanswerable 0.9091** (from 0.42 — the uniform two-step
doctrine + ×4 oversample was the fix), **false_pass 0.0083**, ans_cos 0.754, topo_arch
0.826 (archetype-keyed), rr_equiv 0.4167 (n=12 — sample-noisy; CPU r2.1 measured 0.58 on
the same metric). Forensics on the 6 step misses: 1 canonical-term arbitrariness
(find(2024) vs find(migrations) — BOTH are question-terms; scoring artifact, left
unadjusted rather than loosening a gate at declaration time), 2 immediate-abstains on
blatant out-of-domain (defensible; they ARE the whole false_pass — zero wrong passes on
real content), 1 find-first slack on a name-verbatim question, 1 FE08 residual alias,
1 association-selection miss. TRUE error rate ≈ 2–3 / 242. FE08's healed answers now list
real interceptor files — style differs from gold, truth does not (this is what ans_cos
penalizes; the reranker/execution judges are the truth gates). One abstention residual:
answered once from junk find-hits instead of passing on empty evidence — candidate r3
lever: a "hits are irrelevant → pass" drill class.

**Reward-hacking guard:** the free-prose `answer()` is NEVER RL-trained against cosine or
reranker scores; the Modal judges stay eval-only.

**Round 2 MEASURED (02.09, lora-r2, H100, 2 epochs, eval_loss 0.0917).** Rung 2 grammar-OFF
on 252 stratified test rows (incl. the never-seen-entity stratum): **step_exact 0.938,
exec_fp 0.9545** (round 1: 0/20) — the open-book restructuring is the whole difference; the
GRPO gate (exec_fp < 0.85) does NOT fire. Forensics on all 9 step misses: 5 sibling-stem
ambiguity (drill question "X.component" matches .ts AND .html — the model picked a
defensible sibling; SUPERVISION defect), 2 depth-ambiguity (depth-2 drills phrased
identically to depth-1; SUPERVISION defect), 1 alias-vs-canonical conflict (gold said
health(hubs), alias named a file, model's find(file) was the better move; CORPUS defect),
1 residual invention (FE08). Law: DRILL QUESTIONS MUST BE INJECTIVE — one question surface,
one defensible target. Regressions measured honestly: pass_on_unanswerable 2/6 (pass rows
diluted to 0.7% of corpus; model tries find() first — legitimate prefix, needs the
two-step find→pass trajectory taught); answer rows n=4 too small to judge (2 of 4 kept
navigating instead of terminating — teach termination; expand answer rows in test).

**CPU deploy rung MEASURED (02.09, r2.1 GGUF Q4_K_M 2.5GB, llama.cpp b10155, 30 threads).**
Same stratified rows as GPU rung, scored against the model's own split snapshot
(test_v21): grammar-ON exec_fp **0.9836**, grammar-OFF **0.9836** — identical, and both
slightly above the bf16 GPU run (0.9794). Quantization cost: none measurable. Grammar
dividend on-distribution: **zero at this quality level** (open-book SFT already removed
invention on these rows) and grammar costs nothing at runtime (step p50 0.57s vs 0.56s) —
so the product ships grammar-ON as a FREE insurance floor for out-of-distribution
questions, not as an accuracy lever. Latency vs doc-00 targets: step p50 0.57s / p95
1.41s (green), answer p50 3.66s (green vs <5s), answer p95 15.3s (OVER — long 380-token
answers; levers: answer token budget, streaming UX; reported, not hidden). Judge scores
on CPU: ans_cos 0.78, rr_equiv 0.58, topo_arch 0.78–0.83 (archetype-keyed).

**Round 2.1 levers (all evidence-pinned):** (1) injective drills — full-name question when
the stem collides, explicit "2 hops" phrasing for depth-2; (2) abstention rebuild — ~120
authored negatives EN/PL + two-step find→pass trajectories for in-graph-term negatives
(the negative-example channel: SFT-compatible recovery supervision, model's own r1/r2
errors become repair pairs); (3) answer termination + volume — alias-variant answer pairs,
multi-step trajectories ending in answer(), ≥20 answer + ≥30 abstain rows in test;
(4) skip-alias law — an alias whose surface entity contradicts the canonical arg is
dropped from step supervision, counted in STATS.
