# MODEL FREEZE — CodeMap navigator v1 (2026-09-02, owner directive 11:12)

The v1 navigator is FROZEN. Training rounds stop here; everything after this file builds
AROUND the model (API switch, UI, wizard), never inside it. Unfreezing requires a new
version tag and a full ladder re-run — no silent weight changes, ever.

## The frozen bundle (the four artifacts that must travel together)

| artifact | identity | sha256/16 |
|---|---|---|
| model | `codemap-lora-r22-q4_k_m.gguf` (2.5 GB, Qwen3-4B-Instruct-2507 + r2.2 LoRA merged, Q4_K_M) | `9c454526d7d0d1b0` |
| grammar | `graph/pack/codemap_vocab.gbnf` (1,413 entities + 30 subs; regenerates with pack) | `8df967fc105b85e2` |
| master prompt | `training/master_prompt_v1.txt` (THE SELECTION LAW) — train-serve law D2 | v1 |
| pack | `graph/pack/` per `manifest.json` | `b363691317344649` |

Volume provenance: `codemap-train/lora-r22-qwen3-4b` (adapter + merged + gguf).
Corpus: datagen v2.2 (7,776 pairs + pass ×4 oversample), commit 9403640.

## Why r2.2 and not r3-dpo (the decision, with numbers)

r3-dpo (291 preference pairs) improved every BATCH judge (abstention 0.9091→1.0 GPU,
ans_cos 0.754→0.774, topo 0.826→0.870, rr 0.417→0.583) with exec flat at 0.9752 — and
REGRESSED the product path: autonomous-loop bare-start abstention 0.74→0.58 (8/19
out-of-corpus questions answered — the hallucination-facing failure class), invalids
1→5 (token truncation). Root cause diagnosed: the wrong-abstain pair class was
one-sided — it suppressed bare-start passing indiscriminately, and unanswerable
questions are indistinguishable from answerable ones at the bare start (that
indistinguishability is the whole difficulty of abstention). A frozen product model is
judged on the path users walk: r2.2 ships.

r3-dpo stays on the volume as a recorded experiment. The BALANCED pair set (mirror
class: chosen = doctrine handling of unanswerables, mined from r3-dpo's own 8 failures)
is designed and committed in `dpo_pairs.py` — ready if v2 ever reopens training.
Loop-tier fixes that are model-agnostic (560-token answer budget, per-model output
filenames) are live in `loop_runner.py`.

## The frozen model's measured card (shipping configuration: CPU Q4 + grammar)

exec_fp 0.9752 · abstention-with-evidence 1.0 · false_pass 0.0083 (defensible class) ·
ans_cos 0.7933 · topo_arch 0.8696 · rr_equiv 0.5833 · step p50 0.55 s · answer p50
2.92 s (p95 10.2 s, known amber) · loop strict 0.5692 / bare-start abstention 0.74 /
0 stalls. Full derivation: PIPELINE.md gate table + docs/04-training-story.md.
