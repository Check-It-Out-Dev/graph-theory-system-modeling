# CodeMap — full graph regeneration runbook (Ladybug-first, v1 · 2 September 2026)

The from-scratch path: repositories → indexed entities → three embedding lenses → partition →
curation → organisation → clues → pack → application, with **LadybugDB as the system of record
from the first write**. Neo4j appears nowhere. Use this to (a) prove the whole pipeline on a
fresh store, (b) onboard a NEW codebase, (c) rebuild after a schema change.

Everything below is the SAME chain that produced the current graph. The only differences are the
starting state (an empty store) and MODE full instead of delta.

## Preconditions

- `pip install real-ladybug numpy`. The store is `real_ladybug` 0.15.3 or later; nothing imports
  neo4j (the migrator is provenance-only and is not part of regeneration).
- The Modal embedder is up (`ramzesx--v3-code-embeddings-serve`, Qwen3-Embedding-8B, 4096
  dimensions, scale-to-zero), or any embedder that returns `{"embeddings": [[...]]}` for
  `{"texts": [...]}`. Authoring embeddings are NOT the runtime embedder (the two-embedder rule).
- The four agent manuals are current: `graph/prompts/{HypatiaV5, GrothendieckV5, ErdosNavigatorV5,
  CodeMapConductorV5}.md`. The connection block is the Store; ledger entry L12 binds every worker.
- A fresh store: `python graph/authoring/ladybug_store.py init --db <new>.lbdb` (or the default
  path after moving the old store aside). NEVER regenerate over a live store. Supersession is for
  deltas; a full regeneration starts empty by definition.

## Stage chain (agents in order, each gate-checked by the Conductor)

| # | agent (MODE full) | writes | gate before the next stage |
|---|---|---|---|
| 1 | **HypatiaV5** — indexes both repositories, builds three text views (the "sockets") per file, embeds them via Modal, creates Entity rows through `Store.create` (nid = a stable integer sequence) plus Dep edges and provenance | Entity, Dep | conformance before scale (the first 8 diverse files verified), lens presence 3×N, JSON round-trip probe |
| 2 | **GrothendieckV5** — partition (v4 recompute: typed connections + three-lens fusion + Leiden), hyperedge extraction (metapath v3, IDF), curation session (roles, GROUPs, master index) | Nav, Master, Guides/GuidesChild/Member, Hyperedge, CurationDecision | partition acceptance battery; repository-purity invariant; token economy |
| 3 | **ErdosNavigator** — C1 dossiers → C3 organisation (layers, MacKay trophic heights with a per-component gauge, entry points, spines) → clue bodies at L2/L3 → ClueSnap bi-temporal baseline | Entity annotations, Nav clue fields, ClueSnap | C3 conformance differential against the frozen recipe; gauge acceptance equivalent to 24/24; inversion gate |
| 4 | **mechanical close** — `q_gold_all.py` (goldens; the determinism rule makes them byte-stable), `diag_state.py --run-id <date>-baseline` (first observations), `export_pack.py` (pack and vocabulary GBNF regenerate together), `import_ladybug.py` (runtime DB plus gold M03), `app/test_app.py` and `smoke.py` | mfq, Diag, pack | the FULL acceptance battery of the section below |

Spawn pattern (each agent reads its canonical manual; the Conductor verifies with its own reads):

```
Agent(subagent_type='HypatiaV5',      prompt='MODE full on <repos>; store per manual; report per manual.')
Agent(subagent_type='GrothendieckV5', prompt='MODE full; store per manual; report per manual.')
Agent(subagent_type='ErdosNavigator', prompt='MODE full; store per manual; report per manual.')
```

## Acceptance (what "it worked" means, mechanically)

1. `q_gold_all.py` run twice → identical file hash (run-to-run determinism).
2. `diag_state.py --dry` → the functional catalogue is complete; the baseline is written once.
3. `gauge_acceptance.py` exits 0 (all leaves AGREE between C1 and C3) and
   `inversion_gauge_check_curated.py` exits 0.
4. `export_pack.py` → `import_ladybug.py` reports M03 MATCH → `test_app.py` ALL PASS →
   `smoke.py` PASS.
5. On a RE-generation of the SAME codebase, differences in golds and pack are explainable by
   content (the code changed), never by ordering noise (the determinism rule guarantees this).

## Cost envelope (measured anchors, checkItOut scale: 1,415 files)

- Embeddings: about 4.2k socket texts ≈ 1–2M tokens through the Modal embedder ≈ **$2–5**
  (scale-to-zero A10G minutes).
- Agent wall time: Hypatia 1–2 h, Grothendieck 1–2 h (including curation), Erdős 1–2 h —
  **3–6 h in total**, parallelisable only within a stage, not across stages (each gate blocks the
  next stage by design).
- Mechanical close: about 10 minutes of local CPU.
- **No model retraining** (the navigator speaks the pack contract, not the store) and **no
  application changes**: the frozen v1 model and application consume whatever pack this produces.

## Known dialect rules (bisected; violating any of them means silent data corruption)

See the `ladybug_store.py::Store.create` docstring and ErdosNavigatorV5 ledger entry L12: None and
empty-list parameters cannot be typed; strings that look like literals are re-serialised (the JSON
is destroyed, with no error); lists are cast into STRING notation; NULL is accepted only as a
literal in SET; scan order is never a contract, so sort at the boundary and break ties by name.
