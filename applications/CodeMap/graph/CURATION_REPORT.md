# CURATION_REPORT — CheckItOutV3, GrothendieckV5 batch `2026-09-02-curation`

Phases P2–P5. P1 (the v4 partition, provenance 0.3738 held-out modularity / 18-20 paired wins)
was read as given and never recomputed. No measured property was modified: `v4_subsystem`,
embeddings, hyperedges (92, metapath-v3) and every `assignment_*` field are byte-identical to
their pre-curation state, verified by read after the write.

## 1. Decision table

| sub | action | trigger | evidence (dossier / measured) | measured delta |
|---|---|---|---|---|
| **17** | SPLIT + RETYPE `GROUP` + RENAME | MEGA, share 28.62% > 20% | vertical domain seams dominate: core/auth↔feature/auth 40 edges is the largest internal seam, repeated for demo 10, opportunities 10, support 10, user↔profile 9, subscription↔plan-billing 8. Cohort-parent corroboration: the v3 cut concentrates ≥84% in 6 of 9 parts | vertical **Q=0.604** vs horizontal **Q=0.120** (5.0×); aggregate cohesion 0.489 vs 0.241; largest child share **5.16%** (was 28.62%); fan-out 9 = budget |
| **18** | MERGE → 17, child 177 | MICRO n=1 < 3 | 100.0% of its 156 external edges go to sub-17 — the strongest dominant-seam measurement in the graph; in=156 out=0 | internalises all 156 edges; placed in the platform child because no child dominates consumption (max 23.7%) |
| **13** | MERGE → 4 | LAYER_PURITY (90% Rule) **and** ext 0.938 + dominant partner | LAYER retype **refuted**: in=9 out=361 (**2.4% fan-in**), only **2** distinct consumers — a consumer satellite, not a supplier layer. Dominant partner sub-4 at **60.3%**; v3 overlap 48/49 in sub-4's parent cell | sub-4 cohesion **0.362 → 0.478 (+32%)**, internal edges 515 → 761, size 172 = 12.16% (under MEGA) |
| **16** | RETYPE `LAYER` + RENAME | ext 1.000 | MERGE **refuted** — the trigger needs one dominant partner and there is none: top three tied at 8 edges (**12.7%** each). Fan-in **93.7%**, **12** consumers, 0 internal edges | every merge candidate *lowers* the host: →4 0.362→0.353, →6 0.177→0.173, →9 0.242→0.234, →11 0.489→0.471 |
| **2** | RETYPE `LAYER` + RENAME | ext 0.993 | MERGE **refuted** — no dominant partner (sub-4 33.1% vs sub-5 24.4%, ratio 1.36). Fan-in **86.2%**, **8** consumers | rejected merges gain almost nothing: →4 +0.010, →5 +0.006 |
| **7** | RETYPE `LAYER` + RENAME | none (uniformity) | not in the queue; the same criterion that promoted 2 and 16 selects 7 *more* strongly: fan-in **91.1%**, **15** consumers, max partner 13.4% | no membership change |
| **8** | KEEP + RENAME | ext 0.944 | MERGE-CHECK resolved: max partner sub-4 **26.8%**, a spread not a seam; in/out 104/105 balanced, so neither fragment nor layer | rejected merge 8+4: 0.362 → 0.368 (**+0.006**, negligible) |
| 0,1,3,4,5,6,9,10,11,12,14,15 | KEEP + RENAME | none | see `dossiers/subsystem_<id>.json` | — |
| — | ASSIGNMENT-REVIEW | 14 `assignment_flagged` | all **CONFIRMED**, none moved | 5 of 7 cross-repo affinities land in the child whose BE counterpart the unconstrained kNN named |

**The layer criterion**, applied uniformly to all 19 candidates rather than only to flagged ones:
in-share ≥ 0.85 **and** ≥ 8 distinct consumers **and** no seam partner above 0.40. It selects
exactly {2, 7, 16} and rejects 8 (in-share 0.498) and 13 (0.024). Note this **contradicts the
manual's `purity > 0.85` layer trigger** — measured purity is 67% for sub-16 and 52% for sub-7.
The fan-in signature is the direct evidence for "is this consumed as infrastructure", so per the
manual's own rule the measurement wins and the disagreement is recorded here.

## 2. The sub-17 split

Nine children under a kept parent, not a flat replacement: sub-17's cohesion **0.723** is the
highest of any candidate (it is a real unit), and ledger **L1** requires preserving `sub_id`
identity because MFQ `depends_on_subsystems` stamps key on it.

| child | name | n | cohesion | note |
|---|---|---|---|---|
| 177 | FE shell, i18n & generated client | 74 | 0.521 | absorbs ex-sub-18 |
| 170 | FE auth, 2FA & interceptors | 69 | 0.559 | holds the 40-edge core↔feature auth seam |
| 173 | FE fixtures & E2E harness | 67 | 0.065 | **role LAYER** — 85% Resource purity, 4 internal edges, 23 files of degree 0 |
| 176 | FE profile, settings & admin | 52 | 0.455 | |
| 171 | FE opportunities & applications | 43 | 0.524 | |
| 172 | FE onboarding survey | 42 | **1.000** | zero edges to any sibling |
| 174 | FE plan, billing & consent | 23 | 0.467 | 2 files with cross-repo affinity to BE [11] |
| 175 | FE support & help centre | 20 | 0.529 | |
| 178 | FE demo mode | 16 | 0.824 | |

Louvain on the same subgraph reaches Q=0.815, but only by shattering into 182 communities (178
of them under 3 nodes) — unusable against the fan-out ≤ 9 budget. The vertical scheme keeps 74%
of that modularity with 9 parts. Child 173 was retyped rather than dissolved because the manual
forbids silently keeping a purity-flagged set typed as a slice, and its 0.065 cohesion is the
expected shape of a fixture layer, not a defect.

## 3. Owner questions (2-option, numbers attached)

**Q1 — master fan-out.** The ≤ 9 fan-out budget that forced the sub-17 split is itself violated
one level up: NavigationMaster now GUIDES **17** top-level navigators.
- **(A) Accept 17.** The six index groups already give a reading structure, and changing the
  top level is a P1 re-partition question, which is sealed. Cost: nothing. Risk: L1 answers
  scan a 17-way list.
- **(B) Promote the six index groups to real GROUP navigators**, master fan-out 17 → **6**,
  uniform ≤ 9 at every level. Group sizes would be frontend 406, billing-legal 283, platform
  235, auth-security 203, opportunities 172, data-support 116 (sum 1415). Cost: Erdős must
  clue 6 more nodes; one more hop on every L1→L2 traversal.

**Q2 — `real-login.ts` (low stakes, 1 file).** Placed in child 173 (FE fixtures & E2E harness)
on folder evidence (`e2e-tests/_framework/`), but its `assignment_crossrepo_affinity` is BE
sub-9 (auth journeys), which argues for child 170. The graph cannot settle it: the node has
**degree 0**, so placement changes no cohesion number.
- **(A) Keep in 173** — consistent with the other `e2e-tests/` files.
- **(B) Move to 170** — consistent with the measured cross-repo affinity.
The dissent is recorded on the node as `curation_dissent` either way.

## 4. Verification (every write read back)

| check | result |
|---|---|
| CurationDecision nodes / linked to V3Master | 20 / 20 — equals the decision log |
| leaf CONTAINS_MEMBER (children + non-group navigators) | **1415** |
| distinct EntityDetail covered / total / orphans | 1415 / 1415 / **0** |
| `curated_subsystem` stamped | 455 = 406 split + 49 moved — moved/split nodes only |
| V3Master & NavigationMaster fan-out | 17 and 17 (was 19; 13 and 18 dropped) |
| sub-17 children | 9 |
| ClueSnapshots | 6 → 26 (+19 L2, +1 L1) |
| SUPERSEDED_BY navigator edges | 13→4, 18→177 — both nodes retained, role MERGED |
| measured layer intact | v4_subsystem 1415, embeddings 1415, hyperedges 92, assignment_flagged 41, crossrepo affinities 7, v3_subsystem 1374 |

## 5. Diagnostics drift (`diag_state.py --run-id 2026-09-02-curation`, 60 functionals)

| functional | before → after | verdict |
|---|---|---|
| `l2_census` | `{erdos-v1:14, erdos-v2:5}` → `{curated-v1:28}` | **expected-from-my-changes** — clue_version bumped on all 28 navigators (17 top-level + 9 children + 2 merged) |
| `snapshot_count` | 6 → 26 | **expected-from-my-changes** — 19 L2 snapshots + 1 L1 snapshot from the supersession writes |

All 58 other functionals unchanged, including `node_count` 1415, `edge_count` 4786,
`hyperedge_census` 92, `lens_presence` S=B=T=1415, `reachability` 1415, and every per-subsystem
`subsystem_size` / `external_ratio` / `trophic_service_median`. That last group is the proof the
judgement did not edit the measurement: those functionals are computed from `v4_subsystem`, and
they did not move. **Observation for the catalogue owner** (not a change I made): because they
key on `v4_subsystem`, no functional currently observes the *curated* partition. A
`curated_subsystem_size` functional would be needed for post-curation drift detection.

## 6. Handoff to Erdős

**26 navigators are ready for organisation and clue generation** — 17 top-level (13 SLICE,
3 LAYER, 1 GROUP) plus 9 children of sub-17.

The clue bodies are **stale by construction and this is the next agent's work, not a defect**:
this run changed names, roles and membership but did not regenerate `ai_summary`,
`responsibilities`, `caveats`, `spines` or `contracts`. That is queryable, not just prose —
every navigator carries `clue_body_status`: `STALE-awaiting-erdos` (17), `MISSING` (9 children,
no clue body at all), `SUPERSEDED` (2 merged). `clue_version` reads `curated-v1` to mark the
curation, **not** a regenerated clue.

Subsystems whose membership changed, i.e. the delta input and the MFQ cache invalidation set
(`depends_on_subsystems`): **4** (+49), **17** (+1, now a GROUP), **13** and **18** (dissolved),
and all nine new children **170–178**. Every other subsystem's membership is byte-identical.

Priority order for re-cluing, by how far the clue has drifted from the facts: 170–178 (no clue
at all), then 4 and 17 (membership changed), then 2, 7, 16 (role changed to LAYER — the clue
must now describe what they *supply*, not what they *do*), then the remaining renames.

## 7. Files

- `graph/scripts/c2_curation.py` — the P4 write (decisions, L2 in-place updates, children, membership, merges)
- `graph/scripts/c2_curation_l1.py` — the L1 index and caveat update
- `graph/CURATION_REPORT.md` — this file
