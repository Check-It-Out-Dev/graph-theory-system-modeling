# Experiments catalogue (organised 2026-09-02)

Three tiers. Paths in the papers refer to `experiments/<name>.py`; files moved to a tier
subfolder keep their name — this table is the resolver. Cached arrays (`*.npz`) stay at root
beside their consumers.

## Root = PRODUCTION (the promoted pipeline; still runnable, still consumed)

| script | role | findings |
|---|---|---|
| `meet_merge.py` | THE promoted operator: meet-quotient v2 (T̂+α·Ĉ, Louvain bisected to k) | F105, F106 |
| `promote_champion.py` | wrote `v4_subsystem` + V3Master provenance on all commits | F106 |
| `cochange_eval.py` | the external oracle harness: git co-change, 20 splits, paired wins | F- battery, every promotion |
| `cochange_incremental.py` | delta-oracle for reindex rounds | delta law |
| `cohort_fibers.py` | commit-cohort fiber partition — the meet's second parent | F102 |
| `constrained_partition.py` | constraint machinery + the 332-pair disagreement shortlist generator (C2 prep) | F58, F60 |
| `lens_gate.py` | pre-registered lens acceptance gate (v2-uniform baseline, v3 comparisons) | H6 gates |
| `full_picture.py` | the 4-panel full-topology figure | figures/v3_full_topology.png |
| `build_hierarchy.py` | level-2/3 hierarchy writer + delta-run definitions | S5 |

## `evidence/` — diagnostics that established cited numbers (keep for reproducibility)

| script | established |
|---|---|
| `partition_granularity.py` | granularity-matching control (the confound that kept biting) |
| `resolution_limit.py` | Fortunato–Barthélemy √(L/2) applicability |
| `standard_metrics.py` | TurboMQ / MoJoFM / ARI / NMI field baselines |
| `layer_vs_feature.py` | entity_type = layer coordinate; subsystems are vertical slices (F107 basis) |
| `lexical_vs_neural.py` | RRF ranker of record 0.8432; mechanism rule F92 |
| `metapath_hin.py` | meta-paths: indicators 15.6×, useless rankers (F78/F79); the reversal-bug lesson |
| `hyperedges.py` | hyperedge precision/lift priors (P_R_P 0.491 …) before the single-writer emit |
| `dimension_sweep.py`, `ratio_sweep.py` | embedding-dim and mix-ratio sweeps |
| `model_class.py` | the model-class ladder; rotation lost 0/34 (F- rotation) |
| `fit_rho.py`, `typed_connection.py` | per-type connection refits (S1, F2 settlement) |
| `full_graph_audit.py` | full-graph recomputation discipline (G3) |
| `bigon_stability.py`, `interaction_tensor.py` | stability + tensor probes |
| `grothendieck_vs_content.py` | organizer-vs-content comparison |
| `hypergraph_partition.py` | V6 hypergraph acceptance battery |
| `fiber_bundle_partition.py` | fiber-bundle partition probe |
| `partition.py` | the first S3 partition (superseded by meet_merge) |

## `archive/` — refuted methods (kept per supersession-never-erasure; do not build on these)

| script | refuted by |
|---|---|
| `consensus_partition.py` | F103 — co-association is resolution-discontinuous (13→190 parts in res 0.03) |
| `structural_lens.py` | F42/F47 — 42-invariant feature vector lens |
| `sheaf_laplacian.py` | F99/F104 — vertex-Laplacian family double-sealed |
| `path_composed_shadow.py` | path-composed shadows fell with the geometry arc |
| `topology_shapes.py` | early shape probes superseded by the lens/meet line |
