# Erdős E-run report — first navigation layer over the current (pre-curation) system

**Date**: 2026-09-02 · **Mode**: full · **Executor**: main agent following
`prompts/ErdosNavigatorV5.md` as the spec (first executability test of the prompt).

## What was written (all verified by read)

| layer | artifact | count |
|---|---|---|
| E1 → L3 | `layer`, `local_height` (MacKay, per-subsystem), `entry_point`, `spine_membership`, `org_version` | 1374 nodes; 287 entry points; 129 spine members |
| E2 → L2 | `SubsystemNavigator` ×19, role `CANDIDATE`, provisional names, ai_summary + responsibilities + caveats (authored, dossier-grounded), numeric fields copied from dossiers, sha256 dossier fingerprints | 19 nodes + 1374 `CONTAINS_MEMBER` |
| E3 → L1 | `NavigationMaster`: system summary, 6-group `subsystem_index` (fan-out ≤9), mandated checklist, 4 global caveats | 1 node |

## Gates (E4)

- **Reachability**: 1374/1374 files reachable from L1 in **2 hops** (gate: ≤3). PASS
- **Fan-out**: 6 groups at L1, max 6 subsystems per group, ≤9 everywhere. PASS
- **Conformance** (E1 math-by-script): sub-11 service median reproduced the frozen recipe's
  0.94 exactly. PASS
- **Grounding**: L2 numeric fields are program-copied from dossiers; prose spot-checks
  (16/11/17) match dossier evidence. PASS
- **Provenance**: `clue_version 'erdos-v1'`, generator, timestamp, sha256 dossier
  fingerprint on every written node. PASS

## Token economy (the reason this layer exists)

Answering "what does the billing area do and where do I enter" now costs L1 (~350 words) +
one L2 (~120 words) ≈ **~600 tokens**, against reading a 208-file subsystem. The whole
navigation layer (L1 + 19×L2) is ≈ 3.5k tokens for a ~430k-LOC estate.

## Honesty ledger (what this layer is NOT yet)

- Subsystems are **pre-curation candidates**: 16/2 merges, 13 retype, 17 split are flagged in
  their own caveats and pending GrothendieckV5.
- The scan is stale (11 coverage gaps) — carried as caveats on L1 and affected L2s, not hidden.
- Names are `PROVISIONAL` — curation confirms or renames (F65 law applies then).

## Learnings for the reindex round (what to change in the 3 prompts)

1. `role:'CANDIDATE'` + `name_status:'PROVISIONAL'` worked cleanly — GrothendieckV5's P4
   should UPDATE these nodes (supersede provisional→curated) rather than create parallel ones.
   Add that line to the prompt.
2. The nav-layer rebuild here was delete-and-recreate (acceptable for the first full run);
   delta mode MUST switch to bi-temporal supersession per the prompt — never repeat the
   delete in delta.
3. Dossier fingerprints must be content-stable (sha256, fixed in this run) — Python `hash()`
   is process-salted and broke comparability. Worth a line in ErdosNavigator E4.
4. The E1 conformance assert (recompute vs frozen recipe) is cheap and caught the semantics
   bug earlier in the day (inversion direction) — keep it mandatory.
