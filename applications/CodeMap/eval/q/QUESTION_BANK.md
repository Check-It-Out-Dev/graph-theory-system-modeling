# CodeMap question bank v0 — questions only (gold lands in Q1)

Method: three recon miners (BE repo, FE repo, graph/dossiers) + literature sweep. Every
question carries aliases (incl. one Polish paraphrase — router near-miss training data), a
role, a stratum (G1 keyword-findable / G2 structural 2–4 hops / G3 hidden, no lexical
overlap), and later a worth score 1–3. No answers here by design.

## 1. What people ask about codebases — the general taxonomy (literature-grounded)

**Sillito, Murphy & De Volder (FSE 2006)** — 44 question types from observed programmers, in
four groups that map cleanly onto our graph levels:
1. *Finding focus points* ("where is X?") → `locate` archetype, G1.
2. *Expanding focus points* ("what calls this? what does this call?") → `flow`/`impact`, G2.
3. *Understanding a subgraph* ("how do these types work together?") → `overview`/`flow`, G2.
4. *Questions over groups of subgraphs* ("how does this feature relate to that one?") →
   `boundary`/`health`, G2/G3.

**LaToza & Myers (PLATEAU 2010)** — 179 developers, 371 hard-to-answer questions, 21
categories, 94 distinct questions. The #1 category: **intent and rationale** ("why was it
done this way?") — hard precisely because code alone cannot answer it. Consequence for us:
the `rationale` archetype answers ONLY from stored decisions/docs and says "not recorded"
otherwise. #2-class categories: **implications of a change** ("what will break?") and
debugging counterfactuals — our `impact`/`cohort` archetypes, where the graph (reverse deps +
co-change hyperedges) is strongest.

**Modern AI-assistant usage (2025–26 studies)**: developers reach for the assistant first for
explaining unfamiliar/legacy code, error-first debugging, API clarification; in 100K+ LOC
codebases ~80–90% of developer time is READING code; onboarding with codebase-QA tooling
compresses from weeks to days. Consequence: onboarding-path and overview questions are not
nice-to-have — they are the volume market of the product thesis (doc-00 §1: "programiści
zadają te same pytania setki razy").

Canonical general templates (each becomes many instances):
| # | template | archetype | stratum |
|---|---|---|---|
| T1 | Where is X implemented / which file handles X? | locate | G1 |
| T2 | What happens when ACTION (trace the flow)? | flow | G2 |
| T3 | What breaks / is affected if I change X? | impact | G2/G3 |
| T4 | What does X depend on / what does it call? | flow | G2 |
| T5 | Who calls X / where is X used? | impact | G2 |
| T6 | What changes together with X (historically)? | cohort | G3 |
| T7 | What does subsystem S do? What exists in this system? | overview | G1 |
| T8 | How do S1 and S2 talk to each other (contract)? | boundary | G2 |
| T9 | What is the minimal reading path to understand S? | onboarding_path | G2 |
| T10 | Why is X done this way? | rationale | G3 (honesty rule) |
| T11 | Where are the coupling hotspots / risky boundary files? | health | G3 |
| T12 | How is X configured / where does setting Y live? | locate | G1 |
| T13 | How is X tested / what covers it? | boundary (TESTED_BY) | G2 |
| T14 | Is there already code that does X (avoid duplication)? | locate+cohort | G1/G3 |
| T15 | What is NOT here (feature exists? integration exists?) | overview | G1 |

## 2. Codebase-specific banks (miner output — 104 questions, raw/*.jsonl)

| miner | n | grounding | notable |
|---|---|---|---|
| `raw/qminer_be.jsonl` | 37 | packages, 9 CronJobs, 13 filters/interceptors, ~40 controllers, docs/ | the G3 symptom set: "valid login but 403 everywhere", "accounts vanish after signup", "backend won't boot after DB restore" — all lexically invisible causes (filters, cleanup crons, boot guard). Miner's note says 38; 37 arrived — one lost in its own chunking, accepted. |
| `raw/qminer_fe.jsonl` | 34 | app.routes.ts, app.config.ts, 8 interceptors, generated api (277 files), e2e tiers | G3 gems: Set→`{}` serialization trap, api-frozen orphan types, event-replay hydration, the iter-107 language pin (a true `rationale` question with a recorded source) |
| `raw/qminer_graph.jsonl` | 33 | dossiers + 3 read queries (edge census: cross-sub IMPORTS 1925, INJECTS 292, ALGEBRA_VIOLATION 97) | all G2/G3 — the only-a-graph-can-answer set: blast radii (UserRepository 141 in-edges), articulation points ('api' carries all 150 FE→api edges), cohorts, layer inversions, public→payment reachability |

## 3. Worth-answering analysis for checkItOut (the MFQ-100 seed rule)

Worth = product-thesis fit (asked repeatedly: onboarding/overview) × graph-unique value (G3s
grep cannot answer) × role breadth. Twelve must-answer exemplars (worth=3), spanning all
archetypes:

1. The one-walk system map — every subsystem: name, size, entry point (`overview`, the L1 demo).
2. Where do I start reading the FE / minimal reading path for subscriptions (`onboarding_path`).
3. Blast radius of UserRepository.java — 141 in-edges (`impact`).
4. Valid login but 403 on everything — filter chain, no file named "403" (`flow`, G3).
5. Accounts vanish days after registration — NoConsentAccountCleanupCronJob (`flow`, G3).
6. Multi-select field arrives as `{}` — set-to-array interceptor (`locate`+`flow`, G3).
7. Which emails does the subscription lifecycle trigger (`boundary`, cross-subsystem).
8. What can reach payment resources from a public controller in ≤3 hops (`impact`, security).
9. What co-changes with address.service.ts — hyperedge cohort (`cohort`, G3).
10. HTTP interceptor order and why it matters (`overview`, has a recorded rationale comment).
11. Which cron jobs mutate user data unattended (`health`+`flow`, security).
12. Which subsystem has the worst cross-boundary coupling — sub-16 at 1.0 (`health`).

Seed rule for the remaining ~88 of MFQ-100: take all 104 mined questions, drop near-duplicates
across miners (~8 overlap pairs, e.g. step-up auth appears in BE and FE forms — keep both only
if the gold differs), keep every G3 (33 — they are the product's reason to exist), then top up
G1 during gold-authoring until the strata mix reaches ~40/35/25 (mined mix is 29/39/32 — G1s
are cheapest to author, so balancing happens there, not by deleting G2/G3).

## 4. Balance check (measured on the 104)

Strata: G1 30 (29%) / G2 41 (39%) / G3 33 (32%) — G1 under target by design, topped up in Q1.
Roles: dev 32, architect 22, security 16, onboarder 12, pm 11, em 6, qa 5 — every role ≥5. ✓
Archetypes: all nine covered; `rationale` is intentionally thin (2 questions with recorded
sources — the honesty rule says rationale without a stored source answers "not recorded").
Coverage: BE subsystems, FE, graph-health — every dossier-flagged subsystem (17, 13, 18, 16, 2)
has ≥1 dedicated question. Aliases: 3 per question incl. one Polish paraphrase → 312 alias
pairs for router near-miss training.
