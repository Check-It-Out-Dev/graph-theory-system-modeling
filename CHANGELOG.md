# Changelog

All notable changes to the Graph Theory System Modeling project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 1.2.0, CodeMap Remote

### Added
- Prompt under test, S6 closed: the judge calibrated on 12 anchors (8 baseline runs + 4 degraded variants) against blind reference grades; every gap argued in `judge/reference-scores.json`; rubric r5 (base code in the judge's input, design anchors for duplication and structural breaks) re-measured on the box (Actions run 35986676711): design-fit agreement 0.58 → 0.92 exact, Spearman 0.65 → 0.97, judge noise halved; `put_calibrate extend|score`, `run.py rejudge`, the anchor bench page (D-R29k)
- Prompt under test, S7: the seed's baseline on the self-hosted runner (Actions run 35885709905): score 0.911, delta 0.030, 18/18 hidden tests, `tests_written` 12/18 the main gap (D-R29j)
- Prompt under test, S12: campaign telemetry (`put_telemetry.py`: Claude Code OTLP by role and campaign, per-rule gauges) and the seventh dashboard `codemap-prompt-under-test` (D-R29i)
- Prompt under test: the evidence plane proven by `box-smoke-1` (Actions run 35884706660 on the self-hosted runner), `put_evidence.py` and `RUNS.md`, the runbook, the calibration tool, the PR-reviewer contract sketch, the method sections of `docs/09` (D-R29h)
- Prompt under test, S8 (ahead of S7): `run.py` with the modes smoke, baseline, gepa, certify and report; score, statistics, plateau stopper, GEPA adapter with its guard, certification with the ceiling test, promotion gate, reports; `eval/put/METRICS.md` (D-R29g)
- Prompt under test, S6: the blind judge (`put_judge.py`, rubric r4: correctness, convention fit, design fit, test quality, graph use) (D-R29f)
- Prompt under test, S5: the deterministic checks (`put_checks.py`, `put_diff.py`), 26 fixtures from real worktrees (`put_fixtures.py`: ten references, sixteen single-rule breakages) and `test_put_checks.py` (D-R29e)
- Prompt under test, S4: the coding runner (`put_runner.py`: worktree, restricted session with a Bash allowlist, output-token budget, diff, after/ snapshot, own + hidden + pass-to-pass tests, calls.json) and the prompt renderer with its checksum (`put_prompt.py`); first real run `runs/s4-smoke` (D-R29d)
- Prompt under test, S3: hidden acceptance tests and a reference solution per task, validated fail-on-base / pass-on-reference twice for all ten (40/40 arms); `put_tasks.py`, `put_worktree.py`, `put_maven.py` (D-R29c)
- Prompt under test, S2: instance `backend-conventions`: seed conventions manual v1 (16 rule ids), `contract.json` (predicates, weights, statistics, runner), ten tasks from the real code split 6/4 with every rule tag on both sides, `put_contract.py` + `test_put_contract.py` (D-R29b)
- Prompt under test, S1: the rolled-back ✅ row reverted on the branch, the success story moved to `docs/history/` as superseded, `eval/put/` skeleton and its tracking rules (D-R29a)
- Prompt under test, S0 (`applications/CodeMap/eval/put/probes/`, `docs/09-prompt-under-test.md`): Sonnet and Opus ids pinned, Bash confinement probed (a narrow allowlist is required), worktree and build timings, a self-hosted runner in a group restricted to one workflow on the backend branch `ci/prompt-eval` (D-R29)
- GEPA run `2026-09-17-gepa2` over Erdős's manual: the candidate from iteration 3 scores 0.871 against the seed's 0.776, better on all five problems (finding gaps by listing, a four-kind reading list, grounded negative claims); a behaviour-only reflection prompt, 40-minute reflections, one run per problem per batch, `--finalize` for a stopped run (D-R28)
- Prompt evaluation for Erdős: deterministic adherence checks per manual rule (`eval/erdos/erdos_adherence.py`), a one-answer judge rubric r3 with the query trace (correctness, completeness, architecture fit, graph use), and GEPA optimising the manual against them, seeded with manual 2.1 and a judge-noise measurement (D-R27)
- Erdős 2.1 and judge rubric r2: a required graph pass before any file tool, answer keys with code-built architecture sections, a judge that scores architectural fit and neutralises arm-revealing wording (D-R26)
- Erdős 2.0 (`remote/ladybug_mcp.py`, `.agents/skills/erdos-architect/`): direct read-only Cypher over the pack, an XML operating manual delivered as CLAUDE.md with a loading preflight, and a one-pass task card for both arms (D-R25)
- Erdős pairs 2026-09-17 (`eval/erdos/`): five complex problems with code-built answer keys; Erdős spent 0.42 of the general agent's solve tokens and 0.59 of its time but was judged worse on all five (overall 3.0 against 4.2) (D-R24)
- `.agents/`: LadybugDB-native agent skills (`ladybug-graph`, `hypatia-indexer`, `grothendieck-organizer`, `erdos-architect`) in the Agent Skills format, generated Claude Code shims, schema, tool contract and graph map (`tools/agents/sync_agents.py`); the proposal reviewer reads the whole Grothendieck skill (D-R23)
- `build_release.py` keeps the commits a pack indexes; `pack-1.1.1` records both public main heads (D-R22)
- ~~The README row "Measuring the quality of an AI system in production" is ✅ (D-R21)~~ — withdrawn by the owner 2026-09-17 (the pairs measured the navigator, not the graph); the row is back to 🟡 and the story lives in `applications/CodeMap/docs/history/08-success-story.md` as history (arc 5, S1)
- Campaign nights plan past the budget (`--conversations`); a baseline runs only when the day's budget still pays for its partner (D-R20)
- Pair campaign: a skipped partner is no pair; the runner schedules paired conversations first per persona so the daily budget reaches the partner (D-R19)
- Judge calibration reports raw agreement, prevalence and Gwet's AC1 beside Cohen's κ; a second verdict route for a skewed oracle, named in the artifact (`eval/judge/calibrate.py`, D-R18)
- Full reindex on the box (`graph/delta/{fullscan,reindex}.py`, `extract.py --full`): both repositories at their public main heads, Grothendieck in chunks, apply, reclue with a refined gate, drift, Release `pack-1.1.0` served by the VPS; the oracle answers WHERE only; the night gains a bank pass and paired baselines (arc 3, R1–R2)
- `applications/CodeMap/remote/`: the served MCP (Streamable HTTP JSON-RPC, bearer token, user enum, one event line per call, pointers instead of paths) with a socket-free test suite (S1)
- The navigator tier: Claude Sonnet on the subscription drives the engine through a loopback MCP and answers with pointers; conversations resume their Claude session; `prompts/navigator/v1.md` is built from the pack by `tools/prompt/build_navigator.py` (S2)
- Credits without currency, daily budgets rebuilt from the events file, `GET /metrics` (Prometheus text, OTel GenAI semconv), Influx and Loki push to Grafana Cloud, a 200-row replay fixture (S3)
- `codemap_feedback` (strict validation, verified ratings), `codemap_miss` (the saturation backlog), `codemap_search` (Qwen3 embedding + reranker on Modal over a per-entity socket; index cached per pack version), `POST /feedback` (S4)
- The graph pack as a GitHub Release (`pack-1.0.0`): `tools/pack/build_release.py` + `fetch_pack.py` (sha256-verified); CI fetches it and runs the engine check and the remote suites (S5)
- Hosted on the demo VPS at codemap.checkitout.app: Dockerfile, compose, nginx vhost, `tools/deploy.sh`, dispatch-only `deploy-codemap.yml`; pack reload without restart (`POST /admin/reload` + Releases poll); the active prompt built from template + pack + notes (S6)
- Six synthetic users (`eval/humans/`): role files, bank slices, 30 off-distribution probes, the deterministic night runner with seeded no-CodeMap baselines, Claude Code OTEL to Grafana Cloud, `.claude/agents/` copies; first live conversation recorded (S7)
- Six public Grafana Cloud dashboards as code (`observability/grafana/`, builder + lint + provision, public URLs in `public-urls.md`) (S10)
- The judge (`eval/judge/`): Claude batched against a five-dimension rubric, execution-oracle calibration κ 0.84 on the where-archetypes, frozen anchors with drift detection, the Qwen reranker as a reported second family; `bank_pass.py` for calibration answers (S8)
- Quality rates (`eval/quality/quality.py`): one artifact per night with the dashboard metric names, gains vs the baselines, fixture-replayed in CI, pushed as `codemap_quality_*` (S9)
- Delta digestion, the deterministic half (`graph/delta/{extract,discover}.py`, `repos.json`, `graph-delta.yml`): eligibility, entity heuristic, content fingerprints, structural edges, purge, churn threshold, LadybugDB rebuilt from CSVs (S11)
- Grothendieck on the pull request, as an issue: deterministic candidates + subscription review through a read-only pack MCP + an id checker (`graph/delta/{pack_mcp,propose,issue}.py`); `/codemap accept|move|new-subsystem|reject` applies (`apply.py`, `ci_apply.sh`): bi-temporal ledger rows in `graph/ledger/`, append-only curation notes in the navigator prompt, narrow FAQ invalidation shipped in the pack, `prompts/navigator/active.md`, a new pack Release and a pull request; 48 h timeout job (S12)
- Version drift without a model (`graph/delta/drift.py`): the bank replayed by the engine on the pack before and after a decision, order-free canonical hashes, invalidated rows excluded; `graph/ledger/<version>.drift.json` from the decision job, `codemap_quality_version_drift_rate` on the AI-system dashboard; `find` now reports the curated subsystem (S13a)
- Reclue of the touched subsystems in the decision job (`graph/delta/reclue.py`: dossier from the pack, one tool-less `claude -p` per subsystem, mechanical gates on numbers and file names, snapshot in `graph/ledger/<version>.reclue.json`, untouched lines byte-identical); curation notes ride the pack so the VPS and the pull request build the same prompt (S13b)
- Saturation measured: the extract job's `coverage` (indexed ∩ eligible / eligible per repository) rides the ledger row and the nightly artifact (`codemap_quality_graph_coverage_ratio`), beside the misses counter on the AI-system dashboard (S13c)
- GEPA over the navigator template (`eval/optimize/`): adapter on the real navigator path scored by the execution oracle, constraints that refuse a broken contract, one artifact per run, a promotion gate that writes `v<N>.md` + PROMPT_LOG + PR only on a real win, and the same run inside a Modal CPU container on the subscription (S14)
- Navigator prompt v2, the first promotion: GEPA run 2026-09-16b on Modal (6/6 vs 5/6), confirmed on a fresh twelve-example split (12/12 vs 11/12), PR #6 (S14b); the gate now needs ≥ 18 examples behind a decision (validation + `promote.py --confirm N` on a fresh split) and refuses a regression on the confirmation (S14c)
- Governance docs: MODEL_CARD, EVAL_CARD, DATA_CARD, THREAT_MODEL (OWASP LLM Top 10 2025 → controls), INCIDENTS, the NIST AI RMF map in docs/07, the Art. 50 transparency line, and the `governance-docs` CI gate (`tools/ci/check_governance.py`) (S15a)
- The public quality page (`tools/pages/build_quality.py` → Pages `/quality/`, `nightly.yml`: committed artifacts only, inline SVG, dark and light), README sections in both READMEs with the owner's security paragraph, the team story, the dashboard links and the cards (S15b)
- First full night (2026-09-17): six personas, 18 conversations, 58 asks, 11 misses; judge repeatability and the oracle on rephrased questions recorded as open findings; baselines now scheduled with their CodeMap partner; the pack directory left git (S16)

## [1.0.0] - 2025-09-16

### Added
- Initial release of 6 research papers documenting the methodology
- Complete documentation of the two-stage discovery process (HoTT bootstrap → Graph Theory refinement)
- CheckItOut platform case study with 426 Java files
- NavigationMaster pattern implementation
- 6-Entity behavioral pattern framework
- Comprehensive README with setup instructions
- MIT License for methodology and research
- COMPLIANCE.md explaining legal usage of all tools
- DEVELOPMENT_SETUP.md for team adoption guide
- .gitignore for sensitive data protection

### Methodology Achievements
- Discovered 7 business modules from 20 initial candidates using HoTT/embeddings
- Achieved 73% reduction in AI hallucination rates
- Demonstrated 30-40% improvement in developer productivity
- Reduced onboarding time from 2-3 weeks to 2-3 days
- Created O(1) access patterns through NavigationMaster hub

### Technical Specifications
- Processing speed: 10-20 files/minute (reading), 5-10 files/minute (semantic analysis)
- Graph size: 24,030 nodes, 87,453 relationships for 426 files
- Query performance: <50ms for 3-hop traversals
- AI context windows used: 30 (Claude Sonnet 4) + 3-4 (Claude Opus 4.1)

### Papers Published
1. **Living Documentation HoTT Graph Theory** - Mathematical foundations
2. **Deep Behavioral Modeling** - 6-entity pattern discovery
3. **How to Start for Free** - Neo4j Community Edition guide
4. **On Demand Real Example** - CheckItOut case study
5. **How to Add Seat Model** - AI-driven feature design
6. **Win-Win for Customers and AI Providers** - Business benefits

### Compliance
- Clarified transition from personal research to team deployment:
  - **Research Phase**: Norbert Marchewka used Neo4j Desktop Enterprise with native embeddings (single user, evaluation license)
  - **Team Phase**: Migration to Neo4j Community Edition with separate embedding service (multi-user, GPLv3)
- Documented that Enterprise features were ONLY used on architect's personal computer
- Emphasized complete separation of embeddings from Neo4j in team deployment
- Added comprehensive legal compliance documentation
- Made clear NO Enterprise features are shared with or used by the development team

## [0.9.0] - 2025-09-01 (Pre-release)

### Added
- Initial research using Neo4j Desktop Enterprise trial
- HoTT-based clustering algorithm implementation
- Proof of concept with CheckItOut platform

### Changed
- Refined from 20 subsystem candidates to 7 business modules
- Optimized embedding generation pipeline

### Discovered
- 6-entity pattern emerges universally across subsystems
- Ramsey theory R(3,3)=6 explains pattern prevalence
- Friendship Theorem optimal for navigation topology

## [0.8.0] - 2025-08-01 (Research Phase)

### Added
- Initial graph theory exploration
- Category theory application to code structure
- Sheaf theory for local-global relationships

### Experimental
- Various clustering approaches tested
- Multiple embedding models evaluated
- Different graph topologies analyzed

## Future Roadmap

### [1.1.0] - Planned Q4 2025
- [ ] Language-specific analyzers (Python, JavaScript, Go)
- [ ] Automated CI/CD integration
- [ ] Cloud deployment templates

### [1.2.0] - Planned Q1 2026
- [ ] Multi-repository federation
- [ ] Cross-language dependency tracking
- [ ] Real-time graph updates from Git hooks

### [2.0.0] - Planned Q2 2026
- [ ] AI agent marketplace integration
- [ ] Automated architecture optimization suggestions
- [ ] Predictive refactoring recommendations

---

For more details on each release, see the [GitHub Releases](https://github.com/yourusername/graph-theory-system-modeling/releases) page.
