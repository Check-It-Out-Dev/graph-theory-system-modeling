# Data card — what CodeMap Remote reads, stores and forgets

Last revised 2026-09-16.

## Sources

| data | origin | licence / ownership |
|---|---|---|
| Graph pack (`entities.csv`, `edges.csv`, `hyperedges.csv`, `l1_master.json`, `l2_navigators.jsonl`, `mfq.jsonl`, `codemap.lbdb`) | the checkItOut backend and frontend repositories (public mirrors `Check-It-Out-Dev/checkitout-backend`, `checkitout-frontend`), indexed at the commits in `manifest.json: indexed_sha` | the owner's code; the pack holds names, paths, types, edges and prose about structure — never file bodies |
| Curation notes | written by the delta pipeline from `/codemap` decisions on graph-repo issues | public, append-only |
| Bank and probes | authored for CodeMap v1 (`eval/scripts/`) and this arc | public |
| Persona conversations | six Claude Code agents run by the owner on the owner's subscription | synthetic users; no real person's data |
| Telemetry events | the server, one line per request | the owner's |

## Collection

The server emits one event per request (`remote/telemetry.py`, schema 1): timestamp, user id
(from the enum), tool, tier, terminal, the question and answer text, pointer names, token counts,
credits, session ids. Feedback and miss reports are events too. Personas' own Claude Code
telemetry (OTLP) goes to Grafana Cloud with `role`/`persona` attributes, account id excluded.
The delta pipeline collects nothing from users: it reads public repositories in GitHub Actions.

## PII & consent

Users are personas or team members who chose a name from a public enum; there are no accounts,
passwords or emails in the system. Questions and answers are about code; the code itself is the
owner's. Personas are the owner's own agents. Claude Code's telemetry is sent with
`OTEL_METRICS_INCLUDE_ACCOUNT_UUID=false`. Grafana Cloud dashboards are public by design and
show aggregates by user name; no question text reaches a public panel (Loki lines are not
public). The EU AI Act Art. 50 transparency line is stated where a user meets the system
(`remote/README.md`, the MCP `initialize` instructions).

## Retention

| store | where | kept |
|---|---|---|
| `events.jsonl` | VPS Docker volume | indefinitely until pruned by the owner; conversation sessions pruned after 7 days (`remote/housekeeping.py`) |
| Loki / Prometheus | Grafana Cloud stack 1359921 | per the Grafana Cloud plan's retention |
| Nightly artifacts (`eval/*/runs/`) | git, public | indefinitely (scrubbed fixtures only under `telemetry/fixtures/`) |
| Ledger (`graph/ledger/`) | git, public | never deleted; bi-temporal supersession |
| Pack Releases | GitHub Releases | every version kept |
| Modal volume `codemap-train` | Modal | optimiser run artifacts; deleted on request |

## Splits

The bank is not split for the judge (calibration is against the oracle, not a held-out set). The
optimiser draws train and validation examples from the oracle-checkable bank rows plus probes,
disjoint, seeded (`eval/optimize/adapter.py: dataset`); invalidated rows are excluded.
Personas' nightly questions are bank slices plus probes plus live tasks; the 20 % baseline sample
shares the night's seed.
