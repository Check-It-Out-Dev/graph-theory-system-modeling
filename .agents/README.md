# `.agents/` — the CodeMap graph agents

One source of truth for the agents that build, organise and use the CodeMap code graph of checkItOut, stored in LadybugDB. Each agent is an [Agent Skills](https://agentskills.io/specification) folder: a `SKILL.md` with `name` and `description` frontmatter, and `references/` loaded on demand.

| skill | agent | what it does |
|---|---|---|
| [`ladybug-graph`](skills/ladybug-graph/SKILL.md) | shared | the pack's schema, the Cypher dialect laws, the read-only tools and verification recipes; every agent loads it first |
| [`hypatia-indexer`](skills/hypatia-indexer/SKILL.md) | Hypatia | keeps the graph in step with the code: scan, extract without a model, verify by read, record the indexed commits |
| [`grothendieck-organizer`](skills/grothendieck-organizer/SKILL.md) | Grothendieck | decides where new entities belong, with evidence; proposals become ledger rows and curation notes only through a decision |
| [`erdos-architect`](skills/erdos-architect/SKILL.md) | Erdős | solves hard problems and proposes architectural changes, navigating with the graph first and the files second |

## How each harness loads them

| harness | reads |
|---|---|
| Claude Code, Cursor, VS Code | `.claude/agents/<agent>.md`: generated shims that point at the skill files |
| Codex, GitHub Copilot, Cursor, Gemini CLI, OpenCode, Windsurf, Amp | `.agents/skills/` directly (the cross-client skills location) |
| scripts (`claude -p`) | the skill files by path, for example `graph/delta/propose.py` sends the whole `grothendieck-organizer` skill to its reviewer |

## What is generated

Never edit these by hand; edit the source and run `PYTHONUTF8=1 python applications/CodeMap/tools/agents/sync_agents.py` (`--check` fails on drift, and the test suite runs it):

- `.claude/agents/{hypatia-indexer,grothendieck-organizer,erdos-architect}.md`, from each skill's frontmatter
- `skills/*/references/tools.md`, from the MCP servers and the verb table
- `skills/ladybug-graph/references/schema.md` and `skills/erdos-architect/references/graph-map.md`, from the pack (fetch it first with `applications/CodeMap/tools/pack/fetch_pack.py --latest`)

## Relationship to the V5 manuals

`applications/CodeMap/graph/prompts/{HypatiaV5,GrothendieckV5,ErdosNavigatorV5}.md` were written for the Neo4j authoring graph and stay as the record. Their database-independent rules (invariants, decision triggers, ledger entries) are carried into these skills with their provenance; their Neo4j plumbing is not. What is not LadybugDB-native yet is said inside each skill: the three-socket embeddings and hyperedge emission (Hypatia) and a full re-partition (Grothendieck).
