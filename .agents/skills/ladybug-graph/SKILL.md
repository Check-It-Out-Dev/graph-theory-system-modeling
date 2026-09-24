---
name: ladybug-graph
description: "Read, query and verify the CodeMap code graph of checkItOut stored in LadybugDB (the pack): its schema, the Cypher dialect laws, the engine and pack tools, read-only access and verification recipes. Load it before any agent queries or changes the CodeMap graph; hypatia-indexer, grothendieck-organizer and erdos-architect all start here. Not needed for work that only touches source files."
compatibility: "Python 3.12 with real_ladybug 0.15.3 (pinned; the maintained successor package ladybug 0.20.x uses a newer storage format). The pack is fetched by applications/CodeMap/tools/pack/fetch_pack.py."
metadata:
  version: "1.0.0"
  verified: "2026-09-17 against pack 1.1.1"
---

# The CodeMap graph in LadybugDB

The graph is a navigational skeleton of the checkItOut codebase (a Spring Boot backend and an Angular frontend): which files exist, what kind of thing each is, how they depend on each other, and how they group into subsystems. It holds names, paths, typed edges and short prose — never file bodies. Source code stays in the product checkouts; the graph tells you which few files to open.

## Where it lives

The pack is a directory, `applications/CodeMap/graph/pack/`, published as a GitHub Release (`pack-<version>`) and never committed:

| file | what it is |
|---|---|
| `codemap.lbdb` | the LadybugDB database: node table `Entity`, relationship table `Dep` |
| `entities.csv`, `edges.csv`, `edges_lb.csv`, `hyperedges.csv` | the same graph as CSV (the engine loads these in memory) |
| `l1_master.json` | L1: the subsystem index (groups and subsystems, one line each) |
| `l2_navigators.jsonl` | L2: one navigator per subsystem (summary, responsibilities, entry points, spines, contracts, caveats) |
| `curation_notes.md` | decisions taken on GitHub (`/codemap accept`, `move`, `new-subsystem`, `reject`), the newest word on where things live |
| `mfq.jsonl`, `INVALIDATED_delta.json` | the question bank and the rows a decision made stale |
| `manifest.json` | `pack_version`, `indexed_sha` per repository, counts, file hashes, `built_at` |

Three levels: L1 names about thirty subsystems, L2 explains each one, L3 is the entities and their edges. Read top-down: pick subsystems from L1, confirm with L2, then query entities.

The exact columns, edge kinds and counts are in `references/schema.md` (generated from the database). The dialect laws are in `references/dialect.md`. The tools are in `references/tools.md`.

## Access rules

- Open the database read-only unless you are the pipeline step that rebuilds it: `real_ladybug.Database(path, read_only=True)`. A read-only connection refuses every write, which is the protection you want.
- LadybugDB allows one read-write process or many read-only processes on a database directory, never both. Agents never write `codemap.lbdb` directly: the pipeline scripts rebuild it from the CSVs with `COPY` (`graph/delta/extract.py`, `apply.py`), and those scripts are the only writers.
- Identity: `Entity.file_path` is the primary key. Basenames are not unique (two duplicates exist), so write by `file_path`; the tools accept exact names for reading.
- Paths in the graph carry the authoring box's prefix (`C:/Users/Norbert/IdeaProjects/checkItOut-be2/` for the backend, `.../checkItOut-fe-greenfield/` for the frontend). Strip it to get the path relative to a checkout; `engine_open` does this for you.

## Tools

Two MCP servers expose the pack; both are read-only.

- **engine** (`applications/CodeMap/remote/engine_mcp.py`): `engine_step(dsl)` runs one verb (`map`, `enter`, `find`, `impact`, `flow`, `seam`, `cohort`, `spine`, `health`, `read`), `engine_cypher(stmt)` runs one guarded read-only statement, `engine_open(name)` turns an exact entity name into a pointer (repository, relative path, subsystem).
- **pack** (`applications/CodeMap/graph/delta/pack_mcp.py`): `pack_subsystem(id)`, `pack_entity(name)` (edges both ways and the neighbours' subsystems), `pack_folder(fragment)`, `pack_cypher(stmt)`.

Start either with `CODEMAP_PACK_DIR` pointing at the pack, for example:

```json
{"mcpServers": {"engine": {"command": "python", "args": ["applications/CodeMap/remote/engine_mcp.py"],
                           "env": {"CODEMAP_PACK_DIR": "applications/CodeMap/graph/pack", "PYTHONUTF8": "1"}}}}
```

Copy entity names and subsystem ids exactly from a tool result, the L1/L2 map or the task. An invented name returns nothing, and an empty result is then easy to misread as "no dependents".

## Querying well

1. Know the schema before you write a statement: `CALL show_tables() RETURN *` and `CALL table_info('Entity') RETURN *`.
2. Edge kinds are a property, not a label: `MATCH (a:Entity)-[r:Dep]->(b:Entity) WHERE r.rel = 'IMPORTS'`. Never `[:IMPORTS]`.
3. Always bound the result: `LIMIT`, or `count(*)` / `ORDER BY ... LIMIT` instead of listing.
4. Conditional counts are separate `WHERE`-filtered queries. `CASE` inside `count()` or `sum()` returns wrong numbers on this version.
5. Bound variable-length paths explicitly (`[r:Dep*1..2]`); the default upper bound is 30 hops and walks may repeat edges.
6. Scan order is not guaranteed: sort every result you compare or report, and break ties by name.

## Verification recipes

Each was run on pack 1.1.1 on 2026-09-17.

```cypher
CALL show_tables() RETURN *;                                         -- Entity (NODE), Dep (REL)
MATCH (e:Entity) RETURN e.curated AS sub, count(*) AS n ORDER BY n DESC LIMIT 5;
MATCH (a:Entity)-[r:Dep]->(b:Entity) RETURN r.rel, count(*) AS n ORDER BY n DESC;
MATCH (a:Entity {name:'AccountStatus.java'})<-[r:Dep*1..2]-(b:Entity) RETURN count(DISTINCT b);   -- 121
MATCH (e:Entity) WHERE e.name =~ '.*Consent.*Cron.*' RETURN e.name ORDER BY e.name LIMIT 5;
MATCH (a:Entity {name:$src})-[r:Dep]->(b:Entity {name:$dst}) RETURN r.rel;   -- does this edge exist?
```

Trust a write only after a read confirms it: a tool that reports success is not evidence that anything changed.
