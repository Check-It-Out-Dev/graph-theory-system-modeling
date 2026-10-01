# LadybugDB dialect laws for the CodeMap pack

Hand-written. Every law says where it was earned. "Probed" means run against `real_ladybug` 0.15.3 on pack 1.1.1 (2026-09-17); "docs" means taken from the LadybugDB documentation (docs.ladybugdb.com) and not yet probed here.

## Laws measured on this graph

1. **Package and API.** `import real_ladybug`; `Database(path, read_only=...)` and `Connection(db)`, Kùzu style. `CALL db_version() RETURN *` answers `0.15.3`. (probed)
2. **Typed edges are one table.** Every relationship is a row of `Dep` with a `rel` string; filter with `r.rel = 'X'`. The 17 kinds: ACCESSES, AFFECTS, APPLIES_IN, CALLS, CONFIGURED_BY, CONSTRAINS, EXTENDS, IMPLEMENTS, IMPORTS, INITIATES, INJECTS, MODIFIES, PERFORMS, TESTED_BY, TRIGGERS, USES, VALIDATES. (probed)
3. **Conditional aggregates are wrong.** `count(CASE WHEN b THEN 1 END)` returned 1 where `count(*) ... WHERE b` returned 300 (bisected 2026-09-03). Plain `CASE`, `count(DISTINCT)` and key-grouped `count(*)` are correct; run conditional counts as separate `WHERE`-filtered queries.
4. **Primary key is `file_path`.** Basenames collide (UnifiedStorageConfiguration.java, HashingUtilUnitTest.java); edges whose endpoint name is ambiguous were skipped at import and counted.
5. **COPY into a relationship table** takes the columns in the order FROM, TO, properties; COPY paths use forward slashes on Windows (a backslash is a parser escape).
6. **Bound parameters are typed from their value.** A bound `None` or empty list cannot be typed (leave the property out); a bound string that looks like a list or struct is silently re-serialized (send JSON as an escaped literal); a list bound into a STRING column is cast the same way (`json.dumps` it first); `NULL` in `SET` works only as a literal. (`graph/authoring/ladybug_store.py`)
7. **No `id()` identity to rely on.** Use `file_path` in the pack (`nid` in the authoring store). (store header)
8. **No variable-length paths across different relationship tables.** Pull each table and combine in Python (`diag_state.py`, `reclue_gates.py`). Within `Dep`, `[r:Dep*1..2]` works. (probed)
9. **Determinism.** Scan order is not guaranteed: sort every pull and break top-N ties by name.
10. **Types differ between stores.** `entry_point` is a BOOLEAN in the pack and a STRING in the authoring store; list-valued navigator fields (`spines`, `entry_points`, `contracts`) are JSON strings, some encoded twice.
11. **Membership** is `coalesce(e.curated, e.subsystem)`: `curated` is the decided subsystem, `subsystem` the measured one.
12. **One statement per call** in the guarded tools; substring search is `lower(x) CONTAINS 'y'`, regular expressions use `=~` (probed).

## Verified features (probed 2026-09-17)

- Schema introspection: `CALL show_tables() RETURN *`, `CALL table_info('Entity') RETURN *`, `CALL table_info('Dep') RETURN *`.
- `label(e)` and `labels(e)` both return `Entity`.
- List functions are 1-based: `list_sort(collect(e.layer))[1]`.
- `EXPLAIN <statement>` returns the plan without running it.
- A read-only database raises `Cannot execute write operations in a read-only database!` on `CREATE`.

## Differences from Neo4j (docs, not all probed)

- A node has exactly one table (label). `(a:A:B)` means A or B. There is no APOC and no GDS.
- Not supported: `REMOVE` (use `SET n.p = NULL`), `FOREACH` (use `UNWIND`), `SET n += map`, general `CALL {}` subqueries (only `EXISTS`/`COUNT` subqueries), `LOAD CSV` (use `LOAD FROM` or `COPY FROM`).
- Variable-length patterns default to at most 30 hops with walk semantics (edges may repeat); shortest paths are `* SHORTEST 1..n` and `ALL SHORTEST`.
- `MERGE` supports `ON CREATE` and `ON MATCH`; a node `MERGE` must include the primary key.
- Parameters are `$name`, passed as `execute(query, parameters={...})`; table names cannot be parameters.
- Transactions: auto-commit or `BEGIN TRANSACTION [READ ONLY]` / `COMMIT` / `ROLLBACK`; one write transaction at a time.
- Extensions load per session (`INSTALL x; LOAD x;`): full-text search (BM25), vector indexes (`CREATE_VECTOR_INDEX`, `QUERY_VECTOR_INDEX`) and graph algorithms (k-core, Louvain, PageRank, SCC, WCC; no Leiden). None is used by the pack today.

## The generated spike notes (verbatim, from `graph/scripts/import_ladybug.py` via `graph/pack/DIALECT_NOTES.md`)

- package: real-ladybug (PyPI); import real_ladybug; API Database/Connection (Kuzu style)
- typed rels -> single Dep table with rel STRING property; queries use r.rel = 'X' instead of [:X]
- COPY rel tables requires (FROM, TO, props) column order
- COPY paths must use forward slashes on Windows (backslash = parser escape)
- node PK = file_path (2 duplicate basenames found in 1374: UnifiedStorageConfiguration.java x2, HashingUtilUnitTest.java x2); ambiguous-name edges skipped and counted
- gold M03 fingerprint REPRODUCED on v0.15.3
- CASE WHEN inside an aggregate (count/sum) returns wrong numbers (bisected 2026-09-03: count(CASE WHEN b THEN 1 END) = 1 where count(*) WHERE b = 300); plain CASE, count(DISTINCT) and key-grouped count(*) are all correct — conditional counts must be separate WHERE-filtered queries
