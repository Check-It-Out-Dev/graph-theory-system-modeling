# Ladybug dialect notes (spike)

- package: real-ladybug (PyPI); import real_ladybug; API Database/Connection (Kuzu style)
- typed rels -> single Dep table with rel STRING property; queries use r.rel = 'X' instead of [:X]
- COPY rel tables requires (FROM, TO, props) column order
- COPY paths must use forward slashes on Windows (backslash = parser escape)
- node PK = file_path (2 duplicate basenames found in 1374: UnifiedStorageConfiguration.java x2, HashingUtilUnitTest.java x2); ambiguous-name edges skipped and counted
- gold M03 fingerprint REPRODUCED on v0.15.3
- CASE WHEN inside an aggregate (count/sum) returns wrong numbers (bisected 2026-09-03: count(CASE WHEN b THEN 1 END) = 1 where count(*) WHERE b = 300); plain CASE, count(DISTINCT) and key-grouped count(*) are all correct — conditional counts must be separate WHERE-filtered queries
