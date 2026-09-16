# Partition ledger

One JSON row per applied `/codemap` decision, named by the pack version it produced
(`1.0.1.json`, `1.0.2.json`, ...), written by `graph/delta/apply.py` from the decision job of
`graph-delta.yml`. Rows are bi-temporal: an assignment carries `t_valid`, and when a later decision
moves the same entity again the earlier row's assignment gets `t_invalid` and `superseded_by`
instead of being rewritten. Nothing here is ever deleted; the current partition is the pack, the
history is this directory, and the reasons are the curation notes in the navigator prompt.
