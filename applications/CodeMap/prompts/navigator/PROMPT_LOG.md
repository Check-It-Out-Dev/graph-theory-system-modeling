# Navigator prompt versions

Every version is an immutable file `v<N>.md` built by `tools/prompt/build_navigator.py` from
`template.md` (the prose under optimisation), the pack (L1 index, L2 navigators — data, never
optimised) and `curation_notes.md`. `prompt_version` on every event is `nav@` + sha16 of the file.

`active.md` is the prompt of the latest pack Release: the same template rendered over the pack the
server is running, plus every curation note so far. The delta pipeline regenerates it in the pull
request that follows each `/codemap` decision; CI checks it is fresh. Versions `v<N>.md` change only
when the template changes (a promotion).

| version | built | parent | template sha16 | why | gate |
|---|---|---|---|---|---|
| v1 | 2026-09-16 | — | (see build output) | seed: the API navigator prompt's protocol + the big tier's graph lesson, rewritten for pointers | n/a (baseline) |
