# Navigator prompt versions

Every version is an immutable file `v<N>.md` built by `tools/prompt/build_navigator.py` from
`template.md` (the prose under optimisation), the pack (L1 index, L2 navigators — data, never
optimised) and `curation_notes.md`. `prompt_version` on every event is `nav@` + sha16 of the file.

| version | built | parent | template sha16 | why | gate |
|---|---|---|---|---|---|
| v1 | 2026-09-16 | — | (see build output) | seed: the API navigator prompt's protocol + the big tier's graph lesson, rewritten for pointers | n/a (baseline) |
