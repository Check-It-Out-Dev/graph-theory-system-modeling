# CodeMap — the shape of the runtime (decision of 2 September 2026)

**Owner's decision: for the UI test phase we serve through the browser** — one local
application, no desktop shell. Tauri 2 (document 00, D7) remains the SHIPPING vehicle; it comes
later and wraps the same frontend without changes, which is exactly why Tauri was chosen.

## Test phase (now): one process, the browser as the window

```
┌──────────────────────────────────────────────────────────┐
│ codemap-app (one supervisor process, Python or Node)     │
│                                                          │
│  ├── HTTP :7345  → UI (static React build) + API         │
│  ├── llama-server (sidecar :7346, OpenAI-compatible)     │
│  │     small Qwen GGUF, CPU                              │
│  ├── real-ladybug (embedded, in-process — the .lbdb      │
│  │     file from the graph/pack/ package)                │
│  └── MCP server (stdio) — the SAME code as the API:      │
│        Claude Desktop/Code attach in parallel            │
│        ("one graph, two brains", document 00 week 2)     │
└──────────────────────────────────────────────────────────┘
browser → http://localhost:7345
```

- **No installer in the test phase**: `python -m codemap_app` (or `npx`); the browser serves
  as the window. UI iteration is plain Vite development.
- **The intelligence layer is shared**: recipes, the entry protocol and the MFQ cache live in
  one module served both over HTTP (for the UI and the small model) and over MCP stdio (for
  large models). One implementation, two transports; drift is impossible by construction.
- **Ladybug embedded in-process** — confirmed by the spike: `real-ladybug` 0.15.3, pack loaded
  (1374/4647), gold M03 reproduced its fingerprint exactly. Dialect notes:
  `graph/pack/DIALECT_NOTES.md`.

## Moving to shipping (later, without a rewrite)

Tauri 2 wraps the SAME React build; the llama-server sidecar carries over unchanged (the
official pattern); Ladybug stays embedded; MCP stdio stays. The difference is the shell,
signing and the updater (document 00, section 7, week 4). Mobile (chapter two): sidecar →
FFI/XCFramework.

## What this changes in the plan

- The week-2 "Tauri skeleton" from document 00 splits into: (a) NOW the supervisor server plus
  the UI in the browser (faster to test), (b) LATER the Tauri shell.
- The "five questions" smoke test after a model or graph swap works identically in both shapes.

## Addendum (3 September 2026): a second local sidecar — the graph-native tier

A large instruct GGUF (Qwen3-Next-80B-A3B, auto-detected in `bin/models/`) gets its OWN
llama-server sidecar (`:7352`, lazily booted in `app/big_tier.py`) and speaks the graph's own
language: a read-only `cypher()` verb on the same embedded `.lbdb`, next to the 13 CMDSL verbs.
No user consent is needed, by construction: nothing leaves the machine. The consent card applies
to the cloud only. The engine opens the packaged `.lbdb` read-only, so the benchmark and the
application can share the file across processes.
