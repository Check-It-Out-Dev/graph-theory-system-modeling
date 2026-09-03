# CodeMap DSL — the small model's whole world (design, 2 September 2026)

**Owner's thesis, adopted:** the small model never sees Cypher, Ladybug or the schema. It speaks
one tiny domain-specific language; the MCP compiles that language into whatever the underlying
graph speaks; the ErdosNavigator agent precomputes AI metadata that carries ready-made DSL
continuations. The graph stops being a static database and becomes the other side of a
conversation, held in a language built for exactly this codebase-understanding job. The model is
a translator in a loop: natural language in, DSL out; DSL-shaped results and suggested next steps
in; a natural-language answer at the end.

## 1. Why this beats training on Cypher/Ladybug (evidence, not taste)

- **Unconstrained SQL is hard for small models; tiny grammars are not.** SLM-SQL: a 0.5B model
  reaches 56.9% and a 1.5B model 67.1% execution accuracy on BIRD, which is open-domain SQL over
  arbitrary schemas. FINER-SQL brings 0.5–3B models to the level of 14–70B models through execution
  feedback. Our task is far easier than BIRD: ONE fixed conceptual schema, about 12 verbs, no joins,
  no projections. The hard parts of text-to-SQL are exactly what the DSL removes.
- **Schema linking is what kills small models** (swings of 6.9–46.9% as parameter counts shrink). A
  12-verb DSL has no schema-linking problem: the "schema" fits in the master prompt.
- **Grammar-constrained decoding eliminates syntax errors entirely** (the PICARD line of work;
  llama.cpp GBNF): the model *cannot* emit an invalid expression. Every failure becomes a semantic
  failure — visible, classifiable into the three buckets, trainable.
- **Dialect independence strengthens the replaceability principle.** Neo4j → Ladybug → anything
  else: the model never changes; only the MCP compiler's backend does. Model and graph stay
  independently swappable through `dsl_version` in the manifest.
- Document 00's own research finding: about 600 fine-tuning pairs on one's own tools brings the
  leading small models above 95%, and tool-call syntax IS a DSL.

## 2. CMDSL v0.1 (draft grammar; hard cap: NEVER more than 15 verbs)

| verb | args | compiles to | archetype |
|---|---|---|---|
| `map()` | — | L1 read | overview |
| `enter(sub)` | sub_id \| name | L2 report read | overview |
| `find(term)` | free term | lexical + kNN entity lookup | locate |
| `impact(entity, d?)` | name, depth ≤ 3 | reverse-dependency walk | impact |
| `flow(entity, d?)` | name, depth ≤ 3 | forward walk | flow |
| `seam(a, b)` | two subs | boundary contracts | boundary |
| `cohort(entity)` | name | hyperedge co-change | cohort |
| `spine(sub)` | sub | reading path | onboarding |
| `health(kind)` | enum(7) | census recipes | health |
| `read(entity)` | name | content pointer (file path + hint) | rationale/content |
| `cache(q)` | NL question | MFQ alias match | tier 1 |
| `answer(text)` | NL | terminates the loop | — |
| `pass(reason)` | NL reason | honest abstention; terminates the loop | — |

**`pass()` is in v0 and in the training data from the FIRST LoRA run** (owner directive,
2 September): abstention cannot be retrofitted into a model trained to always answer. In v0 the
MCP renders `pass()` as an honest "beyond me: <reason>" and logs it; the router's risk–coverage
curve (document 00, section 6) is measured from these. LATER — only after the small model has
been pushed as far as it will go (prompts, clues, grammar and MCP iterated; performance measured;
training done) — a key-gated escalation tier turns `pass()` into a router call: with an API key
configured, the MCP forwards the question plus graph context to a larger model (the Claude
Messages API first). OFF by default, never required. That ordering is fixed: get the most out of
the small model first; the API tier is an amplifier, not a crutch.

One expression per turn in v0 (no pipes; pipes are v2 if the ladder proves the need). Results
return as compact typed records PLUS `next:` **affordances** — DSL expressions that the MCP and
the graph suggest as the next step. The model composes almost nothing; it SELECTS and fills.
Selection over generation is decision D5, now at the language level.

## 3. The loop

```
user NL question
   │
   ▼
small model ──DSL──► MCP ──compiled query──► graph (Ladybug embedded)
   ▲                  │                          │
   │                  │◄── rows ─────────────────┘
   │◄── result record + next: [affordances] ─────┘
   │        (loop: model picks the next DSL expression, or answers)
   ▼
answer(NL) ──► user
```

The protocol is enforced by the MCP, not remembered by the model: session start auto-injects the
result of `map()`; `cache(q)` runs implicitly first on every new question (a 58% silent-skip rate
is the measured default when the protocol is optional, so it is not optional).

## 4. AI metadata in DSL (the ErdosNavigator agent's new output field)

Prose stays natural language (humans read answers; clues feed answers). What becomes DSL:
**affordance blocks** on every navigation node —

```
L1:  affordances: [enter(11), enter(17), health(coupling), spine(11)]
L2 (sub 11): affordances: [spine(11), impact(UserRepository.java),
             seam(11,10), cohort(StripeService.java), read(PaymentsDisabledBootGuard.java)]
entry-point node: affordances: [flow(UserRepository.java,1), impact(UserRepository.java)]
```

This is information scent made EXECUTABLE: the label does not merely predict the content, it IS
the next step. The ErdosNavigator agent generates these at stages E2/E3 (grounded: only
affordances whose targets exist and whose recipes are compiled); the MCP validates every
affordance at pack time. A clue that suggests an expression that does not compile fails the pack
gate.

## 5. Division of labour (final form)

| lives in | contents |
|---|---|
| **model** (0.8–4B, LoRA) | NL → DSL translation, result → NL synthesis, loop control (when to stop), the verbs |
| **MCP** | DSL parser and compiler (per-dialect backends), protocol enforcement, cache and invalidation, affordance validation, telemetry, the GBNF grammar file (shipped next to the model) |
| **graph** | facts, clues, DSL affordances, the compiled forms of the recipes, MFQ gold |
| **NOT in the model** | Cypher/Ladybug, the schema, subsystem names and ids (read at runtime from `map()`/`enter()`), file paths |

## 6. What this does to training (it gets easier everywhere)

- Pairs become **NL → one line of DSL**: short outputs, evaluable by exact match, and our 312
  alias pairs become paraphrase-robustness data for free (many NL phrasings, one DSL expression).
- The execution verifier gets stronger: compile, run, compare fingerprints (the 104 golds already
  carry fingerprints). Bad pairs can be filtered mechanically.
- Teacher distillation is a frontier model driving the same MCP loop; its DSL trajectories are the
  imitation corpus (the C5 trajectories still serve as benchmark, corpus and specification at once,
  now in a smaller action space).
- The ablation ladder gains a rung: bare model plus GBNF (no LoRA). Constrained decoding alone may
  carry the G1 stratum; measure before training (rung 1.5).

## 7. Risks and the rules that contain them

1. **DSL growth** — every added verb taxes the smallest model. Rule: at most 15 verbs, ever; new
   needs become recipe parameters, not verbs.
2. **A `raw()` escape hatch** — not in v0. Add only if the "no recipe" failure bucket exceeds 10%,
   gated OFF by default, never in the training set.
3. **Version drift** — `dsl_version` in the pack manifest AND in the model's prompt pin; a mismatch
   means refuse to start (the same rule as for prompt_version).
4. **Affordance rot** — the pack gate recompiles every stored affordance; delta runs re-validate
   the affordances of changed subsystems (riding on the existing invalidation set).
