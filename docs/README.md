# Documentation

Five pages answer the questions most readers arrive with. Everything else is the record behind
them: the papers as they were written, the application's design documents, the agents' manuals and
the governance cards.

## Start here

| If you want to know | Read |
| :-- | :-- |
| How this came to be, and in what order | [The story](STORY.md) |
| Which mathematics runs, which is only on paper, which was withdrawn | [The mathematics](MATHEMATICS.md) |
| Whether it helps — results for and against, and what is next | [The evidence](EVIDENCE.md) |
| What is similar, and when each appeared | [Related work](RELATED-WORK.md) |
| What GEPA did to a team's conventions prompt | [A conventions prompt under test](PROMPT-UNDER-TEST.md) |
| What one question to the graph looks like, step by step | [One question, four hops](https://claude.ai/artifact/GwUEay3qmwnqrANh7yY7pm) — an interactive page; source in [`walkthrough/`](walkthrough/one-question-four-hops.html) |

## The application: CodeMap

| | |
| :-- | :-- |
| [Application README](../applications/CodeMap/README.md) | What it is and how to start it |
| [The navigation language](../applications/CodeMap/docs/03-dsl-interface.md) | The thirteen verbs |
| [Training story](../applications/CodeMap/docs/04-training-story.md) | How a 4B model learned to navigate |
| [Regeneration runbook](../applications/CodeMap/docs/05-regen-runbook.md) | Building the graph for a code base from scratch |
| [Prompt-transfer findings](../applications/CodeMap/docs/06-prompt-transfer-findings.md) | What a prompt alone can teach an untrained model |
| [Quality governance](../applications/CodeMap/docs/07-ai-quality-governance.md) | The decision log of the served system |
| [The served MCP](../applications/CodeMap/remote/README.md) | Connecting, security, running it as a team |
| [Model card](../applications/CodeMap/MODEL_CARD.md) · [evaluation card](../applications/CodeMap/EVAL_CARD.md) · [data card](../applications/CodeMap/DATA_CARD.md) · [threat model](../applications/CodeMap/THREAT_MODEL.md) · [incidents](../applications/CodeMap/INCIDENTS.md) | Governance |
| [The claims gate](../applications/CodeMap/eval/ci/README.md) | How published numbers are tied to artifacts |

## The agents

| | |
| :-- | :-- |
| [`.agents/`](../.agents/README.md) | The live set: indexer, organiser, architect, as agent skills |
| [`Promts/`](../Promts/README.md) | Five generations of prompt contracts, with which are current |

## The papers

Kept as written; each carries a status line saying what later measurement found.

| | |
| :-- | :-- |
| [V3 mathematical foundations](../GraphTheoryInSystemModeling/V3/V3_MathematicalFoundations.md) | The lab notebook: every experiment, the promoted system and the retired constructions |
| [V3 research audit](../GraphTheoryInSystemModeling/V3/V3_ResearchAudit_2026-09.md) | The repository's audit of its own claims |
| [The type algebra](../GraphTheoryInSystemModeling/V3/HypatiaBasis.md) | Roles, typed edges, selection rules |
| [Experiments catalogue](../GraphTheoryInSystemModeling/V3/experiments/CATALOG.md) | Which script established which number |
| Papers [1](../GraphTheoryInSystemModeling/01_Living_Documentation_HoTT_Graph_Theory.md) · [2](../GraphTheoryInSystemModeling/02_Living_Documentation_Deep_Modeling.md) · [3](../GraphTheoryInSystemModeling/03_Living_Documentation_How_To_Start_For_Free.md) · [4](../GraphTheoryInSystemModeling/04_Living_Documentation_On_Demand_Real_Example.md) · [5](../GraphTheoryInSystemModeling/05_Living_Documentation_How_To_Add_Seat_Model_Real_Example.md) · [6](../GraphTheoryInSystemModeling/06_Living_Documentation_Win_Win_For_Customers_And_AI_Providers.md) | The 2025 series; 4 and 5 are worked examples with real output |
| [Chromatic numbers](../GraphTheoryInSystemModeling/ChromaticNumbersInSystemModeling.md) · [Erdős–Lagrangian](../GraphTheoryInSystemModeling/ErdosLagrangianUnification.md) · [Appendix A](../GraphTheoryInSystemModeling/Appendix_A_Mathematical_Bridge.md) · [Appendix C](../GraphTheoryInSystemModeling/Appendix_C_Information_Lensing.md) · [Five detectives](../GraphTheoryInSystemModeling/Five_Independent_Detectives_Method.md) | Theory papers |

## From 2025

[Author's note](AUTHORS-NOTE.md) · [FAQ](FAQ.md) · [development setup on Neo4j](../DEVELOPMENT_SETUP.md) ·
[compliance](../COMPLIANCE.md) · [author's declaration](../AUTHORS_DECLARATION.md) ·
[changelog](../CHANGELOG.md)
