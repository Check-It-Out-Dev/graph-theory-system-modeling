# Frequently asked questions

_Moved here from the README on 2026-09-08._

## What is Homotopy Type Theory (HoTT) in this context?

Homotopy Type Theory is a mathematical framework that treats types as topological spaces. In
this implementation, HoTT enabled the initial clustering of code files into architectural
boundaries by analysing type relationships as geometric structures. This identified 20 initial
subsystem candidates, consolidated into 7 architectural modules through domain expertise.

## What is the 6-entity pattern?

A universal lens for the behavioural relationships between files within any subsystem. Files
consistently organise into six functional roles:

- **Controller** — orchestration and external interfaces
- **Configuration** — settings and parameters
- **Security** — authentication and authorisation
- **Implementation** — core business logic
- **Diagnostics** — monitoring and observability
- **Lifecycle** — state management and temporal coordination

The pattern is argued from Ramsey theory (R(3,3)=6) in
[Paper 2](../GraphTheoryInSystemModeling/02_Living_Documentation_Deep_Modeling.md).

## What is NavigationMaster?

The central hub node of the graph, inspired by the Friendship Theorem. It provides O(1) access
to any component, a maximum 2-hop distance to any node, a betweenness centrality of 1.0, and a
canonical entry point for both human queries and AI agents.

## Why Neo4j?

A graph database represents code relationships that are cumbersome in relational databases: a
dependency query that needs several JOINs in SQL is a pattern match in Cypher. The Community
Edition is sufficient for an internal development tool and free to use that way. The CodeMap
authoring stack has run on LadybugDB (MIT) since 2026; the migration was accepted by
byte-identical gold answers across engines.

## Can this be implemented without deep mathematical understanding?

Yes. Install a graph database, run the indexing agents described in
[Paper 3](../GraphTheoryInSystemModeling/03_Living_Documentation_How_To_Start_For_Free.md),
query the graph. The mathematical principles are embedded in the approach; understanding them
deeply is not required for practical application.

## What are the key mathematical measures used?

- **Chromatic numbers** — minimum dependency exclusions in conflict resolution
- **Betweenness centrality** — critical-path components in the architecture
- **PageRank** — component importance from the dependency network
- **Vector embeddings** — semantic similarity across the codebase
- **Cohomology classes** — H⁰ counts connected components (one for a complete system), H¹
  detects missing feedback loops, H² identifies architectural voids
- **Sheaf cohomology** — local-to-global consistency of system properties
- **Homology groups** — structural features that persist across scales

## Setup requirements

Minimum: Neo4j Community Edition 5.x (or LadybugDB), Python 3.8+ with sentence-transformers,
8 GB RAM, about 2 GB of disk per million lines of code. Recommended: 16 GB RAM, a GPU for
embeddings, Docker for the database. The team-adoption setup, stage by stage, is in
[DEVELOPMENT_SETUP.md](../DEVELOPMENT_SETUP.md).

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -v $HOME/neo4j/data:/data -e NEO4J_AUTH=neo4j/your-password neo4j:5-community
```

Embeddings are computed outside the database and stored as node properties:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(code_text)
```

## Contributing

Areas of interest: language-specific analysers beyond Java, alternative embedding models, query
optimisation patterns, other graph databases. Contributions stay under the MIT licence.
