# Development Setup Guide

## Overview

This guide explains how to set up the graph-theory system modeling environment for your team, transitioning from individual research to team-wide adoption using fully compliant open-source tools.

## Development Stages

### Stage 1: Individual Research & Discovery (Author's Personal Research)

**Conducted by**: Norbert Marchewka (CheckItOut architect) only
**Status**: Completed research phase, not shared with team

For initial research and rapid prototyping:
- **Tool**: Neo4j Desktop Enterprise Edition (evaluation license)
- **Features Used**: 
  - Native vector embeddings (Enterprise feature)
  - Graph Data Science library
  - Advanced clustering algorithms
- **Deployment**: Single computer, personal research only
- **Purpose**: Fast discovery, analysis, and pattern identification
- **Duration**: 30-day trial period
- **Legitimacy**: Explicitly allowed for evaluation
- **Critical Point**: This was NOT shared with other developers

```bash
# Initial discovery metrics from our research:
- 426 Java files analyzed
- 20 subsystem candidates discovered via HoTT/embeddings
- 7 final business modules after consolidation
- 24,030 graph nodes created
- 87,453 relationships mapped
```

### Stage 2: Team Adoption (Recommended Production Setup)

For team-wide deployment:
- **Tool**: Neo4j Community Edition (Docker)
- **Purpose**: Shared team knowledge graph
- **License**: GPLv3 (free for internal use)
- **Vector Embeddings**: Computed separately

## Production Setup Instructions

### 1. Neo4j Community Edition Deployment

```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  neo4j:
    image: neo4j:5-community
    container_name: neo4j_community
    restart: unless-stopped
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - ./neo4j/data:/data
      - ./neo4j/logs:/logs
      - ./neo4j/import:/import
      - ./neo4j/plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/changeme123  # Change this!
      - NEO4J_server_memory_pagecache_size=2G
      - NEO4J_server_memory_heap_initial__size=2G
      - NEO4J_server_memory_heap_max__size=4G
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    networks:
      - neo4j-network

networks:
  neo4j-network:
    driver: bridge
EOF

# Start Neo4j
docker-compose up -d

# Verify it's running
docker logs neo4j_community
```

### 2. Separate Vector Embedding Service

```python
# embedding_service.py
"""
Standalone embedding service - completely separate from Neo4j
This maintains clean separation and legal compliance
"""

from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import hashlib
import json
import os

app = Flask(__name__)

# Load model once at startup
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # MIT licensed model

# Optional: Cache embeddings to avoid recomputation
CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route('/embed', methods=['POST'])
def generate_embedding():
    """Generate embedding for code text"""
    data = request.json
    text = data.get('text', '')
    
    # Check cache first
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/{text_hash}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return jsonify(json.load(f))
    
    # Generate embedding
    embedding = model.encode(text).tolist()
    
    # Cache result
    result = {'embedding': embedding, 'model': 'all-MiniLM-L6-v2'}
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'all-MiniLM-L6-v2'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 3. Graph Population Script

```python
# populate_graph.py
"""
Populate Neo4j with code analysis and embeddings
Keeps embedding generation completely separate
"""

import os
import ast
import requests
from neo4j import GraphDatabase
from pathlib import Path
import json

class GraphBuilder:
    def __init__(self, neo4j_uri, neo4j_auth, embedding_service_url):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.embedding_url = embedding_service_url
        
    def close(self):
        self.driver.close()
    
    def get_embedding(self, text):
        """Get embedding from separate service"""
        response = requests.post(
            f"{self.embedding_url}/embed",
            json={'text': text}
        )
        return response.json()['embedding']
    
    def analyze_file(self, file_path):
        """Analyze Python/Java file and create graph nodes"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get embedding from separate service
        embedding = self.get_embedding(content[:1000])  # First 1000 chars
        
        # Create node in Neo4j
        with self.driver.session() as session:
            session.execute_write(
                self._create_file_node,
                file_path=str(file_path),
                content=content,
                embedding=embedding
            )
    
    @staticmethod
    def _create_file_node(tx, file_path, content, embedding):
        query = """
        MERGE (f:File {path: $path})
        SET f.embedding = $embedding,
            f.analyzed_at = datetime(),
            f.lines = size(split($content, '\n'))
        """
        tx.run(query, path=file_path, content=content, embedding=embedding)
    
    def create_navigation_master(self, namespace):
        """Create the NavigationMaster hub"""
        with self.driver.session() as session:
            session.execute_write(self._create_nav_master, namespace)
    
    @staticmethod
    def _create_nav_master(tx, namespace):
        query = """
        MERGE (nav:NavigationMaster {namespace: $namespace})
        SET nav.created_at = datetime(),
            nav.betweenness_centrality = 1.0
        RETURN nav
        """
        tx.run(query, namespace=namespace)

# Usage
if __name__ == "__main__":
    # Connect to Neo4j Community Edition
    builder = GraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_auth=("neo4j", "changeme123"),
        embedding_service_url="http://localhost:5000"
    )
    
    # Create NavigationMaster
    builder.create_navigation_master("my_project")
    
    # Analyze codebase
    for file_path in Path("./src").rglob("*.java"):
        print(f"Analyzing {file_path}")
        builder.analyze_file(file_path)
    
    builder.close()
```

### 4. Environment Configuration

```bash
# .env file (DO NOT COMMIT THIS)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme123
EMBEDDING_SERVICE_URL=http://localhost:5000

# Optional: OpenAI for better embeddings
OPENAI_API_KEY=sk-...  # Only if using OpenAI
```

### 5. Team Access Configuration

```python
# team_config.py
"""
Configuration for team-wide access to shared knowledge graph
"""

TEAM_CONFIG = {
    'neo4j': {
        'uri': 'bolt://your-server:7687',  # Shared server
        'read_only_user': 'reader',
        'read_only_pass': 'readonly123',
        'admin_user': 'neo4j',
        'admin_pass': 'admin123'
    },
    'embedding_service': {
        'url': 'http://your-server:5000',
        'api_key': None  # Add if you implement authentication
    },
    'roles': {
        'developer': ['read', 'write', 'query'],
        'analyst': ['read', 'query'],
        'admin': ['read', 'write', 'query', 'admin']
    }
}
```

## Migration Path

### From Desktop Enterprise to Community Edition (Current Migration)

**Migration Context**: Moving from Norbert Marchewka's personal Enterprise Desktop research to team-wide Community Edition deployment.

1. **Export from Desktop Enterprise** (removing Enterprise-specific features):
```cypher
// In Neo4j Desktop Enterprise - Norbert's machine only
// Note: Must remove native embeddings before export!

// First, extract embeddings to separate storage
MATCH (n:Node) WHERE exists(n.embedding)
RETURN n.id, n.embedding
// Save these separately for reimport via external service

// Then export graph structure without embeddings
CALL apoc.export.cypher.all('export.cypher', {
    format: 'plain',
    useOptimizations: {type: 'UNWIND_BATCH', unwindBatchSize: 1000}
})
```

2. **Import to Community**:
```bash
# Copy export to import directory
docker cp export.cypher neo4j_community:/import/

# Import via cypher-shell
docker exec -it neo4j_community cypher-shell -u neo4j -p changeme123 \
  -f /import/export.cypher
```

## Performance Optimization

### For Large Codebases (>10K files)

```yaml
# neo4j.conf optimizations
server.memory.heap.initial_size=4G
server.memory.heap.max_size=8G
server.memory.pagecache.size=4G
db.tx_log.rotation.retention_policy=1 days
dbms.checkpoint.interval.time=15m
dbms.checkpoint.interval.tx=100000
```

### Embedding Cache Strategy

```python
# Use Redis for embedding cache (optional)
import redis
import json
import hashlib

class EmbeddingCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def get_or_compute(self, text, compute_func):
        key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Check cache
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        
        # Compute and cache
        embedding = compute_func(text)
        self.redis_client.setex(
            key, 
            86400,  # 24 hour TTL
            json.dumps(embedding)
        )
        return embedding
```

## Monitoring & Maintenance

### Health Checks

```python
# health_check.py
def check_system_health():
    checks = {
        'neo4j': check_neo4j(),
        'embeddings': check_embedding_service(),
        'disk_space': check_disk_space(),
        'memory': check_memory_usage()
    }
    return all(checks.values())

def check_neo4j():
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", 
                                    auth=("neo4j", "changeme123"))
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) LIMIT 1")
            return result.single() is not None
    except:
        return False
```

## Security Considerations

1. **Never commit credentials** - Use environment variables
2. **Use read-only users** for query-only access
3. **Implement API authentication** for embedding service
4. **Run Neo4j behind firewall** for team access
5. **Regular backups** of graph data

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Increase heap size in docker-compose.yml
   - Reduce embedding dimensions (384 vs 768)

2. **Slow Queries**:
   - Create indexes: `CREATE INDEX ON :File(path)`
   - Use query profiling: `PROFILE MATCH ...`

3. **Embedding Service Timeout**:
   - Implement request queuing
   - Use smaller batch sizes
   - Cache aggressively

## Conclusion

This setup provides:
- ✅ Full legal compliance (Neo4j Community + separate embeddings)
- ✅ Team-wide knowledge sharing
- ✅ Scalable architecture
- ✅ Clear separation of concerns
- ✅ Production-ready configuration

Remember: The power isn't in the tools but in the mathematical discovery of your system's structure. The tools just make it queryable.

---

*For questions about setup, please refer to the papers in `/GraphTheoryInSystemModeling/` or create an issue in the repository.*
