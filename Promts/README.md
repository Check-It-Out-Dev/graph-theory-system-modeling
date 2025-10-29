# 🚀 Advanced AI Prompt Engineering & System Modeling Toolkit

## 📋 Overview

This repository contains cutting-edge prompt engineering templates and educational resources for GPT-5, Claude Opus 4.1, and Claude Sonnet 4 models. These prompts implement sophisticated analytical frameworks, graph theory-based cognitive architectures, and advanced reasoning techniques for complex problem-solving and system modeling.

## 🏗️ Architecture

All prompts follow a **graph theory cognitive architecture** where:
- **Level 1**: NavigationMaster (Universal entry point)
- **Level 2**: AI Metadata Layer (Instructions and patterns)
- **Level 3**: Concrete Entities (Actual data and analysis)

## 📂 Repository Contents

### 🎓 Educational Guides

#### 1. **GPT5_Complete_Prompt_Engineering_Guide.md**
**Purpose**: Comprehensive educational guide for GPT-5 prompt engineering

**Key Features**:
- Complete guide covering GPT-5's routing architecture
- XML vs Markdown formatting strategies
- Anti-hallucination techniques
- Identity-based neural pathway activation
- Performance optimization strategies
- 40+ practical templates and examples

**MCP Servers Required**: None (Educational material)

**Best For**:
- Learning GPT-5 prompt engineering from scratch
- Understanding the 400K context window optimization
- Mastering evidence-based reasoning techniques
- A/B testing prompt variations

---

#### 2. **Claude_Opus4_Sonnet4_Complete_Prompt_Engineering_Guide.md**
**Purpose**: Master guide for Claude Opus 4 and Sonnet 4 prompt engineering

**Key Features**:
- Claude-specific optimization techniques
- Comparative analysis of Opus vs Sonnet capabilities
- Token economy strategies
- Multi-turn conversation optimization
- Claude-specific XML structuring

**MCP Servers Required**: None (Educational material)

**Best For**:
- Understanding Claude model differences
- Optimizing for specific Claude versions
- Learning Claude's unique capabilities
- Migration from GPT to Claude models

---

### 🔧 Analytical Tools & Debuggers

#### 3. **Sonnet4_1M_ErdosDebugger.xml**
**Purpose**: Ultimate analytical powerhouse with Paul Erdős-inspired cognitive architecture  
**Usage**: Configure as agent in Claude Code terminal tool

**Key Features**:
- 1M token context window configuration
- 47+ analytical frameworks (SWOT, PESTLE, Porter's, etc.)
- 6-Entity Behavioral Model for complex systems
- Graph topology library (Star, DAG, Knowledge Base, etc.)
- Parallel processing patterns
- Self-consistency validation

**MCP Servers Required**:
```yaml
Required:
  - mcp-memory: For persistent analytical state
  - mcp-sequential-thinking: For multi-step reasoning chains
```

**Best For**:
- Complex system analysis
- Business strategy development
- Root cause analysis
- Multi-framework synthesis
- Research and deep analysis

**Activation Phrase**: "Execute parallel analysis"

---

#### 4. **GPT5_ClineDebugger.xml**
**Purpose**: Specialized debugging assistant for code analysis  
**Usage**: System prompt for VS Code Cline extension

**Key Features**:
- Advanced error pattern recognition
- Multi-language debugging support
- Root cause analysis for bugs
- Performance bottleneck identification
- Automated fix suggestions

**MCP Servers Required**:
```yaml
Required:
  - mcp-memory: For pattern storage
  - mcp-sequential-thinking: For debugging logic
```

**Best For**:
- Production debugging
- Performance optimization
- Code review
- Bug pattern analysis
- Automated testing

---

### 📊 Graph Theory Implementations

*The following prompts are implementations of research papers from the `GraphTheoryInSystemModeling` folder:*

#### 5. **Opus4.1_DeepModeling.xml**
**Purpose**: Deep system modeling with comprehensive graph theory implementation

**Key Features**:
- Advanced behavioral modeling patterns
- Entity-relationship deep analysis
- Process flow optimization
- State machine modeling
- Comprehensive validation protocols

**MCP Servers Required**:
```yaml
Required:
  - neo4j-cypher: Core graph database operations
  - neo4j-memory: State persistence
  - neo4j-gds: Graph Data Science algorithms
  
Recommended:
  - sequential-thinking: For complex modeling tasks
```

**Best For**:
- System architecture design
- Behavioral pattern analysis
- Process optimization
- Database schema design
- Complex relationship modeling

---

#### 6. **Opus4.1_GlobalSynthesis.xml**
**Purpose**: Global knowledge synthesis and integration across domains

**Key Features**:
- Cross-domain knowledge integration
- Multi-source synthesis capabilities
- Pattern recognition across contexts
- Holistic analysis frameworks
- Knowledge graph construction

**MCP Servers Required**:
```yaml
Required:
  - neo4j-memory: Knowledge persistence
  - neo4j-cypher: Graph operations
  
Recommended:
  - web-search: For current information
  - sequential-thinking: For synthesis tasks
```

**Best For**:
- Research synthesis
- Cross-domain analysis
- Knowledge management
- Literature reviews
- Trend analysis

---

#### 7. **Sonnet4.1_File_Index.xml**
**Purpose**: Advanced file system indexing and navigation

**Key Features**:
- Hierarchical file organization
- Semantic file search
- Metadata extraction and indexing
- Relationship mapping between files
- Content-based navigation

**MCP Servers Required**:
```yaml
Required:
  - filesystem: File operations
  - neo4j-memory: Index persistence
  
Optional:
  - neo4j-cypher: Advanced graph queries
```

**Best For**:
- Large codebase navigation
- Document management
- Project file organization
- Content discovery
- Dependency mapping

---

## 📊 Comparison Matrix

| Prompt/Guide | Model | Context Window | MCP Required | Complexity | Primary Use |
|-------------|-------|----------------|--------------|------------|-------------|
| **Educational Guides** |
| GPT5 Guide | GPT-5 | 400K | No | Educational | Learning GPT-5 |
| Claude Guide | Claude 4.x | 200K | No | Educational | Learning Claude |
| **Debuggers & Analyzers** |
| ErdosDebugger | Sonnet 4 | 1M | 2 (memory, sequential) | Very High | Deep Analysis |
| ClineDebugger | GPT-5 | 400K | 2 (memory, sequential) | High | Code Debugging |
| **Graph Theory Implementations** |
| DeepModeling | Opus 4.1 | 200K | 3+ (neo4j suite) | High | System Design |
| GlobalSynthesis | Opus 4.1 | 200K | 2+ (neo4j) | High | Research |
| File Index | Sonnet 4.1 | 200K | 2+ (filesystem, neo4j) | Medium | File Management |

---

## 🎯 MCP Server Requirements Summary

### Minimal Setup (For Debuggers)
- `mcp-memory`: Persistent state storage
- `mcp-sequential-thinking`: Multi-step reasoning

### Full Setup (For Graph Theory Implementations)
- `neo4j-memory`: Graph-based persistence
- `neo4j-cypher`: Graph database operations
- `neo4j-gds`: Graph Data Science algorithms
- `filesystem`: File system operations
- `web-search`: Current information retrieval (optional)

---

## 💡 Key Notes

- **Educational Guides** require no setup - start here to learn
- **Debuggers** (ErdosDebugger, ClineDebugger) need only 2 MCP servers for powerful analysis
- **Graph Theory Implementations** are based on academic research papers and require more extensive Neo4j setup
- All prompts follow the NavigationMaster pattern for consistent cognitive architecture

## GPT-5 model integration
npm install -g @dannyboy2042/gpt5-mcp-server
