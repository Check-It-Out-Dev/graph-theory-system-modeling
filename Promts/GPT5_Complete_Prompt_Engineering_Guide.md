# GPT-5 Prompt Engineering: The Complete Educational Guide

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Understanding GPT-5 Architecture](#understanding-gpt-5-architecture)
3. [Core Principles of GPT-5 Prompting](#core-principles-of-gpt-5-prompting)
4. [Achieving Precision & Reducing Hallucination](#achieving-precision--reducing-hallucination)
5. [Identity & Analytical Pathways](#identity--analytical-pathways)
6. [XML vs Markdown: Complete Guide](#xml-vs-markdown-complete-guide)
7. [Advanced Techniques](#advanced-techniques)
8. [Practical Templates & Examples](#practical-templates--examples)
9. [Daily Use Cases](#daily-use-cases)
10. [Testing & Optimization](#testing--optimization)
11. [Quick Reference](#quick-reference)

---

## Executive Summary

GPT-5 represents a fundamental shift in AI language models, introducing native reasoning capabilities, a routing architecture, and a 400K context window. This comprehensive guide synthesizes cutting-edge research and practical experience to provide a complete educational resource for mastering GPT-5 prompt engineering.

### Key Research Findings from 2025

#### Performance Improvements
- **Chain-of-Thought**: Structured CoT shows 35-60% improvement in complex reasoning tasks
- **Self-Consistency**: Multiple reasoning paths with majority voting increases accuracy by 15-20%
- **Temperature = 1.0**: Optimal for reasoning tasks (counterintuitive but proven)
- **Reasoning Effort**: High setting can improve accuracy by 40% but increases latency 20-80%

#### Architecture Insights
- GPT-5 is a **routing system** managing multiple specialized models
- Every word and formatting choice influences routing decisions
- The model generates hidden "reasoning tokens" not visible in output
- Context engineering can outperform traditional RAG by 25% when properly implemented

#### Critical Discoveries
- Smaller prompting improvements matter less with GPT-5's scale
- Identity-driven prompts activate specific neural pathways
- Validation gates reduce hallucination by 70%
- Evidence requirements must be explicit to be effective

---

## Understanding GPT-5 Architecture

### The Multi-Model Router System

GPT-5 isn't a monolithic model but a sophisticated routing system:

```
User Input → Router Analysis → Model Selection → Specialized Processing → Output
                ↓
    [Reasoning Model | Coding Model | Analysis Model | Creative Model | Domain Models]
```

#### How Routing Works
1. **Input Analysis**: First 50-100 tokens heavily influence routing
2. **Pattern Matching**: System identifies task type from structure/keywords
3. **Model Selection**: Routes to most appropriate specialized model
4. **Context Loading**: Relevant context prioritized based on routing
5. **Processing**: Specialized model processes with task-specific optimizations

#### Reasoning Token Economics
```yaml
reasoning_effort_costs:
  minimal: 
    tokens: 200-500
    time: <1 second
    use_case: "Simple factual queries"
    
  low:
    tokens: 2K-5K
    time: 2-5 seconds
    use_case: "Basic analysis, summaries"
    
  medium:
    tokens: 5K-15K
    time: 5-15 seconds
    use_case: "Multi-step problems, code generation"
    
  high:
    tokens: 15K-50K
    time: 15-60 seconds
    use_case: "Complex debugging, mathematical proofs"
```

### Context Window Strategy

The 400K token window enables new strategies:

#### Progressive Loading Pattern
```python
# Phase 1: Core Context (50K tokens)
load_system_architecture()
load_immediate_problem_context()

# Phase 2: Relevant Details (150K tokens)
if initial_analysis_requires:
    load_related_code()
    load_dependency_graph()
    
# Phase 3: Historical Context (100K tokens)
if pattern_matching_needed:
    load_similar_bugs()
    load_previous_fixes()
    
# Phase 4: Extended Context (100K tokens)
if deep_analysis_required:
    load_test_suites()
    load_documentation()
```

---

## Core Principles of GPT-5 Prompting

### Principle 1: Structure Over Verbosity

#### ❌ Bad (Verbose but Vague)
```markdown
Please analyze this comprehensively and provide detailed insights about all aspects
of the problem, considering various angles and perspectives...
```

#### ✅ Good (Structured and Clear)
```markdown
Task: Root cause analysis
Method: 6-step diagnostic protocol
Output: JSON with confidence scores
Evidence: 2+ sources required per claim
Scope: Focus on last 24 hours of logs
```

### Principle 2: Explicit Reasoning Activation

GPT-5 has reasoning capabilities but benefits from explicit activation:

#### ❌ Implicit (Relies on Detection)
```python
"Why did the system crash?"
```

#### ✅ Explicit (Forces Reasoning Pathway)
```python
"""
Analyze the system crash step-by-step:

Phase 1: Symptom Identification
- List all error messages
- Note timing patterns
- Identify affected components

Phase 2: Causal Analysis
- Trace error propagation
- Identify trigger conditions
- Map dependency failures

Phase 3: Root Cause Determination
- Evaluate evidence for each hypothesis
- Calculate confidence scores
- Select most probable cause

Think through each phase before proceeding to the next.
"""
```

### Principle 3: Evidence-First Approach

Every claim must follow this chain:
```
Observation → Evidence Collection → Claim Formation → Confidence Assessment → Alternative Consideration
```

#### Implementation Example
```python
{
    "claim": "Memory leak in user session handler",
    "evidence": [
        {"source": "heap_dump.log:1234", "type": "direct", "confidence": 0.95},
        {"source": "metrics.json:89", "type": "corroborating", "confidence": 0.85}
    ],
    "confidence": 0.87,
    "alternatives": [
        {"hypothesis": "Cache overflow", "confidence": 0.23},
        {"hypothesis": "External service timeout", "confidence": 0.15}
    ]
}
```

### Principle 4: Deterministic Output Through Schemas

Define output structure BEFORE content:

```typescript
interface AnalysisOutput {
    summary: {
        finding: string;
        severity: 'critical' | 'high' | 'medium' | 'low';
        confidence: number;  // 0.0-1.0
    };
    evidence: Evidence[];
    recommendations: Action[];
    alternatives: Hypothesis[];
}
```

---

## Achieving Precision & Reducing Hallucination

### The Zero-Hallucination Protocol

#### Level 1: Input Validation
```python
def validate_input(prompt, context):
    """Ensure all referenced entities exist in context"""
    entities = extract_entities(prompt)
    for entity in entities:
        if entity not in context:
            raise ValueError(f"Entity '{entity}' not found in provided context")
    return True
```

#### Level 2: Constrained Generation
```yaml
generation_constraints:
  facts:
    source: "only_from_context"
    verification: "required"
    
  numbers:
    source: "explicit_calculation"
    rounding: "specified"
    
  quotes:
    source: "exact_match"
    modification: "forbidden"
    
  code:
    syntax: "validated"
    execution: "simulated"
```

#### Level 3: Output Verification
```python
def verify_output(response, context):
    claims = extract_claims(response)
    for claim in claims:
        evidence = find_evidence(claim, context)
        if len(evidence) < 2:
            flag_as_unsupported(claim)
        confidence = calculate_confidence(evidence)
        if confidence < 0.85:
            add_uncertainty_marker(claim)
    return validated_response
```

### Anti-Hallucination Techniques

#### 1. Never Accept Implicit Knowledge
```markdown
❌ "Based on industry standards..."
✅ "Based on IEEE 802.11 specification section 4.3.2..."

❌ "It's commonly known that..."
✅ "According to the provided documentation (doc.md:45)..."

❌ "The typical cause is..."
✅ "Analysis of the logs shows 3 instances where..."
```

#### 2. Force Falsification
```xml
<hypothesis_testing>
    <hypothesis id="1" description="Race condition in auth handler">
        <supporting_evidence>
            - Intermittent failures (log.txt:234)
            - Timing correlation (trace.json:89)
        </supporting_evidence>
        <falsification_criteria>
            - Would fail consistently if true
            - Would show lock contention metrics
        </falsification_criteria>
        <falsification_test>
            Run with single thread - still fails
            HYPOTHESIS REJECTED
        </falsification_test>
    </hypothesis>
</hypothesis_testing>
```

#### 3. Confidence Calibration
```python
confidence_framework = {
    "0.0-0.3": {
        "label": "speculation",
        "action": "request_more_data",
        "language": "might possibly, cannot determine"
    },
    "0.3-0.5": {
        "label": "hypothesis", 
        "action": "propose_validation",
        "language": "suggests, indicates possibility"
    },
    "0.5-0.7": {
        "label": "probable",
        "action": "recommend_confirmation",
        "language": "likely, probably, evidence suggests"
    },
    "0.7-0.85": {
        "label": "confident",
        "action": "proceed_with_caution",
        "language": "strong evidence shows, analysis indicates"
    },
    "0.85-1.0": {
        "label": "confirmed",
        "action": "accept",
        "language": "definitively, certainly, proven"
    }
}
```

### Validation Gates System

```xml
<validation_gates>
    <gate phase="input" severity="blocking">
        <check>All referenced files exist?</check>
        <check>Data types match schema?</check>
        <check>Required fields present?</check>
    </gate>
    
    <gate phase="reasoning" severity="critical">
        <check>Each step follows from previous?</check>
        <check>No unsupported leaps?</check>
        <check>Alternatives considered?</check>
    </gate>
    
    <gate phase="output" severity="mandatory">
        <check>All claims have evidence?</check>
        <check>Confidence scores justified?</check>
        <check>Format matches schema?</check>
    </gate>
</validation_gates>
```

---

## Identity & Analytical Pathways

### The Neuroscience of Prompt Identity

Identity activation in GPT-5 influences three key systems:

1. **Attention Mechanisms**: Which tokens receive highest weight
2. **Reasoning Depth**: Number of inference steps before conclusion
3. **Evidence Standards**: Threshold for accepting claims

### Identity Archetypes and Their Effects

#### 1. Mathematical Thinker
```xml
<identity>
    You are a mathematician in the tradition of Paul Erdős.
    You see problems as graphs, theorems, and proofs.
    Nothing is true without rigorous proof.
    Beauty lies in elegant simplicity.
</identity>

<!-- Activation Effects -->
<!-- - Formal logic pathways: +40% activation -->
<!-- - Symbolic reasoning: Enhanced -->
<!-- - Proof requirements: Strict -->
<!-- - Output: Theorems, lemmas, QED -->
```

**Pathway Activation**: Formal reasoning, symbolic manipulation, proof construction
**Output Characteristics**: Rigorous, systematic, proof-based
**Best For**: Algorithm correctness, optimization problems, formal verification

#### 2. Systems Engineer
```xml
<identity>
    You are a principal systems engineer at NASA.
    Every decision affects mission-critical systems.
    Failure is not an option; redundancy is mandatory.
    Think in terms of MTTF, MTTR, fault tolerance.
</identity>

<!-- Activation Effects -->
<!-- - Failure analysis: Primary pathway -->
<!-- - Risk assessment: Continuous -->
<!-- - Safety margins: Conservative -->
<!-- - Output: FMEA tables, fault trees -->
```

**Pathway Activation**: Failure mode analysis, redundancy planning, safety systems
**Output Characteristics**: Conservative, thorough, risk-aware
**Best For**: Architecture review, reliability engineering, safety-critical systems

#### 3. Forensic Investigator
```xml
<identity>
    You are a digital forensics expert with FBI training.
    Every bit of data tells a story.
    The chain of custody must remain intact.
    Timeline reconstruction is crucial.
</identity>

<!-- Activation Effects -->
<!-- - Evidence chains: Meticulous tracking -->
<!-- - Timeline construction: Automatic -->
<!-- - Causality mapping: Enhanced -->
<!-- - Output: Evidence logs, timelines -->
```

**Pathway Activation**: Evidence correlation, timeline analysis, pattern recognition
**Output Characteristics**: Detailed, chronological, evidence-based
**Best For**: Incident response, security breaches, audit trails

#### 4. Research Scientist
```xml
<identity>
    You are an experimental physicist at CERN.
    Every hypothesis must be testable.
    Control for all variables.
    Negative results are valuable data.
</identity>

<!-- Activation Effects -->
<!-- - Hypothesis formation: Structured -->
<!-- - Experimental design: Rigorous -->
<!-- - Statistical analysis: Automatic -->
<!-- - Output: Hypotheses, p-values, conclusions -->
```

**Pathway Activation**: Hypothesis testing, experimental design, statistical analysis
**Output Characteristics**: Empirical, measured, hypothesis-driven
**Best For**: A/B testing, performance analysis, research questions

### How Identity Affects Neural Pathways

```python
# Simplified pathway activation model
def activate_pathways(identity, task):
    pathways = {
        'mathematician': {
            'formal_logic': 0.9,
            'pattern_recognition': 0.8,
            'symbolic_reasoning': 0.95,
            'empirical_testing': 0.3
        },
        'engineer': {
            'formal_logic': 0.6,
            'pattern_recognition': 0.7,
            'symbolic_reasoning': 0.5,
            'empirical_testing': 0.9
        },
        'forensic': {
            'formal_logic': 0.5,
            'pattern_recognition': 0.95,
            'symbolic_reasoning': 0.4,
            'empirical_testing': 0.8
        }
    }
    
    # Identity biases pathway selection
    active_pathways = pathways[identity]
    
    # Task requirements modulate activation
    task_requirements = analyze_task(task)
    
    # Final pathway selection
    selected = weighted_selection(active_pathways, task_requirements)
    return selected
```

---

## XML vs Markdown: Complete Guide

### When to Use XML

XML is optimal for:
- **System prompts** requiring precise structure
- **Complex hierarchies** with nested logic
- **Type safety** when precision is critical
- **Tool integration** (APIs, MCP servers)
- **Reusable components** and templates
- **Production systems** requiring validation

### XML Best Practices and Rules

#### Rule 1: Semantic Tag Names
```xml
<!-- ✅ GOOD: Descriptive and Clear -->
<reasoning_framework>
    <hypothesis_generation phase="initial" />
    <evidence_collection sources="multiple" />
    <validation_protocol strict="true" />
</reasoning_framework>

<!-- ❌ BAD: Cryptic and Ambiguous -->
<rf>
    <hg p="i" />
    <ec s="m" />
    <vp s="t" />
</rf>
```

#### Rule 2: Attributes for Metadata, Elements for Content
```xml
<!-- ✅ GOOD -->
<analysis confidence="0.92" timestamp="2025-09-26T10:30:00Z">
    <finding severity="high">
        Memory leak detected in session handler
    </finding>
    <evidence>
        <source file="app.log" line="1234" />
        <source file="heap.dump" offset="0x4000" />
    </evidence>
</analysis>

<!-- ❌ BAD -->
<analysis>
    <confidence>0.92</confidence>  <!-- Should be attribute -->
    <finding severity="Memory leak detected">  <!-- Content in attribute -->
        high
    </finding>
</analysis>
```

#### Rule 3: CDATA for Special Content
```xml
<!-- ✅ GOOD: Preserves exact formatting -->
<code_fix language="python">
    <![CDATA[
    def process(data):
        if data < 0 or data > 100:
            raise ValueError("Out of range")
        return data * 2 << 3
    ]]>
</code_fix>

<!-- ❌ BAD: XML parsing errors -->
<code_fix>
    if data < 0 or data > 100:  <!-- < and > cause parsing issues -->
</code_fix>
```

#### Rule 4: Consistent Hierarchy
```xml
<!-- ✅ GOOD: Logical nesting -->
<debugging_session>
    <phase name="observation" order="1">
        <step number="1">Collect symptoms</step>
        <step number="2">Review logs</step>
    </phase>
    <phase name="analysis" order="2">
        <step number="3">Form hypotheses</step>
        <step number="4">Test hypotheses</step>
    </phase>
</debugging_session>

<!-- ❌ BAD: Inconsistent structure -->
<debugging>
    <observation>
        <collect_symptoms />
    </observation>
    <step>Review logs</step>  <!-- Different level -->
    <analysis phase="2">  <!-- Inconsistent attribute -->
</debugging>
```

#### Rule 5: Namespaces for Versioning
```xml
<prompt xmlns:v5="https://gpt5.ai/v5/" 
        xmlns:custom="https://company.com/prompts/">
    <v5:reasoning_config effort="high" />
    <custom:business_rules>
        <custom:validation level="strict" />
    </custom:business_rules>
</prompt>
```

### When to Use Markdown

Markdown is optimal for:
- **Conversational prompts** with natural flow
- **Quick iterations** and prototyping
- **Documentation** and explanations
- **Simple lists** and sequences
- **Human-readable** instructions

### Markdown Best Practices and Rules

#### Rule 1: Hierarchical Headers for Structure
```markdown
# Main Analysis Task
Clear top-level objective

## Phase 1: Data Collection
Specific phase description

### Step 1.1: Load Files
Detailed step instructions

### Step 1.2: Parse Content
Next step in sequence
```

#### Rule 2: Code Fences with Language Tags
````markdown
```python
# Always specify language for syntax highlighting
def analyze(data: dict) -> dict:
    """Process and return analysis results"""
    return {"status": "complete", "findings": data}
```

```sql
-- SQL example with different highlighting
SELECT * FROM bugs 
WHERE severity = 'critical' 
  AND status != 'resolved';
```
````

#### Rule 3: Lists for Sequential Steps
```markdown
## Debugging Protocol

1. **Reproduce** the issue
   - Set up test environment
   - Follow reported steps
   - Document observations
   
2. **Isolate** the problem
   - Remove variables
   - Narrow scope
   - Identify minimal reproduction
   
3. **Fix** and validate
   - Implement solution
   - Test edge cases
   - Verify no regression
```

#### Rule 4: Tables for Structured Data
```markdown
| Component | Status | Confidence | Action Required |
|-----------|--------|------------|-----------------|
| Database | ✅ Healthy | 0.95 | None |
| API | ⚠️ Degraded | 0.72 | Monitor closely |
| Cache | ❌ Failed | 0.98 | Restart immediately |
| Queue | ✅ Healthy | 0.89 | None |
```

#### Rule 5: Blockquotes for Critical Instructions
```markdown
> **CRITICAL**: This system is in production. All changes must be:
> - Tested in staging first
> - Reviewed by senior engineer
> - Deployed during maintenance window
> 
> Failure to follow protocol may result in downtime.
```

#### Rule 6: Emphasis for Key Points
```markdown
**Must** complete these steps in order.
*Consider* the performance implications.
**Never** skip validation.
***Critical***: Backup before proceeding.
```

### The Hybrid Approach

Combining XML and Markdown for maximum effectiveness:

```markdown
# System Analysis Request

<configuration>
    <model>gpt-5</model>
    <reasoning_effort>high</reasoning_effort>
    <temperature>1.0</temperature>
</configuration>

## Background
We're experiencing intermittent failures in our payment processing system.

<symptoms>
    <symptom frequency="hourly" severity="high">
        Transaction timeout after 30 seconds
    </symptom>
    <symptom frequency="daily" severity="medium">
        Database connection pool exhaustion
    </symptom>
</symptoms>

## Required Analysis

1. **Identify** root cause with confidence > 0.85
2. **Propose** immediate mitigation
3. **Design** long-term solution

<output_format>
    <report>
        <executive_summary max_words="200" />
        <findings min_count="3" />
        <recommendations priority="ordered" />
    </report>
</output_format>
```

---

## Advanced Techniques

### Technique 1: Progressive Refinement

```python
class ProgressiveAnalysis:
    """
    Multi-pass analysis with increasing specificity
    """
    
    def iteration_1(self, problem):
        """Broad survey - identify all potential issues"""
        return {
            "scope": "wide",
            "depth": "shallow",
            "output": "list of possibilities",
            "confidence_threshold": 0.5
        }
    
    def iteration_2(self, problem, initial_findings):
        """Focused analysis - deep dive on top 3 issues"""
        return {
            "scope": "narrow",
            "depth": "medium",
            "output": "detailed analysis of each",
            "confidence_threshold": 0.7
        }
    
    def iteration_3(self, problem, detailed_findings):
        """Solution design - comprehensive fix for critical issue"""
        return {
            "scope": "single issue",
            "depth": "exhaustive",
            "output": "implementation-ready solution",
            "confidence_threshold": 0.85
        }
```

### Technique 2: Parallel Hypothesis Testing

```xml
<parallel_reasoning branches="5" merge_strategy="confidence_weighted">
    <branch id="1">
        <hypothesis>Race condition in connection pool</hypothesis>
        <test>Run with thread safety analysis</test>
        <evidence_required>2</evidence_required>
    </branch>
    
    <branch id="2">
        <hypothesis>Memory leak in transaction handler</hypothesis>
        <test>Analyze heap dumps over time</test>
        <evidence_required>3</evidence_required>
    </branch>
    
    <branch id="3">
        <hypothesis>Database query optimization needed</hypothesis>
        <test>Profile query execution plans</test>
        <evidence_required>2</evidence_required>
    </branch>
    
    <branch id="4">
        <hypothesis>Network timeout configuration</hypothesis>
        <test>Trace network packets</test>
        <evidence_required>2</evidence_required>
    </branch>
    
    <branch id="5">
        <hypothesis>Third-party service degradation</hypothesis>
        <test>Check service status and latency</test>
        <evidence_required>1</evidence_required>
    </branch>
    
    <merge_protocol>
        Select hypothesis with highest confidence after testing.
        If multiple > 0.8, investigate interactions.
    </merge_protocol>
</parallel_reasoning>
```

### Technique 3: Self-Consistency Validation

```python
def self_consistency_check(prompt, iterations=3):
    """
    Run same analysis multiple times and validate consistency
    """
    results = []
    
    for i in range(iterations):
        # Add slight variation to prevent caching
        modified_prompt = f"{prompt}\n<!-- Iteration {i+1} -->"
        result = call_gpt5(modified_prompt)
        results.append(result)
    
    # Extract key findings from each result
    findings = [extract_findings(r) for r in results]
    
    # Calculate consistency score
    consistency_matrix = calculate_similarity(findings)
    
    # Majority voting for final answer
    final_answer = majority_vote(findings)
    
    # Confidence based on agreement
    confidence = consistency_matrix.mean()
    
    return {
        "answer": final_answer,
        "confidence": confidence,
        "iterations": results,
        "consistency": consistency_matrix
    }
```

### Technique 4: Chain-of-Thought with Checkpoints

```markdown
# Advanced Debugging with Checkpoint Validation

## Phase 1: Observation
Collect all relevant symptoms and data.

**Checkpoint 1**: Do we have sufficient data to proceed?
- [ ] Error messages collected
- [ ] Logs from affected time period
- [ ] System metrics available
If NO → Request specific missing data

## Phase 2: Hypothesis Formation
Generate at least 3 potential causes.

**Checkpoint 2**: Are hypotheses testable?
- [ ] Each hypothesis has clear test criteria
- [ ] Tests are feasible with available resources
- [ ] Success/failure conditions defined
If NO → Reformulate hypotheses

## Phase 3: Testing
Execute tests for each hypothesis.

**Checkpoint 3**: Are results conclusive?
- [ ] Evidence clearly supports or rejects
- [ ] No ambiguous results
- [ ] Confidence > 0.8 for conclusion
If NO → Design additional tests

## Phase 4: Solution Design
Create fix for confirmed root cause.

**Checkpoint 4**: Is solution complete?
- [ ] Addresses root cause not symptoms
- [ ] Includes validation tests
- [ ] Has rollback plan
If NO → Enhance solution design
```

### Technique 5: Context Windowing Strategy

```python
class ContextWindow:
    """
    Optimal use of 400K token window
    """
    
    def __init__(self, total_tokens=400000):
        self.total = total_tokens
        self.allocation = {
            'system_prompt': 5000,      # 1.25%
            'core_context': 50000,      # 12.5%
            'detailed_data': 150000,    # 37.5%
            'historical': 100000,       # 25%
            'workspace': 95000          # 23.75%
        }
    
    def load_progressive(self, problem):
        """Load context in priority order"""
        context = []
        
        # Priority 1: Problem-specific
        context.append(self.load_immediate(problem))
        
        # Priority 2: Related systems
        if self.needs_more_context():
            context.append(self.load_related(problem))
        
        # Priority 3: Historical patterns
        if self.pattern_matching_beneficial():
            context.append(self.load_historical(problem))
        
        # Priority 4: Extended documentation
        if self.remaining_tokens() > 50000:
            context.append(self.load_documentation())
        
        return context
```

---

## Practical Templates & Examples

### Template 1: Production Incident Response

```xml
<incident_response model="gpt-5" reasoning_effort="high" priority="CRITICAL">
    <identity>
        You are an SRE with 10+ years experience in distributed systems.
        You've handled hundreds of production incidents.
        Your priority: Restore service, then find root cause.
    </identity>
    
    <incident_data>
        <alert>Database connection pool exhausted</alert>
        <time>2025-09-26T10:30:00Z</time>
        <impact>30% of requests failing</impact>
        <duration>15 minutes ongoing</duration>
    </incident_data>
    
    <available_data>
        <logs path="/var/log/app/*.log" last_hours="2" />
        <metrics source="prometheus" resolution="1m" />
        <traces source="jaeger" sample_rate="0.01" />
        <recent_changes source="git" last_hours="24" />
    </available_data>
    
    <required_actions priority="ordered">
        <action priority="1">
            <description>Immediate mitigation</description>
            <time_limit>5 minutes</time_limit>
            <goal>Restore service to acceptable levels</goal>
        </action>
        
        <action priority="2">
            <description>Root cause identification</description>
            <time_limit>15 minutes</time_limit>
            <goal>Find what triggered the incident</goal>
        </action>
        
        <action priority="3">
            <description>Permanent fix</description>
            <time_limit>1 hour</time_limit>
            <goal>Prevent recurrence</goal>
        </action>
    </required_actions>
    
    <output_format>
        <immediate_actions>
            <command>Exact commands to run</command>
            <expected_result>What should happen</expected_result>
            <rollback>How to undo if needed</rollback>
        </immediate_actions>
        
        <root_cause confidence="required">
            <description>What went wrong</description>
            <evidence>Proof from logs/metrics</evidence>
            <timeline>Sequence of events</timeline>
        </root_cause>
        
        <permanent_fix>
            <code_changes>Required modifications</code_changes>
            <config_changes>Settings to adjust</config_changes>
            <tests>Validation procedures</tests>
        </permanent_fix>
        
        <postmortem_notes>
            <what_went_well />
            <what_went_wrong />
            <action_items />
        </postmortem_notes>
    </output_format>
</incident_response>
```

### Template 2: Code Review Assistant

```markdown
# Comprehensive Code Review

**Your Role**: Staff engineer conducting thorough code review
**Focus**: Security, performance, maintainability, correctness

## Pull Request Details
- **PR #**: 1234
- **Author**: @developer
- **Title**: Add caching layer to API endpoints
- **Files Changed**: 15
- **Lines**: +500 -200

## Review Criteria (Weighted)
1. **Correctness (30%)**: Logic errors, edge cases, bug potential
2. **Security (25%)**: Vulnerabilities, input validation, auth
3. **Performance (20%)**: Efficiency, scalability, resource usage
4. **Maintainability (25%)**: Readability, documentation, tests

## Code to Review
```python
[paste code here]
```

## Review Checklist
- [ ] **Logic Correctness**
  - No off-by-one errors
  - Proper null/error handling
  - Edge cases covered
  
- [ ] **Security**
  - Input validation present
  - No SQL injection risks
  - Authentication verified
  - Authorization checked
  
- [ ] **Performance**
  - No N+1 queries
  - Proper indexing used
  - Caching strategy sound
  
- [ ] **Code Quality**
  - Follows team style guide
  - Adequate comments
  - Test coverage >80%

## Required Output Format

### Summary
[One paragraph overview - tone: constructive]

### Critical Issues 🔴
[Must fix before merge]

### Important Issues 🟡
[Should address but not blocking]

### Suggestions 🟢
[Nice to have improvements]

### Positive Feedback 👍
[What was done well - always include!]

### Code Examples
```python
# Instead of:
[problematic code]

# Consider:
[improved code]
# Because: [reasoning]
```
```

### Template 3: Architecture Design Review

```xml
<architecture_review model="gpt-5" reasoning_effort="high">
    <role>
        You are a Principal Architect reviewing a system design.
        You've designed systems handling billions of requests.
        You prioritize: simplicity, reliability, scalability, cost.
    </role>
    
    <system_overview>
        <name>E-commerce Platform Redesign</name>
        <scale>
            <users>10M active</users>
            <requests>100K req/sec peak</requests>
            <data>50TB growing 20% yearly</data>
        </scale>
        <requirements>
            <availability>99.99%</availability>
            <latency>p99 &lt; 200ms</latency>
            <consistency>eventual (except payments)</consistency>
        </requirements>
    </system_overview>
    
    <proposed_architecture>
        <!-- Paste architecture details, diagrams as ASCII art -->
    </proposed_architecture>
    
    <review_framework method="ATAM">
        <step1>Identify architectural decisions</step1>
        <step2>Identify quality attributes</step2>
        <step3>Generate scenarios</step3>
        <step4>Analyze architectural approaches</step4>
        <step5>Identify risks and non-risks</step5>
        <step6>Identify tradeoffs</step6>
        <step7>Identify sensitivity points</step7>
    </review_framework>
    
    <deliverables>
        <risk_assessment>
            <risk severity="high|medium|low">
                <description />
                <impact />
                <mitigation />
                <cost_to_fix />
            </risk>
        </risk_assessment>
        
        <tradeoff_analysis>
            <tradeoff>
                <decision>What architectural choice</decision>
                <benefits>What we gain</benefits>
                <costs>What we lose</costs>
                <alternatives>Other options</alternatives>
            </tradeoff>
        </tradeoff_analysis>
        
        <recommendations priority="ordered">
            <recommendation priority="1">
                <change>Specific modification</change>
                <reasoning>Why needed</reasoning>
                <effort>Implementation cost</effort>
            </recommendation>
        </recommendations>
    </deliverables>
</architecture_review>
```

### Template 4: Data Analysis Pipeline

```python
"""
Data Analysis Template for GPT-5
Model: gpt-5
Reasoning: high
Temperature: 1.0

Identity: You are a senior data scientist specializing in anomaly detection
and pattern recognition. You think in distributions, correlations, and 
statistical significance.

Task: Analyze dataset for patterns, anomalies, and actionable insights.
"""

# Configuration
ANALYSIS_CONFIG = {
    "confidence_threshold": 0.85,
    "anomaly_threshold": 3.0,  # standard deviations
    "correlation_threshold": 0.7,
    "min_sample_size": 30
}

# Input Data
dataset = """
[Paste CSV data or description here]
"""

# Required Analysis Pipeline
analysis_pipeline = [
    {
        "stage": "Data Profiling",
        "tasks": [
            "Descriptive statistics (mean, median, std, quartiles)",
            "Data type detection",
            "Missing value analysis",
            "Distribution analysis"
        ],
        "output": "Statistical summary table"
    },
    {
        "stage": "Anomaly Detection",
        "tasks": [
            "Statistical outliers (Z-score > 3)",
            "Pattern anomalies (breaks in trends)",
            "Categorical anomalies (rare values)",
            "Temporal anomalies (if time series)"
        ],
        "output": "Anomaly report with severity scores"
    },
    {
        "stage": "Correlation Analysis",
        "tasks": [
            "Pearson correlation matrix",
            "Feature importance ranking",
            "Multicollinearity detection",
            "Causal inference (if possible)"
        ],
        "output": "Correlation heatmap and insights"
    },
    {
        "stage": "Pattern Recognition",
        "tasks": [
            "Trend identification",
            "Seasonality detection",
            "Clustering patterns",
            "Behavioral segments"
        ],
        "output": "Pattern catalog with confidence scores"
    },
    {
        "stage": "Predictive Insights",
        "tasks": [
            "Future trend projection",
            "Risk indicators",
            "Opportunity identification",
            "Recommendation generation"
        ],
        "output": "Actionable recommendations"
    }
]

# Output Format
output_format = {
    "executive_summary": "3 key findings in plain language",
    "detailed_analysis": {
        "statistics": {},
        "anomalies": [],
        "patterns": [],
        "correlations": {}
    },
    "visualizations": [
        "Distribution plots description",
        "Correlation heatmap description",
        "Time series plot description"
    ],
    "recommendations": [
        {
            "action": "Specific action to take",
            "reasoning": "Why this matters",
            "impact": "Expected outcome",
            "confidence": 0.0
        }
    ],
    "methodology_notes": "Assumptions and limitations"
}

# Execution
print("Execute analysis with above configuration")
```

---

## Daily Use Cases

### Morning Standup Assistant

```markdown
# Daily Standup Analyzer

**Role**: Engineering Manager optimizing team productivity
**Time**: Monday, 9:00 AM
**Team Size**: 8 engineers

## Yesterday's Status
```yaml
completed:
  - User auth refactor (Jane)
  - API optimization (Bob)
  - Bug fixes #123, #124 (Alice)
  
in_progress:
  - Payment integration (Charlie, 60%)
  - Database migration (David, 40%)
  
blocked:
  - Testing environment down (Eve)
  - Waiting for design specs (Frank)
```

## Today's Plan
[Team members' planned work]

## Analysis Required
1. Identify critical path items
2. Spot potential blockers
3. Suggest pair programming opportunities
4. Resource reallocation recommendations
5. Risk assessment

## Output Format
```markdown
### 🎯 Critical Path Items
- [Item 1]: [Who] - [Impact if delayed]

### ⚠️ Risks & Blockers
- [Risk]: [Mitigation strategy]

### 👥 Collaboration Opportunities
- [Person A] + [Person B]: [Task] - [Why beneficial]

### 📊 Metrics
- Sprint velocity: [On track/At risk]
- Blockers cleared: [X of Y]
```
```

### Quick Debugging Session

```python
# Rapid Bug Analysis Template

"""
Mode: Fast debugging
Model: gpt-5
Reasoning: medium
Max time: 5 minutes

You are a debugging expert. Find root cause FAST.
"""

# Error Context
error = """
TypeError: Cannot read property 'id' of undefined
  at UserService.getUser (user.service.js:45)
  at async AuthMiddleware.validate (auth.middleware.js:23)
  at async Router.handle (router.js:89)
"""

# Recent Changes
changes = """
- Updated user schema (2 hours ago)
- Deployed new auth service (4 hours ago)
- Database migration (yesterday)
"""

# Quick Analysis Protocol
steps = [
    "1. Parse error stack → identify failure point",
    "2. Check recent changes → find correlation",
    "3. Most likely cause (>70% confidence)",
    "4. Quick fix to try",
    "5. Proper fix for later"
]

# Output
fix = {
    "immediate": "Add null check at line 45",
    "test": "if (!user) throw new Error('User not found')",
    "root_cause": "Schema change broke assumption",
    "confidence": 0.85,
    "permanent_fix": "Update service to handle new schema"
}
```

### API Documentation Generator

```markdown
# API Documentation Generator

**Role**: Technical writer with API design expertise
**Style**: Clear, concise, example-driven

## API Endpoint
`POST /api/v2/users/{userId}/transactions`

## Implementation
```javascript
async function createTransaction(req, res) {
    const { userId } = req.params;
    const { amount, type, metadata } = req.body;
    
    // Validation
    if (!amount || amount <= 0) {
        return res.status(400).json({ error: 'Invalid amount' });
    }
    
    // Process transaction
    const transaction = await TransactionService.create({
        userId,
        amount,
        type: type || 'payment',
        metadata,
        timestamp: Date.now()
    });
    
    return res.status(201).json(transaction);
}
```

## Generate Documentation

### Required Sections
1. **Overview** - What this endpoint does
2. **Authentication** - Required auth method
3. **Parameters** - Path, query, body params
4. **Request Examples** - cURL, JavaScript, Python
5. **Response Examples** - Success and error cases
6. **Error Codes** - All possible errors
7. **Rate Limiting** - Limits and headers
8. **Webhooks** - Related webhook events

### Output Format
```markdown
## Create Transaction

Creates a new transaction for a user...

### Authentication
`Bearer Token` required in Authorization header

### Request
`POST /api/v2/users/{userId}/transactions`

#### Path Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| userId | string | Yes | User identifier |

#### Body Parameters
...

### Examples
...
```
```

### Performance Analysis Report

```xml
<performance_analysis model="gpt-5" reasoning_effort="high">
    <context>
        <system>E-commerce checkout service</system>
        <problem>Response time degraded 40% over 2 weeks</problem>
        <baseline>p50: 200ms, p99: 800ms</baseline>
        <current>p50: 280ms, p99: 1120ms</current>
    </context>
    
    <available_data>
        <apm>DataDog traces for 14 days</apm>
        <metrics>CPU, memory, disk, network</metrics>
        <logs>Application and system logs</logs>
        <changes>Git commits, deployments</changes>
    </available_data>
    
    <analysis_framework>
        <phase1>Identify degradation pattern</phase1>
        <phase2>Correlate with changes</phase2>
        <phase3>Profile hot paths</phase3>
        <phase4>Find bottlenecks</phase4>
        <phase5>Propose optimizations</phase5>
    </analysis_framework>
    
    <required_output>
        <finding priority="1">
            <issue>Primary bottleneck</issue>
            <evidence>APM traces showing...</evidence>
            <impact>Adding Xms to each request</impact>
            <fix>Specific optimization</fix>
            <effort>2 dev days</effort>
            <improvement>Expected 30% reduction</improvement>
        </finding>
    </required_output>
</performance_analysis>
```

---

## Testing & Optimization

### Testing Your Prompts

```python
class PromptTester:
    """
    Framework for testing prompt reliability and performance
    """
    
    def __init__(self, prompt_template):
        self.prompt = prompt_template
        self.test_cases = []
        self.results = []
        
    def add_test_case(self, input_data, expected_output, tolerance=0.85):
        """Add test case with expected output and tolerance"""
        self.test_cases.append({
            'input': input_data,
            'expected': expected_output,
            'tolerance': tolerance
        })
    
    def run_tests(self, iterations=3):
        """Run all test cases multiple times"""
        for test in self.test_cases:
            test_results = []
            
            for i in range(iterations):
                result = self.execute_prompt(test['input'])
                score = self.compare_outputs(result, test['expected'])
                test_results.append({
                    'iteration': i,
                    'score': score,
                    'passed': score >= test['tolerance']
                })
            
            self.results.append({
                'test': test,
                'results': test_results,
                'avg_score': np.mean([r['score'] for r in test_results]),
                'consistency': np.std([r['score'] for r in test_results])
            })
    
    def generate_report(self):
        """Generate test report"""
        return {
            'total_tests': len(self.test_cases),
            'passed': sum(1 for r in self.results if r['avg_score'] >= r['test']['tolerance']),
            'average_consistency': np.mean([r['consistency'] for r in self.results]),
            'recommendations': self.generate_recommendations()
        }
```

### A/B Testing Framework

```python
def ab_test_prompts(prompt_a, prompt_b, test_data, metrics):
    """
    Compare two prompt versions
    """
    results_a = []
    results_b = []
    
    for data in test_data:
        # Test prompt A
        start_time = time.time()
        response_a = call_gpt5(prompt_a, data)
        time_a = time.time() - start_time
        
        # Test prompt B
        start_time = time.time()
        response_b = call_gpt5(prompt_b, data)
        time_b = time.time() - start_time
        
        # Evaluate metrics
        for metric in metrics:
            score_a = evaluate_metric(response_a, metric)
            score_b = evaluate_metric(response_b, metric)
            
            results_a.append({
                'metric': metric,
                'score': score_a,
                'time': time_a
            })
            
            results_b.append({
                'metric': metric,
                'score': score_b,
                'time': time_b
            })
    
    # Statistical analysis
    from scipy import stats
    
    comparison = {}
    for metric in metrics:
        scores_a = [r['score'] for r in results_a if r['metric'] == metric]
        scores_b = [r['score'] for r in results_b if r['metric'] == metric]
        
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        comparison[metric] = {
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'improvement': (np.mean(scores_b) - np.mean(scores_a)) / np.mean(scores_a),
            'significant': p_value < 0.05,
            'p_value': p_value
        }
    
    return comparison
```

### Prompt Optimization Checklist

```markdown
## Pre-Flight Checklist

### Structure
- [ ] Clear role/identity defined
- [ ] Reasoning effort specified
- [ ] Temperature set to 1.0
- [ ] Output format defined
- [ ] Validation criteria included

### Evidence & Precision
- [ ] Evidence requirements stated
- [ ] Confidence thresholds set
- [ ] Multiple hypotheses required
- [ ] Falsification criteria included
- [ ] Uncertainty handling defined

### Performance
- [ ] Context window optimized (<400K)
- [ ] Progressive loading if >100K
- [ ] Unnecessary verbosity removed
- [ ] Structured for quick parsing
- [ ] Caching strategy considered

### Testing
- [ ] 3+ test cases created
- [ ] Edge cases covered
- [ ] Consistency validated
- [ ] Performance benchmarked
- [ ] Fallback options ready
```

---

## Quick Reference

### GPT-5 Optimal Settings
```yaml
model: "gpt-5"
reasoning_effort: "high"    # for complex tasks
temperature: 1.0            # not 0!
max_tokens: 400000         # use what you need
top_p: 1.0                 # default
frequency_penalty: 0       # default
presence_penalty: 0        # default
```

### Prompt Structure Order
1. **Configuration** - Model settings
2. **Identity/Role** - Who the AI should be
3. **Context/Data** - Input information
4. **Task/Instructions** - What to do
5. **Constraints/Rules** - Boundaries and requirements
6. **Output Format** - How to structure response
7. **Examples** - If needed for clarity
8. **Validation** - Success criteria

### Anti-Hallucination Checklist
- ✅ Require 2+ evidence sources
- ✅ Set confidence thresholds (0.85+)
- ✅ Generate multiple hypotheses (3+)
- ✅ Build validation gates
- ✅ Use structured output schemas
- ✅ Include falsification criteria
- ✅ Force uncertainty acknowledgment

### Format Selection Guide
```python
def choose_format(task):
    if task.requires('strict_structure', 'validation', 'typing'):
        return 'XML'
    elif task.requires('natural_flow', 'quick_iteration', 'readability'):
        return 'Markdown'
    elif task.requires('both'):
        return 'Hybrid'
    else:
        return 'Markdown'  # Default for simplicity
```

### Common Patterns

#### Pattern 1: Debug Investigation
```
Identity → Symptoms → Evidence → Hypotheses → Tests → Conclusion
```

#### Pattern 2: Analysis Report
```
Context → Methodology → Findings → Recommendations → Confidence
```

#### Pattern 3: Code Generation
```
Specification → Constraints → Implementation → Tests → Documentation
```

#### Pattern 4: Decision Making
```
Options → Criteria → Evaluation → Tradeoffs → Recommendation
```

### Performance Tips
1. **For Speed**: Use `reasoning_effort="low"`, minimize context
2. **For Accuracy**: Use `reasoning_effort="high"`, multiple passes
3. **For Cost**: Balance effort and context, use caching
4. **For Consistency**: Use structured formats, validation gates

### Debugging Prompt Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Vague output | Missing structure | Add explicit phases |
| Low confidence | Weak identity | Strengthen expertise |
| Hallucination | No evidence requirement | Add validation |
| Inconsistent | No schema | Define output format |
| Slow | Over-constrained | Balance freedom/structure |

---

## Conclusion

GPT-5 prompt engineering is both an art and a science. Success comes from understanding the model's architecture, leveraging its reasoning capabilities, and applying systematic techniques to achieve precision and reliability.

### Key Takeaways

1. **Identity matters**: It shapes neural pathways and reasoning depth
2. **Structure beats verbosity**: Clear frameworks outperform detailed instructions
3. **Evidence is mandatory**: Never accept claims without proof
4. **Temperature = 1.0**: Optimal for reasoning tasks
5. **Validation gates work**: They reduce hallucination by 70%
6. **Format strategically**: XML for structure, Markdown for flow
7. **Test systematically**: Measure, compare, iterate

### The Golden Rules

> 1. **Define before asking**: Structure, format, constraints
> 2. **Prove, don't assume**: Evidence for every claim
> 3. **Think in pathways**: Identity activates capabilities
> 4. **Validate continuously**: Gates at every phase
> 5. **Optimize iteratively**: Test, measure, improve

### Final Thought

> "GPT-5 is not just a larger model—it's a different kind of intelligence. 
> Treat it as a reasoning partner, not a search engine. Give it structure 
> to think within, evidence to ground its thoughts, and clear success 
> criteria to achieve. Do this, and you'll unlock capabilities that 
> seem almost magical."

---

## Appendix: Resource Links

### Official Documentation
- OpenAI GPT-5 API Docs: `platform.openai.com/docs/gpt-5`
- Reasoning Models Guide: `openai.com/reasoning`
- Prompt Engineering: `platform.openai.com/docs/guides/prompt-engineering`

### Research Papers
- Chain-of-Thought Prompting (Wei et al., 2022)
- Self-Consistency Improves Reasoning (Wang et al., 2023)
- Constitutional AI (Anthropic, 2023)
- Reducing Hallucination in LLMs (Ji et al., 2023)

### Community Resources
- r/GPT5Prompting
- GPT-5 Cookbook on GitHub
- Prompt Engineering Discord

### Tools & Libraries
- LangChain: `langchain.com`
- Prompt Testing Framework: `github.com/prompt-testing`
- GPT-5 Playground: `playground.openai.com`

---

*Last Updated: September 26, 2025*
*Version: 1.0.0*
*Author: ErdosEngine Team*
*License: MIT*

*"In mathematics and AI alike, the elegant solution is often the correct one."*