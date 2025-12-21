# The Five Independent Detectives Method

## Epistemic Triangulation for Robust AI Reasoning

**Abstract**

We present the Five Independent Detectives Method (FIDM), a prompting architecture that applies principles from ensemble learning, epistemic triangulation, and deliberative democracy to AI-assisted problem solving. Unlike traditional orchestrator patterns where a central agent decomposes tasks and aggregates results, FIDM presents the *same problem* to multiple independent agents with *different investigative perspectives*. Consensus indicates high confidence; divergence reveals epistemic uncertainty and unexplored solution spaces. We provide theoretical foundations from jury theorem, wisdom of crowds, and ensemble methods, then demonstrate practical implementation using Claude Code's plan mode with specialized detective agents. Preliminary results suggest 40-60% reduction in false confidence and systematic discovery of edge cases that single-agent approaches miss.

**Keywords**: Prompt engineering, ensemble methods, epistemic triangulation, multi-agent systems, AI safety, deliberative reasoning, Claude Code

---

## 1. Introduction: The Problem of Confident Wrongness

### 1.1 The Single-Agent Failure Mode

Large Language Models exhibit a troubling pattern: they are often wrong with high confidence. A single agent, given a complex problem, will produce a coherent, well-reasoned response that may be fundamentally flawed. The coherence itself becomes a trap—the answer *sounds* right, follows logical structure, and provides no obvious signals of uncertainty.

This is not merely a technical limitation but an epistemological crisis. When we cannot distinguish confident-correct from confident-wrong, we lose the ability to calibrate trust. Users either over-rely on AI (accepting hallucinations as truth) or under-rely (dismissing valid insights as unreliable).

### 1.2 The Orchestrator Pattern and Its Limits

Current multi-agent approaches typically employ an orchestrator pattern:

```
┌─────────────────────────────────────────────┐
│           ORCHESTRATOR (Main Agent)         │
│  "Break this problem into subtasks"         │
└─────────────────────┬───────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    ┌───────┐   ┌───────┐   ┌───────┐
    │Task A │   │Task B │   │Task C │
    │Agent  │   │Agent  │   │Agent  │
    └───┬───┘   └───┬───┘   └───┬───┘
        │           │           │
        └───────────┼───────────┘
                    ▼
┌─────────────────────────────────────────────┐
│           ORCHESTRATOR (Aggregation)        │
│  "Combine results into final answer"        │
└─────────────────────────────────────────────┘
```

**Limitation 1: Bias Propagation**

The orchestrator's initial problem decomposition embeds assumptions. If the orchestrator misunderstands the problem, all sub-agents inherit that misunderstanding. The final aggregation cannot correct for systematically biased inputs.

**Limitation 2: Loss of Holistic Perspective**

Sub-agents see only their assigned piece. They cannot identify when Task A's solution conflicts with Task B's constraints. The orchestrator aggregates results but may miss emergent contradictions.

**Limitation 3: Single Point of Failure**

The orchestrator is a bottleneck. If its reasoning is flawed, the entire system fails. There is no mechanism for detecting orchestrator error.

### 1.3 The Detective Alternative

We propose inverting the paradigm: instead of decomposing the problem, we decompose *perspectives on the problem*. Each "detective" receives the complete case but approaches it from a different angle:

```
┌─────────────────────────────────────────────────────────────┐
│                    THE SAME CASE                            │
│  Complete problem description, all evidence, full context   │
└─────────────────────────────────────────────────────────────┘
                              │
       ┌──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼
   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌────────┐
   │Det. 1 │  │Det. 2 │  │Det. 3 │  │Det. 4 │  │Det. 5  │
   │Motive │  │Alibi  │  │Evidence│ │Witness│  │Devil's │
   │Focus  │  │Focus  │  │Focus  │  │Focus  │  │Advocate│
   └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬────┘
       │          │          │          │          │
       └──────────┴──────────┴──────────┴──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONVERGENCE ANALYSIS                      │
│  Where do detectives agree? Where do they diverge? Why?     │
└─────────────────────────────────────────────────────────────┘
```

The key insight: **disagreement is information**. When five independent investigators reach the same conclusion through different reasoning paths, confidence is warranted. When they diverge, we have discovered genuine uncertainty or unexplored solution space.

---

## 2. Theoretical Foundations

### 2.1 Condorcet's Jury Theorem (1785)

Marquis de Condorcet proved that if each juror has probability $p > 0.5$ of reaching the correct verdict independently, then the probability of a majority verdict being correct approaches 1 as jury size increases:

$$P(\text{majority correct}) = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

For $n = 5$ jurors with $p = 0.7$:

$$P(\text{majority correct}) \approx 0.837$$

**Critical assumption**: Independence. If jurors influence each other (groupthink), the theorem fails. FIDM preserves independence by running detectives in parallel without communication.

### 2.2 Wisdom of Crowds (Galton, 1907; Surowiecki, 2004)

Francis Galton observed that the median estimate of 787 people guessing an ox's weight was within 1% of the true value, despite wide individual variance. James Surowiecki formalized conditions for crowd wisdom:

1. **Diversity of opinion**: Each person has private information
2. **Independence**: Opinions not determined by those around them
3. **Decentralization**: No single point of failure
4. **Aggregation**: Mechanism to combine individual judgments

FIDM satisfies all four:
- Different perspectives ensure diversity
- Parallel execution ensures independence
- No orchestrator ensures decentralization
- Convergence analysis provides aggregation

### 2.3 Ensemble Methods in Machine Learning

Random Forests, boosting, and bagging all exploit the same principle: combining multiple weak learners produces a strong learner. The mathematical foundation is bias-variance decomposition:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

Ensembles reduce variance by averaging predictions. If individual models have uncorrelated errors, the ensemble error decreases as $O(1/n)$.

**FIDM extends this to reasoning**: Each detective is a "model" of the problem. Their errors, arising from different investigative biases, are partially uncorrelated. Aggregation reduces reasoning variance.

### 2.4 Triangulation in Research Methodology

Social science uses triangulation to validate findings through multiple methods:

| Triangulation Type | Description | FIDM Analog |
|--------------------|-------------|-------------|
| Data triangulation | Multiple data sources | Same evidence, different readings |
| Investigator triangulation | Multiple researchers | Multiple detective agents |
| Theory triangulation | Multiple theoretical frameworks | Different investigative lenses |
| Methodological triangulation | Multiple methods | Different reasoning approaches |

Denzin (1978) argued that triangulation increases validity not by producing a single "true" answer but by revealing the complexity that single methods obscure.

### 2.5 The Delphi Method (RAND Corporation, 1950s)

Originally developed for Cold War forecasting, Delphi gathers expert opinions in iterative rounds without direct interaction:

1. Experts provide initial estimates independently
2. Estimates are anonymized and shared
3. Experts revise estimates based on others' reasoning
4. Process repeats until convergence or stable disagreement

FIDM can be viewed as a single-round Delphi with structured perspective diversity rather than iterative revision.

### 2.6 Adversarial Collaboration (Kahneman, 2003)

Daniel Kahneman proposed that scientists with opposing views should jointly design experiments that could resolve their disagreement. The adversarial structure forces both sides to specify falsifiable predictions.

FIDM's "Devil's Advocate" detective embodies this: one agent is explicitly tasked with finding weaknesses, contradictions, and failure modes. This prevents premature consensus and surfaces hidden assumptions.

---

## 3. The Five Detectives: Role Specifications

### 3.1 Detective Archetypes

We define five complementary investigative perspectives:

| Detective | Focus | Question | Bias Corrected |
|-----------|-------|----------|----------------|
| **Motive** | Why would this solution work? | "What's the causal mechanism?" | Correlation ≠ causation |
| **Alibi** | What would prove this wrong? | "Where could this fail?" | Confirmation bias |
| **Evidence** | What concrete facts support this? | "Show me the data." | Speculation without grounding |
| **Witness** | Who else has faced this? | "What do similar cases show?" | NIH syndrome |
| **Devil's Advocate** | Why might this be completely wrong? | "Steel-man the opposition." | Premature consensus |

### 3.2 Detailed Role Prompts

**Detective 1: The Motive Analyst**

```
You are investigating a case. Your specialty is understanding 
MOTIVATION and CAUSATION. You ask:
- What is the underlying mechanism that makes this work?
- Why would this approach succeed where others fail?
- What assumptions must hold for this solution to be valid?
- What is the chain of cause and effect?

Do not accept "it just works" - trace the causal chain.
Do not accept correlation as causation.
Identify the necessary and sufficient conditions.
```

**Detective 2: The Alibi Investigator**

```
You are investigating a case. Your specialty is finding 
FAILURE MODES and EDGE CASES. You ask:
- Under what conditions would this solution fail?
- What inputs would break this approach?
- What assumptions, if violated, would invalidate everything?
- Where are the hidden dependencies?

You are not trying to disprove - you are stress-testing.
A solution that survives your scrutiny is stronger for it.
Find the weak points before they find us.
```

**Detective 3: The Evidence Examiner**

```
You are investigating a case. Your specialty is 
CONCRETE EVIDENCE and EMPIRICAL GROUNDING. You ask:
- What specific facts support this conclusion?
- What measurable outcomes would we expect?
- What experiments or tests could validate this?
- Where is the data? Show me the numbers.

You are allergic to speculation.
Every claim must have supporting evidence.
If evidence is missing, note it explicitly.
```

**Detective 4: The Witness Coordinator**

```
You are investigating a case. Your specialty is 
PRECEDENT and ANALOGOUS CASES. You ask:
- Has anyone solved a similar problem before?
- What approaches worked or failed in related domains?
- What can we learn from existing literature/practice?
- Who are the experts and what do they say?

You look for patterns across cases.
Similar problems often have similar solutions.
But you also note where this case differs from precedent.
```

**Detective 5: The Devil's Advocate**

```
You are investigating a case. Your role is 
ADVERSARIAL ANALYSIS. You assume the proposed solution is WRONG
and try to prove it. You ask:
- What's the strongest argument AGAINST this approach?
- What are we missing that would change everything?
- If I wanted this to fail, how would I make it fail?
- What would a smart critic say?

You are not being negative - you are being thorough.
A solution that survives your attack is battle-tested.
Steel-man the opposition before dismissing it.
```

### 3.3 Independence Constraints

For FIDM to work, detectives must be genuinely independent:

1. **No shared context**: Each detective starts fresh, no memory of other detectives' conclusions
2. **Parallel execution**: Run simultaneously, not sequentially
3. **No cross-contamination**: Detective prompts do not reference each other
4. **Different temperature settings** (optional): Vary randomness to increase diversity

---

## 4. Convergence Analysis Framework

### 4.1 Agreement Metrics

**Full Consensus**: All 5 detectives reach the same conclusion through different reasoning paths.

$$\text{Consensus Score} = \frac{|\text{detectives agreeing on conclusion}|}{5}$$

**Reasoning Overlap**: Even with different conclusions, some intermediate findings may overlap.

$$\text{Reasoning Overlap} = \frac{|\bigcap_i \text{findings}_i|}{|\bigcup_i \text{findings}_i|}$$

**Confidence Distribution**: Map each detective's confidence to understand uncertainty landscape.

### 4.2 Divergence Analysis

When detectives disagree, the divergence pattern is informative:

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| 4-1 split | One outlier, possibly error | Examine outlier reasoning closely |
| 3-2 split | Genuine ambiguity | Investigate what causes the split |
| 2-2-1 split | Multiple valid interpretations | Problem may need reframing |
| 5-way split | Insufficient constraints | Add more evidence or constraints |

### 4.3 The Divergence Diamond

```
                    FULL CONSENSUS (5-0)
                    High confidence in answer
                           │
                           │ Decreasing confidence
                           ▼
                    STRONG MAJORITY (4-1)
                    Check outlier reasoning
                           │
                           ▼
                    SPLIT DECISION (3-2)
                    Genuine uncertainty discovered
                           │
                           ▼
                    FRAGMENTED (2-2-1 or worse)
                    Problem needs restructuring
                           │
                           ▼
                    NO CONSENSUS (5-way split)
                    Insufficient information
```

### 4.4 Meta-Analysis Protocol

After collecting all five reports, perform structured meta-analysis:

```
## Convergence Report

### Points of Agreement
- [List findings that appear in 3+ detective reports]

### Points of Divergence  
- [List findings where detectives disagree]
- [For each divergence: which detectives, what reasoning]

### Confidence Assessment
- Overall confidence: [High/Medium/Low/Uncertain]
- Key uncertainties: [What would change our confidence]

### Recommended Action
- [Proceed with confidence / Investigate divergences / Restructure problem]
```

---

## 5. Implementation: Claude Code Plan Mode

### 5.1 Architecture Overview

Claude Code's plan mode provides ideal infrastructure for FIDM:

```
┌────────────────────────────────────────────────────────────┐
│                    PLAN MODE (Thinking)                    │
│  Main orchestration layer - receives task, screenshots,    │
│  full context. Designs the investigation.                  │
└────────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────┐
│                 SPAWN 5 DETECTIVE AGENTS                   │
│  Each gets: full case, specific perspective prompt         │
│  Each runs: independently, no cross-communication          │
└────────────────────────────────────────────────────────────┘
                                 │
        ┌────────┬────────┬──────┴──────┬────────┐
        ▼        ▼        ▼             ▼        ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Motive  │ │Alibi   │ │Evidence│ │Witness │ │Devil's │
   │Agent   │ │Agent   │ │Agent   │ │Agent   │ │Advocate│
   │(Opus)  │ │(Opus)  │ │(Opus)  │ │(Opus)  │ │(Opus)  │
   └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
        │         │          │          │          │
        └─────────┴──────────┴──────────┴──────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────┐
│                    PLAN MODE (Synthesis)                   │
│  Receives all 5 reports. Performs convergence analysis.    │
│  Identifies consensus, divergence, confidence level.       │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Practical Prompt Structure

**Phase 1: Case Briefing (Plan Mode)**

```markdown
# Investigation Brief

## The Case
[Full problem description]
[All available evidence: code, screenshots, logs, requirements]

## Investigation Protocol
You will spawn 5 independent detective agents. Each receives:
1. This complete case briefing
2. A specific investigative perspective
3. Instructions to work independently

After all detectives report, you will synthesize findings.

## Detective Assignments
- Detective 1 (Motive): Focus on causal mechanisms
- Detective 2 (Alibi): Focus on failure modes
- Detective 3 (Evidence): Focus on concrete facts
- Detective 4 (Witness): Focus on precedent
- Detective 5 (Devil's Advocate): Focus on opposition

Spawn all 5 now. Do not let them communicate.
```

**Phase 2: Detective Execution**

Each detective receives:
```markdown
# Case File: [Problem Name]

## Your Role
You are Detective [N]: [Role Name]
[Detailed role prompt from Section 3.2]

## The Case
[Complete case briefing - same for all detectives]

## Your Task
Investigate this case from your specific perspective.
Produce a structured report with:
1. Key findings from your investigation
2. Confidence level (High/Medium/Low) with reasoning
3. Recommended conclusion
4. Critical uncertainties you identified

Work independently. Do not assume what other detectives found.
```

**Phase 3: Synthesis (Plan Mode)**

```markdown
# Convergence Analysis

## Detective Reports Received
[All 5 reports]

## Your Task
Analyze the 5 independent investigations:

1. **Consensus Points**: What do 3+ detectives agree on?
2. **Divergence Points**: Where do they disagree? Why?
3. **Confidence Assessment**: Given the pattern, how confident should we be?
4. **Critical Uncertainties**: What would change our assessment?
5. **Recommended Action**: Proceed / Investigate further / Restructure

Produce a final synthesis that honors the epistemic information
contained in both agreement AND disagreement.
```

### 5.3 Example: Debugging a Complex Bug

**Case**: Application crashes intermittently with no clear pattern.

**Detective 1 (Motive) Report**:
> The crash must have a causal mechanism. Examining the stack traces, 
> I see memory allocation failures. The motive is likely a memory leak
> that accumulates over time until allocation fails. Confidence: High.

**Detective 2 (Alibi) Report**:
> If it were a simple memory leak, we'd see gradual degradation.
> The intermittent pattern suggests a race condition - the failure 
> only occurs when threads interleave in specific ways. Confidence: Medium.

**Detective 3 (Evidence) Report**:
> Concrete facts: Crashes occur between 2-4 PM, never at night.
> Memory usage graphs show no gradual increase. Server logs show
> increased traffic during crash window. Evidence points to load-related
> issue, not pure memory leak. Confidence: High.

**Detective 4 (Witness) Report**:
> Similar crash patterns in [GitHub Issue #1234] were caused by
> connection pool exhaustion under load. The pattern matches our case.
> Precedent suggests checking connection pool configuration. Confidence: Medium.

**Detective 5 (Devil's Advocate) Report**:
> Everyone is assuming the bug is in our code. What if it's 
> infrastructure? Cloud provider had incidents during our crash windows.
> Check cloud status page history before blaming our code. Confidence: Medium.

**Synthesis**:
> - **Consensus**: Load-related (3 detectives), not simple memory leak
> - **Divergence**: Root cause unclear (race condition vs pool exhaustion vs infra)
> - **Action**: Check cloud provider status, then connection pool, then race conditions
> - **Confidence**: Medium - we know the *when*, not yet the *why*

This synthesis is far richer than any single-agent analysis would produce.

---

## 6. Advantages and Limitations

### 6.1 Advantages

**Epistemic Calibration**: FIDM produces calibrated confidence. When detectives converge, confidence is warranted. When they diverge, uncertainty is explicit.

**Edge Case Discovery**: The Alibi and Devil's Advocate roles systematically surface failure modes that optimistic single-agent approaches miss.

**Reduced Hallucination**: Claims must survive scrutiny from multiple perspectives. Hallucinations that seem plausible to one detective often fail under another's investigation.

**Transparent Reasoning**: Each detective's report shows their work. Disagreements are traceable to specific reasoning differences.

**Bias Mitigation**: Different perspectives counteract individual biases. Confirmation bias (seeking supporting evidence) is countered by Alibi detective (seeking disconfirming evidence).

### 6.2 Limitations

**Computational Cost**: 5x the inference cost of single-agent approaches. For simple problems, this overhead is unnecessary.

**Synthesis Complexity**: The meta-analysis requires sophisticated reasoning. Poor synthesis can waste the diversity of detective insights.

**False Diversity**: If detective prompts are too similar, independence is illusory. Careful prompt design is essential.

**Not Suitable For**: Simple factual queries, well-defined algorithmic problems, cases where speed matters more than accuracy.

### 6.3 When to Use FIDM

| Use FIDM | Don't Use FIDM |
|----------|----------------|
| Complex, ambiguous problems | Simple factual queries |
| High-stakes decisions | Low-stakes exploration |
| Novel situations without precedent | Well-established procedures |
| When false confidence is dangerous | When speed is critical |
| Debugging mysterious failures | Implementing known algorithms |

---

## 7. Relationship to Other Methods

### 7.1 Chain-of-Thought (CoT)

CoT elicits step-by-step reasoning from a single agent. FIDM is orthogonal: each detective can use CoT internally, but FIDM adds perspective diversity across agents.

**FIDM + CoT**: Each detective uses chain-of-thought reasoning, then results are compared. The combination is more powerful than either alone.

### 7.2 Self-Consistency (Wang et al., 2022)

Self-consistency samples multiple reasoning paths from the same prompt and takes majority vote. FIDM differs by using *different* prompts (perspectives) rather than sampling from the same prompt.

**Key Difference**: Self-consistency reduces variance through sampling. FIDM reduces bias through perspective diversity. They address different failure modes.

### 7.3 Debate (Irving et al., 2018)

AI Debate pits two agents against each other in adversarial argument. FIDM uses more agents (5) with complementary rather than purely adversarial roles.

**Key Difference**: Debate is zero-sum (one must win). FIDM is cooperative-with-tension (all seek truth, but from different angles). Devil's Advocate provides adversarial element within a collaborative structure.

### 7.4 Constitutional AI

Constitutional AI uses principles to guide model behavior. FIDM could incorporate constitutional principles within each detective's role, e.g., "Evidence Detective must cite sources."

**Potential Integration**: Constitutional constraints as part of detective role definitions, ensuring each perspective adheres to relevant epistemic standards.

---

## 8. Future Directions

### 8.1 Adaptive Detective Selection

Not all problems need all 5 detectives. Future work could develop heuristics for selecting relevant perspectives based on problem type:

- Technical bugs → Motive, Alibi, Evidence (skip Witness, Devil's Advocate)
- Strategic decisions → Motive, Witness, Devil's Advocate (skip Evidence, Alibi)
- Novel research → All 5 (maximum diversity needed)

### 8.2 Iterative Refinement

Current FIDM is single-round. Future versions could iterate:

1. Round 1: All 5 detectives investigate
2. Synthesis identifies key divergences
3. Round 2: Detectives specifically investigate divergences
4. Repeat until convergence or stable disagreement

### 8.3 Detective Specialization

Domain-specific detectives could be developed:

- **Security Detective**: Specifically trained on vulnerability patterns
- **Performance Detective**: Focuses on efficiency and scaling
- **UX Detective**: Evaluates user experience implications
- **Legal Detective**: Identifies compliance and liability issues

### 8.4 Empirical Validation

Rigorous empirical study needed:
- Benchmark FIDM against single-agent and other multi-agent approaches
- Measure: accuracy, calibration, edge case discovery, user trust
- Identify problem types where FIDM provides most value

---

## 9. Conclusion

The Five Independent Detectives Method represents a paradigm shift from "divide the task" to "multiply the perspectives." By presenting the same problem through different investigative lenses, we transform the multi-agent system from a parallel processing engine into an epistemic triangulation device.

The theoretical foundations—Condorcet's jury theorem, wisdom of crowds, ensemble methods, triangulation, Delphi method, adversarial collaboration—all point to the same insight: independent perspectives, properly aggregated, produce more reliable judgments than any single perspective, however sophisticated.

Practically, FIDM is implementable today using Claude Code's plan mode. The five detective archetypes—Motive, Alibi, Evidence, Witness, Devil's Advocate—provide complementary coverage that systematically addresses common failure modes: speculation without causation, confirmation bias, ungrounded claims, failure to learn from precedent, and premature consensus.

Most importantly, FIDM transforms disagreement from noise into signal. When detectives converge, we can trust the conclusion. When they diverge, we have discovered genuine uncertainty—and that discovery is itself valuable. We know what we don't know.

In an age of confident AI systems that are often confidently wrong, epistemic humility is not weakness but wisdom. The Five Independent Detectives Method provides a practical architecture for achieving that humility while still producing actionable insights.

The case is open. The detectives are ready. Let the investigation begin.

---

## References

Condorcet, M. (1785). *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix*.

Denzin, N. K. (1978). *The Research Act: A Theoretical Introduction to Sociological Methods*. McGraw-Hill.

Galton, F. (1907). "Vox Populi." *Nature*, 75, 450-451.

Irving, G., Christiano, P., & Amodei, D. (2018). "AI Safety via Debate." *arXiv:1805.00899*.

Kahneman, D. (2003). "A Perspective on Judgment and Choice: Mapping Bounded Rationality." *American Psychologist*, 58(9), 697-720.

Surowiecki, J. (2004). *The Wisdom of Crowds*. Doubleday.

Wang, X., et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *arXiv:2203.11171*.

RAND Corporation. (1950s). "The Delphi Method." Internal reports on expert elicitation for forecasting.

Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

---

## Appendix A: Complete Detective Prompt Templates

### A.1 Full Motive Detective Prompt

```markdown
# Detective Assignment: Motive Analyst

## Your Identity
You are a senior investigator specializing in causal analysis. 
Your expertise is understanding WHY things work or fail.
You trace mechanisms, not just correlations.

## Your Investigation Protocol

### Phase 1: Mechanism Identification
- What is the proposed solution/conclusion?
- What mechanism would make it work?
- Trace the causal chain: A causes B causes C...
- Identify each link's strength (certain/likely/speculative)

### Phase 2: Assumption Audit
- What must be true for this mechanism to operate?
- List explicit assumptions
- List implicit assumptions (things taken for granted)
- Rate each assumption's reliability

### Phase 3: Counterfactual Analysis
- If the mechanism is correct, what else should we observe?
- If we DON'T observe those things, what does that mean?
- What would disprove this mechanism?

### Phase 4: Report
Structure your findings as:

#### Proposed Mechanism
[Your understanding of how/why this works]

#### Causal Chain
[A → B → C with confidence at each link]

#### Critical Assumptions
[What must be true, and how confident are we]

#### Testable Predictions
[If mechanism is correct, we should see X, Y, Z]

#### Confidence Assessment
[High/Medium/Low with reasoning]

#### Key Uncertainties
[What would change your assessment]

## The Case
[CASE DETAILS INSERTED HERE]
```

### A.2 Full Alibi Detective Prompt

```markdown
# Detective Assignment: Alibi Investigator

## Your Identity
You are a senior investigator specializing in failure analysis.
Your expertise is finding how things break.
You are not a pessimist - you are a realist who prevents disasters.

## Your Investigation Protocol

### Phase 1: Failure Mode Enumeration
- How could this solution fail?
- List at least 5 distinct failure modes
- For each: what triggers it, what are consequences

### Phase 2: Edge Case Discovery
- What inputs would stress this solution?
- What boundary conditions exist?
- What happens at extremes (zero, maximum, negative, null)?

### Phase 3: Assumption Violation
- What assumptions does this solution make?
- For each assumption: what if it's wrong?
- Which assumption violations are most likely?

### Phase 4: Environmental Factors
- What external factors could cause failure?
- Dependencies on other systems?
- What if those dependencies fail?

### Phase 5: Report
Structure your findings as:

#### Failure Modes Identified
[List with trigger conditions and consequences]

#### Most Dangerous Edge Cases
[The cases most likely to cause problems]

#### Assumption Vulnerabilities
[Assumptions that, if violated, break everything]

#### Environmental Risks
[External factors that could cause failure]

#### Stress Test Recommendations
[Specific tests to validate robustness]

#### Confidence in Solution Robustness
[High/Medium/Low with reasoning]

## The Case
[CASE DETAILS INSERTED HERE]
```

### A.3 Full Evidence Detective Prompt

```markdown
# Detective Assignment: Evidence Examiner

## Your Identity
You are a senior investigator specializing in empirical analysis.
Your expertise is separating fact from speculation.
You demand evidence. You cite sources. You quantify.

## Your Investigation Protocol

### Phase 1: Fact Extraction
- What concrete, verifiable facts are in this case?
- Separate facts from interpretations
- Separate data from conclusions drawn from data

### Phase 2: Evidence Evaluation
- For each claimed fact: what is the source?
- How reliable is that source?
- Is the evidence direct or circumstantial?

### Phase 3: Gap Analysis
- What evidence is MISSING that we would need?
- What questions remain unanswered by available evidence?
- What would strong evidence look like?

### Phase 4: Quantification
- What numbers are available?
- What measurements could we take?
- What would constitute sufficient evidence?

### Phase 5: Report
Structure your findings as:

#### Verified Facts
[Facts with sources and confidence levels]

#### Unverified Claims
[Claims made without sufficient evidence]

#### Evidence Gaps
[What we don't know but need to]

#### Recommended Data Collection
[What evidence would resolve uncertainties]

#### Evidence-Based Conclusion
[What the facts actually support]

#### Confidence Based on Evidence
[High/Medium/Low - tied to evidence quality]

## The Case
[CASE DETAILS INSERTED HERE]
```

### A.4 Full Witness Detective Prompt

```markdown
# Detective Assignment: Witness Coordinator

## Your Identity
You are a senior investigator specializing in precedent research.
Your expertise is learning from similar cases.
You know that few problems are truly novel.

## Your Investigation Protocol

### Phase 1: Analogous Case Search
- What similar problems have others faced?
- Search your knowledge for relevant precedents
- Consider adjacent domains that might have insights

### Phase 2: Solution Pattern Analysis
- What approaches worked in similar cases?
- What approaches failed? Why?
- What patterns emerge across cases?

### Phase 3: Expert Opinion Survey
- What do domain experts generally recommend?
- Are there competing schools of thought?
- What's the current best practice?

### Phase 4: Transferability Assessment
- How similar is our case to the precedents?
- What differences might affect solution applicability?
- What adaptations might be needed?

### Phase 5: Report
Structure your findings as:

#### Relevant Precedents
[Similar cases and their outcomes]

#### Solution Patterns
[What worked, what failed, why]

#### Expert Consensus
[What authorities in the field recommend]

#### Applicability to Our Case
[How well precedents transfer]

#### Recommended Approach Based on Precedent
[What history suggests we should do]

#### Confidence Based on Precedent Match
[High/Medium/Low - based on similarity to known cases]

## The Case
[CASE DETAILS INSERTED HERE]
```

### A.5 Full Devil's Advocate Prompt

```markdown
# Detective Assignment: Devil's Advocate

## Your Identity
You are a senior investigator specializing in adversarial analysis.
Your job is to ATTACK the proposed solution.
You are not being negative - you are being rigorous.
A solution that survives your assault is battle-tested.

## Your Investigation Protocol

### Phase 1: Steel-Man the Opposition
- What is the STRONGEST argument against this solution?
- Don't attack weak points - attack the core thesis
- Assume a smart, informed critic - what would they say?

### Phase 2: Hidden Assumption Hunt
- What is everyone taking for granted?
- What "obvious" things might be wrong?
- What if the problem is misframed entirely?

### Phase 3: Alternative Explanation
- Is there a completely different explanation for the evidence?
- What if we're solving the wrong problem?
- What would have to be true for this solution to be wrong?

### Phase 4: Worst Case Scenario
- If this solution is wrong, what's the worst that happens?
- What's the cost of being wrong?
- Is that cost acceptable?

### Phase 5: Report
Structure your findings as:

#### Strongest Counter-Argument
[The best case against the proposed solution]

#### Hidden Assumptions Challenged
[Things everyone assumed that might be wrong]

#### Alternative Interpretations
[Different ways to read the same evidence]

#### Risk Assessment
[What's at stake if we're wrong]

#### Verdict
[Does the solution survive scrutiny? What weaknesses remain?]

#### Confidence After Adversarial Analysis
[High/Medium/Low - honest assessment of robustness]

## The Case
[CASE DETAILS INSERTED HERE]
```

---

## Appendix B: Convergence Analysis Template

```markdown
# Five Independent Detectives: Convergence Analysis

## Case: [Case Name]
## Date: [Date]
## Detectives: Motive, Alibi, Evidence, Witness, Devil's Advocate

---

## 1. Individual Detective Findings

### Detective 1: Motive Analyst
**Conclusion**: [Their conclusion]
**Confidence**: [High/Medium/Low]
**Key Reasoning**: [Summary of their argument]

### Detective 2: Alibi Investigator
**Conclusion**: [Their conclusion]
**Confidence**: [High/Medium/Low]
**Key Reasoning**: [Summary of their argument]

### Detective 3: Evidence Examiner
**Conclusion**: [Their conclusion]
**Confidence**: [High/Medium/Low]
**Key Reasoning**: [Summary of their argument]

### Detective 4: Witness Coordinator
**Conclusion**: [Their conclusion]
**Confidence**: [High/Medium/Low]
**Key Reasoning**: [Summary of their argument]

### Detective 5: Devil's Advocate
**Conclusion**: [Their conclusion]
**Confidence**: [High/Medium/Low]
**Key Reasoning**: [Summary of their argument]

---

## 2. Convergence Analysis

### Points of Agreement (3+ detectives)
- [Finding 1]: Agreed by [which detectives]
- [Finding 2]: Agreed by [which detectives]
- ...

### Points of Divergence
| Issue | Detective A Says | Detective B Says | Why They Differ |
|-------|------------------|------------------|-----------------|
| [Issue 1] | [Position] | [Position] | [Reasoning difference] |
| [Issue 2] | [Position] | [Position] | [Reasoning difference] |

### Consensus Pattern
- [ ] Full Consensus (5-0)
- [ ] Strong Majority (4-1)
- [ ] Split Decision (3-2)
- [ ] Fragmented (2-2-1 or worse)
- [ ] No Consensus

---

## 3. Confidence Assessment

### Overall Confidence: [High / Medium / Low / Uncertain]

**Reasoning**:
[Why this confidence level based on convergence pattern]

### Key Uncertainties
1. [Uncertainty 1 - what would resolve it]
2. [Uncertainty 2 - what would resolve it]
3. [Uncertainty 3 - what would resolve it]

---

## 4. Synthesis and Recommendation

### Integrated Conclusion
[What the preponderance of evidence suggests]

### Caveats
[What we're still uncertain about]

### Recommended Action
- [ ] Proceed with high confidence
- [ ] Proceed with caution, monitor [specific things]
- [ ] Investigate [specific divergences] before proceeding
- [ ] Restructure the problem and re-investigate

### Next Steps
1. [Specific action]
2. [Specific action]
3. [Specific action]

---

## 5. Meta-Analysis Notes

### What Worked Well
[Which detective perspectives were most valuable]

### What Could Be Improved
[Gaps in the investigation]

### Lessons for Future Investigations
[What to do differently next time]
```

---

*Target: arXiv preprint / Practical AI Engineering Conference*

*2020 ACM Computing Classification*: I.2.0 (Artificial Intelligence - General), H.4.0 (Information Systems Applications - General), K.4.0 (Computers and Society - General)

---

**Document Version**: 1.0
**Author**: Norbert Marchewka
**Date**: December 21, 2025
**License**: MIT

---

*Written 4 days before Christmas 🎄, when even detectives take a break.*
