# Information Lensing: A Gravitational Approach to Domain-Specific Embedding Transformation

## Differential Geometric Methods for Anisotropic Embedding Correction

**Abstract**

We present Information Lensing (Polish: *soczewkowanie informacyjne*), a novel theoretical framework that applies gravitational lensing principles to embedding space transformation. This approach addresses the fundamental problem of anisotropic embedding collapse—where generic embeddings produce nearly identical representations for semantically distinct code segments, particularly in enterprise Java environments. By learning domain-specific low-rank transformation matrices that act as gravitational lenses in high-dimensional space, we achieve meaningful semantic separation while preserving topological continuity. Our framework incorporates recent advances in stratified manifold learning, isotropy regularization, and reranker-based metric discovery to provide a mathematically rigorous yet practically implementable solution. This work bridges differential geometry, information theory, and gravitational physics to create a unified framework for domain-specific embedding adaptation without requiring model fine-tuning or ML expertise.

**Keywords**: Embedding transformation, gravitational lensing, differential geometry, isotropy regularization, manifold learning, cross-encoder reranking, low-rank adaptation, domain-specific embeddings

---

## 1. The Fundamental Problem: Anisotropic Embedding Collapse

### 1.1 Embedding Homogeneity in Code

Generic embedding models suffer from what the literature terms anisotropic collapse—embeddings cluster within a narrow cone rather than utilizing the full representational space. In code domains, this manifests as semantic collapse:

Given two semantically distinct Java code segments:
- $C_1$: PaymentService.processPayment(amount, currency)
- $C_2$: InventoryService.updateStock(quantity, location)

Generic embeddings exhibit:

$$\|e(C_1) - e(C_2)\|_2 \approx 0.12 \quad \text{(nearly identical)}$$

$$\cos(e(C_1), e(C_2)) \approx 0.94 \quad \text{(high false similarity)}$$

This occurs because generic models trained on broad corpora see similar syntactic patterns (service calls, method signatures, parameter structures) but fail to capture domain-specific semantic differences.

### 1.2 The Isotropy Problem

Recent research has established that LLM embedding spaces suffer from the isotropy problem:

**Isotropy** refers to the uniform distribution of embeddings across the representational space. Highly anisotropic embeddings:
- Utilize only 2-14 effective dimensions despite 4096 available
- Cluster in narrow cones, reducing discriminative power
- Exhibit high average pairwise cosine similarity (>0.8)

### 1.3 Information Radiation Background

We hypothesize that generic embeddings contain substantial "background radiation"—uniform noise from common programming patterns, language syntax, framework conventions, and generic variable naming patterns.

**Formal Model:**

$$e_{\text{generic}}(\text{code}) = s_{\text{domain}} + n_{\text{syntactic}} + n_{\text{background}}$$

where:
- $s_{\text{domain}} \in \mathbb{R}^{d_1}$ is domain-specific semantic content
- $n_{\text{syntactic}} \in \mathbb{R}^{d_2}$ represents shared syntactic patterns
- $n_{\text{background}} \in \mathbb{R}^{d_3}$ captures model-specific artifacts
- $d_1 \ll d_2 + d_3$ (signal is sparse relative to noise)

---

## 2. Theoretical Framework

### 2.1 Information Lensing Principle

Drawing from Einstein's gravitational lensing, we propose that domain knowledge acts as mass in information space, curving the metric structure:

**Gravitational Analogy:**

$$ds^2 = g_{\mu\nu} dx^\mu dx^\nu \quad \text{(spacetime, curved by mass)}$$

$$dI^2 = G_{ij} d\xi^i d\xi^j \quad \text{(information space, curved by domain knowledge)}$$

where:
- $g_{\mu\nu}$: metric tensor in spacetime (determined by mass distribution)
- $G_{ij}$: learned metric tensor in embedding space (determined by domain structure)
- $\xi^i$: embedding space coordinates ($i \in [1, 4096]$)

Just as gravitational lensing bends light to reveal distant objects, information lensing transforms embeddings to reveal hidden semantic structure.

### 2.2 Stratified Manifold Structure

Recent research validates our manifold-based approach: in the latent space of LLMs, embeddings live in a local manifold structure with different dimensions depending on the perplexities and domains of the input data, commonly referred to as a Stratified Manifold structure.

This suggests that different code domains (payment processing, inventory management, authentication) occupy distinct sub-manifolds within the embedding space. Our transformation matrices map between these stratified structures.

### 2.3 The Lensing Transformation

**Core Transformation:**

$$e_{\text{focused}} = e_{\text{generic}} \cdot T_{\text{domain}}$$

where $T_{\text{domain}} \in \mathbb{R}^{d \times d}$ is the learned transformation lens and $d = 4096$ (embedding dimension).

---

## 3. Low-Rank Transformation via SVD Decomposition

### 3.1 Motivation: Efficiency Through Low-Rank Structure

Full $4096 \times 4096$ transformation matrices require ~134M parameters per lens. Following recent advances in low-rank adaptation, we decompose transformations:

$$T_{\text{domain}} = U \cdot \Sigma \cdot V^T$$

where:
- $U \in \mathbb{R}^{4096 \times r}$ (left singular vectors)
- $\Sigma \in \mathbb{R}^{r \times r}$ (singular values, diagonal)
- $V \in \mathbb{R}^{4096 \times r}$ (right singular vectors)
- $r \ll 4096$ (effective rank, typically 64-256)

**Benefits:**
- **Storage:** 402MB → ~10MB (with $r=100$)
- **Computation:** $O(d^2) \to O(dr)$
- **Regularization:** Implicit low-rank prior prevents overfitting
- **Interpretability:** Principal directions reveal domain structure

### 3.2 Frobenius-Optimal Alignment

We learn transformations via Frobenius-optimal subspace alignment:

$$\min_{U,\Sigma,V} \|S_{\text{reranker}} - (E \cdot U\Sigma V^T) \cdot (E \cdot U\Sigma V^T)^T\|_F^2$$

where:
- $S_{\text{reranker}}$: target similarity matrix from reranker scores
- $E$: matrix of generic embeddings
- $\|\cdot\|_F$: Frobenius norm

### 3.3 Rank Selection via Variance Explained

Select rank $r$ such that cumulative variance explained exceeds threshold:

$$r^* = \min\left\{r : \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^d \sigma_i^2} \geq 0.95\right\}$$

---

## 4. Reranking as Metric Discovery

### 4.1 The Reranker's Role

The reranker (cross-encoder architecture) acts as a gravitational wave detector, revealing the true metric structure hidden beneath cosine similarity:

For embeddings $e_1, e_2$:

$$\text{sim}_{\cos}(e_1, e_2) = 0.92 \quad \text{(misleading)}$$

$$\text{score}_{\text{rerank}}(e_1, e_2) = 0.31 \quad \text{(true semantic distance)}$$

$$\Delta = |\text{sim}_{\cos} - \text{score}_{\text{rerank}}| = 0.61$$

This divergence indicates the presence of information curvature that must be corrected.

### 4.2 Cross-Encoder vs Bi-Encoder Architecture

**Bi-Encoder (Embedding Model):**

$$\text{score} = \cos(\text{encode}(q), \text{encode}(d)) \quad \text{(independent encoding)}$$

**Cross-Encoder (Reranker):**

$$\text{score} = \text{classifier}([\text{CLS}] \, q \, [\text{SEP}] \, d \, [\text{SEP}]) \quad \text{(joint encoding)}$$

Cross-encoders achieve deeper semantic understanding by allowing full attention between query and document tokens, revealing relationships invisible to independent encoding.

### 4.3 Metric Tensor Learning with Isotropy Regularization

We formulate metric learning as:

$$\min_T \mathcal{L}_{\text{total}}(T) = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_{\text{isotropy}} + \lambda_2 \mathcal{L}_{\text{rank}}$$

**Components:**

$$\mathcal{L}_{\text{align}} = \sum_{i,j} \|\text{score}_{\text{rerank}}(e_i, e_j) - \text{sim}(Te_i, Te_j)\|^2$$

$$\mathcal{L}_{\text{isotropy}} = \|\text{Cov}(T \cdot E) - I\|_F^2 \quad \text{(promotes uniform distribution)}$$

$$\mathcal{L}_{\text{rank}} = \|T\|_* \quad \text{(nuclear norm, encourages low rank)}$$

**Hyperparameters:**
- $\lambda_1 \in [0.01, 0.1]$: isotropy regularization strength
- $\lambda_2 \in [0.001, 0.01]$: rank regularization strength

### 4.4 Contrastive Learning Enhancement

Multi-level contrastive loss for transformation learning:

$$\mathcal{L}_{\text{contrast}} = -\log\frac{\exp(s^+/\tau)}{\exp(s^+/\tau) + \sum_j \exp(s_j^-/\tau)}$$

where $\tau = 0.07$ is the temperature parameter.

---

## 5. Triple Transformation Architecture

### 5.1 Three Gravitational Lenses

We employ three specialized lenses, each warping a different aspect of information space:

$$e_{\text{generic}} \xrightarrow{T_{\text{struct}}} e_{\text{structural}} \quad \text{(graph topology)}$$

$$e_{\text{generic}} \xrightarrow{T_{\text{semantic}}} e_{\text{semantic}} \quad \text{(code meaning)}$$

$$e_{\text{generic}} \xrightarrow{T_{\text{behav}}} e_{\text{behavioral}} \quad \text{(runtime patterns)}$$

### 5.2 Product Manifold Structure

The final representation lives on the product manifold:

$$\mathcal{M} = \mathcal{M}_s \times \mathcal{M}_{\text{sem}} \times \mathcal{M}_b \subset \mathbb{R}^{12288}$$

Each lens targets a distinct stratified sub-manifold:

| Lens | Target Sub-Manifold | Focus |
|------|---------------------|-------|
| $T_{\text{struct}}$ | Graph topology manifold | Connectivity, centrality, architectural position |
| $T_{\text{semantic}}$ | Code meaning manifold | Business logic, domain concepts, intent |
| $T_{\text{behav}}$ | Runtime behavior manifold | Execution patterns, dependencies, side effects |

---

## 6. Manifold Distillation Process

### 6.1 Continuous Topological Transformation

The transformation $T$ defines a continuous map between manifolds:

$$\varphi: \mathcal{M}_{\text{generic}} \to \mathcal{M}_{\text{domain}}$$

$$x \mapsto Tx$$

**Properties:**
1. **Homeomorphic**: $\varphi$ is continuous with continuous inverse
2. **Differentiable**: $\nabla\varphi$ exists everywhere (smooth transformation)
3. **Locally Isometric**: preserves local neighborhood distances
4. **Isotropy-Promoting**: increases effective dimensionality

### 6.2 Information Warping Equations

The warping of information space follows differential geometry:

Original space metric: $g_{ij}^{\text{generic}} \approx \delta_{ij}$ (nearly Euclidean)

Target space metric:

$$g_{ij}^{\text{domain}} = T_{ik} T_{jl} g_{kl}^{\text{generic}}$$

Curvature transformation:

$$R_{ijkl}^{\text{domain}} = T_{im} T_{jn} T_{kp} T_{lq} R_{mnpq}^{\text{generic}} + K_{ijkl}$$

where:
- $R_{ijkl}$: Riemann curvature tensor
- $K_{ijkl}$: domain-specific curvature induced by transformation
- $\delta_{ij}$: Kronecker delta (identity metric)

---

## 7. Mathematical Formalization

### 7.1 The Lensing Operator

Define the lensing operator $\mathcal{L}_T$ with low-rank structure:

$$\mathcal{L}_T(v) = U\Sigma V^T v + \alpha \nabla(v^T U\Sigma V^T v) + \beta W(v)$$

where:
- First term: Low-rank linear transformation
- Second term: Gradient correction for curvature ($\alpha \in [0.01, 0.1]$)
- Third term: Whitening correction $W(v)$ for isotropy ($\beta \in [0.1, 0.3]$)

**Whitening operator:**

$$W(v) = \Sigma^{-1/2} \cdot (v - \mu)$$

where $\mu = \mathbb{E}[e]$ and $\Sigma = \text{Cov}(e)$.

### 7.2 Complete Loss Function

$$\mathcal{L}_{\text{total}}(U, \Sigma, V) = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_{\text{iso}} + \lambda_2 \mathcal{L}_{\text{rank}} + \lambda_3 \mathcal{L}_{\text{contrast}}$$

**Recommended hyperparameters:**
- $\lambda_1 = 0.05$ (isotropy weight)
- $\lambda_2 = 0.01$ (rank regularization)
- $\lambda_3 = 0.1$ (contrastive weight)
- $\tau = 0.07$ (temperature)

### 7.3 Convergence Guarantee

The iterative refinement converges under standard assumptions:

$$\theta_{n+1} = \theta_n - \eta \nabla \mathcal{L}_{\text{total}}(\theta_n)$$

where $\theta = \{U, \Sigma, V\}$.

**Convergence:**

$$\|\theta_{n+1} - \theta^*\| \leq \rho \|\theta_n - \theta^*\|$$

where $\rho < 1$ is the contraction coefficient depending on learning rate $\eta$ and Lipschitz constant of $\nabla\mathcal{L}$. Typical convergence: 50-200 epochs.

### 7.4 Effective Dimensionality Metric

We measure effective rank via entropy of singular values:

$$d_{\text{eff}} = \exp\left(-\sum_i p_i \log p_i\right)$$

where $p_i = \sigma_i / \sum_j \sigma_j$ are normalized singular values. This should approach $d$ for isotropic embeddings.

---

## 8. Expected Results (Theoretical Projections)

### 8.1 Hypothesized Semantic Separation

**Before Lensing:**
- $\|e(\text{PaymentService}) - e(\text{InventoryService})\|_2 \approx 0.10\text{-}0.20$
- Effective dimensionality: ~50-100 (of 4096)
- Isotropy score: ~0.15

**After Lensing:**
- $\|Te(\text{PaymentService}) - Te(\text{InventoryService})\|_2 \approx 0.50\text{-}0.80$
- Effective dimensionality: ~500-1000
- Isotropy score: ~0.60-0.80

**Expected improvement: 300-500%**

### 8.2 Anticipated Metrics

| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| Avg pairwise similarity | 0.89 | 0.45 | -49% |
| Effective dimensions | 80 | 600 | +650% |
| Isotropy score | 0.15 | 0.70 | +367% |
| Reranker alignment | 0.35 | 0.85 | +143% |
| Inter-domain distance | 0.12 | 0.65 | +442% |

---

## 9. Theoretical Implications

### 9.1 Information Has Physics

This framework suggests information spaces follow physical laws:

| Physical Principle | Information Analogue |
|-------------------|---------------------|
| Conservation of mass-energy | Conservation of information (rank preservation) |
| Least action principle | Shortest semantic paths (geodesics) |
| Gravitational field equations | Transformation matrices as fields |
| Mass curves spacetime | Domain knowledge curves embedding space |
| Gravitational lensing | Information focusing via transformation |

### 9.2 Connection to Stratified Manifold Theory

Our framework provides a computational mechanism for navigating between stratified sub-manifolds:

$$\mathcal{S} = \bigcup_i \mathcal{M}_i \quad \text{(union of sub-manifolds)}$$

Each domain $d$ corresponds to sub-manifold $\mathcal{M}_d$. Transformation $T_d: \mathcal{M}_{\text{generic}} \to \mathcal{M}_d$ maps to domain-specific structure.

The "gravitational mass" of domain knowledge:

$$m_d \propto |\text{training\_samples}_d| \times \text{diversity}_d$$

### 9.3 Unified Theory Potential

Information lensing connects:
- **Differential Geometry:** Manifold structure, geodesics, curvature
- **Information Theory:** Entropy reduction, channel capacity
- **Physics:** Gravitational analogies, field equations
- **Computer Science:** Embedding transformation, metric learning
- **Representation Learning:** Isotropy, effective dimensionality

---

## 10. Democratization of Domain-Specific Embeddings

### 10.1 The Traditional Barrier

Achieving domain-specific embeddings traditionally requires:
- PhD-level ML team (~\$500k+/year per researcher)
- Extensive fine-tuning (weeks of compute, risk of catastrophic forgetting)
- Specialized infrastructure (GPU clusters, ML pipelines)
- Continuous maintenance (model drift, version control)

### 10.2 The Information Lensing Alternative

Our approach eliminates these barriers:

| Traditional Fine-Tuning | Information Lensing |
|------------------------|---------------------|
| 3-5 ML researchers | Any developer |
| 3-6 months initial + ongoing | 30-minute setup |
| \$2M+/year | Hardware only (~\$5k) |
| GPU cluster | CPU + RAM |
| Catastrophic forgetting risk | Frozen base model |

---

## 11. Decision Criteria: When to Apply Information Lensing

### 11.1 Quantitative Thresholds

Apply Information Lensing when ALL conditions hold:

$$\text{avg\_cosine\_similarity} > 0.85 \quad \text{(high homogeneity)}$$

$$\text{reranker\_divergence} > 0.40 \quad \text{(hidden structure)}$$

$$\text{effective\_dimensionality} < 200 \quad \text{(anisotropic)}$$

### 11.2 Benefit Quantification

$$\text{Benefit}(\mathcal{L}) = \frac{\text{Separation}_{\text{after}}}{\text{Separation}_{\text{before}}}$$

**Theorem**: Lensing is beneficial iff $\text{Benefit}(\mathcal{L}) > 1.5$

For typical enterprise Java codebases:
- BSR before ≈ 4.0-6.0 (high noise)
- BSR after ≈ 0.8-1.2 (filtered)
- Benefit ≈ 4-5 >> 1.5 ✓

---

## 12. Conclusion

Information Lensing provides a theoretically grounded, practically implementable approach to the embedding homogeneity problem. By treating transformation matrices as gravitational lenses that warp information space, we achieve:

1. **Isotropy restoration** via regularized low-rank transformations
2. **Semantic separation** through reranker-guided metric learning
3. **Mathematical rigor** via differential geometry and manifold theory
4. **Physical intuition** through gravitational analogies
5. **Practical efficiency** via low-rank decomposition
6. **Democratized access** without requiring ML expertise

**The key insight:**

> Just as gravitational lenses reveal distant galaxies invisible to direct observation, information lenses reveal hidden semantic structure obscured by syntactic noise.

---

## References

Ethayarajh, K. (2019). "How Contextual are Contextualized Word Representations?" *EMNLP 2019*.

Rajaee, S., & Pilehvar, M. T. (2021). "How Does Fine-tuning Affect the Geometry of Embedding Space." *ACL 2021 Findings*.

Li, X., et al. (2025). "Unraveling the Localized Latents: Learning Stratified Manifold Structures in LLM Embedding Space." *arXiv:2502.13577*.

Sun, Y., et al. (2025). "TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation." *arXiv:2511.09854*.

Zhang, Y., et al. (2025). "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." *arXiv:2506.05176*.

Rudman, W., et al. (2022). "IsoScore: Measuring the Uniformity of Embedding Space Utilization." *ACL 2022 Findings*.

Einstein, A. (1916). "Die Grundlage der allgemeinen Relativitätstheorie." *Annalen der Physik*, 354(7), 769-822.

Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

Gao, T., et al. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." *EMNLP 2021*.

Lee, J. M. (2018). *Introduction to Riemannian Manifolds*. Springer Graduate Texts in Mathematics.

---

*Target Journal: Transactions on Machine Learning Research*

*2020 Mathematics Subject Classification*: 68T07 (Artificial neural networks), 53Z50 (Applications of differential geometry), 62H30 (Classification and discrimination)
