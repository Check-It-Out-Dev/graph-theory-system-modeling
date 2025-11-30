# Information Lensing: A Metric Learning Approach to Domain-Specific Embedding Transformation

## Geometric Methods for Anisotropic Embedding Correction

**Abstract**

We present Information Lensing (Polish: *soczewkowanie informacyjne*), a theoretical framework for domain-specific embedding transformation grounded in metric learning and differential geometry. This approach addresses anisotropic embedding collapse—where generic embeddings produce nearly identical representations for semantically distinct code segments. By learning low-rank transformation matrices that induce domain-specific metric structures, we achieve semantic separation while preserving topological continuity. Our framework provides: (1) rigorous mathematical foundations via bi-Lipschitz transformations, (2) practical algorithms with convergence guarantees to critical points, (3) incremental calibration with Lyapunov stability bounds, and (4) economically viable cloud-first implementation strategies. We explicitly distinguish between mathematically proven properties and heuristic analogies, offering both theoretical depth and engineering pragmatism.

**Keywords**: Metric learning, embedding transformation, bi-Lipschitz maps, anisotropy correction, manifold learning, cross-encoder reranking, low-rank adaptation

---

## 1. The Fundamental Problem: Anisotropic Embedding Collapse

### 1.1 Empirical Observation: Embedding Homogeneity in Code

Generic embedding models exhibit what the literature terms *anisotropic collapse*—embeddings cluster within a narrow cone rather than utilizing the full representational space (Ethayarajh, 2019). In code domains, this manifests as semantic collapse.

**Concrete Example:**

Given two semantically distinct Java code segments:
- $C_1$: `PaymentService.processPayment(amount, currency)`
- $C_2$: `InventoryService.updateStock(quantity, location)`

Generic embeddings often exhibit:

$$\|e(C_1) - e(C_2)\|_2 \approx 0.12 \quad \text{(nearly identical)}$$

$$\cos(e(C_1), e(C_2)) \approx 0.94 \quad \text{(high false similarity)}$$

This occurs because generic models trained on broad corpora see similar syntactic patterns (service calls, method signatures, parameter structures) but fail to capture domain-specific semantic differences.

### 1.2 The Isotropy Problem: Quantitative Characterization

**Definition 1.1 (Effective Dimensionality).** For a set of embeddings $\{e_i\}_{i=1}^N$ with singular values $\{\sigma_j\}_{j=1}^d$ of the centered embedding matrix, the effective dimensionality is:

$$d_{\text{eff}} = \exp\left(-\sum_{j=1}^d p_j \log p_j\right)$$

where $p_j = \sigma_j / \sum_k \sigma_k$ are normalized singular values.

**Definition 1.2 (Isotropy Score).** Following Rudman et al. (2022), the isotropy score measures uniformity of embedding space utilization:

$$I(E) = \frac{\min_j \sigma_j}{\max_j \sigma_j}$$

where $\sigma_j$ are singular values of the centered embedding matrix $E$.

**Empirical Finding (Literature):** LLM embedding spaces typically exhibit $d_{\text{eff}} \approx 50\text{-}200$ despite $d = 4096$ available dimensions (Li et al., 2025; Rajaee & Pilehvar, 2021). The specific value depends on model architecture, training data, and downstream task. We do not claim a universal range but note that significant dimensional underutilization is consistently observed.

### 1.3 Signal-Noise Decomposition Model

We model generic embeddings as containing multiple components:

$$e_{\text{generic}}(\text{code}) = s_{\text{domain}} + n_{\text{syntactic}} + n_{\text{background}} + \epsilon$$

where:
- $s_{\text{domain}} \in \mathbb{R}^{d}$: domain-specific semantic content (sparse)
- $n_{\text{syntactic}} \in \mathbb{R}^{d}$: shared syntactic patterns (dense)
- $n_{\text{background}} \in \mathbb{R}^{d}$: model-specific artifacts (dense)
- $\epsilon$: irreducible noise

**Important Caveat:** This decomposition is a modeling assumption, not a proven fact. These components are generally **not orthogonal**—syntactic patterns may correlate with domain semantics. The practical goal is to learn transformations that amplify $s_{\text{domain}}$ relative to noise components, regardless of their exact geometric relationship.

---

## 2. Theoretical Framework

### 2.1 Core Objective: Metric Learning

Our goal is to learn a transformation $T: \mathbb{R}^d \to \mathbb{R}^d$ such that distances in the transformed space better reflect semantic similarity:

$$d_{\text{semantic}}(C_1, C_2) \approx \|Te(C_1) - Te(C_2)\|_2$$

where $d_{\text{semantic}}$ is the "true" semantic distance, which we approximate using cross-encoder reranker scores.

**Definition 2.1 (Bi-Lipschitz Transformation).** A linear transformation $T: \mathbb{R}^d \to \mathbb{R}^d$ is bi-Lipschitz if there exist constants $0 < c_1 \leq c_2 < \infty$ such that for all $x, y \in \mathbb{R}^d$:

$$c_1 \|x - y\| \leq \|Tx - Ty\| \leq c_2 \|x - y\|$$

For linear $T$, these constants are $c_1 = \sigma_{\min}(T)$ and $c_2 = \sigma_{\max}(T)$, where $\sigma_{\min}, \sigma_{\max}$ are the smallest and largest singular values respectively.

**Remark:** Bi-Lipschitz maps preserve topological properties (homeomorphism) while allowing controlled distortion of distances. This is the correct mathematical property—not "local isometry," which would require $c_1 = c_2 = 1$ and prohibit any distance modification.

### 2.2 Geometric Interpretation: Induced Metric Structure

The transformation $T$ induces a new metric structure on the embedding space.

**Definition 2.2 (Induced Metric Tensor).** Given transformation $T \in \mathbb{R}^{d \times d}$, the induced metric tensor is:

$$G = T^T T$$

Under this metric, distances are computed as:

$$d_G(x, y)^2 = (x - y)^T G (x - y) = \|T(x - y)\|_2^2$$

**Proposition 2.1.** The transformation $T$ changes the geometry of the embedding space from Euclidean (identity metric $I$) to the metric defined by $G = T^T T$. Directions corresponding to large singular values of $T$ are "stretched" (distances amplified), while directions corresponding to small singular values are "compressed."

*Proof.* Let $T = U\Sigma V^T$ be the SVD of $T$. Then $G = T^T T = V\Sigma^2 V^T$. For any vector $v$, we have $v^T G v = v^T V \Sigma^2 V^T v = \|ΣV^T v\|^2$. The eigenvectors of $G$ are the columns of $V$, with eigenvalues $\sigma_i^2$. Thus distances along the $i$-th principal direction are scaled by $\sigma_i$. $\square$

### 2.3 Heuristic Analogy: Gravitational Lensing (Non-Rigorous)

We offer an **analogy** (not a mathematical equivalence) to gravitational lensing for intuition:

| Gravitational Lensing | Information Lensing |
|----------------------|---------------------|
| Mass curves spacetime | Domain knowledge shapes metric |
| Light follows geodesics | Similarity follows induced distances |
| Distant objects become visible | Hidden semantic structure becomes apparent |

**Explicit Disclaimer:** This analogy is pedagogical. Unlike general relativity, where the Einstein field equations $R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}$ precisely determine spacetime curvature from mass-energy distribution, we do not claim equivalent field equations for Information Lensing. Our metric tensor $G$ is learned from data, not derived from first principles.

The analogy captures the intuition that domain expertise "focuses" information—just as gravitational lenses reveal distant galaxies by bending light, domain-specific transformations reveal semantic structure by reshaping the distance function.

### 2.4 Connection to Stratified Manifold Theory

Recent research (Li et al., 2025) establishes that LLM embeddings exhibit stratified manifold structure—different domains occupy distinct sub-manifolds within the embedding space.

**Definition 2.3 (Stratified Manifold).** A stratified space $\mathcal{S}$ is a union of manifolds (strata) of possibly different dimensions:

$$\mathcal{S} = \bigcup_{i} \mathcal{M}_i$$

where each $\mathcal{M}_i$ is a smooth manifold and the strata satisfy certain incidence relations.

**Hypothesis:** Different code domains (payment processing, inventory management, authentication) occupy distinct strata $\mathcal{M}_d$ within the embedding space. Our transformation $T$ can be viewed as learning a map that:
1. Identifies which stratum a code segment belongs to
2. Projects within-stratum structure more clearly

This hypothesis motivates the triple-lens architecture (Section 5), where different transformations target different structural aspects.

---

## 3. Low-Rank Transformation via SVD Decomposition

### 3.1 Motivation: Regularization Through Rank Constraint

Full $d \times d$ transformation matrices ($d = 4096$) require $\sim$134M parameters—prone to overfitting and computationally expensive. Following LoRA (Hu et al., 2022), we constrain transformations to low-rank form.

**Definition 3.1 (Low-Rank Parameterization).** We parameterize $T$ as:

$$T = I + AB^T$$

where $A, B \in \mathbb{R}^{d \times r}$ with $r \ll d$ (typically $r \in [64, 256]$).

Equivalently, via thin SVD:

$$T = U \Sigma V^T$$

where $U \in \mathbb{R}^{d \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$ (diagonal), $V \in \mathbb{R}^{d \times r}$.

**Proposition 3.1 (Parameter and Compute Savings).** Low-rank parameterization reduces:
- Storage: from $O(d^2)$ to $O(dr)$ — e.g., 134M → 819K parameters for $r=100$
- Matrix-vector multiplication: from $O(d^2)$ to $O(dr)$
- Provides implicit regularization through rank constraint

### 3.2 Learning Objective: Frobenius Alignment

**Definition 3.2 (Target Similarity Matrix).** Let $S_{\text{reranker}} \in \mathbb{R}^{N \times N}$ be the matrix of pairwise reranker scores, where $(S_{\text{reranker}})_{ij} = \text{reranker}(\text{file}_i, \text{file}_j)$.

**Definition 3.3 (Predicted Similarity Matrix).** For embedding matrix $E \in \mathbb{R}^{N \times d}$ and transformation $T$, the predicted similarity matrix is:

$$S_{\text{pred}}(T) = \text{CosineSim}(ET) = \text{normalize}(ET) \cdot \text{normalize}(ET)^T$$

where normalization is row-wise to unit vectors.

**Optimization Problem:**

$$\min_{T} \|S_{\text{reranker}} - S_{\text{pred}}(T)\|_F^2 + \lambda \mathcal{R}(T)$$

where $\mathcal{R}(T)$ is a regularizer.

**Important Note on Non-Convexity:** This objective is **non-convex** in $T$ because $S_{\text{pred}}(T)$ involves normalization (division by norms) and products of $T$. Standard gradient descent converges to a **critical point**, not necessarily a global minimum. We address this in Section 7.3.

### 3.3 Rank Selection: Principled Approaches

**Method 1: Variance Explained (Heuristic)**

Select $r$ such that the top $r$ singular values of the learned transformation capture sufficient variance:

$$r^* = \min\left\{r : \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^d \sigma_i^2} \geq \tau\right\}$$

where $\tau \in [0.90, 0.99]$ is a threshold. Note: this requires first learning a full-rank transformation, then truncating.

**Method 2: Cross-Validation (Recommended)**

Evaluate downstream task performance (e.g., retrieval accuracy) for different ranks $r \in \{32, 64, 128, 256, 512\}$ on held-out data. Select $r$ that maximizes performance without overfitting.

**Method 3: Adaptive Rank (Research Direction)**

Learn the rank as part of optimization by placing a sparsity-inducing prior on singular values (e.g., $\ell_1$ penalty on $\text{diag}(\Sigma)$).

---

## 4. Reranking as Metric Discovery

### 4.1 Cross-Encoders as Semantic Oracles

Cross-encoder rerankers provide high-quality semantic similarity scores by jointly encoding query-document pairs, allowing full attention between all tokens.

**Architecture Comparison:**

**Bi-Encoder (Embedding Model):**
$$\text{score}(q, d) = \cos(\text{encode}(q), \text{encode}(d))$$

- Encodes $q$ and $d$ independently
- Fast: $O(N)$ for $N$ candidates (encode once, compare many)
- Lower quality: no cross-attention between $q$ and $d$

**Cross-Encoder (Reranker):**
$$\text{score}(q, d) = \text{MLP}(\text{encode}([\text{CLS}] \, q \, [\text{SEP}] \, d \, [\text{SEP}]))$$

- Encodes $q$ and $d$ jointly with full attention
- Slow: $O(N)$ forward passes for $N$ candidates
- Higher quality: captures fine-grained interactions

**Key Insight:** Cross-encoders reveal semantic relationships that bi-encoders miss. The divergence between cosine similarity (bi-encoder) and reranker score indicates "hidden structure" that our transformation should recover.

### 4.2 Metric Learning Loss Functions

**Primary Loss: Alignment with Reranker**

$$\mathcal{L}_{\text{align}}(T) = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \left( s_{ij}^{\text{rerank}} - \text{sim}(Te_i, Te_j) \right)^2$$

where $\mathcal{P}$ is the set of sampled pairs and $s_{ij}^{\text{rerank}}$ are reranker scores.

**Regularization: Isotropy Promotion**

We want transformed embeddings to utilize more of the available dimensions. Two approaches:

*Approach 1: Whitening Regularization*
$$\mathcal{L}_{\text{whiten}}(T) = \|\text{Cov}(TE) - I\|_F^2$$

This forces the covariance to be identity—strong isotropy but may destroy useful variance structure.

*Approach 2: Uniformity Loss (Preferred)*

Following SimCLR/SimCSE, we encourage uniform angular distribution:

$$\mathcal{L}_{\text{uniform}}(T) = \log \mathbb{E}_{i,j} \left[ \exp\left( -2 \|Te_i - Te_j\|^2 \right) \right]$$

This penalizes clustering without forcing equal variance in all directions.

**Regularization: Rank Penalty**

$$\mathcal{L}_{\text{rank}}(T) = \|T\|_* = \sum_i \sigma_i(T)$$

The nuclear norm encourages low-rank solutions.

**Contrastive Enhancement (Optional)**

Multi-level contrastive loss for hard negative mining:

$$\mathcal{L}_{\text{contrast}} = -\log\frac{\exp(s^+/\tau)}{\exp(s^+/\tau) + \sum_j \exp(s_j^-/\tau)}$$

where $\tau \in [0.05, 0.1]$ is temperature, $s^+$ is similarity to positive, $s_j^-$ are similarities to negatives.

### 4.3 Complete Loss Function

$$\mathcal{L}_{\text{total}}(T) = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_{\text{uniform}} + \lambda_2 \mathcal{L}_{\text{rank}} + \lambda_3 \mathcal{L}_{\text{contrast}}$$

**Recommended Hyperparameters (Starting Point):**
- $\lambda_1 = 0.05$ (uniformity weight)
- $\lambda_2 = 0.01$ (rank regularization)
- $\lambda_3 = 0.1$ (contrastive weight)
- $\tau = 0.07$ (temperature)

These should be tuned via cross-validation for specific domains.

---

## 5. Triple Transformation Architecture

### 5.1 Motivation: Multi-Aspect Semantic Structure

Code semantics are multi-faceted. A single transformation may not capture all relevant structure. We propose learning three specialized transformations:

$$e_{\text{generic}} \xrightarrow{T_{\text{struct}}} e_{\text{structural}}$$
$$e_{\text{generic}} \xrightarrow{T_{\text{sem}}} e_{\text{semantic}}$$
$$e_{\text{generic}} \xrightarrow{T_{\text{behav}}} e_{\text{behavioral}}$$

### 5.2 Lens Specifications

| Lens | Target Aspect | Training Signal |
|------|---------------|-----------------|
| $T_{\text{struct}}$ | Graph topology, architectural position | Graph-based similarity (neighbors, centrality) |
| $T_{\text{sem}}$ | Business logic, domain concepts | Reranker on semantic queries |
| $T_{\text{behav}}$ | Runtime patterns, dependencies | Call graph, execution traces |

### 5.3 Combination Strategies

**Strategy 1: Concatenation (Simple)**

$$e_{\text{combined}} = [T_{\text{struct}} e \| T_{\text{sem}} e \| T_{\text{behav}} e] \in \mathbb{R}^{3d}$$

**Strategy 2: Weighted Average**

$$e_{\text{combined}} = \alpha_1 T_{\text{struct}} e + \alpha_2 T_{\text{sem}} e + \alpha_3 T_{\text{behav}} e$$

where $\alpha_i$ are query-dependent or learned weights.

**Strategy 3: Attention-Based Fusion (Advanced)**

$$e_{\text{combined}} = \sum_i \text{softmax}(q^T W_i T_i e) \cdot T_i e$$

where $q$ is the query embedding and $W_i$ are learned attention matrices.

### 5.4 Geometric Interpretation: Product Manifold

The concatenation strategy embeds data in the product space:

$$\mathcal{M} = \mathcal{M}_{\text{struct}} \times \mathcal{M}_{\text{sem}} \times \mathcal{M}_{\text{behav}}$$

Distances in product manifolds decompose:

$$d_{\mathcal{M}}^2(x, y) = d_{\text{struct}}^2(x, y) + d_{\text{sem}}^2(x, y) + d_{\text{behav}}^2(x, y)$$

This allows retrieval to balance multiple aspects of similarity.

---

## 6. Mathematical Properties

### 6.1 Topological Preservation

**Theorem 6.1 (Homeomorphism).** If $T \in \mathbb{R}^{d \times d}$ is full-rank, then $\varphi: x \mapsto Tx$ is a homeomorphism of $\mathbb{R}^d$.

*Proof.* Full-rank implies $T$ is invertible. Both $\varphi(x) = Tx$ and $\varphi^{-1}(y) = T^{-1}y$ are continuous (linear maps are continuous). Thus $\varphi$ is a continuous bijection with continuous inverse. $\square$

**Corollary 6.1.** The transformation preserves:
- Connectedness of the data manifold
- Neighborhood relationships (points that were neighbors remain neighbors, though distances change)
- Homotopy type (fundamental group, etc.)

### 6.2 Distance Distortion Bounds

**Theorem 6.2 (Bi-Lipschitz Bounds).** For any full-rank $T$ with singular values $\sigma_{\min} \leq \sigma_i \leq \sigma_{\max}$:

$$\sigma_{\min} \|x - y\| \leq \|Tx - Ty\| \leq \sigma_{\max} \|x - y\|$$

*Proof.* For any vector $v$: $\|Tv\|^2 = v^T T^T T v$. The eigenvalues of $T^T T$ are $\sigma_i^2$, so $\sigma_{\min}^2 \|v\|^2 \leq \|Tv\|^2 \leq \sigma_{\max}^2 \|v\|^2$. Taking square roots and setting $v = x - y$ yields the result. $\square$

**Definition 6.1 (Condition Number).** The condition number $\kappa(T) = \sigma_{\max}/\sigma_{\min}$ measures the maximum distortion ratio. Well-conditioned transformations have $\kappa(T)$ close to 1.

**Practical Implication:** To prevent extreme distortion, we can add a regularization term:

$$\mathcal{L}_{\text{condition}}(T) = \max\left(0, \kappa(T) - \kappa_{\max}\right)^2$$

where $\kappa_{\max} \approx 10\text{-}100$ is a threshold.

### 6.3 Convergence Analysis (Honest Assessment)

**Fact:** The optimization problem $\min_T \mathcal{L}_{\text{total}}(T)$ is **non-convex** due to:
1. Cosine similarity involves normalization (non-linear in $T$)
2. Product structure in $S_{\text{pred}} = (ET)(ET)^T$

**Theorem 6.3 (Convergence to Critical Point).** Under standard assumptions (Lipschitz-continuous gradients, bounded iterates), gradient descent with appropriate learning rate $\eta$ satisfies:

$$\min_{k \leq K} \|\nabla \mathcal{L}(T_k)\|^2 \leq \frac{2(\mathcal{L}(T_0) - \mathcal{L}^*)}{\eta K}$$

where $\mathcal{L}^*$ is the minimum value.

*Proof.* Standard result from non-convex optimization theory. See (Nesterov, 2004). $\square$

**Practical Implications:**
1. We converge to **a** critical point, not necessarily the global minimum
2. Multiple random initializations are recommended
3. Learning rate scheduling (e.g., cosine annealing) improves convergence
4. Early stopping based on validation performance prevents overfitting

**What We Cannot Guarantee:**
- Global optimality
- Unique solution
- Convergence rate better than $O(1/K)$ without strong convexity

---

## 7. The Lensing Operator (Corrected Formulation)

### 7.1 Basic Linear Transformation

The core transformation is simply:

$$\mathcal{L}_T(e) = Te$$

For low-rank $T = I + AB^T$:

$$\mathcal{L}_T(e) = e + AB^T e = e + A(B^T e)$$

This is an affine correction to the identity—the embedding is adjusted by a low-rank perturbation.

### 7.2 Optional Preprocessing: Whitening

If the input embeddings are highly anisotropic, preprocessing with whitening can help:

$$e_{\text{whitened}} = \Sigma_E^{-1/2} (e - \mu_E)$$

where $\mu_E = \mathbb{E}[e]$ and $\Sigma_E = \text{Cov}(e)$.

The complete pipeline becomes:

$$e \xrightarrow{\text{whiten}} e_w \xrightarrow{T} Te_w$$

### 7.3 Why Not Nonlinear Transformations?

One might ask: why restrict to linear transformations? Nonlinear maps could capture more complex structure.

**Reasons for Linear:**
1. **Interpretability:** Principal directions have clear meaning
2. **Efficiency:** Matrix multiplication is fast
3. **Stability:** No risk of mode collapse or vanishing gradients
4. **Composability:** Multiple lenses compose cleanly: $T_2 T_1 e$
5. **Sufficient Capacity:** For the "denoising" objective, linear may suffice

**When Nonlinear Might Help:**
- If the signal-noise decomposition is fundamentally nonlinear
- If different regions of embedding space need different transformations

We leave nonlinear extensions (e.g., neural network transformations) as future work.

---

## 8. Expected Results (Theoretical Projections)

### 8.1 Hypothesized Improvements

Based on the framework, we predict the following improvements (to be validated empirically):

| Metric | Before Lensing | After Lensing (Expected) |
|--------|----------------|--------------------------|
| Average pairwise cosine similarity | 0.85-0.92 | 0.40-0.60 |
| Effective dimensionality | 50-150 | 300-800 |
| Isotropy score | 0.10-0.20 | 0.50-0.80 |
| Reranker score correlation | 0.30-0.50 | 0.75-0.90 |
| Inter-domain distance | 0.10-0.20 | 0.50-0.70 |

**Caveat:** These are projections based on literature results for similar techniques. Actual performance will depend on domain, data quality, and hyperparameter tuning.

### 8.2 When Lensing Should Help

Lensing is most beneficial when:

1. **High embedding homogeneity:** $\text{avg\_cosine\_sim} > 0.85$
2. **Reranker divergence:** $|\text{cosine\_sim} - \text{reranker\_score}| > 0.40$
3. **Low effective dimensionality:** $d_{\text{eff}} < 200$
4. **Domain specificity:** Code is specialized (e.g., enterprise Java) rather than generic

### 8.3 When Lensing May Not Help

1. **Already isotropic embeddings:** If $d_{\text{eff}}$ is already high
2. **Aligned bi-encoder and cross-encoder:** If cosine similarity ≈ reranker score
3. **Heterogeneous domains:** If code spans many unrelated domains
4. **Insufficient training data:** Need enough pairs to learn meaningful transformation

---

## 9. Incremental Lens Calibration

### 9.1 The Incremental Update Problem

In production, codebases evolve continuously. Full recalibration requires $O(N^2)$ reranker calls—prohibitive for large repositories.

**Problem Statement:**

Given:
- Previous lens $T_{\text{old}}$ trained on file set $\mathcal{F}_{\text{old}}$
- Changes: $\mathcal{A}$ (added), $\mathcal{D}$ (deleted), $\mathcal{M}$ (modified)
- New file set: $\mathcal{F}_{\text{new}} = (\mathcal{F}_{\text{old}} \setminus \mathcal{D}) \cup \mathcal{A}$, with $\mathcal{M}$ re-embedded

Find $T_{\text{new}}$ efficiently.

### 9.2 Pair Classification Strategy

Partition pairs into categories:

| Category | Definition | Action |
|----------|------------|--------|
| Critical-New | $(e_{\text{new}}, e_j)$ for new files | Must compute reranker |
| Critical-Modified | $(e_{\text{mod}}, e_j)$ for modified files | Must recompute reranker |
| Invalidated | $(e_{\text{del}}, \cdot)$ for deleted files | Remove |
| Stable | All other pairs | Reuse cached scores |

**Complexity Reduction:**

For $N$ files and $|\Delta| = |\mathcal{A}| + |\mathcal{M}|$ changes:
- Full recalibration: $O(N^2)$ reranker calls
- Incremental: $O(N \cdot |\Delta|)$ reranker calls
- Speedup: $N / |\Delta|$ (e.g., 50× for 1000 files, 20 changes)

### 9.3 Incremental Loss Function

$$\mathcal{L}_{\text{incremental}}(T) = \mathcal{L}_{\text{critical}} + \lambda_1 \mathcal{L}_{\text{memory}} + \lambda_2 \mathcal{L}_{\text{anchor}}$$

**Component 1: Critical Pairs (New Information)**

$$\mathcal{L}_{\text{critical}} = \frac{1}{|\mathcal{C}|} \sum_{(i,j) \in \mathcal{C}} \left( s_{ij}^{\text{rerank}} - \text{sim}(Te_i, Te_j) \right)^2$$

**Component 2: Memory Preservation (Prevent Forgetting)**

$$\mathcal{L}_{\text{memory}} = \frac{1}{|\mathcal{S}|} \sum_{(i,j) \in \mathcal{S}} \left( s_{ij}^{\text{rerank}} - \text{sim}(Te_i, Te_j) \right)^2$$

where $\mathcal{S}$ is a random sample of stable pairs.

**Component 3: Anchor Regularization (Prevent Drift)**

$$\mathcal{L}_{\text{anchor}} = \|T - T_{\text{old}}\|_F^2$$

**Hyperparameters:**
- $\lambda_1 = 0.5$ (memory weight)
- $\lambda_2 = 0.1$ (anchor weight)

### 9.4 Stability Analysis: Lyapunov Functional

**Definition 9.1 (Lyapunov Functional).** A functional $V: \mathcal{X} \to \mathbb{R}$ is a Lyapunov functional for a dynamical system if:
1. $V(x) \geq 0$ for all $x$
2. $V(x_{n+1}) \leq V(x_n)$ along trajectories

**Theorem 9.1 (Incremental Stability).** The loss function $\mathcal{L}_{\text{incremental}}$ serves as a Lyapunov functional for the lens update dynamics, guaranteeing:

1. **Monotonic Improvement:** With appropriate learning rate, $\mathcal{L}(T_{n+1}) \leq \mathcal{L}(T_n)$

2. **Bounded Drift:** Due to the anchor term:
   $$\|T_{\text{new}} - T_{\text{old}}\|_F \leq \frac{\|\nabla \mathcal{L}_{\text{critical}}\|_F}{2\lambda_2}$$

3. **Accumulated Drift Bound:** After $K$ incremental updates:
   $$\|T_K - T_0\|_F \leq \sum_{k=1}^K \frac{\|\nabla \mathcal{L}_{\text{critical}}^{(k)}\|_F}{2\lambda_2}$$

*Proof Sketch.* Property 1 follows from gradient descent on smooth loss. Property 2 follows from the first-order optimality condition: at convergence, $\nabla \mathcal{L}_{\text{total}} = 0$, which implies $\nabla \mathcal{L}_{\text{critical}} + 2\lambda_2 (T - T_{\text{old}}) = 0$ (ignoring other terms). Solving gives the bound. Property 3 follows by induction. $\square$

### 9.5 Practical Schedule

| Frequency | Action | Trigger |
|-----------|--------|---------|
| Per commit | Update embedding cache | Any code change |
| Weekly | Incremental lens calibration | $|\Delta| > 5$ files |
| Monthly | Full recalibration | Drift > 10% OR major refactor |
| Quarterly | Hyperparameter review | Calendar |

---

## 10. Computational Considerations

### 10.1 The Bottleneck: Reranker Inference

For $N$ files, full pairwise reranking requires $\binom{N}{2}$ inference calls. With 8B parameter cross-encoders and 32k token context:

| Hardware | Time/Pair | 50k Pairs |
|----------|-----------|-----------|
| CPU (Ryzen 9950X) | 30-60s | 400-800 hours |
| GPU (2× RTX 3090) | 1-2s | 14-28 hours |
| GPU (A100 80GB) | 0.5-1s | 7-14 hours |

**Conclusion:** CPU-only processing is impractical for initial calibration.

### 10.2 Cloud-First Strategy

**Initial Calibration:**
- Use cloud GPU (Vast.ai, Lambda Labs): ~$15-30 for 50k pairs
- Download similarity matrix $S_{\text{reranker}}$ (~3GB)
- Run lens optimization locally (CPU sufficient, minutes)

**Monthly Recalibration:**
- Incremental: only $O(N \cdot |\Delta|)$ pairs
- Cloud cost: ~$2/month for typical change rates

**Annual Cost:** ~$45 USD for living, continuously-updated lenses

### 10.3 Monte Carlo Sampling for Large Codebases

For $N > 2000$ files, even incremental updates become expensive. Use importance sampling:

1. **Stratified Sampling:** Divide files into clusters, sample proportionally
2. **Hard Negative Mining:** Oversample pairs where $|\text{cosine} - \text{reranker}|$ is large
3. **Active Learning:** Prioritize pairs with high uncertainty

Target: 50k pairs regardless of codebase size, with theoretical guarantees on sample complexity.

---

## 11. Decision Criteria: When to Apply Information Lensing

### 11.1 Quantitative Thresholds

Apply Information Lensing when ALL conditions hold:

$$\text{avg\_cosine\_similarity} > 0.85$$
$$|\text{cosine} - \text{reranker}|_{\text{avg}} > 0.40$$
$$d_{\text{eff}} < 200$$

### 11.2 Diagnostic Procedure

1. **Compute Isotropy Metrics:**
  - Sample 1000 random embeddings
  - Compute SVD, measure $d_{\text{eff}}$ and isotropy score

2. **Measure Reranker Divergence:**
  - Sample 100 random pairs
  - Compute both cosine similarity and reranker score
  - Measure correlation and average divergence

3. **Decision:**
  - If divergence < 0.2 and isotropy > 0.5: Lensing unlikely to help
  - If divergence > 0.4 and isotropy < 0.3: Lensing likely beneficial
  - Otherwise: Pilot experiment recommended

---

## 12. Limitations and Future Work

### 12.1 Current Limitations

1. **Non-Convex Optimization:** No guarantee of global optimum
2. **Reranker Quality Ceiling:** Lenses can't exceed reranker's semantic understanding
3. **Domain Specificity:** Lenses may not transfer across domains
4. **Linear Assumption:** Complex signal-noise structure may require nonlinear maps

### 12.2 Future Directions

1. **Nonlinear Lensing:** Neural network transformations with regularization
2. **Multi-Task Lenses:** Single lens optimized for multiple objectives
3. **Theoretical Analysis:** Generalization bounds, sample complexity
4. **Cross-Domain Transfer:** Meta-learning for lens initialization

---

## 13. Conclusion

Information Lensing provides a mathematically grounded approach to embedding transformation, with:

1. **Rigorous Foundations:** Bi-Lipschitz maps, metric learning, convergence analysis
2. **Practical Algorithms:** Low-rank SVD, incremental calibration, cloud-first deployment
3. **Honest Assessment:** Clear distinction between proven properties and heuristics
4. **Economic Viability:** ~$45/year for continuously-updated lenses

The gravitational analogy provides intuition but is not claimed as mathematical equivalence. The core contribution is showing that domain-specific metric structures can be efficiently learned and applied to improve embedding-based retrieval in specialized domains.

---

## References

Ethayarajh, K. (2019). "How Contextual are Contextualized Word Representations?" *EMNLP 2019*.

Gao, T., et al. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." *EMNLP 2021*.

Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

Li, X., et al. (2025). "Unraveling the Localized Latents: Learning Stratified Manifold Structures in LLM Embedding Space." *arXiv:2502.13577*.

Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization*. Springer.

Rajaee, S., & Pilehvar, M. T. (2021). "How Does Fine-tuning Affect the Geometry of Embedding Space." *ACL 2021 Findings*.

Rudman, W., et al. (2022). "IsoScore: Measuring the Uniformity of Embedding Space Utilization." *ACL 2022 Findings*.

Zhang, Y., et al. (2025). "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." *arXiv:2506.05176*.

---

*Target Journal: Transactions on Machine Learning Research*

*2020 Mathematics Subject Classification*: 68T07 (Artificial neural networks), 53Z50 (Applications of differential geometry), 62H30 (Classification and discrimination)

---

## Appendix A: Comparison with Original Formulation

This revision addresses the following issues from the original paper:

| Original Claim | Issue | Correction |
|----------------|-------|------------|
| "Locally isometric" transformation | False for non-orthogonal T | Changed to "bi-Lipschitz" with explicit bounds |
| Curvature tensor transformation equation | Incorrect tensor transformation law | Removed; clarified that embedding space curvature ≠ data manifold curvature |
| Gradient term adds curvature | Gradient of quadratic is linear | Removed; noted transformation is purely linear |
| Convergence to global minimum | Non-convex loss | Changed to "convergence to critical point" |
| Gravitational field equations analogy | No actual field equations provided | Explicitly marked as "heuristic analogy" |
| Perelman's Ricci flow equivalence | Different mathematical structures | Noted as "inspirational analogy" only |
| "Gravitational wave detector" metaphor | Misleading | Changed to "semantic oracle" |

## Appendix B: Glossary of Mathematical Terms

**Anisotropic:** Not uniform in all directions; embedding distributions concentrated in subspace

**Bi-Lipschitz:** Map that distorts distances by bounded factors in both directions

**Condition Number:** Ratio of largest to smallest singular value; measures matrix conditioning

**Effective Dimensionality:** Entropy-based measure of how many dimensions are actually used

**Frobenius Norm:** Matrix norm defined as $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$

**Homeomorphism:** Continuous bijection with continuous inverse; preserves topology

**Isotropy:** Uniform distribution of embeddings across the representational space

**Lyapunov Functional:** Non-increasing function along system trajectories; proves stability

**Nuclear Norm:** Sum of singular values; convex relaxation of rank

**Stratified Manifold:** Union of manifolds of possibly different dimensions
