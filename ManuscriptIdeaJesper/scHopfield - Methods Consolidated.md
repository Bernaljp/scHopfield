# Methods — scHopfield

## Overview of Notation and Symbols
To ensure clarity and consistency, Table 1 summarizes the major mathematical symbols and their dimensions used throughout the scHopfield framework. Matrices are denoted by uppercase italics (e.g., $W$, $X$), vectors by bold lowercase (e.g., $\mathbf{x}$, $\mathbf{I}$), and scalars by lowercase italics.

**Table 1: Table of Symbols and Dimensions**

| Symbol | Description | Dimension |
| :--- | :--- | :--- |
| $N$ | Total number of genes in the network | Scalar |
| $M$ | Total number of cells in the dataset or cluster | Scalar |
| $X$ | Gene expression matrix | $M \times N$ |
| $\dot{X}$ | RNA velocity matrix | $M \times N$ |
| $\mathbf{x}, \mathbf{x}_i$ | Gene expression vector for a single cell (cell $i$) | $N \times 1$ |
| $W$ | Regulatory interaction matrix (weights) | $N \times N$ |
| $\Gamma$ | Diagonal matrix of degradation rates | $N \times N$ |
| $\gamma_i$ | Degradation rate for gene $i$ | Scalar |
| $\mathbf{I}$ | Bias vector (external regulatory inputs) | $N \times 1$ |
| $\varphi(\cdot)$ | Hill activation function | $\mathbb{R} \to [0,1)$ |
| $k_i$ | Hill half-saturation constant for gene $i$ | Scalar |
| $n_i$ | Hill coefficient for gene $i$ | Scalar |
| $E(\mathbf{x})$ | Hopfield energy of cell state $\mathbf{x}$ | Scalar |
| $J(\mathbf{x})$ | Jacobian matrix evaluated at state $\mathbf{x}$ | $N \times N$ |
| $c, l$ | Indices representing specific cell-type clusters | Scalar |
| $W^{(c)}$ | Cluster-specific regulatory interaction matrix | $N \times N$ |

## scHopfield Pipeline Workflow
The complete end-to-end framework of scHopfield from single-cell data to lineage analysis is summarized in Algorithm 1.

**Algorithm 1: scHopfield Framework**
1. **Input:** Expression matrix $X$, Velocity matrix $\dot{X}$, prior scaffold $S$ (optional).
2. **Parameter Estimation:** For each gene $i \in \{1 \dots N\}$:
   - Estimate Hill parameters $(k_i, n_i)$ from ECDF of expression (Section 2).
   - Estimate degradation rate $\gamma_i$ from velocity data.
3. **Network Inference:** 
   - Compute Hill activations $\varphi(X)$.
   - For each cluster $c$, construct augmented matrix $A$ and solve for interaction matrix $W^{(c)}$ and bias $\mathbf{I}^{(c)}$ via Moore-Penrose pseudoinverse or scaffold-guided optimization (Section 3).
4. **Energy Landscape & Dynamics:**
   - Compute cellular energy $E(\mathbf{x})$ and identify local attractors (Section 4, 5).
   - Visualize merged energy landscapes using UMAP projection (Section 5).
   - Compute Jacobian $J(\mathbf{x})$ for local stability and rotational flow (Section 6).
5. **Network Analysis & Perturbation Simulation:**
   - Identify candidate Driver TFs via network centrality and energy correlation (Section 7).
   - Simulate in silico TF knockouts (KO) by integrating Hopfield ODEs (Section 8).
   - Quantify lineage biases, perturbation scores, and synergy (Section 8).
6. **Output:** Inferred GRNs, cell-specific energy states, simulated perturbation trajectories, and prioritized TF regulators.

## 1. Model Framework

### 1.1 Continuous Hopfield Dynamics

scHopfield models single-cell gene regulatory dynamics as a continuous Hopfield network [1,2]. The expression level of gene $i$ evolves according to:

$$\frac{dx_i}{dt} = \sum_{j=1}^{N} W_{ij}\,\varphi_j(x_j) - \gamma_i x_i + I_i \tag{1}$$

where $x_i \in \mathbb{R}_{\geq 0}$ is the (smoothed spliced) mRNA count of gene $i$; $W_{ij}$ is the regulatory weight from gene $j$ to gene $i$ (positive = activation, negative = repression); $\gamma_i > 0$ is a gene-specific degradation rate; and $I_i$ is a constant external input (bias) capturing regulation from genes outside the modeled network.

In matrix form, with $X \in \mathbb{R}^{M \times N}$ denoting the $M$-cell by $N$-gene expression matrix:

$$\dot{X} = \varphi(X)W^T - X\Gamma + \mathbf{1}\mathbf{I}^T \tag{2}$$

where $\varphi(X)$ applies the Hill function element-wise, $\Gamma$ is a diagonal matrix of degradation rates, and $\mathbf{I} \in \mathbb{R}^N$ is the bias vector broadcast across cells.

### 1.2 Hill Function Activation

The activation function for each gene is the Hill (sigmoidal) function:

$$\varphi_j(x_j) = \frac{x_j^{n_j}}{x_j^{n_j} + k_j^{n_j}} \tag{3}$$

where $x \geq 0$ is the gene expression level, $k > 0$ is the half-saturation constant, and $n \geq 1$ is the Hill coefficient. The parameter $k$ represents the half-maximal threshold—the expression level at which the transcription factor exerts half its maximal regulatory effect—and is directly related to the binding affinity of the transcription factor to its target promoters.

The parameter $n$ quantifies cooperativity in transcriptional regulation: $n > 1$ indicates positive cooperativity (switch-like behavior), and $n = 1$ corresponds to simple Michaelis–Menten kinetics. Unlike standard formulations where $n < 1$ might denote negative cooperativity, our model restricts $n \geq 1$ because repressive interactions are explicitly captured by negative weights in the regulatory network, $W_{ij} < 0$. Overall, this function smoothly maps non-negative expression values to the unit interval, $[0,1)$.

The derivative of the Hill function, required for the Jacobian analysis, is:

$$\varphi_j'(x_j) = \varphi_j(x_j)\left[1 - \varphi_j(x_j)\right] \frac{n_j}{x_j} = \frac{n_j \cdot k_j^{n_j} \cdot x_j^{n_j-1}}{\left(k_j^{n_j} + x_j^{n_j}\right)^2} \tag{4}$$

Crucially, restricting the domain of the Hill coefficient to $n \geq 1$ ensures that the derivative remains well-behaved and finite at $x = 0$, avoiding numerical instability during the evaluation of the Jacobian.

## 2. Estimation of the Hill Function Parameters

For each gene $g^{(i)}$ in a dataset of $n_c$ cells, the Hill parameters $(k_i, n_i)$ are estimated directly from the empirical cumulative distribution function (ECDF) of single-cell expression values via the following five-step procedure:

1. **Thresholding.** A minimum expression threshold is set to $\tau = 0.05 \cdot \max_j g^{(i)}_j$ to exclude near-zero (noise-dominated) observations. The fraction of cells below this threshold, $\text{offset} = |\{j: g^{(i)}_j < \tau\}| / n_c$, is recorded. The $m$ values above the threshold form the set of active observations $\{x_j\}_{j=1}^m$.

2. **ECDF construction.** The active observations are sorted in ascending order and assigned uniform cumulative probabilities $y_j = (j-1)/(m-1)$ for $j = 1,\ldots,m$, approximating the marginal ECDF of non-zero expression.

3. **Log-linear transformation.** Taking logarithms of the Hill equation $\varphi(x) = y$ yields the linear relationship:

$$n\,\tilde{x} + b = \tilde{y} \tag{5}$$

where $\tilde{x} = \log x$, $\tilde{y} = \log\!\left(\frac{y}{1-y}\right)$, and $b = -n\log k$. Points where $\tilde{x}$ or $\tilde{y}$ are non-finite (arising from $y = 0$ or $y = 1$) are discarded prior to regression.

4. **Ordinary least squares.** The exponent $n$ and intercept $b$ are estimated by minimizing $\|\tilde{y} - (n\tilde{x} + b)\|_2^2$.

5. **Parameter recovery.** The half-maximal threshold is recovered as:

$$k = e^{-b/n} \tag{6}$$

The procedure is computationally efficient ($O(m \log m)$ dominated by sorting), runs independently for each gene, and does not require iterative optimization.

## 3. Gene Regulatory Network Inference

### 3.1 Moore–Penrose Pseudoinverse (Fast, $L^2$-minimal)

For a system with $N$ genes and $M$ cells, the Hopfield dynamics can be written as an overdetermined (or underdetermined) linear system in the unknown parameters $W$ and $\mathbf{I}$. The minimum-Frobenius-norm solution is obtained via the Moore–Penrose pseudoinverse:

$$P \approx A^+ B \tag{7}$$

where the explicitly augmented design matrix $A = [\varphi(X) \mid \mathbf{1}] \in \mathbb{R}^{M \times (N+1)}$ is formed by concatenating the $M \times N$ activation matrix with an $M \times 1$ column vector of ones. The target matrix is $B = \dot{X} + X\Gamma \in \mathbb{R}^{M \times N}$. The resulting parameter matrix is $P = \begin{bmatrix} W^T \\ \mathbf{I}^T \end{bmatrix} \in \mathbb{R}^{(N+1) \times N}$, whose first $N$ rows represent the transposed interaction matrix $W^T$, and whose final row represents the bias vector $\mathbf{I}^T$.

This yields the minimum-Frobenius-norm least-squares solution and coincides with the $\lambda \to 0^+$ limit of Tikhonov (ridge) regularization. When the number of cells exceeds the augmented features dimension ($M > N+1$), the system is overdetermined and the pseudoinverse minimizes the reconstruction residual; when $M < N+1$ it is underdetermined and the pseudoinverse provides the solution with minimum parameter norm.

### 3.2 Scaffold-Guided Regularized Optimization

When prior knowledge of regulatory interactions is available (e.g., from curated transcription-factor databases such as ENCODE or JASPAR), a scaffold-guided neural-network optimization is used. A binary scaffold matrix $S \in \{0,1\}^{N \times N}$ encodes known interactions: $S_{ij} = 1$ if gene $j$ is a known regulator of gene $i$.

**Network parametrization.** The interaction matrix $W$ is a learnable $N \times N$ weight matrix; the bias vector $I \in \mathbb{R}^N$ is learnable. The degradation rates $\gamma_i$ are log-parameterized as $\gamma_i = \exp(\tilde{\gamma}_i)$ to enforce positivity; by default, they are pre-estimated and held fixed.

**Scaffold mask.** A column-level mask is derived: gene $j$ is considered a candidate transcription factor (Prior TF) if it has at least one prior edge in the scaffold, i.e., $\mathbf{1}_{[\sum_i S_{ij} > 0]}$. The matrix $\bar{S} = \mathbf{1} - S$ (element-wise complement) identifies $W$ entries not supported by prior knowledge.

**Loss function.** The optimization objective is:

$$\mathcal{L} = \lambda_{\text{rec}}\,\mathcal{L}_{\text{rec}} + \lambda_{\text{scaffold}}\,\mathcal{L}_{\text{scaffold}} + \lambda_{\text{bias}}\,\mathcal{L}_{\text{bias}} \tag{8}$$

with components:
- **Reconstruction loss**: $\mathcal{L}_{\text{rec}} = \|\dot{X} - (\varphi(X)W^T - X\Gamma + \mathbf{1}I^T)\|_p$ where $p \in \{1, 2\}$ (L1 or MSE, configurable).
- **Scaffold regularization** (penalizes interactions not in the prior network):
$$\mathcal{L}_{\text{scaffold}} = \|W \odot \bar{S}\|_2 + \|W \odot \bar{S}\|_1 \tag{9}$$
The elastic-net combination promotes both small and sparse deviations from the scaffold.
- **Bias regularization**: $\mathcal{L}_{\text{bias}} = \|I\|_2^2$, preventing overfitting through unbounded external inputs.

**Optimization.** Parameters are optimized using mini-batch gradient descent via the Adam optimizer [3]. By default, the reconstruction error is quantified using a Mean Squared Error (MSE) loss criterion. Regularization penalties are integrated into the total loss to constrain the parameter space. To stabilize training and ensure convergence, an optional `ReduceLROnPlateau` scheduler is employed; this monitors the total loss and scales down the learning rate by a fixed factor if the loss plateaus for a specified number of consecutive epochs, down to a predefined minimum learning rate. Following training convergence, a hard thresholding step is applied: interaction weights with an absolute magnitude below a user-defined threshold are pruned to zero, enforcing sparsity in the final inferred regulatory network.

### 3.3 Limitations and Parameter Identifiability

The inference of interaction matrices ($W$), degradation rates ($\Gamma$), and bias terms ($\mathbf{I}$) from expression and velocity data is generally non-unique. In systems where the number of cells is smaller than the number of genes plus one ($M < N+1$), the linear system is underdetermined. The Moore-Penrose pseudoinverse provides the specific solution with the minimum parameter norm among infinitely many valid solutions. For scaffold-guided optimization, the solution heavily depends on the prior knowledge matrix $S$ to constrain the search space. Furthermore, the accuracy of the inferred GRN is fundamentally bounded by the quality of the upstream RNA velocity estimates. As previously noted, typical biological GRNs are asymmetric, which explicitly removes the guarantee of formal Lyapunov stability ($dE/dt \leq 0$) and allows for complex dynamical behaviors like limit cycles and chaos that symmetric networks cannot produce.

## 4. Energy Function

### 4.1 Derivation

For the continuous Hopfield dynamics with Hill function activation, the corresponding Lyapunov (energy) function is [1,2]:

$$E(\mathbf{x}) = -\frac{1}{2}\,\boldsymbol{\sigma}^T W\,\boldsymbol{\sigma} + \sum_i \gamma_i \int_0^{\sigma_i} \varphi_i^{-1}(z)\,dz - \mathbf{I}^T\boldsymbol{\sigma} \tag{10}$$

where $\boldsymbol{\sigma} = \varphi(\mathbf{x})$ denotes the vector of Hill-function activations, $\sigma_i = \varphi_i(x_i)$, and $\varphi_i^{-1}$ is the inverse Hill function $\varphi_i^{-1}(y) = k_i\left(y/(1-y)\right)^{1/n_i}$. When $W$ is symmetric, $dE/dt \leq 0$ along every trajectory, ensuring that the energy is non-increasing and that the system converges to equilibrium points corresponding to local minima of the energy landscape. 

Importantly, biological gene regulatory networks are inherently asymmetric ($W_{ij} \neq W_{ji}$), which breaks the formal Lyapunov property ($dE/dt \leq 0$). For asymmetric $W$, the computed $E(\mathbf{x})$ serves as an approximate heuristic landscape describing relative expression stability, rather than a strictly monotonic gradient-flow potential. 

### 4.2 Three Energy Components

The total energy decomposes into three biologically interpretable components:

**Interaction energy** (gene–gene coupling):
$$E_{\text{int}} = -\frac{1}{2}\,\boldsymbol{\sigma}^T W\,\boldsymbol{\sigma} = -\frac{1}{2}\sum_{i,j}\sigma_i\,W_{ij}\,\sigma_j \tag{11}$$

A large (negative) interaction energy indicates that the cell's gene expression pattern is strongly aligned with a dominant eigenvector of $W$, characteristic of a stable attractor state.

**Degradation energy** (intracellular balance):
$$E_{\text{deg}} = \sum_i \gamma_i \int_0^{\sigma_i} \varphi_i^{-1}(z)\,dz \tag{12}$$

This term is computed analytically using the Gauss hypergeometric function $_2F_1$ [4]:

$$\int_0^{\sigma} \varphi^{-1}(z)\,dz = \frac{-n\,k\,(1-\sigma)^{(n-1)/n}}{n-1}\;{}_2F_1\!\left(-\frac{1}{n},\frac{n-1}{n};\frac{2n-1}{n};\,1-\sigma\right) - C_0 \tag{13}$$

where the constant $C_0 = \frac{-n\,k}{n-1}\;{}_2F_1\!\left(-\frac{1}{n},\frac{n-1}{n};\frac{2n-1}{n};\,1\right)$ ensures the integral vanishes at $\sigma = 0$.

**Bias energy** (external regulation):
$$E_{\text{bias}} = -\mathbf{I}^T\boldsymbol{\sigma} = -\sum_i I_i\,\sigma_i \tag{14}$$

A large (negative) bias energy reflects strong external activation driving expression.

### 4.3 Gene-wise Decomposition

Each energy component can be decomposed gene-wise, yielding per-cell, per-gene energy contributions. This enables the identification of genes that make the largest contributions to a cell-type's energy landscape, facilitating targeted experimental follow-up.

### 4.4 Attractors and Stable States

In the scHopfield framework, **attractors** represent terminal or metastable cell fates (e.g., fully differentiated states or self-renewing progenitor pools). A **basin of attraction** is the set of all initial cell states whose trajectories eventually converge to the same attractor. The overall topology of these attractors and basins constitutes the **attractor landscape**, analogous to Waddington's epigenetic landscape.

In practice, local stable states are identified by finding regions in the projected 2D expression manifold where the cell-state velocity approaches zero and the computed Hopfield energy is minimized. These empirical minima represent the dominant modes of the local regulatory network.

## 5. Energy Landscape Visualization

### 5.1 Embedding-Based Landscape Reconstruction

Direct visualization of the $N$-dimensional energy function is infeasible. scHopfield projects it onto a two-dimensional embedding space using the following procedure:

1. **UMAP embedding.** A UMAP model $T: \mathbb{R}^N \to \mathbb{R}^2$ is fitted to the gene expression profiles of all cells using 30 nearest neighbors and a minimum distance of 0.1, retaining the inverse transform capability.

2. **Grid construction.** A regular $50 \times 50$ grid of points $\{p_i\}_{i=1}^{2500}$ is created in the two-dimensional embedding space, spanning the bounding box of all projected cells.

3. **Inverse projection.** Each grid point $p_i$ is mapped back to gene expression space using UMAP approximate inverse transform, yielding: $q_i = T^{-1}(p_i) \in \mathbb{R}^N$.

4. **Energy evaluation.** The Hopfield energy $E(q_i)$ is computed at each inverse-projected point using the cluster-specific parameters $(W, \gamma, I)$.

5. **Surface visualization.** The energy grid is plotted as a two-dimensional surface (heatmap or contour), with cells overlaid at their UMAP coordinates colored by their directly-computed energy $E(\mathbf{x}_c)$.

### 5.2 Multi-Cell-Type Landscape Merging

When a dataset contains multiple cell types, each with its own inferred GRN, the cell-type-specific energy landscapes are merged into a single, continuous landscape using a Gaussian kernel weighting scheme. It is crucial to note that this merged landscape is purely a **visualization object** designed to help interpret cross-cluster relationships; it interpolates across local networks and does not represent a single mathematically coherent global energy function. An extended grid covering all cell types is constructed, and the energy at each extended grid point $i$ is the weighted average over all cell-type-specific grids:

$$E_i = \frac{\displaystyle\sum_l \sum_j E_j^l\,r_{ij}^l}{\displaystyle\sum_l \sum_j r_{ij}^l} \tag{15}$$

where $l$ indexes cell types, $j$ indexes grid points within cell-type-specific landscape $l$, and the Gaussian weight is:

$$r_{ij}^l = \exp\!\left(-\frac{d_{ij}^2}{2\sigma^2}\right) \tag{16}$$

with $d_{ij}$ the Euclidean distance in 2D embedding space between extended grid point $i$ and cell-type grid point $j$, and $\sigma$ set to the length of a pixel diagonal in the extended grid. This bandwidth choice ensures that contributions decay rapidly beyond the nearest grid spacing, producing smooth interpolation without long-range blurring.

## 6. Jacobian Analysis

### 6.1 Jacobian Matrix

The Jacobian of the velocity field $\mathbf{f}(\mathbf{x}) = W\varphi(\mathbf{x}) - \gamma\mathbf{x} + \mathbf{I}$ evaluated at cell state $\mathbf{x}$ is:

$$J_{ij}(\mathbf{x}) = \frac{\partial f_i}{\partial x_j} = W_{ij}\,\varphi_j'(x_j) - \gamma_i\,\delta_{ij} \tag{17}$$

where $\varphi_j'(x_j) = n_j \cdot \varphi_j(x_j)\left[1 - \varphi_j(x_j)\right] / x_j$ and $\delta_{ij}$ is the Kronecker delta. The Jacobian linearizes the gene regulatory dynamics around each cell's current state, capturing instantaneous network effects.

### 6.2 Divergence (Local Volume Expansion)

The divergence of the velocity field quantifies the rate at which state-space volume expands or contracts locally:

$$\nabla \cdot \mathbf{f}(\mathbf{x}) = \mathrm{Tr}(J(\mathbf{x})) = \sum_{i=1}^N J_{ii}(\mathbf{x}) = \sum_{k=1}^N \lambda_k(\mathbf{x}) \tag{18}$$

where $\lambda_k$ are the eigenvalues of $J(\mathbf{x})$. Positive divergence indicates local state-space expansion (unstable, proliferating phase); negative divergence indicates contraction (convergence towards an attractor).

### 6.3 Eigenvalue Analysis (Stability and Oscillations)

The eigenvalues $\lambda_k(\mathbf{x})$ of $J(\mathbf{x})$ characterize the local stability of the dynamical system:
- $\text{Re}(\lambda_k) < 0$: trajectories locally converge along the $k$-th eigendirection.
- $\text{Re}(\lambda_k) > 0$: trajectories locally diverge (saddle point or unstable focus).
- $\text{Im}(\lambda_k) \neq 0$: the linearized system exhibits local rotational dynamics. 

It is important to emphasize that complex eigenvalues merely indicate local "spiraling" behavior near the evaluated point; they do not automatically imply the existence of global, sustained oscillatory trajectories (limit cycles) in the full nonlinear system.

### 6.4 Vorticity (Local Rotation Rate)

The skew-symmetric part of the Jacobian captures the local rotational component of the flow:

$$A(\mathbf{x}) = \frac{1}{2}\left(J(\mathbf{x}) - J(\mathbf{x})^T\right) \tag{19}$$

The Frobenius norm measures the magnitude of local rotation:

$$\|A(\mathbf{x})\|_F = \sqrt{\frac{1}{2}\sum_{i,j}\left(\frac{\partial f_i}{\partial x_j} - \frac{\partial f_j}{\partial x_i}\right)^2} \tag{20}$$

Large vorticity indicates that the local flow field has a strong rotational component. However, similar to complex eigenvalues, large local skew-symmetric components do not guarantee the presence of macroscopic limit cycles or macroscopic oscillations.

## 7. Network Analysis

### 7.1 Symmetricity

Because the analytical energy function is guaranteed to be a Lyapunov function only when $W$ is symmetric, we quantify how close the inferred network is to symmetry using:

$$\operatorname{Symm}(M) = \frac{\|M^S\| - \|M^A\|}{\|M^S\| + \|M^A\|} \tag{21}$$

where $M^S = \frac{1}{2}(M + M^T)$ and $M^A = \frac{1}{2}(M - M^T)$ are the symmetric and antisymmetric parts, respectively, and $\|\cdot\|$ is any matrix norm, in this manuscript we use the Frobenius norm. The measure equals $+1$ for fully symmetric matrices and $-1$ for fully antisymmetric matrices.

### 7.2 Network Centrality Measures

To identify key regulatory genes, four standard centrality metrics are computed for the inferred gene regulatory network (GRN). For the following definitions, let $W$ denote the weighted directed adjacency matrix of the inferred GRN, and let $B$ denote its binarized counterpart (where $B_{ij} = 1$ if an edge exists, and $0$ otherwise).

**Degree Centrality (Unweighted)** — direct connectivity, computed on the binarized digraph $B$:
$$C_{D_{\text{in}}}(v) = \sum_i B_{iv}, \qquad C_{D_{\text{out}}}(v) = \sum_i B_{vi} \tag{22}$$

In-degree quantifies the discrete number of regulators targeting a gene; out-degree measures the number of genes it directly regulates.

**Weighted Degree (Strength)** — cumulative regulatory magnitude, computed using the absolute values of the weighted digraph $W$:
$$S_{\text{in}}(v) = \sum_i |W_{iv}|, \qquad S_{\text{out}}(v) = \sum_i |W_{vi}| \tag{23}$$

This metric extends degree centrality by quantifying the total absolute strength of the regulation a gene receives (in-strength) or exerts on others (out-strength).

**Eigenvector Centrality** — influence accounting for neighbor importance, computed on the weighted digraph $W$:
$$W\mathbf{x} = \lambda_{\max}\mathbf{x} \tag{24}$$

Because the directed matrix $W$ can be asymmetric and yield complex eigenvalues, the centrality of gene $v$ is given by the real part of its corresponding component in the eigenvector, $\text{Re}(\mathbf{x}_v)$, associated with the largest eigenvalue $\lambda_{\max}$. We use the real part because it captures the magnitude of the dominant long-term dynamical mode of influence, effectively discarding phase shifts caused by imaginary components.

**Betweenness Centrality** — network bridging role, computed using shortest paths on the binarized digraph $B$:
$$g(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}} \tag{25}$$

where $\sigma_{st}$ is the total number of shortest paths from gene $s$ to gene $t$ in the binarized network, and $\sigma_{st}(v)$ is the number of those paths passing through gene $v$.

### 7.3 Eigendecomposition of the Interaction Matrix

The eigendecomposition $W = V\Lambda V^{-1}$ decomposes the interaction matrix into its principal regulatory modes. The eigenvectors $v_{\max}$ and $v_{\min}$ corresponding to the eigenvalues with the largest positive and most negative real parts, respectively, highlight the genes driving the dominant positive and negative feedback loops. The magnitude of each gene's component in these eigenvectors indicates its contribution to the corresponding dynamical mode.

### 7.4 Cross-Cell-Type Network Similarity

To compare GRNs across cell types (with matrices of equal shape $N \times N$ but potentially different numbers of cells), the following metrics are used:

**For expression matrices** (different cell counts): the RV coefficient [5] and its modified version [6]:
$$RV(X,Y) = \frac{\mathrm{tr}(XX^TYY^T)}{\sqrt{\mathrm{tr}[(XX^T)^2]\,\mathrm{tr}[(YY^T)^2]}} \tag{26}$$

**For interaction matrices** (same shape):
- Jaccard index: $J(W^{(1)},W^{(2)}) = \|B^{(1)} \wedge B^{(2)}\|_1 / \|B^{(1)} \vee B^{(2)}\|_1$ (shared vs. total edges)
- Hamming distance: $H(W^{(1)},W^{(2)}) = \|\mathrm{Vec}(B^{(1)}) \veebar \mathrm{Vec}(B^{(2)})\|_1$
- Frobenius distance: $\|W^{(1)}-W^{(2)}\|_F$
- Pearson correlation: $P(W^{(1)},W^{(2)}) = \frac{\mathrm{Cov}(\mathrm{Vec}(W^{(1)}),\mathrm{Vec}(W^{(2)}))}{\sqrt{\mathrm{Var}(\mathrm{Vec}(W^{(1)}))\,\mathrm{Var}(\mathrm{Vec}(W^{(2)}))}}$

### 7.5 Energy–Gene Correlation

The Pearson correlation between cellular energy and the expression of gene $k$ identifies genes whose expression level co-varies with the system's energy state:

$$\mathrm{Corr}(g_k, E) = \frac{\mathrm{Cov}(E, g_k)}{\sigma_E\,\sigma_{g_k}} \tag{27}$$

where $E$ is the energy vector across the selected set of cells and $g_k$ is the corresponding expression vector. A high positive correlation indicates that high expression of gene $k$ is associated with high-energy (undifferentiated or transitional) states; a high negative correlation identifies genes associated with low-energy (committed) states. Correlations can be computed for the total energy or for each of its three components independently.

## 8. Perturbation Simulation

All perturbation simulations use cell-type-specific GRN parameters $(W^{(c)}, \gamma^{(c)}, I^{(c)})$ by default, so that the propagated perturbation effect is conditioned on the regulatory program of each cell's annotated type.

### 8.1 GRN Signal Propagation

To simulate the downstream effects of a genetic perturbation, scHopfield implements an iterative signal-propagation algorithm inspired by the CellOracle framework [7].

For each cell, the perturbation is applied by setting the expression of the target genes to their prescribed values ($x_k^{\text{perturb}} = v_k$ for perturbed gene $k$). Then, the perturbation signal is propagated through the GRN in $R$ steps:

**Step 1** (direct effects only): the expression change propagated from the $K$ perturbed genes $\{k_1,\ldots,k_K\}$ is:
$$\Delta x_i^{(1)} = \mathrm{d}t \sum_{k \in \mathcal{P}} W_{ik}\left[\varphi_k\!\left(x_k^{\text{current}}\right) - \varphi_k\!\left(x_k^{\text{original}}\right)\right] \tag{28}$$

$$x_i^{(1)} = x_i^{\text{current}} + \Delta x_i^{(1)} \tag{29}$$

**Steps $r = 2,\ldots,R$** (cascade effects): propagation continues from all active transcription-factor genes $\mathcal{T} = \{j : \sum_i |W_{ij}| > 0\}$ (Inferred TFs with any non-zero outgoing edges):
$$x_i^{(r)} = x_i^{(r-1)} + \mathrm{d}t \sum_{j \in \mathcal{T}} W_{ij}\left[\varphi_j\!\left(x_j^{(r-1)}\right) - \varphi_j\!\left(x_j^{\text{original}}\right)\right] \tag{30}$$

At each step, expression values are clipped to $[0, x_{\max}]$ where $x_{\max}$ is set to twice the 99th-percentile of the observed expression range per gene to prevent numerical divergence. The default number of propagation steps is $R = 3$ and the scaling factor is $\mathrm{d}t = 1.0$; these control the reach of indirect regulatory effects. By default, the perturbed genes are held fixed at their prescribed values throughout all steps. The final perturbation effect is $\Delta X = X^{(R)} - X^{\text{original}}$.

### 8.2 ODE Integration

An alternative, continuous-time perturbation simulation integrates the Hopfield ODE (Equation 1) directly for each cell from its current state, with perturbed genes optionally held fixed as constraints.

Each cell is integrated over a total time of $T = 5.0$ using $100$ uniform time steps, giving a step size of $\Delta t = 0.05$. Three integration methods are available: (i) forward Euler with per-step clipping (default, numerically stable), (ii) `scipy.integrate.odeint`, and (iii) `scipy.integrate.solve_ivp` with RK45 or other adaptive methods. The 99th percentile upper bound is applied to each gene at each step to prevent divergence. Simulations are run in parallel across all cells using multi-threaded execution.

**GPU acceleration.** The dataset-wide ODE perturbation (`simulate_shift_ode`) supports GPU-batched integration via [torchdiffeq](https://github.com/rtqichen/torchdiffeq). When a CUDA-capable device is available, all cells belonging to a given cell type are stacked into a single tensor and integrated in parallel on GPU, which substantially reduces wall-clock time for large datasets. The integration method is selected via the `device` parameter (`'cpu'`, `'cuda'`, or `None` for automatic detection); when `device=None` and CUDA is available, GPU is used automatically for torchdiffeq-compatible methods (`euler`, `rk4`, `midpoint`, `dopri5`, etc.). If a GPU out-of-memory error occurs mid-run, the simulation falls back gracefully to CPU for the remaining cell types. GPU memory is explicitly released after each cell-type batch via `torch.cuda.empty_cache()` to prevent accumulation across the screen.

### 8.3 Embedding Flow Projection

Perturbation-induced expression changes $\Delta X$ (from either GRN propagation or ODE simulation) are projected onto the two-dimensional embedding to produce a vector field visualization. scHopfield implements two complementary projection strategies:

**Method 1 — Gene-space dot-product (Hopfield-native).** For each cell $i$, the $k$ nearest neighbors $\mathcal{N}(i)$ are identified in gene expression space. A cosine-similarity weight is computed between the perturbation vector $\Delta\mathbf{x}_i$ and the expression difference $\mathbf{x}_j - \mathbf{x}_i$ to each neighbor $j$:
$$w_{ij} = \exp\!\left(-\frac{d_{ij}}{\tilde{d}_i}\right)\cdot \max\!\left(\frac{\Delta\mathbf{x}_i \cdot (\mathbf{x}_j - \mathbf{x}_i)}{\|\Delta\mathbf{x}_i\|\,\|\mathbf{x}_j - \mathbf{x}_i\|}, 0\right)$$
where $d_{ij}$ is the gene-space distance and $\tilde{d}_i$ its median. Only neighbors with positive alignment contribute. The 2D displacement is the weighted average:
$$\mathbf{u}_i = \frac{\sum_{j \in \mathcal{N}(i)} w_{ij}\,(\mathbf{e}_j - \mathbf{e}_i)}{\sum_{j \in \mathcal{N}(i)} w_{ij}} \tag{31}$$
where $\mathbf{e}_i$ is the 2D embedding coordinate of cell $i$.

**Method 2 — Embedding-space correlation (CellOracle-style).** Neighbors are identified in 2D embedding space. For each neighbor $j$ of cell $i$, the Pearson correlation $\rho(\Delta\mathbf{x}_i,\, \mathbf{x}_j - \mathbf{x}_i)$ is computed in gene space. Transition probabilities are obtained via a softmax-like kernel:
$$p_{ij} \propto \exp\!\left(\rho(\Delta\mathbf{x}_i,\, \mathbf{x}_j - \mathbf{x}_i) \,/\, \sigma_\rho\right)$$
where $\sigma_\rho$ is a bandwidth parameter. The embedded flow is the probability-weighted mean displacement minus the uniform-KNN baseline, following the transition-probability formulation of RNA velocity [ref]. For computational efficiency, a random fraction of neighbors is sampled per cell.

The two methods are accessed via `method='hopfield'` (Method 1) and `method='celloracle'` (Method 2) in `sch.tl.calculate_flow`. Method 1 uses the Hopfield model's own velocity field and aligns perturbations with the gene-space geometry; Method 2 replicates the CellOracle/scVelo embedding approach and is provided for comparability.

### 8.4 TF Candidate Prioritization (`score_driver_tfs`)

Prior to running KO screens, transcription factors are prioritized as lineage-driver candidates using a composite rank-sum score derived from GRN structure. For each gene $g$ and each lineage (A or B), three signals are computed by averaging over the lineage's constituent cell types:

1. **W-matrix regulatory strength** ($W_{\mathrm{norm}}$): mean L2-norm of row $g$ across the cell-type-specific $W^{(c)}$ matrices (total outgoing regulatory strength).
2. **Out-degree centrality** ($D_{\mathrm{out}}$): mean normalized out-degree from `compute_network_centrality`.
3. **|Energy–gene correlation|** ($|r_E|$): mean absolute Pearson correlation between cellular energy and gene expression from `energy_gene_correlation`.

Each signal is converted to an integer rank across all genes. The composite scores are:
$$\text{score}_A = \mathrm{rank}(W_{\mathrm{norm}, A}) + \mathrm{rank}(D_{\mathrm{out}, A}) + \mathrm{rank}(|r_E|_A)$$
and analogously for lineage B. The equal-weight rank aggregation provides a non-parametric method to balance structural evidence (W-matrix norm and out-degree) with functional evidence (energy correlation) without assuming a specific parametric scale for any individual metric. The **lineage bias** $\Delta = \text{score}_A - \text{score}_B$ is positive for genes that are more regulatory-influential in lineage A than B, and negative for the converse. Candidate genes for KO screening are selected by taking the top-$n$ genes by $\text{score}_A$ and top-$n$ by $\text{score}_B$.

### 8.5 Perturbation Effect Quantification

After running a KO simulation, several complementary metrics quantify its magnitude:

**Per-cluster effect score.** For each cell-type cluster $c$, let $C_c$ denote the set of cells belonging to that cluster, and let $N$ be the total number of genes (as defined in Section 1.1). The mean absolute expression change is:
$$S_c = \frac{1}{|C_c| \cdot N} \sum_{i \in C_c} \|\Delta\mathbf{x}_i\|_1$$
or alternatively using the L2 norm, median, or maximum across cells. This produces a cluster × gene matrix of perturbation effects.

**Cell transition score.** The per-cell perturbation magnitude is the L2-norm of the expression-change vector:
$$m_i = \|\Delta\mathbf{x}_i\|_2$$
Large values indicate cells whose expression state is strongly displaced by the perturbation.

### 8.6 Lineage Bias Score (`compute_lineage_bias`)

To quantify how strongly a perturbation biases differentiation toward a specific lineage, perturbation flow vectors $\Delta X$ are projected to the embedding using the dot-product method (§8.3, Method 1), yielding per-cell embedded flows $\mathbf{u}_i$. These are compared with the wild-type Hopfield velocity field $\mathbf{v}_i^{\mathrm{WT}}$ (computed once on the unperturbed adata) via cosine similarity:
$$\rho_i = \frac{\mathbf{v}_i^{\mathrm{WT}} \cdot \mathbf{u}_i}{\|\mathbf{v}_i^{\mathrm{WT}}\|\,\|\mathbf{u}_i\|} \tag{32}$$
Cosine similarity is utilized here because it isolates directional alignment independent of the perturbation's magnitude, ensuring that large but undirected perturbations do not artificially inflate the lineage score. The lineage score for lineage $\ell$ is the mean cosine similarity over cells belonging to $\ell$:
$$\text{score}_\ell = \frac{1}{|C_\ell|}\sum_{i \in C_\ell} \rho_i$$
Positive $\text{score}_\ell$ indicates that the perturbation flow aligns with the natural differentiation direction toward lineage $\ell$; negative values indicate opposition. The **lineage bias** is:
$$\Delta_{\mathrm{bias}} = \text{score}_A - \text{score}_B$$

### 8.7 Pseudotime-Based Perturbation Score (`compute_perturbation_score`)

An alternative perturbation score, analogous to the CellOracle perturbation score [7], uses the pseudotime gradient as a reference direction for differentiation. Note that this score explicitly measures alignment with an external, empirically derived pseudotime gradient, rather than alignment with the inferred Hopfield dynamical system itself. The algorithm proceeds as follows:

1. **Pseudotime surface.** A smooth pseudotime field is estimated on the $n_g \times n_g$ regular 2D grid by fitting a degree-3 polynomial regression to observed pseudotime values at cell positions.
2. **Gradient field.** The 2D gradient $\nabla \tau(p)$ of the pseudotime surface is computed at each grid point via finite differences, giving the local direction of increasing pseudotime (i.e., the expected differentiation direction).
3. **Flow interpolation to grid.** KO perturbation flow vectors are interpolated from cells to grid points using Gaussian-kernel-weighted KNN averaging (bandwidth $\sigma$ = median nearest-neighbor distance). Grid points with cell density below $\rho_{\min}$ (default 1% of maximum) are masked.
4. **Perturbation score.** At each non-masked grid point $p$, the perturbation score is the cosine similarity between the interpolated KO flow and the pseudotime gradient:
$$\mathrm{PS}(p) = \frac{\hat{\mathbf{u}}(p) \cdot \nabla\tau(p)}{\|\hat{\mathbf{u}}(p)\|\,\|\nabla\tau(p)\|} \tag{33}$$
$\mathrm{PS} > 0$: perturbation promotes differentiation; $\mathrm{PS} < 0$: perturbation opposes differentiation. The summary ranking metric is the sum of negative PS values across non-masked grid points, matching CellOracle's convention [7].

### 8.8 KO Screening and Synergy Analysis

To systematically identify lineage-driver genes and pairs, scHopfield provides two screening functions:

**Single-gene KO screen** (`run_ko_screen`): for each candidate gene $g$, runs `simulate_shift_ode({g: 0})` on a copy of the dataset, then computes lineage bias (§8.6) and per-cluster effects (§8.5). Results are returned as dictionaries keyed by gene name.

**Pairwise KO screen** (`run_pairwise_ko_screen`): for each candidate pair $(g_1, g_2)$, runs `simulate_shift_ode({g_1: 0, g_2: 0})` and computes the same metrics. This enables identification of gene pairs that act cooperatively to bias lineage commitment.

**Synergy score.** Given single-KO lineage biases $\Delta_{g_1}$ and $\Delta_{g_2}$ and the double-KO lineage bias $\Delta_{g_1,g_2}$, the expected additive effect is the sum of their individual biases ($\Delta_{g_1} + \Delta_{g_2}$). The deviation from this additive baseline (the cancellation error) is:
$$\mathrm{CE}(g_1, g_2) = \Delta_{g_1,g_2} - (\Delta_{g_1} + \Delta_{g_2}) \tag{34}$$
To yield a directional synergy score that is universally positive for synergistic (super-additive) interactions and negative for antagonistic (redundant or buffering) interactions, this deviation is multiplied by the sign of the primary anchor gene's single-KO bias:
$$\mathrm{Syn}(g_1, g_2) = \mathrm{CE}(g_1, g_2) \cdot \operatorname{sgn}(\Delta_{g_1}) \tag{35}$$
This explicitly measures synergy against an additive null model. The additive baseline assumes that the two genes operate in independent, parallel pathways; therefore, any significant deviation (synergy or antagonism) provides strong evidence for functional interaction or mechanistic saturation between the targeted nodes.

## 9. Pseudotime-Based Velocity Estimation

When splicing kinetics are unavailable (e.g., datasets with only pseudotime and a neighbor graph), RNA velocity is estimated from pseudotime ordering.

**Forward-restricted graph.** A directed adjacency matrix is constructed from the existing kNN graph by retaining only edges pointing forward in pseudotime: $P_{ij} = A_{ij}$ if $t_j > t_i$, and 0 otherwise.

**Row normalization.** $P$ is row-normalized to $P_{\text{norm}} = \mathrm{diag}(\mathbf{P}\mathbf{1})^{-1} P$, so that each row sums to one (cells with no forward neighbors are assigned unit self-weight).

**Velocity estimation.** The velocity of cell $i$ is estimated as the average expression change per unit pseudotime:

$$\mathbf{v}_i = \frac{\displaystyle\sum_j p_{ij}\left(\mathbf{x}_j - \mathbf{x}_i\right)}{\displaystyle\sum_j p_{ij}\left(t_j - t_i\right) + \varepsilon} \tag{36}$$

where $p_{ij}$ are the row-normalized forward-graph weights, $\mathbf{x}_j - \mathbf{x}_i$ is the expression difference between cells $j$ and $i$, $t_j - t_i$ is the corresponding pseudotime interval, and $\varepsilon = 10^{-6}$ prevents division by zero. Cells without forward pseudotime neighbors receive zero velocity.

## 10. Datasets

**Pancreatic endocrinogenesis** (Bastidas-Ponce et al., 2019) [8]. This dataset contains 3,696 cells from embryonic day E15.5 mouse pancreas, spanning the secondary transition of endocrine commitment. Transcriptome profiles were generated from cells differentiating from endocrine progenitors into four major cell types: $\alpha$, $\beta$, $\delta$, and $\varepsilon$ cells. RNA velocity was estimated using Dynamo [9] with the conventional experiment type (steady-state labeling). Following velocity calculation, 2,000 dynamically informative genes were retained for all subsequent analyses.

**Human hematopoiesis** (Qiu et al., 2022) [9]. This dataset comprises 1,947 CD34+ hematopoietic stem and progenitor cells cultured for one week under differentiation conditions, profiled with 4sU metabolic labeling. The dataset captures a branching differentiation hierarchy from multipotent progenitors through MEP (Megakaryocyte/Erythrocyte Progenitor) and GMP (Granulocyte/Macrophage Progenitor) intermediates to five terminal fates: Erythrocytes, Megakaryocytes, Monocytes, Basophils, and Neutrophils. RNA velocity was estimated with Dynamo using the one-shot experiment type appropriate for metabolic labeling. After velocity calculation, 1,956 genes were retained.

**Mouse hematopoiesis** (Paul et al., 2015) [10]. This dataset of mouse bone-marrow hematopoietic progenitors was used for perturbation analysis benchmarking, allowing comparison of predicted transcription-factor knockout effects against known lineage-commitment phenotypes.

## 11. Preprocessing and Velocity

All datasets were preprocessed using the Dynamo package [9]. Gene and cell selection were performed using the Monocle recipe [11,12,13] with default parameters, which selects highly variable genes, filters low-quality cells, and normalizes expression. Velocity-related parameters were estimated using Dynamo's `DynamicsRecovery` module; the `experiment_type` parameter was set to conventional for standard ligation-based libraries and one_shot for 4sU metabolic labeling experiments. Smoothed spliced expression (the Ms layer from scVelo-style moment calculations) was used as the gene expression input $X$ for all scHopfield analyses.

## 12. Software Implementation

scHopfield is implemented in Python ($\geq 3.8$). Key dependencies include:

- **PyTorch** — differentiable computation and GPU acceleration for scaffold-guided optimization
- **NumPy / SciPy** — pseudoinverse inference, energy calculation (hyp2f1), Jacobian analysis
- **AnnData** — all inputs and outputs follow the AnnData convention: inferred networks in `adata.varp`, per-cell quantities in `adata.obs`, gene-level parameters in `adata.var`, and energy landscapes in `adata.uns`
- **UMAP-learn** — embedding and inverse transform for landscape visualization
- **joblib** — thread-based parallelization of ODE simulation across cells
- **h5py** — serialization of fitted models for reproducibility

The package provides a high-level API organized into preprocessing, tools, dynamics, and plotting modules that integrate with standard single-cell analysis workflows. Full documentation and source code are available at the project repository.

## 13. Default Parameters

Table 2 summarizes the default parameter values used throughout the framework.

**Table 2: Default Parameter Values**

| Parameter Category | Parameter Name | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Hill Estimation** | Minimum expression threshold | $5\%$ of max | Excludes near-zero noise |
| **Optimization** | Loss criterion | MSE ($L^2$) | Quantifies reconstruction error |
| | Optimizer | Adam | Mini-batch gradient descent |
| | Pruning threshold | User-defined (e.g., $10^{-3}$) | Hard threshold for $W$ sparsity |
| **UMAP Embedding** | Number of neighbors | 30 | Defines local neighborhood size |
| | Minimum distance | 0.1 | Controls local packing |
| **Energy Merging** | Grid resolution | $50 \times 50$ | Resolution of 2D energy landscape |
| | Gaussian bandwidth ($\sigma$) | 1 pixel diagonal | Smoothing factor for interpolation |
| **Perturbation** | Number of steps ($R$) | 3 | Iterations for signal propagation |
| | Time step ($\mathrm{d}t$) | 1.0 | Signal propagation scale |
| | Clipping threshold | $2 \times 99$th percentile | Prevents numerical divergence |
| **ODE Integration** | Total time ($T$) | 5.0 | Total simulation time per cell |
| | Number of steps | 100 ($\Delta t = 0.05$) | Default uniform time grid |
| | Integration method | Forward Euler | Numerical scheme (with clipping) |

## References

[1] Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.
[2] Hopfield, J. J. (1984). Neurons with graded response have collective computational properties like those of two-state neurons. *PNAS*, 81(10), 3088–3092.
[3] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
[4] Abramowitz, M., & Stegun, I. A. (1972). *Handbook of Mathematical Functions*. Dover.
[5] Robert, P., & Escoufier, Y. (1976). A unifying tool for linear multivariate statistical methods: the RV-coefficient. *Applied Statistics*, 25(3), 257–265.
[6] Smilde, A. K., et al. (2009). Matrix correlations for high-dimensional data: the modified RV-coefficient. *Bioinformatics*, 25(3), 401–405.
[7] Kamimoto, K., et al. (2023). Dissecting cell identity via network inference and in silico gene perturbation. *Nature*, 614, 742–751.
[8] Bastidas-Ponce, A., et al. (2019). Comprehensive single cell mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis. *Development*, 146(12), dev173849.
[9] Qiu, X., et al. (2022). Mapping transcriptomic vector fields of single cells. *Cell*, 185(4), 690–711.
[10] Paul, F., et al. (2015). Transcriptional heterogeneity and lineage commitment in myeloid progenitors. *Cell*, 163(7), 1663–1677.
[11] Qiu, X., et al. (2017). Reversed graph embedding resolves complex single-cell trajectories. *Nature Methods*, 14(10), 979–982.
[12] Trapnell, C., et al. (2014). The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells. *Nature Biotechnology*, 32(4), 381–386.
[13] Cao, J., et al. (2019). The single-cell transcriptional landscape of mammalian organogenesis. *Nature*, 566, 496–502.
