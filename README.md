# scHopfield

**Single-cell Hopfield Network Analysis**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for analyzing single-cell RNA-seq data using Hopfield network models. Infer gene regulatory networks, compute energy landscapes, analyze network topology, perform stability analysis, and simulate cellular dynamics—all with a **scanpy-style API** that integrates seamlessly with AnnData objects.

## Overview

scHopfield models gene regulatory networks (GRNs) as continuous Hopfield networks, where gene expression dynamics follow:

```
dx/dt = W·σ(x) - γ·x + I
```

**Key components:**
- **W**: Interaction matrix encoding gene-gene regulatory relationships
- **σ(x)**: Sigmoid activation function fitted to expression data
- **γ**: Degradation rates (mRNA decay)
- **I**: Bias vector representing external inputs/basal expression

This formulation allows:
- **Energy landscapes** that quantify cellular state stability
- **Jacobian analysis** for local stability and bifurcation detection
- **Network topology** analysis via centrality metrics and eigenanalysis
- **Trajectory simulation** for perturbation experiments and cell fate prediction

## Features

### Core Functionality
- **Preprocessing**: Sigmoid function fitting to gene expression distributions
- **Network Inference**: Learn interaction matrices from RNA velocity using gradient descent
- **Energy Landscapes**: Compute total energy and decompose into interaction, degradation, and bias components

### Network Analysis
- **Topology Analysis**: Network centrality metrics (degree, betweenness, eigenvector centrality)
- **Eigenanalysis**: Eigenvalue decomposition of interaction matrices with spectral visualization
- **Network Comparison**: Compare GRN structures across cell types using multiple similarity metrics
- **GRN Visualization**: Interactive network graphs with customizable layouts and styling

### Stability & Dynamics
- **Jacobian Analysis**: Compute Jacobian matrices at each cell state
- **Stability Metrics**: Eigenvalue spectra, trace, rotational components, and instability counts
- **Trajectory Simulation**: Simulate gene expression dynamics from any cell state
- **Perturbation Analysis**: In-silico gene knockouts and overexpression experiments

### Correlation Analysis
- **Energy-Gene Correlations**: Identify genes driving energy landscapes
- **Cell Type Similarity**: RV coefficient analysis for network similarity
- **Future State Prediction**: Correlate current states with predicted future states

### Visualization
- **Energy plots**: Landscapes, boxplots, scatter plots, and component decomposition
- **Network plots**: Interaction matrices, GRN graphs, centrality rankings
- **Stability plots**: Jacobian eigenvalue spectra, element heatmaps on UMAP
- **Dynamics plots**: Trajectory visualization in gene expression space
- **Correlation plots**: Gene-energy scatter plots and grid comparisons

## Prerequisites

Before using scHopfield, you need:

1. **Single-cell RNA-seq data** in AnnData format
2. **RNA velocity** computed (e.g., using [scVelo](https://scvelo.readthedocs.io/))
   - `adata.layers['Ms']` - spliced counts
   - `adata.layers['velocity_S']` - RNA velocity
   - `adata.var['gamma']` - degradation rates
3. **Cell type annotations** (e.g., `adata.obs['cell_type']`)
4. **Highly variable genes** selected (recommended: 2000-3000 genes)

## Installation

### From source

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install -e .
```

### Dependencies

**Core:**
- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- anndata >= 0.8.0

**Computation:**
- torch >= 1.9.0 (CPU or GPU)
- scikit-learn >= 1.0.0

**Visualization:**
- matplotlib >= 3.4.0
- seaborn (optional, for boxplots)
- networkx (for graph layouts)

**Network Analysis:**
- igraph (recommended for fast centrality computation, falls back to networkx)

**Other:**
- umap-learn >= 0.5.0
- tqdm >= 4.62.0
- hoggorm >= 0.13.0
- h5py >= 3.0.0 (for Jacobian storage)

## Quick Start

### Basic Workflow

```python
import scHopfield as sch
import scanpy as sc
import numpy as np

# Load your data (requires RNA velocity: e.g., from scVelo)
adata = sc.read_h5ad('data.h5ad')
# Required: adata.layers['Ms'], adata.layers['velocity_S'], adata.var['gamma']

# 1. Preprocessing: Fit sigmoid functions
sch.pp.fit_all_sigmoids(adata, genes=highly_variable_genes)
sch.pp.compute_sigmoid(adata)

# 2. Network Inference: Learn interaction matrices
sch.inf.fit_interactions(
    adata,
    cluster_key='cell_type',
    n_epochs=1000,
    device='cuda'  # or 'cpu'
)

# 3. Energy Analysis
sch.tl.compute_energies(adata, cluster_key='cell_type')
sch.tl.energy_gene_correlation(adata, cluster_key='cell_type')

# 4. Network Analysis
sch.tl.compute_network_centrality(adata, cluster_key='cell_type')
sch.tl.compute_eigenanalysis(adata, cluster_key='cell_type')
sch.tl.network_correlations(adata, cluster_key='cell_type')

# 5. Stability Analysis
sch.tl.compute_jacobians(adata, cluster_key='cell_type', device='cuda')
sch.tl.compute_jacobian_stats(adata)

# 6. Visualization
sch.pl.plot_energy_landscape(adata, cluster='HSC')
sch.pl.plot_interaction_matrix(adata, cluster='HSC', top_n=30)
sch.pl.plot_grn_network(adata, cluster='HSC', topn=50)
sch.pl.plot_jacobian_eigenvalue_spectrum(adata, cluster_key='cell_type')

# 7. Dynamics Simulation
trajectory = sch.dyn.simulate_trajectory(
    adata, cluster='HSC', cell_idx=0,
    t_span=np.linspace(0, 10, 100)
)
sch.pl.plot_trajectory(trajectory, t_span)
```

### Advanced: Perturbation Experiments

```python
# Simulate gene knockout
perturbed = sch.dyn.simulate_perturbation(
    adata,
    cluster='HSC',
    cell_idx=0,
    gene_perturbations={'GATA1': 0.0},  # knockout
    t_span=np.linspace(0, 20, 200)
)

# Compare with wild-type
wt = sch.dyn.simulate_trajectory(adata, cluster='HSC', cell_idx=0,
                                  t_span=np.linspace(0, 20, 200))
```

## API Reference

### Preprocessing (`sch.pp`)

- `fit_all_sigmoids(adata, genes, ...)` - Fit sigmoid functions to gene expression distributions
- `compute_sigmoid(adata, ...)` - Compute sigmoid-transformed expression for all cells

### Inference (`sch.inf`)

- `fit_interactions(adata, cluster_key, ...)` - Infer gene regulatory networks from velocity data

### Tools (`sch.tl`)

#### Energy Analysis
- `compute_energies(adata, cluster_key, ...)` - Calculate total energy landscapes for all cells
- `decompose_degradation_energy(adata, cluster, ...)` - Gene-wise degradation energy decomposition
- `decompose_bias_energy(adata, cluster, ...)` - Gene-wise bias energy decomposition
- `decompose_interaction_energy(adata, cluster, ...)` - Gene-wise interaction energy decomposition

#### Network Analysis
- `network_correlations(adata, cluster_key, ...)` - Compare GRN similarity across cell types
- `get_network_links(adata, cluster_key, ...)` - Extract network edges as DataFrame
- `compute_network_centrality(adata, cluster_key, ...)` - Compute centrality metrics for all genes
- `get_top_genes_table(adata, cluster, metric, n, ...)` - Get ranked table of top genes by centrality
- `compute_eigenanalysis(adata, cluster_key, ...)` - Eigenvalue decomposition of interaction matrices
- `get_top_eigenvector_genes(adata, cluster, k, n, ...)` - Extract top genes from specific eigenvectors
- `get_eigenanalysis_table(adata, cluster, k, n, ...)` - Formatted table of eigenanalysis results

#### Correlation Analysis
- `energy_gene_correlation(adata, cluster_key, ...)` - Correlate energies with gene expression
- `celltype_correlation(adata, cluster_key, ...)` - Cell type similarity via RV coefficient
- `future_celltype_correlation(adata, cluster_key, ...)` - Correlate current and future states

#### Embedding
- `compute_umap(adata, spliced_key, ...)` - Compute UMAP embedding from expression
- `energy_embedding(adata, resolution, ...)` - Project energy landscape onto 2D embedding
- `save_embedding(adata, filename)` - Save embedding and grid to file
- `load_embedding(adata, filename)` - Load embedding and grid from file

#### Jacobian & Stability Analysis
- `compute_jacobians(adata, cluster_key, device, ...)` - Compute Jacobian matrices and eigenvalues
- `save_jacobians(adata, filename, ...)` - Save Jacobians to HDF5 file
- `load_jacobians(adata, filename, ...)` - Load Jacobians from HDF5 file
- `compute_jacobian_stats(adata, ...)` - Compute summary statistics from eigenvalues
- `compute_jacobian_elements(adata, gene_pairs, ...)` - Compute specific partial derivatives
- `compute_rotational_part(adata, ...)` - Compute antisymmetric component of Jacobian

#### Velocity
- `compute_reconstructed_velocity(adata, cluster_key, ...)` - Compute model-predicted velocity
- `validate_velocity(adata, cluster_key, ...)` - Compare predicted vs observed velocity

### Plotting (`sch.pl`)

#### Energy Plots
- `plot_energy_landscape(adata, cluster, ...)` - 2D energy landscape on embedding
- `plot_energy_components(adata, cluster, ...)` - Grid of all energy components
- `plot_energy_boxplots(adata, cluster_key, order, ...)` - Boxplots of energy distributions
- `plot_energy_scatters(adata, cluster_key, order, ...)` - Energy scatter plots by component

#### Network Plots
- `plot_interaction_matrix(adata, cluster, top_n, ...)` - Heatmap of W matrix
- `plot_network_centrality_rank(adata, cluster_key, metric, ...)` - Ranked centrality plot
- `plot_centrality_comparison(adata, clusters, metric, ...)` - Compare centrality across clusters
- `plot_gene_centrality(adata, genes, cluster_key, ...)` - Multi-panel gene centrality plots
- `plot_centrality_scatter(adata, cluster, metric_x, metric_y, ...)` - Scatter two centrality metrics
- `plot_eigenvalue_spectrum(adata, cluster, ...)` - Eigenvalues in complex plane
- `plot_eigenvector_components(adata, cluster, k, ...)` - Sorted eigenvector components
- `plot_eigenanalysis_grid(adata, cluster_key, ...)` - Comprehensive eigenanalysis grid
- `plot_grn_network(adata, cluster, topn, ...)` - Full GRN graph visualization
- `plot_grn_subset(adata, cluster, selected_genes, ...)` - Focused GRN subnetwork

#### Jacobian Plots
- `plot_jacobian_eigenvalue_spectrum(adata, cluster_key, ...)` - Jacobian eigenvalues per cluster
- `plot_jacobian_eigenvalue_boxplots(adata, cluster_key, ...)` - Boxplots of eigenvalue parts
- `plot_jacobian_stats_boxplots(adata, cluster_key, ...)` - Boxplots of stability metrics
- `plot_jacobian_element_grid(adata, gene_pairs, ...)` - Partial derivatives on UMAP

#### Correlation Plots
- `plot_gene_correlation_scatter(adata, genes, cluster, ...)` - Gene vs energy scatter
- `plot_correlations_grid(adata, clusters, genes, ...)` - Grid of correlation plots

#### Other Plots
- `plot_sigmoid_fit(adata, gene, ...)` - Show sigmoid fit quality for a gene
- `plot_trajectory(trajectory, t_span, genes, ...)` - Plot simulated gene expression trajectories

### Dynamics (`sch.dyn`)

- `ODESolver` - ODE solver class for Hopfield dynamics simulation
- `create_solver(adata, cluster, ...)` - Create pre-configured solver instance
- `simulate_trajectory(adata, cluster, cell_idx, t_span, ...)` - Simulate from a cell's initial state
- `simulate_perturbation(adata, cluster, cell_idx, gene_perturbations, t_span, ...)` - Simulate with gene knockouts/overexpression

## Data Storage Conventions

scHopfield follows standard AnnData conventions and stores results systematically:

### `adata.var` (gene-level data)

**Sigmoid Parameters:**
- `sigmoid_threshold`, `sigmoid_exponent`, `sigmoid_offset`, `sigmoid_mse` - Fitted sigmoid parameters
- `scHopfield_used` - Boolean mask of genes used in analysis

**Network Parameters (per cluster):**
- `I_{cluster}` - Bias vector for each cluster
- `gamma_{cluster}` - Cluster-specific degradation rates (if refitted)

**Network Centrality (per cluster):**
- `degree_all_{cluster}` - Total degree (in + out)
- `degree_centrality_all_{cluster}` - Normalized total degree
- `degree_in_{cluster}`, `degree_out_{cluster}` - In/out degree
- `degree_centrality_in_{cluster}`, `degree_centrality_out_{cluster}` - Normalized in/out degree
- `betweenness_centrality_{cluster}` - Betweenness centrality
- `eigenvector_centrality_{cluster}` - Eigenvector centrality

**Correlations (per cluster):**
- `correlation_energy_total_{cluster}` - Total energy vs gene expression
- `correlation_energy_interaction_{cluster}` - Interaction energy correlation
- `correlation_energy_degradation_{cluster}` - Degradation energy correlation
- `correlation_energy_bias_{cluster}` - Bias energy correlation

### `adata.obs` (cell-level data)

**Energy Values (shared across all cells):**
- `energy_total` - Total Hopfield energy
- `energy_interaction` - Interaction component
- `energy_degradation` - Degradation component
- `energy_bias` - Bias component

**Jacobian Statistics:**
- `jacobian_eig1_real`, `jacobian_eig1_imag` - Leading eigenvalue components
- `jacobian_positive_evals` - Count of positive real eigenvalues
- `jacobian_negative_evals` - Count of negative real eigenvalues
- `jacobian_trace` - Trace of Jacobian
- `jacobian_rotational` - Frobenius norm of antisymmetric part
- `jacobian_df_{gene_i}_dx_{gene_j}` - Specific partial derivatives (if computed)

### `adata.varp` (gene-gene matrices)
- `W_{cluster}` - Interaction matrix for each cluster (n_genes × n_genes, sparse)

### `adata.obsm` (cell embeddings)
- `X_umap` - UMAP coordinates (n_obs × 2)
- `jacobian_eigenvalues` - Jacobian eigenvalues for all cells (n_obs × n_genes, complex)

### `adata.varm` (gene-dimensional arrays)
- `highD_grid` - High-dimensional grid points for energy embedding

### `adata.layers`
- `sigmoid` - Sigmoid-transformed expression (n_obs × n_genes)
- `velocity_reconstructed` - Model-predicted velocity (if computed)
- `velocity_umap` - Velocity projected to UMAP space (if computed)

### `adata.uns['scHopfield']` (metadata & results)

**Configuration:**
- `cluster_key` - Name of cluster key used
- `genes_used` - Indices of genes used in analysis

**Models & Embeddings:**
- `embedding` - UMAP model object
- `models` - Trained optimizer models (dictionary by cluster)

**Energy Grids (per cluster):**
- `grid_X_{cluster}`, `grid_Y_{cluster}` - 2D grid coordinates
- `grid_energy_{cluster}` - Total energy on grid
- `grid_energy_interaction_{cluster}` - Interaction energy on grid
- `grid_energy_degradation_{cluster}` - Degradation energy on grid
- `grid_energy_bias_{cluster}` - Bias energy on grid

**Network Analysis:**
- `network_correlations` - Dictionary of similarity metrics between clusters
  - `'jaccard'`, `'hamming'`, `'euclidean'`, `'pearson'`, `'pearson_bin'`, `'mean_col_corr'`, `'singular'`
- `celltype_correlation` - RV coefficient matrix
- `future_celltype_correlation` - Future state correlation matrix

**Eigenanalysis (per cluster):**
- `eigenanalysis[f'eigenvalues_{cluster}']` - Eigenvalues of W
- `eigenanalysis[f'eigenvectors_{cluster}']` - Eigenvectors of W

## Typical Workflow

A complete analysis typically follows this sequence:

1. **Preprocessing** → Fit sigmoid activation functions to gene expression
2. **Network Inference** → Learn cluster-specific interaction matrices from RNA velocity
3. **Energy Analysis** → Compute energy landscapes and identify genes driving cell states
4. **Network Analysis** → Analyze GRN topology via centrality and eigenanalysis
5. **Stability Analysis** → Compute Jacobians to assess local stability at each cell
6. **Visualization** → Generate publication-ready plots
7. **Dynamics** → Simulate trajectories and test perturbations

Each step builds on the previous, with results stored in the AnnData object for seamless integration.

## Performance Tips

- **GPU Acceleration**: Use `device='cuda'` for `fit_interactions()`, `compute_jacobians()`, etc.
- **Jacobian Storage**: Use HDF5 files (`save_jacobians()`) instead of keeping full matrices in memory
- **Network Centrality**: For large networks (>1000 genes), install `igraph` for 10-100× speedup
- **Energy Embedding**: Reduce `resolution` parameter if computation is slow
- **Batch Processing**: Process clusters separately for large datasets

## Use Cases

scHopfield is designed for:

- **Cell fate analysis**: Identify attractors and transition paths in differentiation
- **Perturbation prediction**: Simulate gene knockouts/overexpression effects
- **Network comparison**: Compare GRN rewiring across cell types or conditions
- **Stability analysis**: Detect unstable/transitional states via Jacobian eigenvalues
- **Driver gene identification**: Find genes with high centrality or energy correlation
- **Trajectory inference**: Predict future cell states from current expression

## Citation

If you use scHopfield in your research, please cite:

```
[Citation information to be added]
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install -e ".[dev]"
```

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/Bernaljp/scHopfield/issues
- **Documentation**: https://schopfield.readthedocs.io (coming soon)

## Acknowledgments

This package was developed for analyzing single-cell RNA-seq data using dynamical systems theory and Hopfield network models. It integrates concepts from:
- Continuous Hopfield networks for modeling gene regulatory dynamics
- RNA velocity for inferring regulatory interactions
- Energy landscape theory for understanding cell state stability
- Network science for analyzing GRN topology
