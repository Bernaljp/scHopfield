# scHopfield

**Single-cell Hopfield Network Analysis**

A Python package for analyzing single-cell RNA-seq data using Hopfield network models to infer gene regulatory networks, compute energy landscapes, and simulate cellular dynamics.

## Overview

scHopfield models gene regulatory networks as continuous Hopfield networks, where:
- Gene expression states evolve according to: `dx/dt = W·σ(x) - γ·x + I`
- `W` is the interaction matrix (gene-gene regulatory interactions)
- `σ(x)` is a sigmoid activation function fitted to expression data
- `γ` represents degradation rates
- `I` is a bias vector representing external inputs

The package provides a **scanpy-style API** that integrates seamlessly with AnnData objects.

## Features

- **Preprocessing**: Sigmoid function fitting to gene expression distributions
- **Network Inference**: Learn interaction matrices from RNA velocity data
- **Energy Landscapes**: Compute and visualize cellular energy landscapes
- **Dynamics Simulation**: Simulate gene expression trajectories and perturbations
- **Stability Analysis**: Jacobian analysis and eigenvalue computation
- **Visualization**: Publication-ready plots for networks, energies, and dynamics

## Installation

### From source

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install -e .
```

### Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- anndata >= 0.8.0
- torch >= 1.9.0
- umap-learn >= 0.5.0
- scikit-learn >= 1.0.0
- tqdm >= 4.62.0
- hoggorm >= 0.13.0
- h5py >= 3.0.0

## Quick Start

```python
import scHopfield as sch
import scanpy as sc

# Load your data
adata = sc.read_h5ad('data.h5ad')

# Ensure you have RNA velocity computed (e.g., using scVelo)
# adata.layers['Ms'] - spliced counts
# adata.layers['velocity_S'] - RNA velocity
# adata.var['gamma'] - degradation rates

# Step 1: Fit sigmoid functions to gene expression
sch.pp.fit_all_sigmoids(adata, genes=highly_variable_genes)
sch.pp.compute_sigmoid(adata)

# Step 2: Infer gene regulatory network
sch.inf.fit_interactions(
    adata,
    cluster_key='celltype',
    n_epochs=1000,
    device='cuda'  # or 'cpu'
)

# Step 3: Compute energy landscapes
sch.tl.compute_energies(adata)
sch.tl.energy_gene_correlation(adata)

# Step 4: Create UMAP embedding and energy landscape
sch.tl.compute_umap(adata)
sch.tl.energy_embedding(adata, resolution=50)

# Step 5: Visualize results
sch.pl.plot_energy_landscape(adata, cluster='HSC')
sch.pl.plot_interaction_matrix(adata, cluster='HSC')

# Step 6: Simulate dynamics
trajectory = sch.dyn.simulate_trajectory(
    adata,
    cluster='HSC',
    cell_idx=0,
    t_span=np.linspace(0, 10, 100)
)
sch.pl.plot_trajectory(trajectory, t_span)
```

## API Reference

### Preprocessing (`sch.pp`)

- `fit_all_sigmoids(adata, genes, ...)` - Fit sigmoid functions to gene expression
- `compute_sigmoid(adata, ...)` - Compute sigmoid-transformed expression

### Inference (`sch.inf`)

- `fit_interactions(adata, cluster_key, ...)` - Infer gene regulatory networks

### Tools (`sch.tl`)

**Energy Analysis:**
- `compute_energies(adata, ...)` - Calculate energy landscapes
- `decompose_degradation_energy(adata, cluster, ...)` - Gene-wise degradation energy
- `decompose_bias_energy(adata, cluster, ...)` - Gene-wise bias energy
- `decompose_interaction_energy(adata, cluster, ...)` - Gene-wise interaction energy

**Correlation Analysis:**
- `energy_gene_correlation(adata, ...)` - Correlate energies with gene expression
- `celltype_correlation(adata, ...)` - Cell type similarity via RV coefficient
- `future_celltype_correlation(adata, ...)` - Future state correlation

**Embedding:**
- `compute_umap(adata, ...)` - Compute UMAP embedding
- `energy_embedding(adata, ...)` - Project energy landscape onto embedding
- `save_embedding(adata, filename)` - Save embedding to file
- `load_embedding(adata, filename)` - Load embedding from file

**Jacobian Analysis:**
- `compute_jacobians(adata, ...)` - Compute Jacobian matrices and eigenvalues
- `save_jacobians(adata, filename)` - Save Jacobians to HDF5
- `load_jacobians(adata, filename)` - Load Jacobians from HDF5

**Network Analysis:**
- `network_correlations(adata, ...)` - Compare networks across cell types

### Plotting (`sch.pl`)

- `plot_energy_landscape(adata, cluster, ...)` - Visualize energy landscape
- `plot_energy_components(adata, cluster, ...)` - Plot all energy components
- `plot_interaction_matrix(adata, cluster, ...)` - Heatmap of interaction matrix
- `plot_sigmoid_fit(adata, gene, ...)` - Show sigmoid fit for a gene
- `plot_trajectory(trajectory, t_span, ...)` - Plot simulated trajectories

### Dynamics (`sch.dyn`)

- `ODESolver` - ODE solver class for Hopfield dynamics
- `create_solver(adata, cluster, ...)` - Create configured solver
- `simulate_trajectory(adata, cluster, cell_idx, ...)` - Simulate from cell state
- `simulate_perturbation(adata, cluster, cell_idx, gene_perturbations, ...)` - Simulate with gene knockouts/overexpression

## Data Storage Conventions

scHopfield follows standard AnnData conventions:

### `adata.var` (gene-level data)
- `sigmoid_threshold`, `sigmoid_exponent`, `sigmoid_offset`, `sigmoid_mse` - Sigmoid parameters
- `scHopfield_used` - Boolean marking genes used in analysis
- `I_{cluster}` - Bias vector for each cluster (one column per cluster)
- `gamma_{cluster}` - Refitted degradation rates (if `refit_gamma=True`)
- `correlation_*_{cluster}` - Gene-energy correlations

### `adata.obs` (cell-level data)
- `energy_total_{cluster}` - Total energy per cell
- `energy_interaction_{cluster}` - Interaction energy
- `energy_degradation_{cluster}` - Degradation energy
- `energy_bias_{cluster}` - Bias energy

### `adata.varp` (gene-gene matrices)
- `W_{cluster}` - Interaction matrix for each cluster

### `adata.obsm` (cell embeddings)
- `X_umap` - UMAP coordinates
- `jacobian_eigenvalues` - Jacobian eigenvalues (n_obs × n_genes, complex)

### `adata.varm` (gene-dimensional matrices)
- `highD_grid` - High-dimensional grid points for energy embedding

### `adata.layers`
- `sigmoid` - Sigmoid-transformed expression
- `velocity_umap` - Velocity in UMAP space (if computed)

### `adata.uns['scHopfield']` (metadata)
- `cluster_key` - Name of cluster key used
- `embedding` - UMAP model object
- `models` - Trained optimizer models (if using scaffold)
- `grid_X_{cluster}`, `grid_Y_{cluster}` - Energy grid coordinates
- `grid_energy_{cluster}` - Energy values on grid
- `network_correlations` - Network similarity metrics
- `celltype_correlation` - RV coefficient matrix

## Citation

If you use scHopfield in your research, please cite:

```
[Citation information to be added]
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- GitHub Issues: https://github.com/Bernaljp/scHopfield/issues
- Documentation: https://schopfield.readthedocs.io (coming soon)

## Acknowledgments

This package was developed for analyzing single-cell RNA-seq data using dynamical systems theory and Hopfield network models.
