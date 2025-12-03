"""
Hematopoiesis scHopfield Analysis
==================================

This script replicates the analysis from the original Jupyter notebook using
the new scHopfield package API.

Analysis workflow:
1. Load and preprocess hematopoiesis data
2. Load CellOracle scaffold for network constraint
3. Fit gene regulatory network with scaffold
4. Compute energy landscapes
5. Correlation and network analyses
6. Jacobian stability analysis
7. Energy embedding and visualization
8. Gene perturbation simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp
from scipy.spatial.distance import squareform
import anndata
import dynamo as dyn
import celloracle as co

# Import the new scHopfield package
import scHopfield as sch

# ============================================================================
# Configuration
# ============================================================================

# Data configuration
DATA_PATH = '/path/to/your/data/'  # Update this path
DATASET_NAME = 'Hematopoiesis'
DATASET_FILE = 'hematopoiesis.h5ad'  # Update filename

# Analysis parameters
CLUSTER_KEY = 'celltype'  # Update to your cluster column name
VELOCITY_KEY = 'velocity_S'
SPLICED_KEY = 'Ms'
DEGRADATION_KEY = 'gamma'
DYNAMIC_GENES_KEY = 'use_for_dynamics'

# Order for plotting (update with your cell types)
CELL_TYPE_ORDER = ['HSC', 'MEP', 'Ery', 'Mega', 'GMP', 'Mono', 'Neu', 'pDC']

# Network inference parameters
N_EPOCHS = 1000
BATCH_SIZE = 128
W_THRESHOLD = 1e-12
SCAFFOLD_REGULARIZATION = 1e-2
DEVICE = 'cuda'  # or 'cpu'

# Visualization parameters
FIGSIZE_LARGE = (15, 10)
FIGSIZE_MEDIUM = (10, 6)

print("=" * 80)
print("scHopfield Analysis: Hematopoiesis Dataset")
print("=" * 80)

# ============================================================================
# 1. Load and Preprocess Data
# ============================================================================

print("\n1. Loading data...")
adata = dyn.read_h5ad(DATA_PATH + DATASET_FILE)
print(f"   Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# Remove genes with NaN velocities (Hematopoiesis-specific)
if DATASET_NAME == 'Hematopoiesis':
    print("   Removing genes with NaN velocities...")
    bad_genes = np.unique(np.where(np.isnan(adata.layers[VELOCITY_KEY].A))[1])
    adata = adata[:, ~np.isin(range(adata.n_vars), bad_genes)]
    print(f"   After filtering: {adata.n_obs} cells × {adata.n_vars} genes")

# Get genes to use for analysis
genes_to_use = adata.var[DYNAMIC_GENES_KEY].values
n_genes = genes_to_use.sum()
print(f"   Using {n_genes} dynamic genes for analysis")

# ============================================================================
# 2. Load Scaffold from CellOracle
# ============================================================================

print("\n2. Loading CellOracle scaffold...")
base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
base_GRN.drop(['peak_id'], axis=1, inplace=True)

# Create scaffold matrix
scaffold = pd.DataFrame(
    0,
    index=adata.var.index[genes_to_use],
    columns=adata.var.index[genes_to_use]
)

# Convert gene names to lowercase for case-insensitive comparison
tfs = list(set(base_GRN.columns.str.lower()) & set(scaffold.index.str.lower()))
target_genes = list(set(base_GRN['gene_short_name'].str.lower().values) & set(scaffold.columns.str.lower()))

# Map original names for assignment
index_map = {gene.lower(): gene for gene in scaffold.index}
col_map = {gene.lower(): gene for gene in scaffold.columns}

# Fill scaffold with 1s where connections exist
for tf in tfs:
    tf_original = index_map[tf]
    tf_base_GRN = [col for col in base_GRN.columns if col.lower() == tf][0]

    for target in base_GRN[base_GRN[tf_base_GRN] == 1]['gene_short_name']:
        if target.lower() in target_genes:
            target_original = col_map[target.lower()]
            scaffold.loc[tf_original, target_original] = 1

print(f"   Scaffold created: {scaffold.sum().sum()} potential connections")
print(f"   TFs: {len(tfs)}, Target genes: {len(target_genes)}")

# ============================================================================
# 3. Fit Gene Regulatory Network
# ============================================================================

print("\n3. Fitting gene regulatory network with scHopfield...")

# Step 3a: Fit sigmoid functions
print("   3a. Fitting sigmoid functions...")
sch.pp.fit_all_sigmoids(
    adata,
    genes=genes_to_use,
    spliced_key=SPLICED_KEY,
    min_th=0.05
)
sch.pp.compute_sigmoid(adata, spliced_key=SPLICED_KEY)
print("   ✓ Sigmoid fitting complete")

# Step 3b: Infer interaction matrices
print("   3b. Inferring interaction matrices...")
sch.inf.fit_interactions(
    adata,
    cluster_key=CLUSTER_KEY,
    spliced_key=SPLICED_KEY,
    velocity_key=VELOCITY_KEY,
    degradation_key=DEGRADATION_KEY,
    w_threshold=W_THRESHOLD,
    w_scaffold=scaffold.values,
    scaffold_regularization=SCAFFOLD_REGULARIZATION,
    only_TFs=True,
    infer_I=True,
    refit_gamma=False,
    pre_initialize_W=False,
    n_epochs=N_EPOCHS,
    criterion='MSE',
    batch_size=BATCH_SIZE,
    device=DEVICE,
    skip_all=True,
    get_plots=False
)
print("   ✓ Network inference complete")

# ============================================================================
# 4. Compute Energy Landscapes
# ============================================================================

print("\n4. Computing energy landscapes...")
sch.tl.compute_energies(
    adata,
    spliced_key=SPLICED_KEY,
    degradation_key=DEGRADATION_KEY
)
print("   ✓ Energy computation complete")

# Compute energy summary statistics
print("\n   Energy Summary Statistics:")
energy_cols = [CLUSTER_KEY, 'energy_total_HSC', 'energy_interaction_HSC',
               'energy_degradation_HSC', 'energy_bias_HSC']
available_cols = [col for col in energy_cols if col in adata.obs.columns]

if len(available_cols) > 1:
    summary_stats = adata.obs[available_cols].groupby(CLUSTER_KEY).describe()
    for energy in summary_stats.columns.levels[0]:
        if energy != CLUSTER_KEY:
            summary_stats[(energy, 'cv')] = summary_stats[(energy, 'std')] / summary_stats[(energy, 'mean')]
    print(summary_stats.head())

# ============================================================================
# 5. Correlation Analyses
# ============================================================================

print("\n5. Running correlation analyses...")

# Energy-gene correlations
sch.tl.energy_gene_correlation(adata, spliced_key=SPLICED_KEY)
print("   ✓ Energy-gene correlations computed")

# Cell type correlations
sch.tl.celltype_correlation(adata, spliced_key=SPLICED_KEY, modified=True)
print("   ✓ Cell type correlations computed")

# Future cell type correlations
sch.tl.future_celltype_correlation(adata, spliced_key=SPLICED_KEY, modified=True)
print("   ✓ Future cell type correlations computed")

# Network correlations
sch.tl.network_correlations(adata)
print("   ✓ Network correlations computed")

# ============================================================================
# 6. UMAP and Energy Embedding
# ============================================================================

print("\n6. Computing UMAP and energy embedding...")

# Compute UMAP if not present
if 'X_umap' not in adata.obsm:
    sch.tl.compute_umap(adata, spliced_key=SPLICED_KEY, n_neighbors=30, min_dist=0.1)
    print("   ✓ UMAP computed")
else:
    print("   Using existing UMAP coordinates")

# Compute energy embedding
sch.tl.energy_embedding(
    adata,
    basis='umap',
    resolution=50,
    spliced_key=SPLICED_KEY,
    degradation_key=DEGRADATION_KEY
)
print("   ✓ Energy embedding computed")

# ============================================================================
# 7. Jacobian Analysis (Optional - can be slow)
# ============================================================================

COMPUTE_JACOBIANS = False  # Set to True to run Jacobian analysis

if COMPUTE_JACOBIANS:
    print("\n7. Computing Jacobian matrices and eigenvalues...")
    sch.tl.compute_jacobians(
        adata,
        spliced_key=SPLICED_KEY,
        degradation_key=DEGRADATION_KEY,
        compute_eigenvectors=False,  # Set True if you need eigenvectors
        device=DEVICE
    )
    print("   ✓ Jacobian analysis complete")

    # Save Jacobians to file
    sch.tl.save_jacobians(adata, 'jacobians_hematopoiesis.h5')
    print("   ✓ Jacobians saved to file")
else:
    print("\n7. Skipping Jacobian analysis (set COMPUTE_JACOBIANS=True to run)")

# ============================================================================
# 8. Visualization
# ============================================================================

print("\n8. Creating visualizations...")

# Setup colors for cell types
colors = plt.cm.tab10(np.linspace(0, 1, len(CELL_TYPE_ORDER)))
color_dict = dict(zip(CELL_TYPE_ORDER, colors))

# 8a. UMAP with cell types
fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
for i, ct in enumerate(CELL_TYPE_ORDER):
    mask = adata.obs[CLUSTER_KEY] == ct
    ax.scatter(
        adata.obsm['X_umap'][mask, 0],
        adata.obsm['X_umap'][mask, 1],
        c=[color_dict[ct]],
        label=ct,
        s=10,
        alpha=0.5
    )
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('Hematopoiesis - Cell Types')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('umap_celltypes.pdf', dpi=300, bbox_inches='tight')
print("   ✓ UMAP plot saved")

# 8b. Energy landscapes for each cell type
for cluster in CELL_TYPE_ORDER[:4]:  # Plot first 4 as examples
    if f'grid_energy_{cluster}' in adata.uns['scHopfield']:
        fig, ax = plt.subplots(figsize=(8, 6))
        sch.pl.plot_energy_landscape(adata, cluster=cluster, basis='umap', ax=ax)
        plt.savefig(f'energy_landscape_{cluster}.pdf', dpi=300, bbox_inches='tight')
        print(f"   ✓ Energy landscape for {cluster} saved")

# 8c. Interaction matrices
for cluster in CELL_TYPE_ORDER[:2]:  # Plot first 2 as examples
    fig, ax = plt.subplots(figsize=(10, 10))
    sch.pl.plot_interaction_matrix(adata, cluster=cluster, ax=ax, vmin=-1, vmax=1)
    plt.savefig(f'interaction_matrix_{cluster}.pdf', dpi=300, bbox_inches='tight')
    print(f"   ✓ Interaction matrix for {cluster} saved")

# 8d. Cell type correlation dendrogram
if 'celltype_correlation' in adata.uns['scHopfield']:
    rv_matrix = adata.uns['scHopfield']['celltype_correlation']

    fig, ax = plt.subplots(figsize=(10, 4))
    Z = scp.cluster.hierarchy.linkage(squareform(1 - rv_matrix), 'complete')
    scp.cluster.hierarchy.dendrogram(Z, labels=rv_matrix.index, ax=ax)
    ax.set_title('Cell Type Correlation (RV Coefficient)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('celltype_correlation_dendrogram.pdf', dpi=300, bbox_inches='tight')
    print("   ✓ Cell type correlation dendrogram saved")

# 8e. Network similarity dendrogram
if 'network_correlations' in adata.uns['scHopfield']:
    pearson = adata.uns['scHopfield']['network_correlations']['pearson']

    fig, ax = plt.subplots(figsize=(10, 4))
    Z = scp.cluster.hierarchy.linkage(squareform(1 - pearson), 'complete')
    scp.cluster.hierarchy.dendrogram(Z, labels=pearson.index, ax=ax)
    ax.set_title('Network Similarity (Pearson Correlation)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('network_similarity_dendrogram.pdf', dpi=300, bbox_inches='tight')
    print("   ✓ Network similarity dendrogram saved")

# ============================================================================
# 9. Gene Perturbation Simulations (Optional)
# ============================================================================

RUN_SIMULATIONS = False  # Set to True to run simulations

if RUN_SIMULATIONS:
    print("\n9. Running gene perturbation simulations...")

    # Example: Simulate GATA1 knockout in HSC
    cluster = 'HSC'
    cell_idx = np.where(adata.obs[CLUSTER_KEY] == cluster)[0][0]
    t_span = np.linspace(0, 10, 100)

    # Baseline trajectory
    trajectory_baseline = sch.dyn.simulate_trajectory(
        adata,
        cluster=cluster,
        cell_idx=cell_idx,
        t_span=t_span,
        spliced_key=SPLICED_KEY,
        degradation_key=DEGRADATION_KEY
    )

    # GATA1 knockout (0.1x expression)
    trajectory_knockout = sch.dyn.simulate_perturbation(
        adata,
        cluster=cluster,
        cell_idx=cell_idx,
        gene_perturbations={'GATA1': 0.1},  # 90% knockdown
        t_span=t_span,
        spliced_key=SPLICED_KEY,
        degradation_key=DEGRADATION_KEY
    )

    # Plot trajectories for key genes
    key_genes = ['GATA1', 'GATA2', 'FLI1', 'KLF1']
    genes_used = adata.var.index[adata.var['scHopfield_used'].values]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (gene, ax) in enumerate(zip(key_genes, axes.flat)):
        if gene in genes_used:
            gene_idx = np.where(genes_used == gene)[0][0]
            ax.plot(t_span, trajectory_baseline[:, gene_idx], label='Baseline', lw=2)
            ax.plot(t_span, trajectory_knockout[:, gene_idx], label='GATA1 KO', lw=2, ls='--')
            ax.set_xlabel('Time')
            ax.set_ylabel('Expression')
            ax.set_title(f'{gene} Expression')
            ax.legend()

    plt.tight_layout()
    plt.savefig('gata1_knockout_simulation.pdf', dpi=300, bbox_inches='tight')
    print("   ✓ GATA1 knockout simulation complete")
else:
    print("\n9. Skipping simulations (set RUN_SIMULATIONS=True to run)")

# ============================================================================
# 10. Save Results
# ============================================================================

print("\n10. Saving results...")

# Save the annotated AnnData object
adata.write_h5ad('hematopoiesis_schopfield_analysis.h5ad')
print("   ✓ AnnData object saved")

# Save energy embedding
sch.tl.save_embedding(adata, 'energy_embedding.pkl')
print("   ✓ Energy embedding saved")

# Export key results to CSV
if 'correlation_total_HSC' in adata.var.columns:
    # Top genes correlated with total energy
    corr_cols = [col for col in adata.var.columns if col.startswith('correlation_total_')]
    if corr_cols:
        correlations = adata.var[corr_cols].copy()
        correlations.to_csv('energy_gene_correlations.csv')
        print("   ✓ Energy-gene correlations exported")

# Export network statistics
if 'network_correlations' in adata.uns['scHopfield']:
    network_stats = pd.DataFrame({
        'Pearson': adata.uns['scHopfield']['network_correlations']['pearson'].values.diagonal(),
        'Jaccard': adata.uns['scHopfield']['network_correlations']['jaccard'].values.diagonal(),
        'Hamming': adata.uns['scHopfield']['network_correlations']['hamming'].values.diagonal(),
    }, index=adata.uns['scHopfield']['network_correlations']['pearson'].index)
    network_stats.to_csv('network_statistics.csv')
    print("   ✓ Network statistics exported")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print("\nGenerated files:")
print("  - hematopoiesis_schopfield_analysis.h5ad")
print("  - energy_embedding.pkl")
print("  - energy_gene_correlations.csv")
print("  - network_statistics.csv")
print("  - umap_celltypes.pdf")
print("  - energy_landscape_*.pdf")
print("  - interaction_matrix_*.pdf")
print("  - celltype_correlation_dendrogram.pdf")
print("  - network_similarity_dendrogram.pdf")
if RUN_SIMULATIONS:
    print("  - gata1_knockout_simulation.pdf")
if COMPUTE_JACOBIANS:
    print("  - jacobians_hematopoiesis.h5")

print("\nNext steps:")
print("  1. Examine the energy landscapes and interaction matrices")
print("  2. Investigate genes with high energy correlations")
print("  3. Compare network architectures across cell types")
print("  4. Run perturbation simulations for genes of interest")
print("\n" + "=" * 80)
