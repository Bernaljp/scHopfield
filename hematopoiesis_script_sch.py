# %%
import anndata
import celloracle as co
import dynamo as dyn
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy as scp
from scipy import sparse
from scipy.integrate import solve_ivp
import scipy.interpolate as interp
from scipy.signal import convolve2d
from scipy.spatial.distance import squareform
import scHopfield as sch
import seaborn as sns
import sys
from tqdm import tqdm

# %%
%matplotlib inline

# %%
config_path = '/home/bernaljp/KAUST'
sys.path.append(config_path)
import config

# %%
name = 'Hematopoiesis'

# %%
dataset = config.datasets[name]
cluster_key = config.cluster_keys[name]
velocity_key = config.velocity_keys[name]
spliced_key = config.spliced_keys[name]
title = config.titles[name]
order = config.orders[name]
dynamic_genes_key = config.dynamic_genes_keys[name]
degradation_key = config.degradation_keys[name]

adata = dyn.read_h5ad(config.data_path+dataset) if dataset.split('.')[1]=='h5ad' else dyn.read_loom(config.data_path+dataset)

# %%
if name=='Hematopoiesis':
    bad_genes = np.unique(np.where(np.isnan(adata.layers[velocity_key].A))[1])
    adata = adata[:,~np.isin(range(adata.n_vars),bad_genes)]

adata

# %%
dyn.pl.scatters(adata, color=cluster_key, basis="umap", show_legend="on data", figsize=(15,10), save_show_or_return='return', pointsize=2, alpha=0.35)
plt.show()

# %%
def change_spines(ax):
    for ch in ax.get_children():
        try:
            text = ch.get_text().split('_')
            if text[0]=='umap':
                ch.set_text(r'UMAP$_{'+text[1]+'}$')
        except:
            pass

# %%
ax = dyn.pl.streamline_plot(adata, color=cluster_key, basis="umap", show_legend="on data", show_arrowed_spines=True, size=(15,10), save_show_or_return='return')
ax[0].set_title(title)
change_spines(ax[0])
plt.show()

# %%
colors = {k:ax[0].get_children()[0]._facecolors[np.where(adata.obs[cluster_key]==k)[0][0]] for k in adata.obs[cluster_key].unique()}
for k in colors:
    colors[k][3] = 1

# %% [markdown]
# ## Fit sigmoid and infer interactions using scHopfield

# %%
# Loading Scaffold
base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
base_GRN.drop(['peak_id'], axis=1, inplace=True)

# Ensure case-insensitive handling of gene names
genes_to_use = list(adata.var['use_for_dynamics'].values)
scaffold = pd.DataFrame(0, index=adata.var.index[adata.var['use_for_dynamics']], columns=adata.var.index[adata.var['use_for_dynamics']])

# Convert gene names to lowercase for case-insensitive comparison
tfs = list(set(base_GRN.columns.str.lower()) & set(scaffold.index.str.lower()))
target_genes = list(set(base_GRN['gene_short_name'].str.lower().values) & set(scaffold.columns.str.lower()))

# Map original names for assignment
index_map = {gene.lower(): gene for gene in scaffold.index}
column_map = {gene.lower(): gene for gene in scaffold.columns}

for gene in tfs:
    for target in target_genes:
        if gene in index_map and target in column_map:
            scaffold.loc[index_map[target], column_map[gene]] = 1

# %%
# Fit sigmoids (preprocessing)
sch.pp.fit_all_sigmoids(adata,
                         spliced_key=spliced_key,
                         genes=adata.var['use_for_dynamics'].values)

# %%
# Fit interactions using scHopfield
sch.inf.fit_interactions(adata,
                         cluster_key=cluster_key,
                         spliced_key=spliced_key,
                         velocity_key=velocity_key,
                         degradation_key=degradation_key,
                         w_threshold=1e-12,
                         w_scaffold=scaffold.values,
                         scaffold_regularization=1e-2,
                         only_TFs=True,
                         infer_I=True,
                         refit_gamma=False,
                         pre_initialize_W=False,
                         n_epochs=1000,
                         criterion='MSE',
                         batch_size=128,
                         skip_all=True,
                         use_scheduler=True,
                         get_plots=False,
                         device='cuda')

# %% [markdown]
# # Energies

# %%
# Compute energies using scHopfield
sch.tl.compute_energies(adata, cluster_key=cluster_key)

# %%
summary_stats = adata.obs[[cluster_key,'Total_energy','Interaction_energy','Degradation_energy','Bias_energy']].groupby(cluster_key).describe()
for energy in summary_stats.columns.levels[0]:
    summary_stats[(energy,'cv')] = summary_stats[(energy,'std')]/summary_stats[(energy,'mean')]
summary_stats['Total_energy']

# %%
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[colors[i] for i in order])
sch.pl.plot_energy_components(adata, cluster_key=cluster_key, order=order, colors=colors, figsize=(22,11))
plt.show()

sch.pl.plot_energy_landscape(adata, cluster_key=cluster_key, order=order, figsize=(15,15))
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.2))
plt.show()

# %% [markdown]
# # Dendrograms

# %% [markdown]
# ## Cell type dendrogram

# %%
# Compute celltype correlation using scHopfield
sch.tl.celltype_correlation(adata, cluster_key=cluster_key)

# %%
cells_correlation = adata.uns['scHopfield']['celltype_correlation']
plt.figure(figsize=(9, 3))
Z = scp.cluster.hierarchy.linkage(squareform(1-cells_correlation), 'complete')
fig,axs = plt.subplots(1,1,figsize=(10, 4), tight_layout=True)
scp.cluster.hierarchy.dendrogram(Z, labels = cells_correlation.index, ax=axs)
axs.get_yaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.set_title('Celltype RV score')
plt.show()

# %% [markdown]
# ## Network dendrogram

# %%
# Compute network correlations using scHopfield
sch.tl.network_correlations(adata, cluster_key=cluster_key)

# %%
pearson = adata.uns['scHopfield']['network_correlations']['pearson']
hamming = adata.uns['scHopfield']['network_correlations']['hamming']
pearson_bin = adata.uns['scHopfield']['network_correlations']['pearson_bin']

# %%
fig,axs = plt.subplots(1,1,figsize=(9, 3), tight_layout=True)
axs.get_yaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)

Z = scp.cluster.hierarchy.linkage(squareform(1-pearson), 'complete')
scp.cluster.hierarchy.dendrogram(Z, labels = pearson.index)
plt.show()

# %%
fig,axs = plt.subplots(1,1,figsize=(10, 4), tight_layout=True)
axs.get_yaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.set_title('Network Hamming Distance Dendrogram')

Z = scp.cluster.hierarchy.linkage(squareform(hamming), 'complete')
scp.cluster.hierarchy.dendrogram(Z, labels = hamming.index)
plt.show()

# %%
fig,axs = plt.subplots(1,1,figsize=(10, 3), tight_layout=True)
axs.get_yaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)

Z = scp.cluster.hierarchy.linkage(squareform(1-pearson_bin), 'complete')
scp.cluster.hierarchy.dendrogram(Z, labels = pearson_bin.index)
plt.show()

# %% [markdown]
# # Symmetricity

# %%
def symmetricity(A, norm=2):
    S = np.linalg.norm((A+A.T)/2, ord=norm)
    As = np.linalg.norm((A-A.T)/2, ord=norm)
    return (S-As)/(S+As)

# Get interaction matrices from scHopfield storage
W = {}
genes_used = adata.var['use_for_dynamics'].values
gene_names = adata.var_names[genes_used]
for cluster in order:
    W[cluster] = adata.varp[f'W_{cluster}'][genes_used][:,genes_used]

syms = np.array([symmetricity(W[k], norm=2) for k in order])
idxs = np.argsort(syms)
plt.figure(figsize=(5,4), tight_layout=True)
plt.scatter(range(len(W)), syms, s=200, marker='*', c=[colors[i] for i in order])
plt.xticks(range(len(W)), np.array(order))
plt.ylabel('Symmetricity')
plt.xticks(rotation=-30)
plt.title('Distribution of Symmetricity across Weights')
plt.show()

# %% [markdown]
# # Model Analysis

# %%
# Extract gamma and I from scHopfield storage
gamma = {}
I = {}
for cluster in order:
    gamma[cluster] = adata.var[f'gamma_{cluster}'].values[genes_used] if f'gamma_{cluster}' in adata.var.columns else adata.var[degradation_key].values[genes_used]
    I[cluster] = adata.var[f'I_{cluster}'].values[genes_used] if f'I_{cluster}' in adata.var.columns else np.zeros(genes_used.sum())

# %%
fig,axs = plt.subplots(2,4,figsize=(20,10))
for cl,ax in zip(order,axs.flatten()):
    ax.scatter(gamma[cl], adata.var[degradation_key][genes_used], color=colors[cl], s=2)
    ax.set_title(cl)
    max_gamma = max(np.concatenate([gamma[cl], adata.var[degradation_key][genes_used]]))
    ax.set_xlabel('Refitted gamma')
    ax.set_ylabel('Original gamma')
    ax.plot([0, max_gamma], [0, max_gamma], color='k', ls='--', lw=1)
plt.show()

# %%
fig,axs = plt.subplots(2,4,figsize=(20,10))
for cl,ax in zip(order,axs.flatten()):
   sns.histplot(I[cl].flatten(), bins=100,ax=ax, color=colors[cl])
plt.show()

# %%
# Compute reconstructed velocity
def hopfield_model(adata, cluster, cluster_key, spliced_key, genes_used):
    """Compute reconstructed velocity for a cluster."""
    cluster_cells = adata.obs[cluster_key] == cluster
    W = adata.varp[f'W_{cluster}'][genes_used][:,genes_used]
    I_vec = adata.var[f'I_{cluster}'].values[genes_used] if f'I_{cluster}' in adata.var.columns else np.zeros(genes_used.sum())
    gamma_vec = adata.var[f'gamma_{cluster}'].values[genes_used] if f'gamma_{cluster}' in adata.var.columns else adata.var[degradation_key].values[genes_used]

    # Get expression data
    X = adata.layers[spliced_key][cluster_cells][:,genes_used].A

    # Compute sigmoid
    sigmoid_vals = adata.layers['sigmoid'][cluster_cells][:,genes_used].A if 'sigmoid' in adata.layers else X

    # Compute velocity: W * sigmoid(X) - gamma * X + I
    reconstructed_v = (W @ sigmoid_vals.T).T - gamma_vec * X + I_vec

    return reconstructed_v

fig,axs = plt.subplots(2,4,figsize=(20,10))
for cl,ax in zip(order,axs.flatten()):
    reconstructed_v = hopfield_model(adata, cl, cluster_key, spliced_key, genes_used)
    original_v = adata.layers[velocity_key][adata.obs[cluster_key]==cl][:,genes_used]
    mse = np.mean((reconstructed_v-original_v.A)**2)
    ax.scatter(reconstructed_v.flatten(), original_v.A.flatten(), color=colors[cl], s=2)
    ax.set_title(f'{cl} - MSE: {mse:.2f}')
    ax.set_xlabel('Reconstructed velocity')
    ax.set_ylabel('Original velocity')
    min_v = min(np.concatenate([reconstructed_v.flatten(), original_v.A.flatten()]))
    max_v = max(np.concatenate([reconstructed_v.flatten(), original_v.A.flatten()]))
    ax.plot([min_v, max_v], [min_v, max_v], c='k', ls='--', lw=1)
plt.show()

# %% [markdown]
# # Correlations

# %%
# Compute energy-gene correlations using scHopfield
sch.tl.energy_gene_correlation(adata, cluster_key=cluster_key)

# %%
def get_correlation_table(adata, n_top_genes=20, which_correlation='total'):
    """Get correlation table from scHopfield results."""
    corr_key = f'energy_gene_correlation_{which_correlation}' if which_correlation.lower()!='total' else 'energy_gene_correlation'
    assert corr_key in adata.uns['scHopfield'], f'No {corr_key} found in adata.uns["scHopfield"]'

    corrs_dict = adata.uns['scHopfield'][corr_key]
    df = pd.DataFrame(index=range(n_top_genes), columns=pd.MultiIndex.from_product([order, ['Gene', 'Correlation']]))

    for k in order:
        corrs = corrs_dict[k]
        indices = np.argsort(corrs)[::-1][:n_top_genes]
        genes = gene_names[indices]
        corrs_sorted = corrs[indices]
        df[(k, 'Gene')] = genes
        df[(k, 'Correlation')] = corrs_sorted
    return df

df_correlations = get_correlation_table(adata, n_top_genes=100, which_correlation='total')
df_correlations

# %%
# Plot correlation grids using custom plotting
correlation = adata.uns['scHopfield']['energy_gene_correlation']

corner_genes = np.array([])
clus1_low = -0.4
clus1_high = 0.4
clus2_low = -0.4
clus2_high = 0.4
nn = 5

for corr1,corr2 in itertools.combinations(order, 2):
    corr1_vals = correlation[corr1]
    corr2_vals = correlation[corr2]
    positions_corners = np.logical_or(np.logical_and(corr1_vals >= clus1_high, corr2_vals <= clus2_low),
                                      np.logical_and(corr1_vals <= clus1_low, corr2_vals >= clus2_high))

    corr_corners = np.where(positions_corners)[0]
    corr_indices = np.argsort((corr1_vals[corr_corners])**2 + (corr2_vals[corr_corners])**2)[-nn:]
    corr_corners = corr_corners[corr_indices]
    corner_genes = np.concatenate((corner_genes, gene_names[corr_corners]))

corner_genes = np.unique(corner_genes)
df_corr_corners = pd.DataFrame.from_dict(correlation)
df_corr_corners.drop('all', axis=1, inplace=True, errors='ignore')
df_corr_corners.index = gene_names
df_corr_corners = df_corr_corners.loc[corner_genes]
df_corr_corners.to_csv(f'/home/bernaljp/KAUST/corner_genes_{name}.csv', index=True)

# %% [markdown]
# # Network scores

# %%
def get_links_dict(adata, cluster_key, gene_names, genes_used):
    """Create links dictionary from scHopfield results."""
    links = {}
    clusters = adata.obs[cluster_key].unique()

    for k in clusters:
        if k == 'all':
            continue
        W = adata.varp[f'W_{k}'][genes_used][:,genes_used]
        links[k] = pd.DataFrame(W.T, index=gene_names, columns=gene_names).reset_index()
        links[k] = links[k].melt(id_vars='index', value_vars=links[k].columns, var_name='target', value_name='coef_mean')
        links[k] = links[k][links[k]['coef_mean'] != 0]
        links[k].rename(columns={'index': 'source'}, inplace=True)
        links[k]['coef_abs'] = np.abs(links[k]['coef_mean'])
        links[k]['p'] = 0
        links[k]['-logp'] = np.nan
    return links

def plot_scores_as_rank(links, clusters=None, axs=None, n_gene=50, values=None, colors=None, skip_first_n=0, return_table=False):
    """
    Pick up top n-th genes with high-network scores and make plots.
    """
    values = links.merged_score.columns.drop('cluster') if values is None else values
    n_cols = len(values)
    clusters = links.cluster if clusters is None else np.array([clusters]).ravel().tolist()
    n_rows = len(clusters)
    size_per_gene = 0.2

    _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*n_gene*size_per_gene), tight_layout=True) if axs is None else (None,axs)

    df_table = pd.DataFrame(index=range(n_gene), columns=pd.MultiIndex.from_product([clusters, values, ['Gene','Value']], names=['cluster', 'score', 'values']))

    for i, (axs_row, cluster) in enumerate(zip(axs, clusters)):
        color = colors[cluster] if colors is not None else 'tab:blue'
        for j, (ax, value) in enumerate(zip(axs_row, values)):
            res = links.merged_score[links.merged_score.cluster == cluster]
            res = res[value].sort_values(ascending=False)
            res = res[skip_first_n:n_gene+skip_first_n]

            ax.scatter(res.values, range(len(res)), color=color)
            ax.set_yticks(range(len(res)), res.index.values)
            if i == 0:
                if skip_first_n == 0:
                    ax.set_title(f" {value.replace('_',' ').capitalize()} \n top {n_gene}")
                else:
                    ax.set_title(f" {value.replace('_',' ').capitalize()} \n top {n_gene} (skip {skip_first_n})")
            if j == 0:
                ax.set_ylabel(cluster)
            ax.invert_yaxis()

            if return_table:
                df_table[(cluster, value, 'Gene')] = res.index.values
                df_table[(cluster, value, 'Value')] = res.values
    if return_table:
        return df_table

# %%
def _plot_goi(x, y, goi, args_annot, scatter=False, x_shift=0.1, y_shift=0.1, ax=None):
    """Plot annotation to highlight one point in scatter plot."""
    default = {"size": 10}
    default.update(args_annot)
    args_annot = default.copy()

    arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "black"}
    plotter = plt if ax is None else ax
    if scatter:
        plotter.scatter(x, y, c="none", edgecolor="black")
    plotter.annotate(f"{goi}", xy=(x, y), xytext=(x+x_shift, y+y_shift),
                 color="black", arrowprops=arrow_dict, **args_annot)

def plot_score_comparison_2D(links, value, cluster1, cluster2, percentile=99, annot_shifts=None, save=None, fillna_with_zero=True, ignore_genes=[], plt_show=True, ax=None):
    """Make scatter plot showing relationship of network score in two groups."""
    res = links.merged_score[links.merged_score.cluster.isin([cluster1, cluster2])][[value, "cluster"]]
    res = res.reset_index(drop=False)
    piv = pd.pivot_table(res, values=value, columns="cluster", index="index")
    piv.drop(ignore_genes, inplace=True, errors='ignore')
    if fillna_with_zero:
        piv = piv.fillna(0)
    else:
        piv = piv.fillna(piv.mean(axis=0))

    goi1 = piv[piv[cluster1] > np.percentile(piv[cluster1].values, percentile)].index
    goi2 = piv[piv[cluster2] > np.percentile(piv[cluster2].values, percentile)].index

    gois = np.union1d(goi1, goi2)
    not_gois = np.setdiff1d(piv.index, gois)
    piv_gois = piv.loc[gois]
    piv_not_gois = piv.loc[not_gois]

    x, y = piv_not_gois[cluster1], piv_not_gois[cluster2]
    plotter = plt if ax is None else ax
    plotter.scatter(x, y, c='lightgray', s=2)
    x, y = piv_gois[cluster1], piv_gois[cluster2]
    plotter.scatter(x, y, c="none", edgecolors='b')

    if annot_shifts is None:
        x_shift, y_shift = (x.max() - x.min())*0.03, (y.max() - y.min())*0.03
    else:
        x_shift, y_shift = annot_shifts
    for goi in gois:
        x, y = piv.loc[goi, cluster1], piv.loc[goi, cluster2]
        _plot_goi(x, y, goi, {}, scatter=False, x_shift=x_shift, y_shift=y_shift, ax=ax)

    if ax is None:
        plt.xlabel(cluster1)
        plt.ylabel(cluster2)
    else:
        ax.set_xlabel(cluster1)
        ax.set_ylabel(cluster2)

    if plt_show:
        plt.show()

def plot_score_comparison_grid(links, score="eigenvector_centrality", colors=None, order=None, ignore_genes=[], annotate_percentile=99, **kwargs):
    """Plots matrix where diagonal shows cell types and off-diagonal shows gene correlation scatter plots."""
    cell_types = links.cluster if order is None else order
    n = len(cell_types)
    figsize = kwargs.pop('figsize', (15, 15))
    fontsize = kwargs.pop('fontsize', 30)
    tight_layout = kwargs.get('tight_layout', True)
    colors = colors if colors is not None else links.palette.to_dict()['palette']
    fig, axs = plt.subplots(n, n, figsize=figsize, tight_layout=tight_layout)
    fig.suptitle(score.replace('_', ' ').capitalize(), fontsize=fontsize)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                text = cell_types[i]
                for spine in axs[i, j].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_color(colors[text])
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].set_facecolor(colors[text])
                text = text.replace(' ', '\n', 1)
                text = text.replace('-', '-\n')
                axs[i, j].text(0.5, 0.5, text, ha='center', va='center', fontsize=18, fontweight='bold', fontname='serif', transform=axs[i, j].transAxes)
                continue

            axs[i, j].axis('off')
            plot_score_comparison_2D(links, value=score,
                               cluster1=cell_types[i], cluster2=cell_types[j],
                               percentile=annotate_percentile, ax=axs[j, i], plt_show=False, ignore_genes=ignore_genes, **kwargs)
            axs[j, i].set_xlabel('')
            axs[j, i].set_ylabel('')
            if i != 0:
                axs[j, i].set_yticks([])
            if j != n - 1:
                axs[j, i].set_xticks([])

# %%
links_dict = get_links_dict(adata, cluster_key, gene_names, genes_used)

links = co.network_analysis.links_object.Links(name=cluster_key, links_dict=links_dict)

co.trajectory.oracle_utility._check_color_information_and_create_if_not_found(adata, cluster_key, 'umap')

try:
    links.palette = pd.DataFrame.from_dict(adata.uns[f'{cluster_key}_colors'], orient='index', columns=['palette'])
except:
    links.palette = pd.DataFrame(adata.uns[f'{cluster_key}_colors'], index=adata.obs[cluster_key].unique(), columns=['palette'])

links.filter_links(p=0.001, weight="coef_abs", threshold_number=40000)
links.get_network_score()

# %%
values = [
          'degree_centrality_out',
          'betweenness_centrality',
          'eigenvector_centrality',
          ]
df_scores = plot_scores_as_rank(links, clusters=order, values=values, colors=colors, n_gene=20, skip_first_n=0, return_table=True)
plt.show()

# %%
plot_score_comparison_grid(links, score='eigenvector_centrality', colors=colors, order=order, ignore_genes=[], annotate_percentile=99)
plt.show()

# %% [markdown]
# # Eigenanalysis

# %%
def linspace_iterator(start, stop, num):
    """Generate linearly spaced values."""
    step = (stop - start) / (num - 1)
    for i in range(num):
        yield start + i * step

def annotate_points(ax, x_data, y_data, labels, offset_x_fraction=0.1, offset_y_fraction=0.1):
    """Annotate points on a plot."""
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset_x = x_range * offset_x_fraction
    offset_y = y_range * offset_y_fraction

    for x, y, label in zip(x_data, y_data, labels):
        ax.annotate(label, (x, y), xytext=(x+offset_x, y+offset_y), fontsize=8,
                   arrowprops=dict(arrowstyle='->', lw=0.5))

# %%
# Eigenvalue analysis
exclude_genes = []
part = 'real'
n_genes = 10
n_genes_table = 20

e_vals = {}
e_vecs = {}
v_svd = {}
s2_svd = {}
ut_svd = {}

for cell_type in order:
    e_vals[cell_type], e_vecs[cell_type] = np.linalg.eig(W[cell_type])
    v_svd[cell_type], s2_svd[cell_type], ut_svd[cell_type] = np.linalg.svd(W[cell_type])

# %%
genes_in = np.where(~np.isin(gene_names, exclude_genes))[0]

df_rankings = pd.DataFrame(index=range(len(gene_names)))

fig, axs = plt.subplots(len(order), 3, figsize=(15, 3 * len(order)))
df_eigenvalues = pd.DataFrame(index=range(1,n_genes_table+1),columns=pd.MultiIndex.from_product([order, ['EV gene', 'EV value', 'Score gene', 'Score value']]))
top_evalues = {}

for i, cell_type in tqdm(enumerate(order)):
    evals, evecs = e_vals[cell_type], e_vecs[cell_type]

    # Plot eigenvalues (real vs imaginary)
    ax = axs[i, 0]
    ax.scatter(evals.real, evals.imag, label=cell_type, color=colors[cell_type])
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.legend()
    if i == 0:
        ax.set_title(f'Eigenvalues')

    # Plot sorted eigenvector components
    eigenvector = evecs[:, np.argmax(evals.real)]
    top_evalues[cell_type] = evals[np.argmax(evals.real)]
    sorted_indices_abs = np.argsort(np.abs(eigenvector))[::-1]
    sorted_indices = np.argsort(eigenvector)
    indices_for_table = sorted_indices_abs[:n_genes_table]
    indices_to_plot = sorted_indices_abs[:n_genes]

    x_data = [np.where(sorted_indices==idx)[0][0] for idx in indices_to_plot]
    y_data = eigenvector[indices_to_plot]
    names = gene_names[indices_to_plot]

    df_eigenvalues[cell_type, 'EV gene'] = gene_names[indices_for_table]
    df_eigenvalues[cell_type, 'EV value'] = list(map(lambda x: f'{np.real(x):.3f}', eigenvector[indices_for_table]))

    ax = axs[i, 1]
    ax.plot(eigenvector[sorted_indices], '.', color=colors[cell_type])
    ax.set_ylabel('Component value')
    annotate_points(ax, x_data, y_data, names, offset_x_fraction=0.2, offset_y_fraction=0.1)
    ax.set_xticks([])
    if i == 0:
        ax.set_title(f'First Eigenvector components')

    # Plot real eigenvector score sorted
    evecs_part = evecs.real if part=='real' else evecs.imag
    evals_part = evals.real if part=='real' else evals.imag

    e_score = evecs_part @ evals_part

    sorted_indices = np.argsort(e_score)
    sorted_indices_absolute = np.argsort(np.abs(e_score))[::-1]
    indices_to_plot = sorted_indices_absolute[:n_genes]
    indices_for_table = sorted_indices_abs[:n_genes_table]

    df_rankings[f'{cell_type} {part} score genes'] = gene_names[sorted_indices]
    df_rankings[f'{cell_type} {part} score'] = e_score[sorted_indices]

    x_data = [np.where(sorted_indices==idx)[0][0] for idx in indices_to_plot]
    y_data = e_score[indices_to_plot]
    names = gene_names[indices_to_plot]

    df_eigenvalues[cell_type, 'Score gene'] = gene_names[indices_for_table]
    df_eigenvalues[cell_type, 'Score value'] = list(map(lambda x: f'{np.real(x):.3f}', e_score[indices_for_table]))

    ax = axs[i, 2]
    ax.plot(np.sort(e_score),'.',color=colors[cell_type])
    ax.set_ylabel('Component score')
    ax.set_xticks([])
    annotate_points(ax, x_data, y_data, names, offset_x_fraction=0.2, offset_y_fraction=0.1)
    if i == 0:
        ax.set_title(f'Eigenvector score')

plt.tight_layout()
plt.show()

print("Top eigenvalues:", top_evalues)
df_eigenvalues.head(10)

# %% [markdown]
# # Jacobian Analysis

# %%
# Compute Jacobians using scHopfield
sch.tl.compute_jacobians(adata, cluster_key=cluster_key)

# Load jacobians
jacobians = sch.tl.load_jacobians(adata)

# %%
print(f"Jacobians shape: {jacobians['jacobians'].shape}")
print(f"Eigenvalues shape: {jacobians['eigenvalues'].shape}")

# %%
# Average Jacobian per cluster
mean_jacobian = {}
for k in order:
    cell_idx = np.where(adata.obs[cluster_key] == k)[0]
    mean_jacobian[k] = np.mean(jacobians['jacobians'][cell_idx], axis=0)

# %% [markdown]
# # Network Visualization

# %%
from matplotlib.colors import LinearSegmentedColormap

def GRN_graph(
    adata, W, gene_names, score_df, score_size=None, cmap=None,
    size_threshold=0.25, topn=50, ax=None, label_offset=0.0, variable_width=True
):
    """Plot gene regulatory network graph."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Get top genes
    if score_size is not None:
        top_genes = score_df.nlargest(topn, score_size).index.values
    else:
        # Use degree as default
        degrees = np.abs(W).sum(axis=0) + np.abs(W).sum(axis=1)
        top_indices = np.argsort(degrees)[-topn:]
        top_genes = gene_names[top_indices]

    # Create subgraph
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    top_indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    W_sub = W[np.ix_(top_indices, top_indices)]

    # Create graph
    G = nx.DiGraph()
    for i, gene_i in enumerate(top_genes):
        for j, gene_j in enumerate(top_genes):
            if i < len(W_sub) and j < len(W_sub[0]):
                weight = W_sub[i, j]
                if abs(weight) > size_threshold:
                    G.add_edge(gene_i, gene_j, weight=weight)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Node sizes
    if score_size is not None and score_size in score_df.columns:
        node_sizes = []
        for node in G.nodes():
            if node in score_df.index:
                node_sizes.append(score_df.loc[node, score_size] * 1000)
            else:
                node_sizes.append(100)
    else:
        node_sizes = [300] * len(G.nodes())

    # Edge properties
    edges = list(G.edges())
    weights = [abs(G[u][v]['weight']) for u, v in edges]
    if len(weights) > 0:
        weights = np.array(weights)
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10) * 5 + 0.5

    # Edge colors
    edge_colors = []
    for u, v in edges:
        w = G[u][v]['weight']
        if w > 0:
            edge_colors.append('red')
        else:
            edge_colors.append('blue')

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgray',
                          edgecolors='black', linewidths=1.5, ax=ax, alpha=0.9)

    if variable_width:
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights,
                              edge_color=edge_colors, ax=ax, alpha=0.6,
                              arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
    else:
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=1,
                              edge_color=edge_colors, ax=ax, alpha=0.6,
                              arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    ax.axis('off')
    plt.tight_layout()

def plot_subset_grn(adata, W, subset_genes, score_df, score_size=None, ax=None,
                    node_positions=None, selected_edges=None, label_offset=0.08, variable_width=True):
    """Plot subset of GRN with custom positioning."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    subset_indices = [gene_to_idx[g] for g in subset_genes if g in gene_to_idx]
    subset_genes = [g for g in subset_genes if g in gene_to_idx]

    W_sub = W[np.ix_(subset_indices, subset_indices)]

    G = nx.DiGraph()
    for i, gene_i in enumerate(subset_genes):
        G.add_node(gene_i)

    # Add edges
    if selected_edges is not None:
        for u, v in selected_edges:
            if u in subset_genes and v in subset_genes:
                i = subset_genes.index(u)
                j = subset_genes.index(v)
                weight = W_sub[i, j]
                if weight != 0:
                    G.add_edge(u, v, weight=weight)
    else:
        for i, gene_i in enumerate(subset_genes):
            for j, gene_j in enumerate(subset_genes):
                weight = W_sub[i, j]
                if weight != 0:
                    G.add_edge(gene_i, gene_j, weight=weight)

    # Positions
    if node_positions is not None:
        pos = node_positions
    else:
        pos = nx.spring_layout(G)

    # Node sizes
    if score_size is not None and score_size in score_df.columns:
        node_sizes = []
        for node in G.nodes():
            if node in score_df.index:
                node_sizes.append(max(score_df.loc[node, score_size] * 2000, 100))
            else:
                node_sizes.append(100)
    else:
        node_sizes = [500] * len(G.nodes())

    # Draw
    edges = list(G.edges())
    if len(edges) > 0:
        weights = np.array([abs(G[u][v]['weight']) for u, v in edges])
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10) * 5 + 0.5

        edge_colors = []
        for u, v in edges:
            w = G[u][v]['weight']
            edge_colors.append('red' if w > 0 else 'blue')

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgray',
                              edgecolors='black', linewidths=1.5, ax=ax, alpha=0.9)

        if variable_width:
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights,
                                  edge_color=edge_colors, ax=ax, arrows=True,
                                  arrowsize=20, connectionstyle="arc3,rad=0.1")
        else:
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=1,
                                  edge_color=edge_colors, ax=ax, arrows=True,
                                  arrowsize=20, connectionstyle="arc3,rad=0.1")

        adjusted_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}
        nx.draw_networkx_labels(G, adjusted_pos, font_size=12, ax=ax)

    ax.axis('off')
    plt.tight_layout()

# %% [markdown]
# ## Network figures

# %%
colors_graph = ["blue", "lightgray", "red"]
positions = [0, 0.5, 1]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors_graph)))

fig,axs = plt.subplots(4,2, figsize=(20,40),tight_layout=True)
score = 'degree_centrality_out'
topn = 50

for k,ax in zip(order, axs.flat):
    ax.axis('off')
    ax.set_title(score.capitalize().replace('_',' ') + ' - ' + k)
    GRN_graph(adata, W[k], gene_names, links.merged_score[links.merged_score.cluster==k],
              score_size=score, cmap=custom_cmap, size_threshold=0.25, topn=topn, ax=ax)

plt.show()

# %%
fig,axs = plt.subplots(4,2, figsize=(20,15),tight_layout=True)

custom_positions = {
    'CEBPA': (1,1), 'GATA1': (4,1), 'GATA2': (0,0),
    'RUNX1': (2,0), 'KLF1': (3,0), 'FLI1': (5,0)
}

selected_edges = [('CEBPA','GATA2'), ('CEBPA','RUNX1'),
                   ('GATA2','GATA2'), ('GATA2','RUNX1'), ('GATA2','GATA1'), ('RUNX1','GATA2'), ('RUNX1','RUNX1'),
                   ('GATA1', 'KLF1'), ('GATA1','FLI1'), ('GATA1','GATA2'),
                   ('KLF1', 'FLI1'), ('FLI1', 'KLF1'), ('FLI1', 'FLI1')
                   ]

for k,ax in zip(order, axs.flat):
    ax.axis('off')
    ax.set_title('Degree centrality out - ' + k)
    plot_subset_grn(adata, W[k], custom_positions.keys(),
                   links.merged_score[links.merged_score.cluster==k],
                   score_size='degree_centrality_out', ax=ax,
                   node_positions=custom_positions, selected_edges=selected_edges,
                   label_offset=0.08, variable_width=True)

plt.show()

# %% [markdown]
# ## Jacobian-based network figures

# %%
fig,axs = plt.subplots(4,2, figsize=(20,40),tight_layout=True)
score = 'degree_centrality_out'
topn = 50

for k,ax in zip(order, axs.flat):
    ax.axis('off')
    ax.set_title(score.capitalize().replace('_',' ') + ' - ' + k)
    GRN_graph(adata, mean_jacobian[k], gene_names,
             links.merged_score[links.merged_score.cluster==k],
             score_size=None, cmap=custom_cmap, size_threshold=0.25, topn=topn, ax=ax)

plt.show()

# %%
fig,axs = plt.subplots(4,2, figsize=(20,15),tight_layout=True)

for k,ax in zip(order, axs.flat):
    ax.axis('off')
    ax.set_title('Jacobian network - ' + k)
    plot_subset_grn(adata, mean_jacobian[k], custom_positions.keys(),
                   links.merged_score[links.merged_score.cluster==k],
                   score_size=None, ax=ax, node_positions=custom_positions,
                   selected_edges=selected_edges, label_offset=0.08, variable_width=True)

plt.show()

# %%
print("Analysis complete!")
print("All results are stored in the AnnData object:")
print("- adata.varp[f'W_{cluster}']: Interaction matrices")
print("- adata.var[f'I_{cluster}']: Bias vectors")
print("- adata.var[f'gamma_{cluster}']: Degradation rates")
print("- adata.obs: Energy values and Jacobian eigenvalues")
print("- adata.uns['scHopfield']: Correlation matrices and other metadata")
