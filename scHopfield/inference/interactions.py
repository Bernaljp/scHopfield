"""Inference of gene regulatory network interactions."""

import numpy as np
import torch
from typing import Optional, Union, List, Dict
from anndata import AnnData

from .optimizer import ScaffoldOptimizer
from .datasets import CustomDataset
from .._utils.io import get_matrix, to_numpy, parse_genes, get_genes_used


def fit_interactions(
    adata: AnnData,
    cluster_key: str,
    spliced_key: str = 'Ms',
    velocity_key: str = 'velocity_S',
    degradation_key: str = 'gamma',
    w_threshold: float = 1e-5,
    w_scaffold: Optional[np.ndarray] = None,
    scaffold_regularization: float = 1.0,
    only_TFs: bool = False,
    infer_I: bool = False,
    refit_gamma: bool = False,
    pre_initialize_W: bool = False,
    n_epochs: int = 1000,
    criterion: str = 'L2',
    batch_size: int = 64,
    device: str = 'cpu',
    skip_all: bool = False,
    use_scheduler: bool = False,
    scheduler_kws: Optional[Dict] = None,
    get_plots: bool = False,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Infer gene regulatory network interaction matrices.

    Fits interaction matrix W and bias vector I for each cluster by
    solving: velocity = W * sigmoid(expression) - gamma * expression + I

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted sigmoid parameters
    cluster_key : str
        Key in adata.obs containing cluster annotations
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    velocity_key : str, optional (default: 'velocity_S')
        Key in adata.layers for RNA velocity
    degradation_key : str, optional (default: 'gamma')
        Key in adata.var for degradation rates
    w_threshold : float, optional (default: 1e-5)
        Threshold for pruning small interaction weights
    w_scaffold : np.ndarray, optional
        Binary scaffold matrix constraining network topology
    scaffold_regularization : float, optional (default: 1.0)
        Regularization strength for scaffold constraint
    only_TFs : bool, optional (default: False)
        If True, use masked linear layer (requires w_scaffold)
    infer_I : bool, optional (default: False)
        If True, infer bias vector I in least squares
    refit_gamma : bool, optional (default: False)
        If True, refit degradation rates during optimization
    pre_initialize_W : bool, optional (default: False)
        If True, initialize W with least squares solution
    n_epochs : int, optional (default: 1000)
        Number of training epochs
    criterion : str, optional (default: 'L2')
        Loss function: 'L1', 'L2', or 'MSE'
    batch_size : int, optional (default: 64)
        Batch size for training
    device : str, optional (default: 'cpu')
        Device for computation: 'cpu' or 'cuda'
    skip_all : bool, optional (default: False)
        If True, skip fitting on all cells combined
    use_scheduler : bool, optional (default: False)
        If True, use learning rate scheduler
    scheduler_kws : dict, optional
        Keyword arguments for scheduler
    get_plots : bool, optional (default: False)
        If True, show training plots
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata:
        - adata.varp[f'W_{cluster}']: interaction matrix for each cluster
        - adata.var[f'I_{cluster}']: bias vector for each cluster
        - adata.var[f'gamma_{cluster}']: refitted gamma if refit_gamma=True
        - adata.uns['scHopfield']['models'][cluster]: trained models if w_scaffold is provided
    """
    adata = adata.copy() if copy else adata

    # Store cluster_key
    if 'scHopfield' not in adata.uns:
        adata.uns['scHopfield'] = {}
    adata.uns['scHopfield']['cluster_key'] = cluster_key

    # Get gene indices
    genes = get_genes_used(adata)

    # Get data matrices
    x = to_numpy(get_matrix(adata, spliced_key, genes=genes))
    v = to_numpy(get_matrix(adata, velocity_key, genes=genes))
    g = adata.var[degradation_key].values[genes].astype(x.dtype)
    sig = get_matrix(adata, 'sigmoid', genes=genes)

    # Get clusters
    clusters = adata.obs[cluster_key].unique()
    if not skip_all:
        clusters = np.append(clusters, 'all')

    # Initialize storage for models if using scaffold
    if w_scaffold is not None:
        if 'models' not in adata.uns['scHopfield']:
            adata.uns['scHopfield']['models'] = {}

    # Fit for each cluster
    for cluster in clusters:
        print(f"Inferring interaction matrix W and bias vector I for cluster {cluster}")

        # Get cluster indices
        if cluster == 'all':
            idx = np.ones(adata.n_obs, dtype=bool)
        else:
            idx = adata.obs[cluster_key].values == cluster

        # Fit interactions for this cluster
        _fit_interactions_for_cluster(
            adata=adata,
            cluster=cluster,
            x=x[idx, :],
            v=v[idx, :],
            sig=sig[idx, :],
            g=g,
            w_threshold=w_threshold,
            w_scaffold=w_scaffold,
            scaffold_regularization=scaffold_regularization,
            only_TFs=only_TFs,
            infer_I=infer_I,
            refit_gamma=refit_gamma,
            pre_initialize_W=pre_initialize_W,
            n_epochs=n_epochs,
            criterion=criterion,
            batch_size=batch_size,
            device=device,
            use_scheduler=use_scheduler,
            scheduler_kws=scheduler_kws,
            get_plots=get_plots
        )

    return adata if copy else None


def _fit_interactions_for_cluster(
    adata: AnnData,
    cluster: str,
    x: np.ndarray,
    v: np.ndarray,
    sig: np.ndarray,
    g: np.ndarray,
    w_threshold: float,
    w_scaffold: Optional[np.ndarray],
    scaffold_regularization: float,
    only_TFs: bool,
    infer_I: bool,
    refit_gamma: bool,
    pre_initialize_W: bool,
    n_epochs: int,
    criterion: str,
    batch_size: int,
    device: str,
    use_scheduler: bool,
    scheduler_kws: Optional[Dict],
    get_plots: bool
):
    """
    Fit interaction matrix W and bias I for a single cluster.

    This is adapted from Landscape._fit_interactions_for_group.
    Modifies adata in-place.
    """
    if scheduler_kws is None:
        scheduler_kws = {}

    device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

    W = None
    I = None

    # Least squares initialization
    if (w_scaffold is None) or pre_initialize_W:
        rhs = np.hstack((sig, np.ones((sig.shape[0], 1), dtype=x.dtype))) if infer_I else sig
        try:
            WI = np.linalg.lstsq(rhs, v + g[None, :] * x, rcond=1e-5)[0]
            W = WI[:-1, :].T if infer_I else WI.T
            I = WI[-1, :] if infer_I else -np.clip(WI, a_min=None, a_max=0).sum(axis=0)
        except:
            pass

    # Use ScaffoldOptimizer if scaffold provided
    if w_scaffold is not None:
        model = ScaffoldOptimizer(
            g, w_scaffold, device, refit_gamma,
            scaffold_regularization=scaffold_regularization,
            use_masked_linear=only_TFs,
            pre_initialized_W=W,
            pre_initialized_I=I
        )
        train_loader = _create_train_loader(sig, v, x, device, batch_size)
        scheduler_fn = torch.optim.lr_scheduler.StepLR if use_scheduler else None
        scheduler_kwargs = {"step_size": 100, "gamma": 0.4} if scheduler_kws == {} else scheduler_kws
        model.train_model(
            train_loader, n_epochs,
            learning_rate=0.1,
            criterion=criterion,
            scheduler_fn=scheduler_fn,
            scheduler_kwargs=scheduler_kwargs,
            get_plots=get_plots
        )
        W = model.W.weight.detach().cpu().numpy()
        I = model.I.detach().cpu().numpy()
        g = np.exp(model.gamma.detach().cpu().numpy())
        adata.uns['scHopfield']['models'][cluster] = model

    # Threshold and store
    W[np.abs(W) < w_threshold] = 0
    I[np.abs(I) < w_threshold] = 0

    # Store interaction matrix in varp
    adata.varp[f'W_{cluster}'] = W

    # Store bias vector in var (one column per cluster)
    adata.var[f'I_{cluster}'] = 0.0
    gene_indices = get_genes_used(adata)
    adata.var.iloc[gene_indices, adata.var.columns.get_loc(f'I_{cluster}')] = I

    # Store refitted gamma in var if applicable
    if refit_gamma:
        adata.var[f'gamma_{cluster}'] = 0.0
        adata.var.iloc[gene_indices, adata.var.columns.get_loc(f'gamma_{cluster}')] = g


def _create_train_loader(sig, v, x, device, batch_size=64):
    """Helper to create PyTorch DataLoader."""
    dataset = CustomDataset(sig, v, x, device)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
