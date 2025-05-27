import numpy as np
import torch
import logging
from typing import Optional, Dict, List, Union
from anndata import AnnData
from scipy import sparse
from schopfield.utils.data import get_matrix, to_numpy, write_property
from schopfield.utils.math import compute_sigmoid
from schopfield.optimization.optimizer import ScaffoldOptimizer, CustomDataset

logger = logging.getLogger(__name__)

import numpy as np
import scipy.optimize as opt
from typing import Optional, Dict
import logging
from anndata import AnnData
import torch
from scipy.sparse import issparse
from . import analysis
from ..utils.data import to_numpy, get_matrix, write_sigmoids, write_property
from ..optimization.optimizer import ScaffoldOptimizer, CustomDataset

logger = logging.getLogger(__name__)

def fit_sigmoids(landscape: 'Landscape', min_th: float = 0.05, max_th: float = 5.0) -> None:
    """Fit sigmoid parameters for each gene."""
    logger.info("Fitting sigmoid parameters")
    
    x = get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes)
    v = get_matrix(landscape.adata, landscape.velocity_key, genes=landscape.genes)
    g = landscape.adata.var[landscape.gamma_key].iloc[landscape.genes].values
    
    n = len(landscape.genes)
    landscape.threshold = np.zeros(n)
    landscape.exponent = np.zeros(n)
    offset = np.zeros(n)
    mse = np.zeros(n)
    
    for i in range(n):
        th, n, c, err = _fit_sigmoid(x[:, i], v[:, i], g[i], min_th, max_th)
        landscape.threshold[i] = th
        landscape.exponent[i] = n
        offset[i] = c
        mse[i] = err
    
    write_property(landscape.adata, "sigmoid_threshold", landscape.threshold)
    write_property(landscape.adata, "sigmoid_exponent", landscape.exponent)
    write_property(landscape.adata, "sigmoid_offset", offset)
    write_property(landscape.adata, "sigmoid_mse", mse)
    write_sigmoids(landscape)

def _fit_sigmoid(x: np.ndarray, v: np.ndarray, g: float, min_th: float, max_th: float) -> tuple:
    """Fit a single sigmoid function."""
    def err(p: np.ndarray) -> float:
        s, n, c = p
        y = x**n / (x**n + s**n)
        y = np.clip(y, 1e-10, 1 - 1e-10)
        ty = np.log(y / (1 - y))
        return np.mean((ty - (v + c) / g)**2)
    
    res = opt.minimize(err, [0.5 * (min_th + max_th), 2.0, 0.0], bounds=[(min_th, max_th), (0.1, 10.0), (-1.0, 1.0)])
    return res.x[0], res.x[1], res.x[2], res.fun

def fit_interactions(
    landscape: 'Landscape',
    w_threshold: float = 1e-5,
    w_scaffold: Optional[np.ndarray] = None,
    scaffold_regularization: float = 1.0,
    only_TFs: bool = False,
    infer_I: bool = False,
    bias_regularization: float = 10.0,
    refit_gamma: bool = False,
    pre_initialize_W: bool = False,
    n_epochs: int = 1000,
    criterion: str = "L2",
    batch_size: int = 64,
    device: str = "cpu",
    skip_all: bool = False,
    use_scheduler: bool = False,
    scheduler_kws: Dict = {},
    get_plots: bool = False,
) -> None:
    """Fit interaction matrices and biases for each cluster.
    
    Updates landscape.W, landscape.I, and landscape.gamma with fitted parameters.
    
    Args:
        landscape: Landscape object containing adata and parameters.
        w_threshold: Threshold for zeroing small weights in W and I.
        w_scaffold: Scaffold matrix for regularization (optional).
        scaffold_regularization: Regularization strength for scaffold.
        only_TFs: Use masked linear layer for TFs only.
        infer_I: Infer bias vector I.
        bias_regularization: Regularization strength for bias terms.
        refit_gamma: Refit degradation rates.
        pre_initialize_W: Pre-initialize W using least squares.
        n_epochs: Number of training epochs for ScaffoldOptimizer.
        criterion: Loss criterion for optimization ("L2" or other).
        batch_size: Batch size for training.
        device: Device for PyTorch ("cpu" or "cuda").
        skip_all: If True, skip fitting for a synthetic 'all' cluster; otherwise, include it.
        use_scheduler: Use learning rate scheduler.
        scheduler_kws: Keyword arguments for scheduler.
        get_plots: Generate training plots.
    
    Raises:
        ValueError: If cluster_key is missing when required.
    """
    logger.info("Fitting interaction matrices and biases")
    
    # Validate inputs
    x = to_numpy(get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes))
    v = to_numpy(get_matrix(landscape.adata, landscape.velocity_key, genes=landscape.genes))
    g = landscape.adata.var[landscape.gamma_key].iloc[landscape.genes].values.astype(x.dtype)
    sig = get_matrix(landscape.adata, "sigmoid", genes=landscape.genes)
    
    # Initialize dictionaries
    landscape.W = {}
    landscape.I = {}
    landscape.gamma = {}
    if w_scaffold is not None:
        landscape.adata.uns['models'] = {}
    
    # Get clusters
    clusters = []
    if landscape.cluster_key is not None:
        try:
            clusters = landscape.adata.obs[landscape.cluster_key].unique().tolist()
        except KeyError:
            raise ValueError(f"cluster_key '{landscape.cluster_key}' not found in adata.obs")
    if not skip_all or not clusters:
        clusters.append('all')
    
    for cluster in clusters:
        if cluster == 'all':
            idx = np.arange(len(x))
        else:
            idx = np.where(landscape.adata.obs[landscape.cluster_key] == cluster)[0]
        
        _fit_interactions_for_group(landscape, cluster,
            x[idx], v[idx], sig[idx], g, w_threshold, w_scaffold, scaffold_regularization, only_TFs,
            infer_I, bias_regularization, refit_gamma, pre_initialize_W, n_epochs,
            criterion, batch_size, device, use_scheduler, scheduler_kws, get_plots
        )

def _fit_interactions_for_group(
    landscape: 'Landscape',
    group: str,
    x: np.ndarray,
    v: np.ndarray,
    sig: np.ndarray,
    g: np.ndarray,
    w_threshold: float,
    w_scaffold: Optional[np.ndarray],
    scaffold_regularization: float,
    only_TFs: bool,
    infer_I: bool,
    bias_regularization: float,
    refit_gamma: bool,
    pre_initialize_W: bool,
    n_epochs: int,
    criterion: str,
    batch_size: int,
    device: str,
    use_scheduler: bool,
    scheduler_kws: Dict,
    get_plots: bool,
) -> None:
    """Fit interaction matrix W and bias vector I for a group.

    Args:
        landscape: Landscape object to store results.
        group: Cluster label or 'all'.
        x: Spliced expression data.
        v: Velocity data.
        sig: Sigmoid activations.
        g: Degradation rates.
        w_threshold: Threshold for zeroing weights.
        w_scaffold: Scaffold matrix (optional).
        scaffold_regularization: Regularization strength.
        only_TFs: Use masked linear layer for TFs.
        infer_I: Infer bias vector.
        bias_regularization: Bias regularization strength.
        refit_gamma: Refit degradation rates.
        pre_initialize_W: Pre-initialize W.
        n_epochs: Number of training epochs.
        criterion: Loss criterion.
        batch_size: Batch size.
        device: PyTorch device.
        use_scheduler: Use scheduler.
        scheduler_kws: Scheduler arguments.
        get_plots: Generate plots.
    """
    device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")
    
    W = None
    I = None
    if w_scaffold is None or pre_initialize_W:
        rhs = np.hstack((sig, np.ones((sig.shape[0], 1), dtype=x.dtype))) if infer_I else sig
        try:
            WI = np.linalg.lstsq(rhs, v + g[None, :] * x, rcond=1e-5)[0]
            W = WI[:-1, :].T if infer_I else WI
            I = WI[-1, :] if infer_I else -np.clip(WI, a_min=None, a_max=0).sum(axis=0)
        except np.linalg.LinAlgError:
            logger.warning(f"Least squares failed for group '{group}'")
    
    if w_scaffold is not None:
        model = ScaffoldOptimizer(
            g,
            w_scaffold,
            device,
            refit_gamma,
            scaffold_regularization=scaffold_regularization,
            use_masked_linear=only_TFs,
            pre_initialized_W=W,
            pre_initialized_I=I,
            bias_regularization=bias_regularization
        )
        train_loader = _create_train_loader(sig, v, x, device, batch_size=batch_size)
        scheduler_fn = torch.optim.lr_scheduler.StepLR if use_scheduler else None
        scheduler_kwargs = {"step_size": 100, "gamma": 0.4} if not scheduler_kws else scheduler_kws
        model.train_model(
            train_loader,
            n_epochs,
            learning_rate=0.1,
            criterion=criterion,
            scheduler_fn=scheduler_fn,
            scheduler_kwargs=scheduler_kwargs,
            get_plots=get_plots
        )
        W = model.W.weight.detach().cpu().numpy()
        I = model.I.detach().cpu().numpy()
        g = np.exp(model.gamma.detach().cpu().numpy())
        if w_scaffold is not None:
            landscape.adata.uns['models'][group] = model
    
    # Threshold and store results
    W[np.abs(W) < w_threshold] = 0
    I[np.abs(I) < w_threshold] = 0
    landscape.W[group] = W
    landscape.I[group] = I
    landscape.gamma[group] = g if refit_gamma else landscape.adata.var[landscape.gamma_key][landscape.genes].values.astype(W.dtype)

def _create_train_loader(
    sig: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader for training.

    Args:
        sig: Sigmoid activations.
        v: Velocity data.
        x: Expression data.
        device: PyTorch device.
        batch_size: Batch size.

    Returns:
        torch.utils.data.DataLoader: DataLoader for training.
    """
    dataset = CustomDataset(sig, v, x, device)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)