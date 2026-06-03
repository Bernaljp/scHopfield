"""Thin wrapper around the scHopfield ScaffoldOptimizer for validation circuits.

The full ``fit_interactions`` pipeline expects an scRNA-seq context (clusters,
neighbor graphs, sigmoid fitting, hierarchical initialization). For synthetic
circuits we bypass all of that and fit ``ScaffoldOptimizer`` directly on the
single-cluster synthetic data, using the circuit's analytic Hill sigmoid as
the activation rather than scHopfield's empirical-CDF Hill fit (which would
be miscalibrated on toy data with only 2-3 genes).
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..inference.optimizer import ScaffoldOptimizer


def _build_scaffold(W_true: np.ndarray, mode: str, seed: int = 0,
                    false_pos_rate: float = 0.0) -> np.ndarray:
    """Build a scaffold prior G with one of three regimes.

    Parameters
    ----------
    W_true : np.ndarray
        Ground-truth interaction matrix.
    mode : {'none', 'partial', 'full'}
        - 'none': all-ones scaffold (no prior structure imposed).
        - 'partial': retain 50% of true edges, add ``false_pos_rate`` fraction
          of random false-positive edges from the zero-positions of W_true.
        - 'full': scaffold is the indicator of nonzero entries in W_true.
    seed
        RNG seed for the partial-scaffold edge sampling.
    false_pos_rate
        Fraction of off-edges to flip to "in-scaffold" under the partial regime.
    """
    n = W_true.shape[0]
    if mode == "none":
        return np.ones_like(W_true, dtype=np.float32)
    if mode == "full":
        return (np.abs(W_true) > 1e-12).astype(np.float32)
    if mode == "partial":
        rng = np.random.default_rng(seed)
        G = np.zeros_like(W_true, dtype=np.float32)
        true_edges = list(zip(*np.nonzero(np.abs(W_true) > 1e-12)))
        rng.shuffle(true_edges)
        keep = true_edges[: max(1, len(true_edges) // 2)]
        for i, j in keep:
            G[i, j] = 1.0
        # Add false positives from the zero-positions of W_true.
        zero_edges = list(zip(*np.nonzero(np.abs(W_true) < 1e-12)))
        n_fp = int(false_pos_rate * len(zero_edges))
        rng.shuffle(zero_edges)
        for i, j in zero_edges[:n_fp]:
            G[i, j] = 1.0
        return G
    raise ValueError(f"Unknown scaffold mode: {mode}")


def fit_circuit(
    adata,
    scaffold_mode: str = "full",
    scaffold_regularization: float = 1e-2,
    reconstruction_regularization: float = 1.0,
    bias_regularization: float = 1e-2,
    n_epochs: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 5e-2,
    device: str = "cpu",
    seed: int = 0,
    false_pos_rate: float = 0.1,
    return_optimizer: bool = False,
):
    """Fit scHopfield on a synthetic circuit and return the inferred parameters.

    Uses the circuit's ground-truth Hill activation (stored as a callable on
    the circuit object) to sigmoid-transform expression, then fits
    ``W sigma(s) + I - gamma s = v`` against the analytic dx/dt.

    Parameters
    ----------
    adata : AnnData
        Output of ``simulate_circuit``. Must have ``layers['Ms']`` and
        ``layers['velocity_S']``, plus ``uns['ground_truth']`` with the true W.
    scaffold_mode : {'none', 'partial', 'full'}
        Which scaffold regime to use. See ``_build_scaffold``.
    scaffold_regularization, reconstruction_regularization, bias_regularization : float
        Loss weights for the three terms.
    n_epochs, batch_size, learning_rate : training hyperparams.
    device : str
        Torch device.
    seed : int
        RNG seed for scaffold construction and weight initialization.
    false_pos_rate : float
        Used only when scaffold_mode='partial'.
    return_optimizer : bool
        If True, also return the trained ScaffoldOptimizer for inspection.

    Returns
    -------
    result : dict with keys
        'W_inferred'    : (n_genes, n_genes) inferred interaction matrix
        'I_inferred'    : (n_genes,) inferred bias
        'gamma_used'    : (n_genes,) gamma values (held fixed at ground truth here)
        'W_true'        : ground-truth W
        'I_true'        : ground-truth I
        'gamma_true'    : ground-truth gamma
        'scaffold'      : the scaffold matrix used
        'scaffold_mode' : the mode string
        'loss_history'  : list of per-epoch loss values
        'reconstruction_loss_history' : list of per-epoch reconstruction loss
        'optimizer'     : the trained optimizer (only if return_optimizer=True)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    W_true = adata.uns["ground_truth"]["W"]
    I_true = adata.uns["ground_truth"]["I"]
    gamma_true = adata.uns["ground_truth"]["gamma"]

    scaffold = _build_scaffold(W_true, scaffold_mode, seed=seed,
                                false_pos_rate=false_pos_rate)

    # We bypass empirical CDF fitting by precomputing sigma(s) on the circuit's
    # Hill curve, then training ScaffoldOptimizer on (sigma(s), s, v).
    # ScaffoldOptimizer's forward is W(sig) + I - exp(gamma_log) * x.
    # If we already pass sigma(s) as 'sig' and raw s as 'x', that's exactly
    # what we want.
    expression = adata.layers["Ms"]                # raw s
    velocity = adata.layers["velocity_S"]           # analytic dx/dt
    # Compute Hill sigmoid from the circuit equations.
    # The circuit is the one stored in adata.uns; we reproduce it here.
    # For simplicity, accept that we need the user to either pass it in or
    # apply a generic Hill function. We use the same Hill form as the circuit
    # uses (xn / (kn + xn)), with k=1, n=4 as the default for our circuits.
    # Most circuits in this validation suite share these defaults.
    # If a different k or n is needed, override via adata.uns.
    k = adata.uns["ground_truth"].get("k", 1.0)
    n_hill = adata.uns["ground_truth"].get("n", 4)
    xn = np.power(np.maximum(expression, 0.0), n_hill)
    sig = (xn / (k**n_hill + xn)).astype(np.float32)

    sig_t = torch.from_numpy(sig.astype(np.float32))
    x_t = torch.from_numpy(expression.astype(np.float32))
    v_t = torch.from_numpy(velocity.astype(np.float32))

    dataset = TensorDataset(sig_t, x_t, v_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Wrap loader to deliver the (sig, x), v tuple ScaffoldOptimizer expects.
    class _LoaderShim:
        def __init__(self, base):
            self.base = base
        def __iter__(self):
            for sig_b, x_b, v_b in self.base:
                yield (sig_b, x_b), v_b
        def __len__(self):
            return len(self.base)

    opt = ScaffoldOptimizer(
        g=gamma_true.astype(np.float32),
        scaffold=scaffold,
        device=torch.device(device),
        refit_gamma=False,
        scaffold_regularization=scaffold_regularization,
        reconstruction_regularization=reconstruction_regularization,
        bias_regularization=bias_regularization,
        normalize_regularization=True,
    )
    loss_hist, recon_hist = opt.train_model(
        train_loader=_LoaderShim(loader),
        epochs=n_epochs,
        learning_rate=learning_rate,
        criterion="MSE",
        verbose=False,
        get_plots=False,
    )

    W_inferred = opt.W.weight.detach().cpu().numpy().astype(np.float64)
    I_inferred = opt.I.detach().cpu().numpy().astype(np.float64)

    result = {
        "W_inferred": W_inferred,
        "I_inferred": I_inferred,
        "gamma_used": gamma_true,
        "W_true": W_true,
        "I_true": I_true,
        "gamma_true": gamma_true,
        "scaffold": scaffold,
        "scaffold_mode": scaffold_mode,
        "loss_history": loss_hist,
        "reconstruction_loss_history": recon_hist,
    }
    if return_optimizer:
        result["optimizer"] = opt
    return result
