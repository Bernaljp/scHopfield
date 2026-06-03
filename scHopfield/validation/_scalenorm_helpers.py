"""Helpers for the scale-normalized scHopfield validation runners.

Factored out of the per-circuit sweep scripts so the same training and
evaluation logic is shared. Keeps every change additive: the original
``run_*_figure.py`` scripts are not affected.
"""
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from scHopfield.inference.optimizer import ScaffoldOptimizer
from scHopfield.dynamics.solver import ODESolver


def make_weights(train_DX: np.ndarray, p: float, floor_quantile: float = 0.5,
                 floor_mult: float = 0.1):
    """Per-gene weight 1 / std^p, with zero-std species set to weight=0.

    A floor based on a quantile of nonzero stds (default the median) prevents
    explosive weight blowups for low-variance but-nonzero species. Genes with
    exactly zero std (e.g. clamped variables) get weight 0.
    """
    std = train_DX.std(axis=0).astype(np.float32)
    nz = std[std > 1e-6]
    floor = float(np.quantile(nz, floor_quantile)) * floor_mult if nz.size else 1.0
    weight = np.where(std > 1e-6, 1.0 / np.maximum(std, floor) ** p, 0.0).astype(np.float32)
    return std, weight, floor


def train_with_weighted_mse(opt: ScaffoldOptimizer, loader, weights_torch: torch.Tensor,
                            *, epochs: int, lr: float, criterion: str = "MSE",
                            verbose: bool = False):
    """Train ``opt`` with a per-gene weighted MSE loss on the derivative.

    ``loader`` must yield ``((s, x), v)`` so it matches the standard
    ScaffoldOptimizer forward signature. ``weights_torch`` has shape (n_genes,)
    and is broadcast-multiplied into the residual.
    """
    optimizer = optim.Adam(opt.parameters(), lr=lr)
    mask_m = 1.0 - opt.scaffold_raw
    loss_hist, recon_hist = [], []
    for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
        epoch_loss = 0.0
        epoch_recon = 0.0
        for (s_b, x_b), tgt in loader:
            s_b = s_b.to(opt.device)
            x_b = x_b.to(opt.device)
            tgt = tgt.to(opt.device)
            optimizer.zero_grad()
            out = opt((s_b, x_b))
            err = (out - tgt) * weights_torch
            recon = opt.reconstruction_lambda * (err ** 2).mean()
            bs = s_b.shape[0]
            if opt.normalize_regularization:
                graph = opt.scaffold_lambda * ((opt.W.weight * mask_m).norm(2) + (opt.W.weight * mask_m).norm(1)) / bs
                bias = opt.bias_lambda * torch.abs(opt.I + opt.bias_bias).norm(2) / bs
            else:
                graph = opt.scaffold_lambda * ((opt.W.weight * mask_m).norm(2) + (opt.W.weight * mask_m).norm(1))
                bias = opt.bias_lambda * torch.abs(opt.I + opt.bias_bias).norm(2)
            total = recon + graph + bias
            total.backward()
            optimizer.step()
            epoch_loss += total.item()
            epoch_recon += recon.item()
        loss_hist.append(epoch_loss / len(loader))
        recon_hist.append(epoch_recon / len(loader))
    return loss_hist, recon_hist


def make_loader(train_X: np.ndarray, sig_arr: np.ndarray, train_DX: np.ndarray,
                batch_size: int = 128):
    sig_t = torch.from_numpy(sig_arr.astype(np.float32))
    x_t = torch.from_numpy(train_X.astype(np.float32))
    v_t = torch.from_numpy(train_DX.astype(np.float32))
    dataset = TensorDataset(sig_t, x_t, v_t)
    base = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return LoaderShim(base)


class LoaderShim:
    """Wrap DataLoader to deliver the ((sig, x), v) tuple ScaffoldOptimizer expects."""
    def __init__(self, base):
        self.base = base
    def __iter__(self):
        for sig_b, x_b, v_b in self.base:
            yield (sig_b, x_b), v_b
    def __len__(self):
        return len(self.base)


def forward_simulate(opt: ScaffoldOptimizer, x0: np.ndarray, t_eval: np.ndarray,
                     k_hill, n_hill: float):
    """Forward-simulate the fitted model with scHopfield's ODESolver."""
    W_inf = opt.W.weight.detach().cpu().numpy()
    I_inf = opt.I.detach().cpu().numpy()
    gamma_inf = np.exp(np.clip(opt.gamma.detach().cpu().numpy(), -np.inf, 10.0))
    n_genes = W_inf.shape[0]
    threshold = np.asarray(k_hill, dtype=np.float32)
    if threshold.ndim == 0:
        threshold = np.full(n_genes, float(k_hill), dtype=np.float32)
    exponent = np.full(n_genes, float(n_hill), dtype=np.float32)
    solver = ODESolver(W=W_inf, I=I_inf, gamma=gamma_inf,
                       threshold=threshold, exponent=exponent, x_min=0.0)
    traj = solver.solve(x0.astype(np.float64), t_eval, method="euler", clip_each_step=True)
    return traj, W_inf, I_inf, gamma_inf


def per_gene_pearson(X_gt: np.ndarray, X_fit: np.ndarray) -> np.ndarray:
    """Pearson r per column (gene), ignoring constant-std species (returns NaN)."""
    out = np.full(X_gt.shape[1], np.nan)
    for k in range(X_gt.shape[1]):
        a = X_gt[:, k]
        b = X_fit[:, k]
        if a.std() > 1e-9 and b.std() > 1e-9:
            out[k] = np.corrcoef(a, b)[0, 1]
    return out


def average_jacobian(rhs, X_samples: np.ndarray, n_eval: int = 500,
                     eps: float = 1e-5, seed: int = 0) -> np.ndarray:
    """Average Jacobian of the ground-truth RHS over a sample of training points.

    For non-Hopfield circuits (cell cycle MM, JAK-STAT cascade) there is no
    literal ground-truth W. The natural analog is
        $\\bar J_{ij} = \\langle \\partial f_i / \\partial x_j \\rangle_{x \\sim D}$,
    the average linear sensitivity over the data distribution -- the matrix a
    perfect *linear* fit would converge to in L2. Comparing this against the
    inferred $\\hat W$ shows whether the sign pattern of the interactions is
    captured (magnitudes differ because the Hopfield form absorbs sigma'(x)).
    Computed by central finite differences.
    """
    rng = np.random.default_rng(seed)
    n_samples = X_samples.shape[0]
    n_dim = X_samples.shape[1]
    if n_samples > n_eval:
        idx = rng.choice(n_samples, size=n_eval, replace=False)
        X = X_samples[idx]
    else:
        X = X_samples
    J_sum = np.zeros((n_dim, n_dim))
    for x in X:
        for j in range(n_dim):
            xp = x.copy()
            xp[j] += eps
            xm = x.copy()
            xm[j] = max(xm[j] - eps, 0.0)
            actual_eps = xp[j] - xm[j]
            if actual_eps > 0:
                J_sum[:, j] += (rhs(xp) - rhs(xm)) / actual_eps
    return J_sum / X.shape[0]


def amplitude_stability_score(X_gt: np.ndarray, X_fit: np.ndarray) -> float:
    """Max ratio of fit max-amplitude to GT max-amplitude across species.

    Score >> 1 indicates the fit is running away. Score around 1 is
    well-calibrated. Constant species are skipped.
    """
    ratios = []
    for k in range(X_gt.shape[1]):
        gt_amp = X_gt[:, k].max() - X_gt[:, k].min()
        fit_amp = X_fit[:, k].max() - X_fit[:, k].min()
        if gt_amp > 1e-6:
            ratios.append(fit_amp / gt_amp)
    return float(np.max(ratios)) if ratios else 1.0
