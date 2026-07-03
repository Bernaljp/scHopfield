"""Neural network models for parameter optimization."""
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class MaskedLinearLayer(nn.Module):
    """
    Linear layer with masked weights.
    """
    def __init__(self, input_size, output_size, mask, device, pre_initialized_W=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False, device=device)
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, device=device))

        n_in_mask = (mask.sum(dim=0) > 0).sum().sqrt()

        if pre_initialized_W is None:
            nn.init.uniform_(self.linear.weight, -1 / n_in_mask, 1 / n_in_mask)
        else:
            self.linear.weight = nn.Parameter(torch.tensor(pre_initialized_W, dtype=torch.float32, device=device))

        with torch.no_grad():
            self.linear.weight *= self.mask

        self.weight = self.linear.weight
        self.linear.weight.register_hook(self._apply_mask)

    def _apply_mask(self, grad):
        return grad * self.mask

    def forward(self, x):
        return self.linear(x)


class ScaffoldOptimizer(nn.Module):
    """
    A model that learns:  output = W(s) + I - gamma * x
    with a scaffold-based regularization on W. ``gamma`` is stored in log-space and
    exponentiated to a strictly positive degradation rate, then capped at ``gamma_max``
    (default 10.0). The cap prevents a runaway rate when ``refit_gamma=True`` and leaves
    ordinary fixed-gamma fits untouched (data rates are well below the cap).

    Two distinct ways the scaffold constrains W (chosen by ``use_masked_linear``):

    - ``use_masked_linear=False`` (default): W is a free ``nn.Linear`` and off-scaffold
      weights are **softly** penalized in the loss (``scaffold_regularization`` times the
      L1+L2 norm of ``W * (1 - scaffold_raw)``). Edges outside the prior can survive if the
      data demands them.
    - ``use_masked_linear=True`` (set by ``only_TFs``): W is a ``MaskedLinearLayer`` whose
      off-mask entries are zeroed at init and kept at zero by a gradient hook (a **hard**
      constraint). Note the mask here is the set of TF *columns* (any regulator that has at
      least one outgoing scaffold edge), so this restricts *which genes may regulate*,
      which is coarser than the per-edge soft penalty. The soft penalty term is still
      computed but is ~0 for the hard-masked weights.
    """
    def __init__(
        self,
        g: torch.Tensor,
        scaffold: torch.Tensor,
        device: torch.device,
        refit_gamma: bool = False,
        scaffold_regularization: float = 1.0,
        reconstruction_regularization: float = 1.0,
        bias_regularization: float = 1.0,
        bias_bias: float = 0.0,
        bias_penalty: str = 'l1',
        elastic_ratio: float = 0.5,
        use_masked_linear: bool = False,
        pre_initialized_W: torch.Tensor = None,
        pre_initialized_I: torch.Tensor = None,
        normalize_regularization: bool = False,
    ):
        super().__init__()
        self.device = device

        g = torch.log(torch.tensor(g, dtype=torch.float32, device=device)+1e-8)
        self.gamma = nn.Parameter(g) if refit_gamma else g
        self.gamma_max = 10.0  # cap on the linear degradation rate (see forward)

        scaffold = torch.tensor(scaffold, dtype=torch.float32, device=device)
        self.register_buffer("scaffold_raw", scaffold)

        scaffold_tfs = torch.zeros_like(scaffold)
        with torch.no_grad():
            scaffold_tfs[:, scaffold.sum(dim=0) > 0] = 1
        self.register_buffer("scaffold", scaffold_tfs)

        self.scaffold_lambda = scaffold_regularization
        self.reconstruction_lambda = reconstruction_regularization
        self.bias_lambda = bias_regularization
        self.bias_bias = bias_bias
        self.bias_penalty = bias_penalty
        self.elastic_ratio = elastic_ratio
        self.normalize_regularization = normalize_regularization

        n = g.shape[0]

        init_I = torch.rand((n,), device=device) if pre_initialized_I is None else torch.tensor(pre_initialized_I, dtype=torch.float32, device=device)
        self.I = nn.Parameter(init_I)

        if use_masked_linear:
            self.W = MaskedLinearLayer(n, n, self.scaffold, device=device, pre_initialized_W=pre_initialized_W)
        else:
            self.W = nn.Linear(n, n, bias=False, device=device)
            if pre_initialized_W is not None:
                self.W.weight = nn.Parameter(torch.tensor(pre_initialized_W, dtype=torch.float32, device=device))
            else:
                nn.init.xavier_uniform_(self.W.weight)

        # optional Jacobian-consistency target (set via configure_jacobian_consistency)
        self._jac = None
        self._jac_nsub = 512

    def configure_jacobian_consistency(self, x, sig, exponent, jac_data, n_sub=512, seed=0):
        """Store a per-cell data-estimated Jacobian target for the consistency loss.

        The model's local sensitivity is J_model[i,j] = W_ij * sigma'_j(x) (off-diagonal).
        The regularizer pulls this toward ``jac_data`` (a finite-difference velocity
        Jacobian estimated from each cell's neighbors), injecting the identifying
        information that plain velocity reconstruction lacks on low-dimensional
        (e.g. trajectory-confined) data.

        Parameters
        ----------
        x : (n_cells, N) raw expression at the cells.
        sig : (n_cells, N) Hill activations sigma(x) at the cells.
        exponent : (N,) per-gene Hill exponents n_j (for sigma'(x) = n*sigma(1-sigma)/x).
        jac_data : (n_cells, N, N) target Jacobian per cell (only off-diagonal is used).
        n_sub : max cells subsampled per optimization step (bounds memory).
        seed : seed for the per-step cell subsampling, so the Jacobian regularizer is
            reproducible independent of the global RNG consumption order.
        """
        dev = self.device
        n = self.W.weight.shape[0]
        self._jac = {
            "x": torch.as_tensor(x, dtype=torch.float32, device=dev).clamp(min=1e-6),
            "sig": torch.as_tensor(sig, dtype=torch.float32, device=dev),
            "n": torch.as_tensor(exponent, dtype=torch.float32, device=dev).reshape(-1),
            "data": torch.as_tensor(jac_data, dtype=torch.float32, device=dev),
            "offmask": (~torch.eye(n, dtype=torch.bool, device=dev)),
        }
        self._jac_nsub = int(n_sub)
        try:
            self._jac_gen = torch.Generator(device=dev).manual_seed(int(seed))
        except (RuntimeError, TypeError):
            # Some backends (e.g. MPS) don't support device generators; fall back to
            # the global RNG (still seeded by sch.set_seed).
            self._jac_gen = None

    def _jacobian_loss(self):
        j = self._jac
        m = j["x"].shape[0]
        gen = getattr(self, "_jac_gen", None)
        idx = torch.randint(0, m, (min(self._jac_nsub, m),), device=self.device, generator=gen)
        x = j["x"][idx]
        sig = j["sig"][idx]
        sp = j["n"][None, :] * sig * (1.0 - sig) / x          # sigma'(x): (b, N)
        Jmodel = self.W.weight[None, :, :] * sp[:, None, :]    # W[i,j]*sigma'_j: (b, i, j)
        diff = (Jmodel - j["data"][idx]) * j["offmask"][None]
        return diff.pow(2).sum() / idx.shape[0]

    def _bias_loss(self, batch_size=None):
        """Penalty on the bias vector I.

        - ``'l1'`` (default): ``lambda * ||I + bias_bias||_1`` (lasso; sparse -- most
          genes get exactly zero bias, a few can grow. Matches "small under natural
          GRN control, large only on externally forced genes"; see FINDINGS M16).
        - ``'l2'`` (legacy): ``lambda * ||I + bias_bias||_2`` (group shrinkage; drives
          the whole vector uniformly small, so a genuine per-gene input is flattened
          together with the noise).
        - ``'elastic'``: ``lambda * (r*||.||_1 + (1-r)*||.||_2^2)`` (sparse but stable).
        """
        Ib = self.I + self.bias_bias
        if self.bias_penalty == 'l1':
            val = Ib.abs().sum()
        elif self.bias_penalty == 'elastic':
            val = self.elastic_ratio * Ib.abs().sum() + (1.0 - self.elastic_ratio) * Ib.pow(2).sum()
        else:  # 'l2' (legacy default)
            val = Ib.norm(2)
        val = self.bias_lambda * val
        if self.normalize_regularization and batch_size:
            val = val / batch_size
        return val

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): (s, x), both shape (batch_size, n)
        Returns:
            torch.Tensor: Output of shape (batch_size, n)
        """
        s, x = inputs
        gamma_clamped = torch.exp(self.gamma).clamp(max=self.gamma_max)
        # I_clamped = torch.exp(torch.clamp(self.I, max=10.0))
        I_clamped = self.I
        return self.W(s) + I_clamped - gamma_clamped * x

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        criterion: str = "L1",
        scheduler_fn=None,
        scheduler_kwargs={},
        use_plateau_scheduler: bool = False,
        plateau_patience: int = 50,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-6,
        plateau_threshold: float = 1e-4,
        get_plots=False,
        display_epoch=100,
        verbose: bool = True,
        jacobian_lambda: float = 0.0,
    ):
        """
        Train the model.

        jacobian_lambda (default 0.0 = off): weight of the optional Jacobian-consistency
        term (requires configure_jacobian_consistency). Kept off by default: on the
        biophysical validation circuits it did not improve effective-GRN recovery with
        neighbor-estimated Jacobian targets (see FINDINGS M11); broad data coverage and
        the scaffold prior are the effective identifiability fixes.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        epochs : int, optional (default: 1000)
            Number of training epochs
        learning_rate : float, optional (default: 0.001)
            Initial learning rate
        criterion : str, optional (default: "L1")
            Loss function: "L1", "MSE", or "L2"
        scheduler_fn : callable, optional
            Learning rate scheduler function (ignored if use_plateau_scheduler=True)
        scheduler_kwargs : dict, optional
            Keyword arguments for scheduler_fn
        use_plateau_scheduler : bool, optional (default: False)
            If True, use ReduceLROnPlateau scheduler that decreases learning rate
            when the loss plateaus. This overrides scheduler_fn.
        plateau_patience : int, optional (default: 50)
            Number of epochs with no improvement after which learning rate will be reduced
        plateau_factor : float, optional (default: 0.5)
            Factor by which the learning rate will be reduced (new_lr = lr * factor)
        plateau_min_lr : float, optional (default: 1e-6)
            Minimum learning rate
        plateau_threshold : float, optional (default: 1e-4)
            Threshold for measuring the new optimum
        get_plots : bool, optional (default: False)
            Show training plots
        display_epoch : int, optional (default: 100)
            Display progress every N epochs
        verbose : bool, optional (default: True)
            Print training progress

        Returns
        -------
        tuple
            (loss_history, reconstruction_loss_history)
        """
        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        loss_mapping = {"L1": nn.L1Loss, "MSE": nn.MSELoss, "L2": nn.MSELoss}
        if criterion not in loss_mapping:
            raise ValueError(f"Invalid criterion: {criterion}. Choose from {list(loss_mapping.keys())}")
        loss_fn = loss_mapping[criterion]()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Set up scheduler
        if use_plateau_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=plateau_factor,
                patience=plateau_patience,
                threshold=plateau_threshold,
                min_lr=plateau_min_lr,
            )
            is_plateau_scheduler = True
        elif scheduler_fn is not None:
            scheduler = scheduler_fn(optimizer, **scheduler_kwargs)
            is_plateau_scheduler = False
        else:
            scheduler = None
            is_plateau_scheduler = False

        loss_history, reconstruction_loss_history = [], []
        lr_history = []

        mask_m = 1.0 - self.scaffold_raw

        for epoch in tqdm(range(epochs), desc="Training Epochs", disable=not verbose):
            epoch_loss, epoch_reconstruction_loss = 0.0, 0.0

            for (s_batch, x_batch), target in train_loader:
                s_batch, x_batch, target = s_batch.to(self.device), x_batch.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self((s_batch, x_batch))

                reconstruction_loss = self.reconstruction_lambda * loss_fn(output, target)

                # Normalize regularization losses by batch size if requested
                batch_size = s_batch.shape[0]
                if self.normalize_regularization:
                    graph_constr_loss = self.scaffold_lambda * ((self.W.weight * mask_m).norm(2) + (self.W.weight * mask_m).norm(1)) / batch_size
                else:
                    graph_constr_loss = self.scaffold_lambda * ((self.W.weight * mask_m).norm(2) + (self.W.weight * mask_m).norm(1))
                bias_loss = self._bias_loss(batch_size)

                total_loss = reconstruction_loss + graph_constr_loss + bias_loss
                if jacobian_lambda > 0 and self._jac is not None:
                    total_loss = total_loss + jacobian_lambda * self._jacobian_loss()

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()

            avg_loss = epoch_loss / len(train_loader)
            avg_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)
            loss_history.append(avg_loss)
            reconstruction_loss_history.append(avg_reconstruction_loss)

            # Step the scheduler
            if scheduler is not None:
                if is_plateau_scheduler:
                    # ReduceLROnPlateau needs the metric
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()

            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            if verbose and ((epoch % display_epoch == 0) or (epoch == epochs - 1)):
                tqdm.write(f"[Epoch {epoch+1}/{epochs}] "
                           f"Total Loss: {avg_loss:.6f}, "
                           f"Reconstruction Loss: {avg_reconstruction_loss:.6f}, "
                           f"LR: {current_lr:.2e}, "
                           f"Batch size: {s_batch.shape[0]}")

                if get_plots:
                    _,axs = plt.subplots(1,2, figsize=(10,5))

                    axs[0].scatter(output[0].detach().cpu().numpy().flatten(), target[0].detach().cpu().numpy().flatten())
                    x_min, x_max = axs[0].get_xlim()
                    y_min, y_max = axs[0].get_ylim()
                    min_val = min(x_min, y_min)
                    max_val = max(x_max, y_max)
                    axs[0].plot([min_val, max_val], [min_val, max_val], color='k', linestyle='--', linewidth=1)

                    axs[1].imshow(self.W.weight.detach().cpu().numpy())
                    plt.show()

        # Store learning rate history
        self.lr_history = lr_history

        return loss_history, reconstruction_loss_history




class HillScaffoldOptimizer(ScaffoldOptimizer):
    """ScaffoldOptimizer with per-gene trainable Hill parameters (k, n).

    Adds two new parameter vectors of shape (n_genes,):
      log_k -- per-gene Hill threshold (parameterized in log space)
      log_n -- per-gene Hill exponent  (parameterized in log space)

    The forward pass computes sigma_{k,n}(x) internally from raw x. The
    precomputed sigma input from train_loader is ignored. Convergence
    is the known-hard part of this model (variable exponent, variable in
    denominator, ratio); the defaults here are conservative:
      - hard clamp n in [n_min, n_max]
      - lower learning rate for log_k, log_n (hill_lr_factor * lr)
      - warmup_epochs where the Hill params are frozen
      - anchoring L2 regularization toward the initializers
    """

    def __init__(
        self,
        *args,
        k_init,
        n_init: float = 2.0,
        n_min: float = 1.0,
        n_max: float = 8.0,
        hill_anchor_lambda: float = 1e-2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        device = self.device
        n_genes = self.W.weight.shape[0]
        k_init_t = torch.as_tensor(k_init, dtype=torch.float32, device=device)
        if k_init_t.ndim == 0:
            k_init_t = k_init_t.expand(n_genes).clone()
        self.log_k = nn.Parameter(torch.log(k_init_t.clamp(min=1e-6)))
        self.log_n = nn.Parameter(torch.full((n_genes,), math.log(n_init),
                                              dtype=torch.float32, device=device))
        self.register_buffer("log_k_init", self.log_k.detach().clone())
        self.register_buffer("log_n_init", self.log_n.detach().clone())
        self.n_min = n_min
        self.n_max = n_max
        self.hill_anchor_lambda = hill_anchor_lambda
        self._hill_frozen = False

    def freeze_hill(self, frozen: bool = True):
        self._hill_frozen = frozen
        self.log_k.requires_grad_(not frozen)
        self.log_n.requires_grad_(not frozen)

    def hill(self, x):
        k = torch.exp(self.log_k)
        n = torch.exp(self.log_n).clamp(self.n_min, self.n_max)
        x_safe = x.clamp(min=1e-6)
        xn = x_safe.pow(n)
        kn = k.pow(n)
        return xn / (kn + xn)

    def forward(self, inputs):
        _, x = inputs  # precomputed sigma is ignored
        sig = self.hill(x)
        gamma_clamped = torch.exp(self.gamma).clamp(max=self.gamma_max)
        return self.W(sig) + self.I - gamma_clamped * x

    def hill_anchor_loss(self):
        return self.hill_anchor_lambda * (
            ((self.log_k - self.log_k_init) ** 2).mean()
            + ((self.log_n - self.log_n_init) ** 2).mean()
        )

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 1000,
        learning_rate: float = 1e-2,
        hill_lr_factor: float = 0.1,
        warmup_epochs: int = 200,
        criterion: str = "MSE",
        verbose: bool = True,
        display_epoch: int = 200,
        get_plots: bool = False,
    ):
        """Two-stage training: warmup with frozen Hill, then unlock.

        Uses separate Adam param-groups so log_k/log_n get a smaller LR.
        Adds an anchoring L2 toward the initial (k, n) values.
        """
        loss_mapping = {"L1": nn.L1Loss, "MSE": nn.MSELoss, "L2": nn.MSELoss}
        loss_fn = loss_mapping[criterion]()

        hill_params = [self.log_k, self.log_n]
        other_params = [p for n_, p in self.named_parameters()
                        if n_ not in ("log_k", "log_n") and p.requires_grad]
        optimizer = optim.Adam([
            {"params": other_params, "lr": learning_rate},
            {"params": hill_params,  "lr": learning_rate * hill_lr_factor},
        ])

        self.freeze_hill(True)
        loss_history, recon_history = [], []
        mask_m = 1.0 - self.scaffold_raw

        for epoch in tqdm(range(epochs), desc="Training Epochs", disable=not verbose):
            if epoch == warmup_epochs:
                self.freeze_hill(False)

            epoch_loss = 0.0
            epoch_recon = 0.0
            for (s_batch, x_batch), target in train_loader:
                s_batch, x_batch, target = (t.to(self.device) for t in (s_batch, x_batch, target))
                optimizer.zero_grad()
                output = self((s_batch, x_batch))
                reconstruction_loss = self.reconstruction_lambda * loss_fn(output, target)
                bs = s_batch.shape[0]
                if self.normalize_regularization:
                    graph_constr_loss = self.scaffold_lambda * ((self.W.weight * mask_m).norm(2) + (self.W.weight * mask_m).norm(1)) / bs
                else:
                    graph_constr_loss = self.scaffold_lambda * ((self.W.weight * mask_m).norm(2) + (self.W.weight * mask_m).norm(1))
                bias_loss = self._bias_loss(bs)
                total_loss = reconstruction_loss + graph_constr_loss + bias_loss + self.hill_anchor_loss()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                epoch_recon += reconstruction_loss.item()

            loss_history.append(epoch_loss / len(train_loader))
            recon_history.append(epoch_recon / len(train_loader))

            if verbose and ((epoch % display_epoch == 0) or (epoch == epochs - 1)):
                k_now = torch.exp(self.log_k).detach().cpu().numpy()
                n_now = torch.exp(self.log_n).clamp(self.n_min, self.n_max).detach().cpu().numpy()
                tqdm.write(f"[E{epoch+1}/{epochs}] loss={loss_history[-1]:.4g} "
                           f"recon={recon_history[-1]:.4g} "
                           f"k=[{k_now.min():.3g}, {k_now.max():.3g}] "
                           f"n=[{n_now.min():.2f}, {n_now.max():.2f}] "
                           f"frozen={self._hill_frozen}")

        return loss_history, recon_history
