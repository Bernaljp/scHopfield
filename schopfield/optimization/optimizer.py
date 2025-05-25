import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from typing import Optional, Dict, Union, Tuple, List

logger = logging.getLogger(__name__)

class MaskedLinearLayer(nn.Module):
    """Linear layer with masked weights for selective connectivity.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        mask: Binary mask for weights (1 to keep, 0 to mask).
        device: PyTorch device ('cpu' or 'cuda').
        pre_initialized_W: Optional pre-initialized weight matrix.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        mask: Union[np.ndarray, torch.Tensor],
        device: torch.device,
        pre_initialized_W: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False, device=device)
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, device=device))

        n_in_mask = (mask.sum(dim=0) > 0).sum().sqrt()
        if pre_initialized_W is None:
            nn.init.uniform_(self.linear.weight, -1 / n_in_mask, 1 / n_in_mask)
        else:
            self.linear.weight = nn.Parameter(
                torch.tensor(pre_initialized_W, dtype=torch.float32, device=device)
            )

        with torch.no_grad():
            self.linear.weight *= self.mask

        self.weight = self.linear.weight
        self.linear.weight.register_hook(self._apply_mask)

    def _apply_mask(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaffoldOptimizer(nn.Module):
    """Model for learning Hopfield dynamics with scaffold-based regularization.

    Computes output = W(s) + I - clamp(gamma, 0) * x, where W is the interaction
    matrix, s is the sigmoid activation, I is the bias, and gamma is the degradation rate.

    Args:
        g: Initial degradation rates (n_genes,).
        scaffold: Scaffold matrix for regularization (n_genes, n_genes).
        device: PyTorch device ('cpu' or 'cuda').
        refit_gamma: If True, gamma is learnable.
        scaffold_regularization: Regularization strength for scaffold.
        use_masked_linear: If True, uses MaskedLinearLayer for W.
        pre_initialized_W: Optional pre-initialized W matrix.
        pre_initialized_I: Optional pre-initialized bias vector.
        bias_regularization: Regularization strength for bias terms.
    """
    def __init__(
        self,
        g: Union[np.ndarray, torch.Tensor],
        scaffold: Union[np.ndarray, torch.Tensor],
        device: torch.device,
        refit_gamma: bool = False,
        scaffold_regularization: float = 1.0,
        use_masked_linear: bool = False,
        pre_initialized_W: Optional[Union[np.ndarray, torch.Tensor]] = None,
        pre_initialized_I: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bias_regularization: float = 10.0,
    ):
        super().__init__()
        self.device = device

        g = torch.log(torch.tensor(g, dtype=torch.float32, device=device) + 1e-8)
        self.gamma = nn.Parameter(g) if refit_gamma else g

        scaffold = torch.tensor(scaffold, dtype=torch.float32, device=device)
        self.register_buffer("scaffold_raw", scaffold)

        scaffold_tfs = torch.zeros_like(scaffold)
        with torch.no_grad():
            scaffold_tfs[:, scaffold.sum(dim=0) > 0] = 1
        self.register_buffer("scaffold", scaffold_tfs)

        self.scaffold_lambda = scaffold_regularization
        n = g.shape[0]

        init_I = (torch.rand((n,), device=device) if pre_initialized_I is None else
                  torch.tensor(pre_initialized_I, dtype=torch.float32, device=device))
        self.I = nn.Parameter(init_I)
        self.bias_regularization = bias_regularization

        if use_masked_linear:
            self.W = MaskedLinearLayer(n, n, self.scaffold, device=device, pre_initialized_W=pre_initialized_W)
        else:
            self.W = nn.Linear(n, n, bias=False, device=device)
            if pre_initialized_W is not None:
                self.W.weight = nn.Parameter(
                    torch.tensor(pre_initialized_W, dtype=torch.float32, device=device)
                )
            else:
                nn.init.xavier_uniform_(self.W.weight)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute model output.

        Args:
            inputs: Tuple of (s, x), where s is sigmoid activation (batch_size, n_genes)
                and x is expression data (batch_size, n_genes).

        Returns:
            torch.Tensor: Output (batch_size, n_genes).
        """
        s, x = inputs
        gamma_clamped = torch.exp(torch.clamp(self.gamma, max=10.0))
        I_clamped = self.I
        return self.W(s) + I_clamped - gamma_clamped * x

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        criterion: str = "L1",
        scheduler_fn: Optional[callable] = None,
        scheduler_kwargs: Optional[Dict] = None,
        get_plots: bool = False,
        display_epoch: int = 100,
    ) -> Tuple[List[float], List[float]]:
        """Train the model using the provided data loader.

        Args:
            train_loader: DataLoader with (s, x), target pairs.
            epochs: Number of training epochs.
            learning_rate: Learning rate for Adam optimizer.
            criterion: Loss function ('L1', 'MSE', 'L2').
            scheduler_fn: Optional learning rate scheduler function.
            scheduler_kwargs: Arguments for scheduler.
            get_plots: If True, logs loss information (plotting deferred to schopfield.plotting).
            display_epoch: Frequency of logging progress.

        Returns:
            Tuple[List[float], List[float]]: Total loss and reconstruction loss histories.
        """
        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        loss_mapping = {"L1": nn.L1Loss, "MSE": nn.MSELoss, "L2": nn.MSELoss}
        if criterion not in loss_mapping:
            raise ValueError(f"Invalid criterion: {criterion}. Choose from {list(loss_mapping.keys())}")
        loss_fn = loss_mapping[criterion]()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = scheduler_fn(optimizer, **scheduler_kwargs) if scheduler_fn else None

        loss_history, reconstruction_loss_history = [], []

        mask_m = 1.0 - self.scaffold_raw
        for epoch in range(epochs):
            epoch_loss, epoch_reconstruction_loss = 0.0, 0.0

            for (s_batch, x_batch), target in train_loader:
                s_batch, x_batch, target = (
                    s_batch.to(self.device),
                    x_batch.to(self.device),
                    target.to(self.device)
                )

                optimizer.zero_grad()
                output = self.forward((s_batch, x_batch))

                reconstruction_loss = loss_fn(output, target)
                graph_constr_loss = self.scaffold_lambda * (
                    (self.W.weight * mask_m).norm(2) + (self.W.weight * mask_m).norm(1)
                )
                bias_loss = self.bias_regularization * torch.abs(self.I).norm(2)
                total_loss = reconstruction_loss + graph_constr_loss + bias_loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            avg_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)
            loss_history.append(avg_loss)
            reconstruction_loss_history.append(avg_reconstruction_loss)

            if (epoch % display_epoch == 0) or (epoch == epochs - 1):
                logger.info(
                    f"Epoch: {epoch+1}/{epochs}, "
                    f"Total Loss: {avg_loss:.6f}, "
                    f"Reconstruction Loss: {avg_reconstruction_loss:.6f}, "
                    f"Batch size: {s_batch.shape[0]}"
                )

        return loss_history, reconstruction_loss_history

class CustomDataset(Dataset):
    """Dataset for Hopfield model training.

    Args:
        s: Sigmoid activations (n_cells, n_genes).
        v: Velocity data (n_cells, n_genes).
        x: Expression data (n_cells, n_genes).
        device: PyTorch device ('cpu' or 'cuda').
    """
    def __init__(
        self,
        s: Union[np.ndarray, torch.Tensor],
        v: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        device: torch.device
    ):
        self.s = torch.tensor(s, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        
        # Validate shapes
        if self.s.shape != self.v.shape or self.s.shape != self.x.shape:
            raise ValueError(
                f"Shape mismatch: s {self.s.shape}, v {self.v.shape}, x {self.x.shape}"
            )

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (self.s[idx], self.x[idx]), self.v[idx]