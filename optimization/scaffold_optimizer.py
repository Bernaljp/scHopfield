"""
Scaffold-based optimization functionality for scHopfield package.
Contains the ScaffoldOptimizer class and related optimization utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Callable, Dict, Any

from ..core.base_models import BaseOptimizer


class MaskedLinearLayer(nn.Module):
    """
    Linear layer with masked weights for scaffold-based optimization.
    """

    def __init__(self, input_size: int, output_size: int, mask: torch.Tensor,
                 device: torch.device, pre_initialized_W: Optional[torch.Tensor] = None):
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

    def _apply_mask(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply mask to gradients during backpropagation."""
        return grad * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ScaffoldOptimizer(nn.Module, BaseOptimizer):
    """
    A model that learns: output = W(s) + I - clamp(gamma, 0) * x
    with a scaffold-based regularization on W.

    This optimizer implements the core learning algorithm for inferring
    interaction matrices and bias vectors in the Hopfield-like system.
    """

    def __init__(self,
                 g: torch.Tensor,
                 scaffold: torch.Tensor,
                 device: torch.device,
                 refit_gamma: bool = False,
                 scaffold_regularization: float = 1.0,
                 use_masked_linear: bool = False,
                 pre_initialized_W: Optional[torch.Tensor] = None,
                 pre_initialized_I: Optional[torch.Tensor] = None):
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

    def forward(self, inputs: tuple) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs: Tuple of (s, x), both shape (batch_size, n)

        Returns:
            Output tensor of shape (batch_size, n)
        """
        s, x = inputs
        gamma_clamped = torch.exp(torch.clamp(self.gamma, max=10.0))
        I_clamped = self.I
        return self.W(s) + I_clamped - gamma_clamped * x

    def train_model(self,
                    train_loader: DataLoader,
                    epochs: int = 1000,
                    learning_rate: float = 0.001,
                    criterion: str = "L1",
                    scheduler_fn: Optional[Callable] = None,
                    scheduler_kwargs: Optional[Dict[str, Any]] = None,
                    get_plots: bool = False) -> None:
        """
        Train the scaffold optimizer model.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            criterion: Loss criterion ("L1" or "L2")
            scheduler_fn: Optional learning rate scheduler function
            scheduler_kwargs: Keywords arguments for scheduler
            get_plots: Whether to generate training plots
        """
        # Set up loss function
        if criterion.upper() == "L1":
            loss_fn = nn.L1Loss()
        else:
            loss_fn = nn.MSELoss()

        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Set up scheduler if provided
        scheduler = None
        if scheduler_fn is not None:
            scheduler_kwargs = scheduler_kwargs or {}
            scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Training loop
        losses = []
        self.train()

        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_loss = 0.0

            for batch_idx, ((sig, x), v) in enumerate(train_loader):
                sig, v, x = sig.to(self.device), v.to(self.device), x.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = self.forward((sig, x))

                # Compute loss
                loss = loss_fn(output, v)

                # Add scaffold regularization
                if hasattr(self.W, 'weight'):
                    scaffold_loss = torch.norm(self.W.weight - self.scaffold_raw) * self.scaffold_lambda
                    loss += scaffold_loss

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            losses.append(epoch_loss / len(train_loader))

        self.eval()

        # Generate plots if requested
        if get_plots:
            self._plot_training_curves(losses)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit the model to data (required by BaseOptimizer).

        This method provides a wrapper around train_model for compatibility.
        """
        # Create a simple dataset and dataloader
        import torch
        device = self.device
        dataset = SimpleDataset(torch.tensor(x, dtype=torch.float32, device=device),
                               torch.tensor(y, dtype=torch.float32, device=device))
        train_loader = DataLoader(dataset, batch_size=kwargs.get('batch_size', 64), shuffle=True)

        # Train the model
        epochs = kwargs.get('epochs', 1000)
        learning_rate = kwargs.get('learning_rate', 0.001)
        criterion = kwargs.get('criterion', 'L2')

        self.train_model(train_loader, epochs, learning_rate, criterion)

        return {'training_completed': True}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model (required by BaseOptimizer).
        """
        import torch
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            # Assuming x contains sigmoid values, predict velocities
            predictions = self.forward(x_tensor, x_tensor)  # Using same input for both sig and x
        return predictions.cpu().numpy()

    def _plot_training_curves(self, losses: list) -> None:
        """Plot training loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


class CustomDataset(Dataset):
    """
    Custom PyTorch dataset for scaffold optimization training.
    """

    def __init__(self, sig: torch.Tensor, v: torch.Tensor, x: torch.Tensor, device: torch.device):
        self.sig = torch.tensor(sig, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.sig)

    def __getitem__(self, idx: int) -> tuple:
        return (self.sig[idx], self.x[idx]), self.v[idx]


class SimpleDataset(Dataset):
    """
    Simple PyTorch dataset for general use.
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


class InteractionMatrixOptimizer(BaseOptimizer):
    """
    Optimizer for inferring interaction matrices using various optimization strategies.
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, device: str = 'cpu'):
        super().__init__(data, targets)
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")
        self.data = data.to(self.device)
        self.targets = targets.to(self.device)

    def optimize(self, method: str = 'adam', **kwargs) -> torch.Tensor:
        """
        Optimize the interaction matrix using the specified method.

        Args:
            method: Optimization method to use
            **kwargs: Additional arguments for the optimization method

        Returns:
            Optimized interaction matrix
        """
        if method.lower() == 'adam':
            return self._optimize_adam(**kwargs)
        elif method.lower() == 'lbfgs':
            return self._optimize_lbfgs(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _optimize_adam(self, lr: float = 0.001, epochs: int = 1000) -> torch.Tensor:
        """Optimize using Adam optimizer."""
        # Implementation would go here
        pass

    def _optimize_lbfgs(self, max_iter: int = 100) -> torch.Tensor:
        """Optimize using L-BFGS optimizer."""
        # Implementation would go here
        pass