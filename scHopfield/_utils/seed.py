"""Global reproducibility helpers.

scHopfield's model fitting draws on three independent RNGs: Python's ``random``,
NumPy's global RNG (used by the neighbor batch sampler), and PyTorch's global RNG
(used for weight initialization and ``DataLoader`` shuffling). Seeding all three
from a single call makes ``fit_interactions`` / ``compute_umap`` reproducible.
"""
import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Parameters
    ----------
    seed : int, optional (default: 0)
        Seed applied to every RNG.
    deterministic : bool, optional (default: False)
        If True, also request deterministic cuDNN kernels
        (``torch.backends.cudnn.deterministic = True``, ``benchmark = False``).
        This can slow down GPU training and is off by default.

    Returns
    -------
    int
        The seed that was applied (useful for logging).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader`` so each worker is seeded reproducibly."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: Optional[int]):
    """Return a seeded ``torch.Generator`` (or None if seed is None)."""
    if seed is None:
        return None
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
