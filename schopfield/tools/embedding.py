import numpy as np
from typing import Optional, Dict
import logging
import pickle
from anndata import AnnData
from schopfield._core.landscape import Landscape
from schopfield.utils.data import to_numpy
from schopfield.preprocessing.embedding import get_embedding
from schopfield.tools.analysis import compute_energies
from schopfield.utils.math import soften

logger = logging.getLogger(__name__)

def energy_embedding(landscape: 'Landscape', which: str = 'UMAP', resolution: int = 50, **kwargs) -> None:
    """Compute and visualize the energy embedding for the dataset.

    Generates a grid-based energy landscape for each cluster using the specified embedding method,
    computes energies for grid points, and stores results in the Landscape object. Also computes
    cell velocities using dynamo.

    Args:
        landscape: Landscape object containing adata, W, and parameters.
        which: The embedding method used (e.g., 'UMAP', 'PCA'). Defaults to 'UMAP'.
        resolution: Resolution of the grid for energy computation (higher values mean finer grids). Defaults to 50.
        **kwargs: Additional keyword arguments for the embedding method (e.g., n_neighbors for UMAP).

    Raises:
        ValueError: If W, spliced_matrix_key, velocity_key, or cluster_key are not initialized.
        ImportError: If dynamo is not installed for velocity computation.

    Notes:
        Requires fitted parameters from schopfield.tools.fitting.fit_interactions.
        Stores results in landscape.grid_X, grid_Y, grid_energy, grid_energy_interaction,
        grid_energy_degradation, grid_energy_bias, and adata.obsm[f'X_{which}'].
        Depends on get_embedding from schopfield.preprocessing.embedding and compute_energies
        from schopfield.tools.analysis.
    """
    logger.info(f"Computing energy embedding with method: {which}, resolution: {resolution}")

    # Validate parameters
    if not landscape.W:
        raise ValueError("Interaction parameters not initialized; run schopfield.tools.fitting.fit_interactions")
    if not all([landscape.spliced_matrix_key, landscape.velocity_key, landscape.cluster_key]):
        raise ValueError("Required keys (spliced_matrix_key, velocity_key, cluster_key) not initialized")

    # Compute the embedding
    get_embedding(landscape, which=which, **kwargs)

    # Initialize dictionaries for grid coordinates
    grid_X, grid_Y = {}, {}

    # Retrieve 2D coordinates from the embedding
    cells2d = landscape.adata.obsm[f'X_{which}']

    # Generate grids for each cluster
    for k in landscape.W:
        cidx = (landscape.adata.obs[landscape.cluster_key] == k
                if k != 'all' else np.arange(landscape.adata.n_obs))
        minx, miny = np.min(cells2d[cidx], axis=0)
        maxx, maxy = np.max(cells2d[cidx], axis=0)
        grid_X[k], grid_Y[k] = np.mgrid[minx:maxx:resolution*1j, miny:maxy:resolution*1j]

    # Transform grid points to high-dimensional space
    grid_points = np.vstack([grid_X[k].ravel() for k in landscape.W] +
                           [grid_Y[k].ravel() for k in landscape.W]).T
    try:
        highD_grid = landscape.embedding.inverse_transform(grid_points)
    except AttributeError:
        raise ValueError(f"Embedding method {which} does not support inverse_transform")

    # Ensure non-negative values
    highD_grid = np.maximum(highD_grid, 0)

    # Compute energies for grid points
    energies = compute_energies(landscape, x=highD_grid)
    E, inter, deg, bias = (energies['total'], energies['interaction'],
                          energies['degradation'], energies['bias'])

    # Initialize dictionaries for softened energies
    Es, inters, degs, biases = {}, {}, {}, {}

    # Soften and reshape energies for each cluster
    for i, k in enumerate(landscape.W):
        reshape_slice = slice(i * resolution**2, (i + 1) * resolution**2)
        Es[k] = soften(E[k][reshape_slice].reshape(grid_X[k].shape))
        inters[k] = soften(inter[k][reshape_slice].reshape(grid_X[k].shape))
        degs[k] = soften(deg[k][reshape_slice].reshape(grid_X[k].shape))
        biases[k] = soften(bias[k][reshape_slice].reshape(grid_X[k].shape))

    # Update landscape attributes
    landscape.grid_X, landscape.grid_Y = grid_X, grid_Y
    landscape.highD_grid = highD_grid
    landscape.grid_energy = Es
    landscape.grid_energy_interaction = inters
    landscape.grid_energy_degradation = degs
    landscape.grid_energy_bias = biases

    # Compute cell velocities
    try:
        import dynamo as dyn
        dyn.tl.cell_velocities(
            landscape.adata,
            X=landscape.adata.layers[landscape.spliced_matrix_key],
            V=landscape.adata.layers[landscape.velocity_key],
            X_embedding=landscape.adata.obsm[f'X_{which}'],
            add_velocity_key=f'velocity_{which}'
        )
    except ImportError:
        logger.warning("Dynamo not installed; skipping cell velocity computation")
        raise ImportError("Install dynamo with 'pip install dynamo' for velocity computation")

def save_embedding(landscape: 'Landscape', filename: str) -> None:
    """Save the embedding and grid coordinates to a pickle file.

    Stores the embedding transformer and grid coordinates for later use.

    Args:
        landscape: Landscape object containing embedding and grid attributes.
        filename: Path to the file where the embedding and grid coordinates will be saved.

    Raises:
        ValueError: If embedding or grid attributes are not initialized.
        IOError: If file writing fails.

    Notes:
        Saves landscape.embedding, grid_X, grid_Y, and highD_grid.
    """
    logger.info(f"Saving embedding to file: {filename}")

    # Validate attributes
    if not hasattr(landscape, 'embedding') or not all([landscape.grid_X, landscape.grid_Y, landscape.highD_grid]):
        raise ValueError("Embedding or grid attributes not initialized; run energy_embedding first")

    # Create dictionary to save
    emb = {
        'embedding': landscape.embedding,
        'grid_X': landscape.grid_X,
        'grid_Y': landscape.grid_Y,
        'highD_grid': landscape.highD_grid
    }

    # Save to file
    try:
        with open(filename, 'wb') as outp:
            pickle.dump(emb, outp, pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        logger.error(f"Failed to save embedding to {filename}: {e}")
        raise

def load_embedding(landscape: 'Landscape', filename: str, which: str = 'UMAP', resolution: int = 50) -> None:
    """Load embedding and grid coordinates from a pickle file and recalculate grid energies.

    Updates the Landscape object with the loaded embedding and recomputes energy landscapes
    using the stored high-dimensional grid points.

    Args:
        landscape: Landscape object to update with loaded embedding.
        filename: Path to the file containing the saved embedding and grid coordinates.
        which: Key under which to store the embedding in adata.obsm. Defaults to 'UMAP'.
        resolution: Resolution used for grid generation. Defaults to 50.

    Raises:
        ValueError: If W, spliced_matrix_key, or genes are not initialized.
        FileNotFoundError: If the specified file does not exist.
        IOError: If file reading fails.
        KeyError: If expected keys are missing in the loaded dictionary.

    Notes:
        Updates landscape.embedding, grid_X, grid_Y, highD_grid, grid_energy,
        grid_energy_interaction, grid_energy_degradation, grid_energy_bias,
        and adata.obsm[f'X_{which}'].
    """
    logger.info(f"Loading embedding from file: {filename}")

    # Validate parameters
    if not landscape.W or not landscape.spliced_matrix_key or not landscape.genes:
        raise ValueError("Required parameters (W, spliced_matrix_key, genes) not initialized")

    # Load the embedding
    try:
        with open(filename, 'rb') as inp:
            emb = pickle.load(inp)
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        raise
    except IOError as e:
        logger.error(f"Failed to load embedding from {filename}: {e}")
        raise

    # Validate loaded dictionary
    required_keys = ['embedding', 'grid_X', 'grid_Y', 'highD_grid']
    if not all(k in emb for k in required_keys):
        missing = [k for k in required_keys if k not in emb]
        raise KeyError(f"Loaded file missing required keys: {missing}")

    # Update landscape attributes
    landscape.embedding = emb['embedding']
    landscape.grid_X = emb['grid_X']
    landscape.grid_Y = emb['grid_Y']
    landscape.highD_grid = emb['highD_grid']

    # Transform cells to embedding space
    X = to_numpy(get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes))
    try:
        cells2d = landscape.embedding.transform(X)
    except AttributeError:
        raise ValueError(f"Embedding method {which} does not support transform")
    landscape.adata.obsm[f'X_{which}'] = cells2d

    # Recalculate grid energies
    energies = compute_energies(landscape, x=landscape.highD_grid)
    E, inter, deg, bias = (energies['total'], energies['interaction'],
                          energies['degradation'], energies['bias'])

    # Initialize dictionaries for softened energies
    Es, inters, degs, biases = {}, {}, {}, {}

    # Soften and reshape energies
    for i, k in enumerate(landscape.W):
        reshape_slice = slice(i * resolution**2, (i + 1) * resolution**2)
        Es[k] = soften(E[k][reshape_slice].reshape(landscape.grid_X[k].shape))
        inters[k] = soften(inter[k][reshape_slice].reshape(landscape.grid_X[k].shape))
        degs[k] = soften(deg[k][reshape_slice].reshape(landscape.grid_X[k].shape))
        biases[k] = soften(bias[k][reshape_slice].reshape(landscape.grid_X[k].shape))

    # Update grid energy attributes
    landscape.grid_energy = Es
    landscape.grid_energy_interaction = inters
    landscape.grid_energy_degradation = degs
    landscape.grid_energy_bias = biases