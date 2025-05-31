import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def plot_embedding(
    adata: AnnData,
    embedding_key: str = "X_umap",
    color_by: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Embedding",
    **kwargs
) -> plt.Figure:
    """Plot low-dimensional embedding of single-cell data.

    Args:
        adata: AnnData object with embedding in adata.obsm.
        embedding_key: Key for embedding in adata.obsm.
        color_by: Column in adata.obs or adata.var to color points (e.g., "cluster").
        ax: Matplotlib Axes object to plot on. If None, creates new figure.
        title: Plot title.
        **kwargs: Additional arguments for scatter plot (e.g., s, cmap).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    logger.info(f"Plotting embedding from {embedding_key}")
    
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")
    
    embedding = adata.obsm[embedding_key]
    if embedding.shape[1] < 2:
        raise ValueError("Embedding must have at least 2 dimensions")
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    if color_by:
        if color_by in adata.obs:
            colors = adata.obs[color_by].values
        elif color_by in adata.var:
            colors = adata.var[color_by].values
        else:
            raise ValueError(f"Color key '{color_by}' not found in adata.obs or adata.var")
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=colors, ax=ax, **kwargs)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], **kwargs)
    
    ax.set_xlabel(f"{embedding_key} 1")
    ax.set_ylabel(f"{embedding_key} 2")
    ax.set_title(title)
    plt.tight_layout()
    
    return fig