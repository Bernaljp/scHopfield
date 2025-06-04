import numpy as np
import anndata as ad
import logging
from sklearn.preprocessing import StandardScaler
import umap
from typing import Optional

logger = logging.getLogger(__name__)

def get_embedding(
    landscape: "Landscape",
    method: str = "umap",
    n_components: int = 2,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
    **kwargs
) -> ad.AnnData:
    """Compute low-dimensional embedding of single-cell data.

    Args:
        landscape: Landscape object containing single-cell data.
        method: Embedding method ("umap" or "pca").
        n_components: Number of embedding dimensions.
        layer: Layer to use (e.g., "spliced"). If None, uses adata.X.
        key_added: Key to store embedding in adata.obsm.
        **kwargs: Additional arguments for embedding method (e.g., n_neighbors for UMAP).

    Returns:
        ad.AnnData: Updated AnnData with embedding in adata.obsm[key_added].
    """
    logger.info(f"Computing {method} embedding with {n_components} components")
    adata = landscape.adata
    # Extract data
    X = adata.layers[layer] if layer else adata.X
    X = X.toarray() if X.__class__.__name__ == 'csr_matrix' else np.asarray(X)
    
    # Normalize data
    X = StandardScaler().fit_transform(X)
    
    # Compute embedding
    if method.lower() == "umap":
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(X)
    elif method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")
    
    # Store embedding
    landscape.embedding = reducer
    key_added = key_added or f"X_{method.lower()}"
    adata.obsm[key_added] = embedding
    
    logger.info(f"Stored embedding in adata.obsm['{key_added}']")
    
    return adata