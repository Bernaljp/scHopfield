import pytest
import numpy as np
from schopfield.preprocessing.embedding import get_embedding

def test_get_embedding(sample_landscape):
    adata = get_embedding(sample_landscape, method="umap", n_components=2, key_added="X_umap")
    assert "X_umap" in adata.obsm
    assert adata.obsm["X_umap"].shape == (100, 2)
    
    adata = get_embedding(sample_landscape, method="pca", layer="Ms", key_added="X_pca")
    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape == (100, 2)
    
    with pytest.raises(ValueError):
        get_embedding(sample_landscape, method="invalid")