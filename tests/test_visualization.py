import pytest
from schopfield.plotting.visualization import plot_embedding
from schopfield.preprocessing.embedding import get_embedding

def test_plot_embedding(sample_landscape):
    adata = get_embedding(sample_landscape, method="umap", key_added="X_umap")
    fig = plot_embedding(adata, embedding_key="X_umap", color_by="cluster")
    assert fig is not None
    
    with pytest.raises(ValueError):
        plot_embedding(adata, embedding_key="invalid_key")
    with pytest.raises(ValueError):
        plot_embedding(adata, color_by="invalid_column")