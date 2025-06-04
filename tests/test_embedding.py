import pytest
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from schopfield._core.landscape import Landscape
from schopfield.tools.embedding import energy_embedding, save_embedding, load_embedding
import os


def test_energy_embedding(sample_landscape):
    landscape = sample_landscape
    energy_embedding(landscape, which='UMAP', resolution=10, n_neighbors=5)

    assert hasattr(landscape, 'grid_X') and len(landscape.grid_X) == 3
    assert 'A' in landscape.grid_energy
    assert landscape.grid_energy['A'].shape == (10, 10)
    assert f'X_umap' in landscape.adata.obsm

def test_save_load_embedding(sample_landscape, tmp_path):
    landscape = sample_landscape
    energy_embedding(landscape, which='UMAP', resolution=10)
    filename = tmp_path / 'embedding.pkl'
    save_embedding(landscape, str(filename))
    assert os.path.exists(filename)
    
    new_landscape = Landscape(
        landscape.adata.copy(),
        spliced_matrix_key='Ms',
        velocity_key='velocity_S',
        gamma_key='gamma',
        cluster_key='cluster',
        genes=np.arange(10)
    )
    new_landscape.W = landscape.W
    new_landscape.I = landscape.I
    new_landscape.gamma = landscape.gamma
    new_landscape.threshold = landscape.threshold
    new_landscape.exponent = landscape.exponent
    load_embedding(new_landscape, str(filename), which='UMAP', resolution=10)
    assert np.allclose(new_landscape.grid_X['A'], landscape.grid_X['A'])
    assert f'X_UMAP' in new_landscape.adata.obsm

def test_load_embedding_invalid_file(sample_landscape):
    landscape = sample_landscape
    with pytest.raises(FileNotFoundError):
        load_embedding(landscape, 'nonexistent.pkl')