import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from unittest.mock import patch
from schopfield._core.landscape import Landscape
from schopfield.tools.analysis import get_energies
from schopfield.tools.correlation import (
    energy_genes_correlation,
    celltype_correlation,
    network_correlations
)

# Mock hoggorm.mat_corr_coeff
class MockMatCorrCoeff:
    @staticmethod
    def RVcoeff(matrices):
        return np.array([[1.0, 0.5], [0.5, 1.0]])
    @staticmethod
    def RV2coeff(matrices):
        return MockMatCorrCoeff.RVcoeff(matrices)


def test_energy_genes_correlation(sample_landscape):
    get_energies(sample_landscape)
    landscape = sample_landscape
    energy_genes_correlation(landscape)
    assert 'A' in landscape.correlation
    assert landscape.correlation['A'].shape == (len(landscape.genes),)

@patch('schopfield.tools.correlation.hoggorm.mat_corr_coeff', MockMatCorrCoeff)
def test_celltype_correlation(sample_landscape):
    landscape = sample_landscape
    celltype_correlation(landscape, modified=True, all_genes=False)
    assert isinstance(landscape.cells_correlation, pd.DataFrame)
    assert landscape.cells_correlation.shape == (2, 2)
    assert landscape.cells_correlation.loc['A', 'B'] == landscape.cells_correlation.loc['B', 'A']

def test_network_correlations(sample_landscape):
    landscape = sample_landscape
    network_correlations(landscape)
    assert all(hasattr(landscape, attr) for attr in [
        'jaccard', 'hamming', 'euclidean', 'pearson', 'pearson_bin', 'mean_col_corr', 'singular'
    ])
    assert landscape.jaccard.shape == (2, 2)
    assert landscape.pearson.loc['A', 'B'] == landscape.pearson.loc['B', 'A']