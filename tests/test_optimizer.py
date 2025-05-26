import pytest
import torch
import numpy as np
from schopfield.optimization.optimizer import MaskedLinearLayer, ScaffoldOptimizer, CustomDataset

def test_masked_linear_layer():
    """Test MaskedLinearLayer initialization and forward pass."""
    device = torch.device("cpu")
    mask = np.eye(5)
    layer = MaskedLinearLayer(5, 5, mask, device)
    
    x = torch.ones((10, 5))
    output = layer(x)
    assert output.shape == (10, 5)
    assert torch.all(layer.weight * (1 - torch.tensor(mask)) == 0)

def test_scaffold_optimizer():
    """Test ScaffoldOptimizer initialization and forward pass."""
    device = torch.device("cpu")
    g = np.ones(5)
    scaffold = np.eye(5)
    model = ScaffoldOptimizer(g, scaffold, device, refit_gamma=True)
    
    s = torch.ones((10, 5))
    x = torch.ones((10, 5))
    output = model((s, x))
    assert output.shape == (10, 5)
    assert model.gamma.shape == (5,)
    assert model.W.weight.shape == (5, 5)

def test_custom_dataset():
    """Test CustomDataset initialization and access."""
    device = torch.device("cpu")
    s = np.ones((100, 5))
    v = np.zeros((100, 5))
    x = np.ones((100, 5))
    
    dataset = CustomDataset(s, v, x, device)
    assert len(dataset) == 100
    (s_item, x_item), v_item = dataset[0]
    assert s_item.shape == (5,)
    assert x_item.shape == (5,)
    assert v_item.shape == (5,)
    
    with pytest.raises(ValueError):
        CustomDataset(s, v[:50], x, device)  # Shape mismatch