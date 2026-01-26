# scHopfield Installation Guide

## Installation Methods

### Method 1: Development Installation (Recommended for local development)

From the project directory:

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install -e .
```

The `-e` flag installs in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

### Method 2: Standard Installation

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install .
```

### Method 3: With Optional Dependencies

For enhanced functionality (seaborn, igraph, dynamo):
```bash
pip install -e ".[optional]"
```

For development tools:
```bash
pip install -e ".[dev]"
```

For documentation building:
```bash
pip install -e ".[docs]"
```

For all optional features:
```bash
pip install -e ".[all,dev,docs]"
```

## Verify Installation

After installation, verify it works:

```python
import scHopfield as sch
print(sch.__version__)  # Should print: 0.1.0

# Check available modules
print(dir(sch))  # Should show: pp, inf, tl, pl, dyn, etc.
```

## Building a Distribution Package

To create a distributable package:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# - dist/scHopfield-0.1.0.tar.gz (source distribution)
# - dist/scHopfield-0.1.0-py3-none-any.whl (wheel distribution)
```

## Installing from Distribution

```bash
pip install dist/scHopfield-0.1.0-py3-none-any.whl
```

## Uninstall

```bash
pip uninstall scHopfield
```

## System Requirements

- **Python**: >= 3.8
- **OS**: Linux, macOS, Windows
- **Memory**: Recommended 16GB+ RAM for large datasets
- **GPU**: Optional (CUDA-compatible GPU for faster training with device='cuda')

## Dependencies

Core dependencies are automatically installed:
- numpy, scipy, pandas (numerical computing)
- matplotlib, seaborn (visualization)
- anndata, scanpy (single-cell analysis)
- torch (deep learning)
- networkx (network analysis and graph layouts)
- umap-learn, scikit-learn (dimensionality reduction)
- tqdm (progress bars)
- h5py (HDF5 file handling)
- hoggorm (multivariate analysis)

**Recommended optional dependencies:**
- `python-igraph` - For 10-100× faster network centrality computation on large networks
  ```bash
  pip install igraph
  ```

## Troubleshooting

### Issue: PyTorch installation fails

Install PyTorch separately first:
```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA version (replace cu118 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then install scHopfield:
```bash
pip install -e .
```

### Issue: UMAP installation fails

Try:
```bash
pip install umap-learn --no-cache-dir
```

### Issue: igraph installation fails

python-igraph requires C libraries. Try:

**On macOS:**
```bash
brew install igraph
pip install python-igraph
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install libigraph0-dev
pip install python-igraph
```

**On Windows:**
```bash
pip install python-igraph
# If that fails, download pre-built wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph
```

**Alternative:** The package will work fine without igraph, it will just use networkx (slower for large networks)

### Issue: Import errors

Make sure you're in a clean Python environment:
```bash
python -c "import sys; print(sys.executable)"
```

If using conda:
```bash
conda create -n schopfield python=3.10
conda activate schopfield
pip install -e .
```

## Testing the Installation

Create a simple test script (`test_install.py`):

```python
import scHopfield as sch
import numpy as np
import anndata as ad

# Create dummy data
n_obs, n_vars = 100, 50
X = np.random.rand(n_obs, n_vars)
adata = ad.AnnData(X)
adata.layers['Ms'] = X
adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]

# Test preprocessing
sch.pp.fit_all_sigmoids(adata)
print("✓ Preprocessing module works!")

# Test sigmoid computation
sch.pp.compute_sigmoid(adata)
print("✓ Sigmoid computation works!")

print("\n✅ scHopfield is successfully installed and working!")
```

Run it:
```bash
python test_install.py
```

## Next Steps

1. Check out the tutorials in the `examples/` directory (to be added)
2. Read the full documentation at https://schopfield.readthedocs.io (coming soon)
3. See the README.md for a quick start guide
4. Explore the API reference in the docstrings
