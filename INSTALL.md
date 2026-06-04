# scHopfield Installation Guide

## Installation Methods

### Method 1: Robust Installation with Conda (Recommended)

This method ensures all complex dependencies (like `gimmemotifs`, `velocyto`, and `celloracle`) are installed correctly without build conflicts.

```bash
# 1. Create and activate a fresh environment
conda create -n schopfield python=3.10.8 -y
conda activate schopfield

# 2. Upgrade core build tools
conda install "setuptools<82" -y
pip install --upgrade pip wheel

# 3. Install core numerical and single-cell dependencies
pip install \
  numpy==1.26.4 \
  pandas==1.5.3 \
  scikit-learn==1.5.2 \
  matplotlib==3.6.3 \
  scipy \
  numba \
  seaborn \
  h5py \
  scanpy \
  anndata \
  networkx \
  umap-learn \
  tqdm \
  cython

# 4. Remove conflicting packages (optional safety step)
pip uninstall -y pims omnipath xarray-dataclasses xarray-schema dask-expr ome-zarr || true

# 5. Install mamba and gimmemotifs
conda install -c conda-forge "mamba>=0.27" -y
mamba install -c bioconda -c conda-forge gimmemotifs -y

# 6. Install velocyto and celloracle without build isolation
pip install --no-build-isolation velocyto
pip install --no-build-isolation celloracle

# 7. Install scHopfield LAST
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install -e .
```

### Method 2: Minimal Installation

If you already have a working environment and just want to install the package directly:

```bash
git clone https://github.com/Bernaljp/scHopfield.git
cd scHopfield
pip install -e .
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
