"""
scHopfield: Single-cell Hopfield network analysis
================================================

A package for analyzing single-cell RNA-seq data using Hopfield network models.

Submodules
----------
pp : preprocessing
    Sigmoid fitting and data preprocessing
inf : inference
    Network parameter inference
tl : tools
    Analysis tools (energy, correlation, embedding, jacobian, networks)
pl : plotting
    Visualization functions
dyn : dynamics
    ODE solving and trajectory simulation
validation : validation
    Synthetic gene circuits with known ground-truth interaction matrices, and the
    metrics that score a recovered network against them

Usage
-----
Import scHopfield with::

    import scHopfield as sch

Then access functions via::

    sch.pp.fit_all_sigmoids(adata)
    sch.inf.fit_interactions(adata, cluster_key='celltype')
    sch.tl.compute_energies(adata)
    sch.pl.plot_energy_landscape(adata, cluster='HSC')

The fitting defaults are the configuration used throughout the paper, so calling
``fit_all_sigmoids`` and ``fit_interactions`` without tuning reproduces the
published method. A prior-knowledge scaffold is the one thing they cannot supply
for you: build one with ``sch.inf.build_scaffold`` and pass it as ``w_scaffold``.
Fitting without a scaffold is a different, and measurably worse, method.

The scaffold is built from a base gene regulatory network, which scHopfield does
not distribute. ``sch.fetch_base_grn`` downloads one from CellOracle on demand and
caches it. That table carries CellOracle's own license, restricted to
non-commercial academic use, and not scHopfield's MIT license; ``DATA_SOURCES.md``
states the restriction and lists the works to cite.
"""

__version__ = '0.1.0'

# Import submodules
from . import preprocessing as pp
from . import inference as inf
from . import tools as tl
from . import plotting as pl
from . import dynamics as dyn
from . import validation

# Expose key classes and functions at top level
from ._utils.seed import set_seed
from .preprocessing import fit_all_sigmoids, compute_sigmoid, prepare_dataset
from .inference import fit_interactions, build_scaffold, fetch_base_grn
from .tools import compute_energies, compute_umap, energy_embedding
from .dynamics import ODESolver, simulate_trajectory
from .workflows import run_pipeline

__all__ = [
    'pp',
    'inf',
    'tl',
    'pl',
    'dyn',
    'validation',
    'set_seed',
    'fit_all_sigmoids',
    'compute_sigmoid',
    'prepare_dataset',
    'fit_interactions',
    'build_scaffold',
    'fetch_base_grn',
    'compute_energies',
    'compute_umap',
    'energy_embedding',
    'ODESolver',
    'simulate_trajectory',
    'run_pipeline',
]
