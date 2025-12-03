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

Usage
-----
Import scHopfield with::

    import scHopfield as sch

Then access functions via::

    sch.pp.fit_all_sigmoids(adata)
    sch.inf.fit_interactions(adata, cluster_key='celltype')
    sch.tl.compute_energies(adata)
    sch.pl.plot_energy_landscape(adata, cluster='HSC')
"""

__version__ = '0.1.0'

# Import submodules
from . import preprocessing as pp
from . import inference as inf
from . import tools as tl
from . import plotting as pl
from . import dynamics as dyn

# Expose key classes and functions at top level
from .preprocessing import fit_all_sigmoids, compute_sigmoid
from .inference import fit_interactions
from .tools import compute_energies, compute_umap, energy_embedding
from .dynamics import ODESolver, simulate_trajectory

__all__ = [
    'pp',
    'inf',
    'tl',
    'pl',
    'dyn',
    'fit_all_sigmoids',
    'compute_sigmoid',
    'fit_interactions',
    'compute_energies',
    'compute_umap',
    'energy_embedding',
    'ODESolver',
    'simulate_trajectory',
]
