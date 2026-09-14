Changelog
=========

Version 1.2.0 (2026-09-14)
--------------------------

The dose sweep is anchored on each cell's own state, so a dose of one perturbs nothing.

- ``dose_fate_bias`` takes ``mode``, defaulting to ``'hybrid'``: at or below dose one each cell is
  held at a multiple of its own observed expression, and above it each cell moves linearly to the
  gene's 99th percentile, reaching it at dose two. Dose zero is the knockout, dose one is the
  unperturbed state exactly, and a dose above one raises every cell, including cells expressing
  almost none of the gene, while a cell already above the percentile is left where it is. The level
  is continuous at dose one and non-decreasing in dose for every cell.
- ``mode='relative'`` scales each cell's own level throughout, so a dose above one cannot switch on
  a gene a cell does not express. ``mode='absolute'`` is the previous behavior, every cell clamped
  to a multiple of the gene's 99th percentile, under which no dose is the unperturbed state.
- The clamp accepts a per-cell level, so ``model_velocity``, ``perturbed_fate`` and the readouts
  built on them can hold a gene at a different level in each cell rather than one level everywhere.

Version 1.1.0 (2026-09-14)
--------------------------

The activation is fitted by maximum likelihood, and every evaluation of the fitted
field uses the Hill component each cell was assigned to. Fitted results differ from
1.0.x on any dataset with bimodal genes.

- ``fit_all_sigmoids`` takes ``method``, defaulting to ``'mle'``: the two-component
  Hill is fitted to expression by maximum likelihood, from two starts, with the
  parameters bounded and one start initialized from a Gaussian mixture on log
  expression. ``method='ecdf'`` is the earlier least-squares fit to the empirical
  distribution function, kept so that objects fitted that way can be reproduced.
- A gene takes two components only when it passes an acceptance gate: Sarle's
  bimodality coefficient on log expression above ``bimodality_min``, thresholds
  differing at least ``min_k_ratio``-fold, a smaller weight of at least
  ``min_weight``, and a mixture that lowers the Bayesian information criterion.
- Each cell is assigned to a component by maximum posterior, and that assignment is
  fixed at the cell's observed state. Integration, Jacobians, knockouts, energies,
  velocity and flow all evaluate a cell in the component it was assigned, rather than
  re-deciding at each point they visit, so a trajectory cannot change which branch of
  the activation it is governed by while it moves. Below ``sigmoid_active_min``, where
  the mixture has no data to speak from, a cell takes the first component.
- ``fit_all_sigmoids`` writes ``sigmoid_nll``, ``sigmoid_ks``, ``sigmoid_bc`` and
  ``sigmoid_active_min``, and records ``sigmoid_method`` and ``sigmoid_assignment`` in
  ``uns['scHopfield']``.
- ``save_model`` persists those columns. ``sigmoid_active_min`` is part of the
  activation rather than a diagnostic, so a checkpoint without it reassigned the
  low-expression cells of every bimodal gene on load, with nothing said.

Version 1.0.1 (2026-08-19)
--------------------------

Documentation only. No behavior changes, and no fitted result differs.

- Progress bars in the tutorial output are collapsed to the frame each one ended on.
  A notebook records every redraw of a bar as a separate output and the documentation
  has no terminal to overwrite them, so the published Getting Started page was 71
  percent progress-bar frames.
- Three non-ASCII characters are removed from docstrings that the API reference
  publishes, one of which left the PDF stating the opposite of the condition it meant.
- The PDF build declares the block characters a progress bar is drawn from, so it draws
  the bar rather than dropping it.

Version 1.0.0 (2026-08-19)
--------------------------

First public release, the version accompanying the manuscript.

- Public API fixed to what each submodule's ``__all__`` declares, reachable as
  ``sch.pp``, ``sch.inf``, ``sch.tl``, ``sch.pl``, ``sch.dyn`` and ``sch.validation``.
- Base regulatory network scaffolds fetched on demand by ``sch.fetch_base_grn`` from a
  pinned upstream commit and checked against a recorded sha256, rather than shipped.
- Every fitting parameter defaults to the value used throughout the paper, ``seed=0``
  included, so a call that tunes nothing reproduces the published configuration.
- ``save_model`` persists the full fitted activation, and ``load_model`` warns when a file
  predates that and carries only the primary Hill component.
- Six executed tutorial notebooks and a documentation site at
  https://schopfield.readthedocs.io.
- ``reproducibility/`` carries the code behind every figure in the paper, with
  ``reproducibility/README.md`` documenting the path from the public raw data to each one.

Version 0.1.0 (2025-01-26)
--------------------------

Initial release of scHopfield.

Features
~~~~~~~~

**Core Functionality**

- Sigmoid function fitting to gene expression distributions
- Network inference from RNA velocity using gradient descent
- Energy landscape computation and decomposition
- GPU acceleration support for training and analysis

**Network Analysis**

- Network centrality metrics (degree, betweenness, eigenvector)
- Eigenvalue decomposition of interaction matrices
- Network comparison across cell types
- GRN visualization with customizable layouts

**Stability Analysis**

- Jacobian matrix computation for all cells
- Eigenvalue analysis for stability assessment
- Rotational component analysis
- Partial derivative computation for gene pairs
- HDF5 storage for large Jacobian matrices

**Visualization**

- Energy landscape plots
- Interaction matrix heatmaps
- GRN network graphs
- Jacobian eigenvalue spectra
- Centrality rankings and comparisons
- Correlation scatter plots

**Dynamics Simulation**

- ODE integration for gene expression trajectories
- Perturbation experiments (knockouts, overexpression)
- Trajectory visualization

**Documentation**

- Complete API reference with numpy-style docstrings
- User guide with detailed tutorials
- ReadTheDocs integration
- Example notebooks

API
~~~

- ``scHopfield.pp`` - Preprocessing
- ``scHopfield.inf`` - Network inference
- ``scHopfield.tl`` - Analysis tools
- ``scHopfield.pl`` - Plotting
- ``scHopfield.dyn`` - Dynamics simulation

Dependencies
~~~~~~~~~~~~

- Core: numpy, scipy, pandas, matplotlib, anndata, scanpy, torch, networkx
- Optional: seaborn, python-igraph, dynamo-release

Future Releases
---------------

Planned features for future versions:

- More example notebooks with real datasets
- Additional network analysis metrics
- Enhanced perturbation analysis
- Integration with trajectory inference tools
- Performance optimizations
- Additional visualization options
