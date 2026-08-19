Changelog
=========

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
