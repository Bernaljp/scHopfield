Changelog
=========

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
