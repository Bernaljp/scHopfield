scHopfield Documentation
========================

**Single-cell Hopfield Network Analysis**

Welcome to scHopfield's documentation! This package provides comprehensive tools for analyzing single-cell RNA-seq data using Hopfield network models.

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Overview
--------

scHopfield models gene regulatory networks (GRNs) as continuous Hopfield networks, where gene expression dynamics follow:

.. math::

   \frac{dx}{dt} = W \cdot \sigma(x) - \gamma \cdot x + I

**Key components:**

- **W**: Interaction matrix encoding gene-gene regulatory relationships
- **σ(x)**: Sigmoid activation function fitted to expression data
- **γ**: Degradation rates (mRNA decay)
- **I**: Bias vector representing external inputs/basal expression

This formulation enables:

- Energy landscapes that quantify cellular state stability
- Jacobian analysis for local stability and bifurcation detection
- Network topology analysis via centrality metrics and eigenanalysis
- Trajectory simulation for perturbation experiments and cell fate prediction

Key Features
------------

**Core Functionality**

- Preprocessing: Sigmoid function fitting to gene expression distributions
- Network Inference: Learn interaction matrices from RNA velocity
- Energy Landscapes: Compute and decompose into interaction, degradation, and bias components

**Network Analysis**

- Topology Analysis: Centrality metrics (degree, betweenness, eigenvector)
- Eigenanalysis: Eigenvalue decomposition of interaction matrices
- Network Comparison: Compare GRN structures across cell types
- GRN Visualization: Interactive network graphs

**Stability & Dynamics**

- Jacobian Analysis: Compute Jacobian matrices at each cell state
- Stability Metrics: Eigenvalue spectra, trace, rotational components
- Trajectory Simulation: Simulate gene expression dynamics
- Perturbation Analysis: In-silico gene knockouts and overexpression

**Visualization**

- Energy plots: Landscapes, boxplots, scatter plots
- Network plots: Interaction matrices, GRN graphs, centrality rankings
- Stability plots: Jacobian eigenvalue spectra, partial derivatives on UMAP
- Dynamics plots: Trajectory visualization

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   preprocessing
   inference
   energy_analysis
   network_analysis
   stability_analysis
   visualization
   dynamics

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/preprocessing
   api/inference
   api/tools
   api/plotting
   api/dynamics

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   data_conventions
   examples
   faq
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
