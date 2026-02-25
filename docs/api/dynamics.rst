Dynamics (sch.dyn)
==================

The dynamics module provides ODE solvers for simulating Hopfield network dynamics and gene expression trajectories.

Classes
-------

.. currentmodule:: scHopfield.dynamics

.. autosummary::
   :toctree: generated/

   ODESolver

ODE-based Simulation
--------------------

.. autosummary::
   :toctree: generated/

   create_solver
   simulate_trajectory
   simulate_perturbation

CellOracle-style GRN Propagation
---------------------------------

These functions implement perturbation simulation using GRN signal propagation,
inspired by CellOracle (Kamimoto et al., 2023).

.. autosummary::
   :toctree: generated/

   simulate_shift
   calculate_perturbation_effect_scores
   calculate_cell_transition_scores
   get_top_affected_genes
   compare_perturbations
