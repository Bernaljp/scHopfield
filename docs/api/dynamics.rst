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
   simulate_perturbation_ode
   simulate_shift_ode
   calculate_trajectory_flow

CellOracle-style GRN Propagation
---------------------------------

Perturbation simulation via GRN signal propagation, inspired by CellOracle
(Kamimoto et al., 2023). ``simulate_shift`` is an alias of
``simulate_perturbation``.

.. autosummary::
   :toctree: generated/

   simulate_perturbation
   simulate_shift
   calculate_perturbation_effect_scores
   calculate_cell_transition_scores
   get_top_affected_genes
   compare_perturbations

Knockout screens
----------------

.. autosummary::
   :toctree: generated/

   run_ko_screen
   score_ko_panel
   run_pairwise_ko_screen
   compute_epistasis
   run_dose_response
