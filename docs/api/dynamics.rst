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

Knockout screens
----------------

.. autosummary::
   :toctree: generated/

   run_ko_screen
   score_ko_panel
   run_pairwise_ko_screen
   compute_epistasis
   run_dose_response
   dose_levels_from_fractions

Propagated Perturbation Readouts
---------------------------------

Readouts that integrate the fitted field with genes clamped, rather than reading the
instantaneous response: where the perturbation drives cells, and how far from wild type it has
driven each cell type over time.

.. autosummary::
   :toctree: generated/

   knockout_displacement_flow
   perturbation_cascade
