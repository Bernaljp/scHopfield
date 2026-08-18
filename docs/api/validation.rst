Validation (sch.validation)
===========================

Synthetic gene circuits whose interaction matrix is known exactly, plus the metrics that
score a recovered network against it. Because the true ``W``, ``I`` and ``gamma`` and the
analytic Hill activation are all known, a fit on these circuits is a clean identifiability
check: it isolates the optimizer from RNA-velocity estimation error.

These are the circuits behind the synthetic-recovery figure, so a reader can run them
directly rather than taking the reported numbers on trust. They need no downloaded data.

Circuits
--------

.. currentmodule:: scHopfield.validation.circuits

.. autosummary::
   :toctree: generated/

   ToggleCircuit
   OscillatorCircuit

Simulation and fitting
----------------------

.. currentmodule:: scHopfield.validation

.. autosummary::
   :toctree: generated/

   simulate_circuit
   fit_circuit
   build_circuit_scaffold

Recovery metrics
----------------

.. autosummary::
   :toctree: generated/

   edge_sign_accuracy
   edge_signed_correlation
   spectral_overlap
   symmetry_index
   frobenius_distance
   summarize_recovery
