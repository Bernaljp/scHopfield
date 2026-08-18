Tutorials
=========

Six notebooks, meant to be read in order. Each one is a Jupyter notebook under
``docs/tutorials/`` in the repository, rendered here from the outputs committed
with it, so every number and every figure on these pages is the output of an
actual run rather than an illustration.

They are runnable. The first one fits the model from
:func:`scvelo.datasets.pancreas`, which downloads on first use, and saves the
fitted checkpoint to ``docs/tutorials/pancreas_fitted.h5sch``. That checkpoint is
committed, so the later tutorials load it in seconds and a reader without a GPU
never has to run the fit.

.. important::

   ``save_model`` stores the whole fitted system: the interaction matrix, the
   bias and the degradation rates, and the activation in full, meaning both Hill
   components of every gene and the weight that mixes them. So a notebook that
   loads a checkpoint can call ``compute_sigmoid`` straight away and get the
   activation the model was fitted with, which is what tutorials 3 to 6 do.
   Checkpoints written before the second component was persisted carry the first
   one alone; ``load_model`` warns when it reads one, and the remedy is to re-run
   ``fit_all_sigmoids`` on the expression data or to write the checkpoint again
   from a current fit.

.. toctree::
   :maxdepth: 1

   Fit a Hopfield system <tutorials/01-getting-started>
   Ground-truth circuits <tutorials/02-ground-truth-circuits>
   Why the scaffold matters <tutorials/03-why-the-scaffold-matters>
   Reading the fitted system <tutorials/04-reading-the-fitted-system>
   Single-gene knockouts <tutorials/05-single-gene-knockouts>
   Combinatorial perturbation <tutorials/06-combinatorial-perturbation>

What each one covers
--------------------

:doc:`1. Getting started <tutorials/01-getting-started>`
    Fits a Hopfield system to pancreatic endocrinogenesis end to end: velocity
    preparation, sigmoid activation fitting, scaffold construction and the GRN
    fit. Produces the checkpoint the rest of the series loads.

:doc:`2. Ground-truth circuits <tutorials/02-ground-truth-circuits>`
    Fits circuits whose interaction matrix is written down in advance, so the
    recovery can be scored rather than argued. Isolates the optimizer from
    velocity-estimation error, which no real dataset can do.

:doc:`3. Why the scaffold matters <tutorials/03-why-the-scaffold-matters>`
    The fit is not free: regulation is restricted to transcription factors, and
    edges an independent base network does not support are penalized. This
    measures what dropping each of those two constraints costs.

:doc:`4. Reading the fitted system <tutorials/04-reading-the-fitted-system>`
    One fit, three readings: the energy landscape, Jacobian stability, and
    network structure. Nothing further is fitted here.

:doc:`5. Single-gene knockouts <tutorials/05-single-gene-knockouts>`
    In-silico perturbation, and what constrains the answer to a question the data
    never asked. Covers both the propagated and the fate-probability readouts.

:doc:`6. Combinatorial perturbation <tutorials/06-combinatorial-perturbation>`
    Double knockouts and departures from additivity, which is where two
    regulators stop acting independently.

See Also
--------

- :doc:`quickstart` - the same pipeline as a page of code
- :doc:`pipeline` - the one-call entry point
- :doc:`api/index` - every public function
