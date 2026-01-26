Dynamics Simulation
===================

Simulate gene expression trajectories using the inferred Hopfield network dynamics.

Theory
------

Gene expression evolves according to:

.. math::

   \frac{dx}{dt} = W \cdot \sigma(x) - \gamma \cdot x + I

scHopfield integrates this ODE system to simulate cellular dynamics.

Basic Trajectory Simulation
----------------------------

Simulate from a cell's initial state:

.. code-block:: python

   import scHopfield as sch
   import numpy as np

   # Simulate trajectory
   trajectory = sch.dyn.simulate_trajectory(
       adata,
       cluster='HSC',
       cell_idx=0,  # Start from cell 0
       t_span=np.linspace(0, 10, 100)  # Time points
   )

   # Plot results
   sch.pl.plot_trajectory(
       trajectory,
       t_span=np.linspace(0, 10, 100),
       genes=['GATA1', 'SPI1', 'FLI1']  # Specific genes
   )

Perturbation Experiments
-------------------------

Gene Knockouts
~~~~~~~~~~~~~~

Simulate gene knockout by setting expression to 0:

.. code-block:: python

   # GATA1 knockout
   ko_trajectory = sch.dyn.simulate_perturbation(
       adata,
       cluster='HSC',
       cell_idx=0,
       gene_perturbations={'GATA1': 0.0},  # Knockout
       t_span=np.linspace(0, 20, 200)
   )

Gene Overexpression
~~~~~~~~~~~~~~~~~~~

Simulate overexpression by setting high expression:

.. code-block:: python

   # GATA1 overexpression
   oe_trajectory = sch.dyn.simulate_perturbation(
       adata,
       cluster='HSC',
       cell_idx=0,
       gene_perturbations={'GATA1': 10.0},  # Overexpression
       t_span=np.linspace(0, 20, 200)
   )

Multiple Perturbations
~~~~~~~~~~~~~~~~~~~~~~

Combine multiple gene perturbations:

.. code-block:: python

   # Double knockout
   double_ko = sch.dyn.simulate_perturbation(
       adata,
       cluster='HSC',
       cell_idx=0,
       gene_perturbations={
           'GATA1': 0.0,
           'SPI1': 0.0
       },
       t_span=np.linspace(0, 20, 200)
   )

Advanced Usage
--------------

Custom ODE Solver
~~~~~~~~~~~~~~~~~

For fine-grained control:

.. code-block:: python

   # Create solver
   solver = sch.dyn.create_solver(
       adata,
       cluster='HSC',
       method='RK45',  # Integration method
       rtol=1e-6,  # Relative tolerance
       atol=1e-9   # Absolute tolerance
   )

   # Set initial condition
   x0 = adata.layers['Ms'][0, :]  # From cell 0

   # Simulate
   t_span = (0, 10)
   t_eval = np.linspace(0, 10, 100)
   solution = solver.solve_ivp(x0, t_span, t_eval)

Analyzing Trajectories
-----------------------

Extract trajectory features:

.. code-block:: python

   # Check if trajectory reaches steady state
   final_state = trajectory[:, -1]
   initial_state = trajectory[:, 0]

   # Compute distance
   import numpy as np
   distance = np.linalg.norm(final_state - initial_state)
   print(f"Distance traveled: {distance}")

   # Find genes with largest changes
   changes = np.abs(final_state - initial_state)
   gene_names = adata.var_names[adata.var['scHopfield_used']]
   top_idx = np.argsort(changes)[-10:]
   print("Top changing genes:", gene_names[top_idx])

Comparing Perturbations
------------------------

Compare wild-type vs perturbation:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Simulate wild-type and knockout
   wt = sch.dyn.simulate_trajectory(...)
   ko = sch.dyn.simulate_perturbation(...)

   # Plot comparison
   fig, axes = plt.subplots(2, 1, figsize=(10, 8))

   gene_idx = adata.var_names.get_loc('GATA1')
   axes[0].plot(t_span, wt[gene_idx, :], label='WT')
   axes[0].plot(t_span, ko[gene_idx, :], label='KO')
   axes[0].set_ylabel('GATA1 Expression')
   axes[0].legend()

   gene_idx2 = adata.var_names.get_loc('SPI1')
   axes[1].plot(t_span, wt[gene_idx2, :], label='WT')
   axes[1].plot(t_span, ko[gene_idx2, :], label='KO')
   axes[1].set_ylabel('SPI1 Expression')
   axes[1].set_xlabel('Time')
   axes[1].legend()

   plt.tight_layout()
   plt.show()

Applications
------------

1. **Cell Fate Prediction**: Simulate from progenitor states
2. **Perturbation Screening**: Test knockout effects in silico
3. **Drug Response**: Model effects of gene expression changes
4. **Transition Paths**: Trace differentiation trajectories
5. **Bifurcation Analysis**: Find decision points

Parameters
----------

**t_span** : array-like
   Time points for simulation

**method** : str (default='RK45')
   ODE integration method: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'

**rtol** : float (default=1e-3)
   Relative tolerance for integration

**atol** : float (default=1e-6)
   Absolute tolerance for integration

See Also
--------

- :doc:`api/dynamics` - Dynamics API
- :doc:`api/plotting` - Trajectory plotting
- :doc:`stability_analysis` - Stability analysis
