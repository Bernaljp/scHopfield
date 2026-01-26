Tools (sch.tl)
==============

The tools module provides analysis functions for energy, networks, correlations, embeddings, Jacobians, and velocity.

Energy Analysis
---------------

.. currentmodule:: scHopfield.tools

.. autosummary::
   :toctree: generated/

   compute_energies
   decompose_degradation_energy
   decompose_bias_energy
   decompose_interaction_energy

Network Analysis
----------------

.. autosummary::
   :toctree: generated/

   network_correlations
   get_network_links
   compute_network_centrality
   get_top_genes_table
   compute_eigenanalysis
   get_top_eigenvector_genes
   get_eigenanalysis_table

Correlation Analysis
--------------------

.. autosummary::
   :toctree: generated/

   energy_gene_correlation
   celltype_correlation
   future_celltype_correlation
   get_correlation_table

Embedding
---------

.. autosummary::
   :toctree: generated/

   compute_umap
   energy_embedding
   save_embedding
   load_embedding

Jacobian & Stability Analysis
------------------------------

.. autosummary::
   :toctree: generated/

   compute_jacobians
   save_jacobians
   load_jacobians
   compute_jacobian_stats
   compute_jacobian_elements
   compute_rotational_part

Velocity
--------

.. autosummary::
   :toctree: generated/

   compute_reconstructed_velocity
   validate_velocity

Detailed API
------------

Energy Analysis
~~~~~~~~~~~~~~~

compute_energies
^^^^^^^^^^^^^^^^

.. autofunction:: compute_energies

decompose_degradation_energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: decompose_degradation_energy

decompose_bias_energy
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: decompose_bias_energy

decompose_interaction_energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: decompose_interaction_energy

Network Analysis
~~~~~~~~~~~~~~~~

network_correlations
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: network_correlations

get_network_links
^^^^^^^^^^^^^^^^^

.. autofunction:: get_network_links

compute_network_centrality
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_network_centrality

get_top_genes_table
^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_top_genes_table

compute_eigenanalysis
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_eigenanalysis

get_top_eigenvector_genes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_top_eigenvector_genes

get_eigenanalysis_table
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_eigenanalysis_table

Correlation Analysis
~~~~~~~~~~~~~~~~~~~~

energy_gene_correlation
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: energy_gene_correlation

celltype_correlation
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: celltype_correlation

future_celltype_correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: future_celltype_correlation

get_correlation_table
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_correlation_table

Embedding
~~~~~~~~~

compute_umap
^^^^^^^^^^^^

.. autofunction:: compute_umap

energy_embedding
^^^^^^^^^^^^^^^^

.. autofunction:: energy_embedding

save_embedding
^^^^^^^^^^^^^^

.. autofunction:: save_embedding

load_embedding
^^^^^^^^^^^^^^

.. autofunction:: load_embedding

Jacobian & Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

compute_jacobians
^^^^^^^^^^^^^^^^^

.. autofunction:: compute_jacobians

save_jacobians
^^^^^^^^^^^^^^

.. autofunction:: save_jacobians

load_jacobians
^^^^^^^^^^^^^^

.. autofunction:: load_jacobians

compute_jacobian_stats
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_jacobian_stats

compute_jacobian_elements
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_jacobian_elements

compute_rotational_part
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_rotational_part

Velocity
~~~~~~~~

compute_reconstructed_velocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_reconstructed_velocity

validate_velocity
^^^^^^^^^^^^^^^^^

.. autofunction:: validate_velocity
