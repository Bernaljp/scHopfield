Contributing
============

We welcome contributions to scHopfield! This guide will help you get started.

Ways to Contribute
------------------

- Report bugs and request features via GitHub Issues
- Improve documentation
- Add new analysis functions
- Contribute example notebooks
- Optimize performance
- Fix bugs

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/scHopfield.git
   cd scHopfield

3. Install in development mode with dev dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

4. Create a new branch for your changes:

.. code-block:: bash

   git checkout -b my-feature-branch

Code Guidelines
---------------

Style
~~~~~

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

We use **Black** for code formatting:

.. code-block:: bash

   black scHopfield/

Documentation
~~~~~~~~~~~~~

All functions must have numpy-style docstrings:

.. code-block:: python

   def my_function(adata, param1, param2='default'):
       """
       Brief description of function.

       Detailed description if needed.

       Parameters
       ----------
       adata : AnnData
           Annotated data object
       param1 : type
           Description of param1
       param2 : str, optional (default: 'default')
           Description of param2

       Returns
       -------
       type or None
           Description of return value
       """
       pass

Testing
~~~~~~~

Add tests for new functions:

.. code-block:: python

   # tests/test_my_module.py
   import pytest
   import scHopfield as sch

   def test_my_function():
       # Test implementation
       assert result == expected

Run tests with:

.. code-block:: bash

   pytest tests/

Pull Request Process
--------------------

1. **Update Documentation**

   - Add docstrings to new functions
   - Update user guide if needed
   - Add to API reference

2. **Add Tests**

   - Write unit tests for new functionality
   - Ensure all tests pass

3. **Update Changelog**

   - Add entry to ``docs/changelog.rst``

4. **Submit Pull Request**

   - Push to your fork
   - Open a pull request to ``main`` branch
   - Describe your changes clearly
   - Reference any related issues

5. **Code Review**

   - Address reviewer feedback
   - Update as needed

Commit Messages
---------------

Use clear, descriptive commit messages:

.. code-block:: text

   Add network centrality computation using igraph

   - Implement compute_network_centrality() function
   - Add support for multiple centrality metrics
   - Include igraph as optional dependency
   - Add tests and documentation

Avoid:

.. code-block:: text

   fix bug
   update code
   changes

Code Organization
-----------------

The package structure:

.. code-block:: text

   scHopfield/
   ├── preprocessing/     # Data preprocessing
   ├── inference/         # Network inference
   ├── tools/            # Analysis tools
   │   ├── energy.py
   │   ├── networks.py
   │   ├── jacobian.py
   │   └── ...
   ├── plotting/         # Visualization
   ├── dynamics/         # ODE simulation
   └── _utils/           # Internal utilities

Adding New Features
-------------------

Analysis Functions
~~~~~~~~~~~~~~~~~~

1. Add function to appropriate module in ``tools/``
2. Update ``tools/__init__.py`` to export it
3. Add docstring with Parameters and Returns
4. Add to API documentation in ``docs/api/tools.rst``
5. Add example to user guide

Plotting Functions
~~~~~~~~~~~~~~~~~~

1. Add function to appropriate module in ``plotting/``
2. Update ``plotting/__init__.py`` to export it
3. Add docstring with complete parameter descriptions
4. Add to API documentation in ``docs/api/plotting.rst``
5. Follow existing plotting patterns (return figure/axes)

Reporting Bugs
--------------

When reporting bugs, include:

1. **scHopfield version**:

   .. code-block:: python

      import scHopfield as sch
      print(sch.__version__)

2. **Python version**:

   .. code-block:: bash

      python --version

3. **Minimal reproducible example**
4. **Error message** (full traceback)
5. **Expected behavior**

Feature Requests
----------------

For feature requests:

1. Check if similar feature already exists
2. Describe the use case clearly
3. Provide example API design if possible
4. Explain why it would be useful

Questions?
----------

- GitHub Issues: https://github.com/Bernaljp/scHopfield/issues
- GitHub Discussions: https://github.com/Bernaljp/scHopfield/discussions

Thank you for contributing!
