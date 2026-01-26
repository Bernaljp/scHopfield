# scHopfield Documentation

This directory contains the Sphinx documentation for scHopfield.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or manually:

```bash
pip install sphinx sphinx-rtd-theme nbsphinx ipykernel
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Live Preview

To build and serve the documentation locally:

```bash
cd docs
make livehtml
```

Then open http://localhost:8000 in your browser.

### Clean Build Files

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── index.rst              # Main index
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── tutorial.rst           # Detailed tutorial
├── preprocessing.rst      # Preprocessing guide
├── inference.rst          # Network inference guide
├── energy_analysis.rst    # Energy analysis guide
├── network_analysis.rst   # Network analysis guide
├── stability_analysis.rst # Stability analysis guide
├── visualization.rst      # Visualization guide
├── dynamics.rst           # Dynamics simulation guide
├── data_conventions.rst   # Data storage conventions
├── examples.rst           # Example notebooks
├── faq.rst                # FAQ
├── changelog.rst          # Version history
├── contributing.rst       # Contributing guide
├── api/                   # API reference
│   ├── preprocessing.rst
│   ├── inference.rst
│   ├── tools.rst
│   ├── plotting.rst
│   └── dynamics.rst
└── conf.py                # Sphinx configuration
```

## ReadTheDocs

The documentation is automatically built and hosted on ReadTheDocs when you push to GitHub.

Configuration file: `.readthedocs.yaml` in the repository root.

Documentation URL: https://schopfield.readthedocs.io

## Updating Documentation

1. Edit the `.rst` files as needed
2. Build locally to test: `make html`
3. Commit and push changes
4. ReadTheDocs will automatically rebuild

## Adding New Pages

1. Create a new `.rst` file in `docs/`
2. Add it to the `toctree` in `index.rst`
3. Build and verify

## API Documentation

API documentation is auto-generated from docstrings using:

- `sphinx.ext.autodoc` - Extract docstrings
- `sphinx.ext.autosummary` - Generate summaries
- `sphinx.ext.napoleon` - Parse numpy-style docstrings

To add a new function to API docs, update the appropriate file in `docs/api/`.
