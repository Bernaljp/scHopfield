# Configuration file for the Sphinx documentation builder.
#
# Full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "scHopfield"
author = "scHopfield Contributors"
copyright = f"{datetime.now():%Y}, {author}"

release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",       # grid/cards on the landing page
    "sphinx_copybutton",   # copy button on code blocks
    "myst_parser",         # Markdown support
    "nbsphinx",            # executed-notebook tutorials
]

# Napoleon (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autosummary / autodoc
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "none"        # keep signatures readable; types live in the docstring
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    # heavy / optional deps so the API builds even if they are absent
    "torch", "torchdiffeq", "scvelo", "umap", "hoggorm", "igraph",
    "leidenalg", "celloracle", "genie3",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "**.ipynb_checkpoints"]

# Markdown + reStructuredText
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- HTML output (pydata-sphinx-theme) ---------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "scHopfield"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sourcelink = False
html_context = {
    "github_user": "Bernaljp",
    "github_repo": "scHopfield",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "light",
}

html_theme_options = {
    "logo": {"text": "scHopfield"},
    "github_url": "https://github.com/Bernaljp/scHopfield",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Bernaljp/scHopfield",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scHopfield/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "show_prev_next": True,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

html_sidebars = {
    "index": [],  # full-width landing page, no left sidebar
}

# -- copybutton --------------------------------------------------------------
# Strip prompts so pasted snippets are runnable.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regex = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_timeout = 600
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None)|string %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        This page was generated from `{{ docname }}`__.

    __ https://github.com/Bernaljp/scHopfield/blob/main/{{ docname }}
"""
