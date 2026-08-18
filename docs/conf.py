# Configuration file for the Sphinx documentation builder.
#
# Full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "scHopfield"
author = "Juan Pablo Bernal Tamayo"
# Pinned to the year in LICENSE rather than computed from the build date, so a
# rebuild in a later year cannot make the docs disagree with the license.
copyright = f"2026, {author}"

release = "1.0.0"
version = "1.0.0"

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
    "myst_nb",             # Markdown support, and notebooks rendered from committed outputs
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md",
                    "**.ipynb_checkpoints", "methods", "archive"]

# reStructuredText, Markdown, and notebooks. myst-nb owns the last two.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
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
    # Page navigation lives in the LEFT sidebar (sidebar-nav-bs), not the top
    # navbar. The header keeps only the logo, search, theme switch, and icons.
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "show_prev_next": True,
    "navigation_with_keys": True,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

# Left sidebar = full site navigation (custom template rendering the global
# toctree). Right sidebar ("On this page") is the in-page TOC (page-toc),
# configured by the theme separately, so the two never mirror each other.
html_sidebars = {
    "**": ["sidebar-main-nav.html"],
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

# -- myst-nb (tutorial notebooks) --------------------------------------------
#
# The tutorials render from the outputs committed alongside them and are never
# executed by the docs build. Two reasons, and both are load-bearing. A build that
# executed them would need a GPU, the pancreas dataset and roughly an hour, so it
# would fail on Read the Docs and on any stranger's checkout. And the outputs are
# the record of a specific run of a specific fit, which is what a reader should be
# reading; regenerating them is the notebook author's job, not the doc builder's.
#
# myst-nb rather than nbsphinx because nbsphinx shells out to pandoc, a system
# binary that pip cannot install. With myst-nb the whole toolchain comes from
# docs/requirements.txt, so the build reproduces from a clean checkout with no
# apt step.
nb_execution_mode = "off"
nb_merge_streams = True
