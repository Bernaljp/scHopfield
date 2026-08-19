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

release = "1.0.1"
version = "1.0.1"

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

# -- LaTeX / PDF output ------------------------------------------------------
#
# `formats: [pdf, epub]` in .readthedocs.yaml means every page also goes through
# pdflatex. pdflatex resolves a character through its font encoding, and the block
# characters a progress bar is drawn from are in none of the encodings the default
# engine loads, so it raises an error per character and prints nothing in its place.
# Read the Docs runs latexmk in a mode that ignores the exit code, so the PDF is
# published with the bar silently missing rather than the build failing.
#
# docs/clean_tutorial_output.py keeps only the frame each bar ended on, so what
# reaches LaTeX is a handful of full blocks rather than thousands. Declaring them
# draws the bar with a rule of the same width instead, which is what the reader of
# the PDF should see.
latex_engine = "pdflatex"
# A block character fills a character cell, or the left fraction of one, so each is drawn
# as a rule of that fraction inside a box exactly one cell wide. The cell is measured from
# the font in force where the character appears rather than fixed here, because the bar
# sits in a verbatim block set smaller than body text, and a box of the wrong width would
# walk the rest of the line out of column.
latex_elements = {
    "preamble": "\n".join(
        [
            r"\newlength{\schopfieldcell}",
            r"\newcommand{\schopfieldblock}[1]{%",
            r"  \settowidth{\schopfieldcell}{0}%",
            r"  \makebox[\schopfieldcell][l]{%",
            r"    \rule[-0.05em]{\dimexpr\schopfieldcell*#1/8\relax}{0.9em}}}",
        ]
        + [
            r"\DeclareUnicodeCharacter{%04X}{\schopfieldblock{%d}}" % (code, eighths)
            for code, eighths in (
                (0x2588, 8),  # FULL BLOCK
                (0x2589, 7),  # LEFT SEVEN EIGHTHS BLOCK
                (0x258A, 6),  # LEFT THREE QUARTERS BLOCK
                (0x258B, 5),  # LEFT FIVE EIGHTHS BLOCK
                (0x258C, 4),  # LEFT HALF BLOCK
                (0x258D, 3),  # LEFT THREE EIGHTHS BLOCK
                (0x258E, 2),  # LEFT ONE QUARTER BLOCK
                (0x258F, 1),  # LEFT ONE EIGHTH BLOCK
            )
        ]
    ),
}
