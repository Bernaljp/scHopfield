"""Filesystem layout for the reproducibility tree.

Single source of truth for where the figure scripts read and write. Every path is
anchored to this file rather than to the working directory, so a script behaves the
same whether it is run from the repository root or from anywhere else.

Three roots are not part of the repository, because the material under them is either
too large to version or is not ours to redistribute. Each is read from an environment
variable and falls back to a sibling of the repository root:

``SCHOPFIELD_REPORTS``
    The per-dataset report tree written by the analysis pipeline. Several figures read
    a fitted ``adata_analyzed.h5ad`` or a pickled cache from under it. It is measured in
    gigabytes and is regenerated, not distributed.

``SCHOPFIELD_DATA``
    The raw and preprocessed single-cell datasets, and the base GRN scaffolds.

``SCHOPFIELD_DYNAMISC_DATA``
    Two of the seven datasets are read from a separate data directory on the machine
    where the analyses were run. Defaults to ``<SCHOPFIELD_DATA>/DynamiSC``.

Outputs go under this directory and are ignored by git, so a run never dirties the
working tree:

``figures/``
    Rendered figures. The default run writes the working figure; ``--submission``
    writes the journal-page variant under ``figures/submission/``.

``output/``
    Generated report pages and their plots.
"""
from __future__ import annotations

import os

#: This directory, ``<repo>/reproducibility``.
REPRO = os.path.dirname(os.path.abspath(__file__))
#: The repository root.
REPO = os.path.dirname(REPRO)

# ----------------------------------------------------------------------------- #
# Inputs that ship with the repository
# ----------------------------------------------------------------------------- #
DATA = os.path.join(REPRO, "data")
CACHE = os.path.join(REPRO, "cache")

DYNGEN = os.path.join(DATA, "dyngen")
ABLATIONS = os.path.join(DATA, "ablations")
IDENTIFIABILITY = os.path.join(DATA, "real_identifiability")
SMALL_CIRCUITS = os.path.join(DATA, "small_circuits")

# ----------------------------------------------------------------------------- #
# Outputs
# ----------------------------------------------------------------------------- #
FIGURES = os.environ.get("SCHOPFIELD_FIGURES") or os.path.join(REPRO, "figures")
FIGURES_SPEC = os.path.join(FIGURES, "submission")
OUTPUT = os.environ.get("SCHOPFIELD_OUTPUT") or os.path.join(REPRO, "output")

# ----------------------------------------------------------------------------- #
# Roots that do not ship
# ----------------------------------------------------------------------------- #
REPORTS = os.environ.get("SCHOPFIELD_REPORTS") or os.path.join(REPO, "reports")
DATASETS = os.environ.get("SCHOPFIELD_DATA") or os.path.join(REPO, "data")
DYNAMISC = os.environ.get("SCHOPFIELD_DYNAMISC_DATA") or os.path.join(DATASETS, "DynamiSC")

# ----------------------------------------------------------------------------- #
# Optional dynamo streamline rendering
#
# One report panel can be drawn by dynamo's streamline_plot, which needs its own
# interpreter and a helper script that is part of the analysis pipeline rather than of
# this repository. Both are optional: when either is unset or missing, the caller falls
# back to a scVelo stream plot.
# ----------------------------------------------------------------------------- #
DYNAMO_PYTHON = os.environ.get("SCHOPFIELD_DYNAMO_PYTHON", "")
DYN_STREAMLINE = os.environ.get("SCHOPFIELD_DYN_STREAMLINE", "")
