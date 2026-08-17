#!/usr/bin/env bash
# Run an R script with the user dyngen library and the glpk shim on LD_LIBRARY_PATH.
#
# This is provenance, not part of the reproduction path. The simulator's own CSV exports
# are committed under reproducibility/data/dyngen/, and 02_build_h5ad.py rebuilds every
# ground-truth matrix from them with no R and no dyngen. This script is kept so the
# simulation that produced those CSVs can be re-run and audited.
#
# Both paths are machine specific. Set them for your install:
#   R_LIBS_USER     the R library holding dyngen
#   SCHOPFIELD_GLPK a directory holding libglpk, if your R build cannot find it
export R_LIBS_USER="${R_LIBS_USER:?set R_LIBS_USER to the R library holding dyngen}"
if [ -n "${SCHOPFIELD_GLPK:-}" ]; then
  export LD_LIBRARY_PATH="${SCHOPFIELD_GLPK}:${LD_LIBRARY_PATH:-}"
fi
exec Rscript "$@"
