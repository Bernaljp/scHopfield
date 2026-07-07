"""Compute Dynamo RNA velocity for a dataset, aligned to the caller's cells+genes.

Runs in the isolated .venv-dyn (dynamo 1.5.x needs anndata<0.11). Reads a minimal
h5ad carrying raw spliced/unspliced counts (X = spliced), runs the standard dynamo
pipeline (recipe_monocle -> dynamics), and writes an npz with the spliced velocity
(cells x dynamics-genes) plus barcode/gene labels so the main env can align it to the
scHopfield gene set by name.

Usage (from .venv-dyn):
    .venv-dyn/bin/python analyses/reports/dyn_velocity.py <in.h5ad> <out.npz> [cores]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import anndata as ad
import dynamo as dyn


def main():
    inp, out = sys.argv[1], sys.argv[2]
    cores = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    a = ad.read_h5ad(inp)
    print(f"[dyn_velocity] loaded {a.shape}; layers={list(a.layers)}", flush=True)

    pp = dyn.preprocessing.Preprocessor()
    pp.preprocess_adata(a, recipe="monocle")
    dyn.tl.dynamics(a, cores=cores)

    if "velocity_S" not in a.layers:
        raise RuntimeError(f"dynamo produced no velocity_S; layers={list(a.layers)}")
    V = a.layers["velocity_S"]
    V = V.toarray() if hasattr(V, "toarray") else np.asarray(V)
    # dynamics genes: nonzero velocity anywhere (others are NaN/0 padding)
    use = np.asarray(a.var["use_for_dynamics"]) if "use_for_dynamics" in a.var else \
        (np.nan_to_num(V).any(0))
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    genes = np.asarray(a.var_names)[use]
    Vd = V[:, use]
    np.savez_compressed(out, velocity=Vd, cells=np.asarray(a.obs_names, dtype=object),
                        genes=np.asarray(genes, dtype=object))
    print(f"[dyn_velocity] wrote {out}: {Vd.shape[0]} cells x {Vd.shape[1]} dynamics genes",
          flush=True)


if __name__ == "__main__":
    main()
