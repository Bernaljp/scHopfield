"""Standalone dynamo streamline plot, run in the isolated .venv-dyn environment.

dynamo requires anndata<0.11, which is incompatible with the main pipeline env
(anndata>=0.12, needed to read the datasets). So the report projects the velocity to the
embedding in the main env (scVelo velocity_embedding) and hands a minimal, clean AnnData
(X_umap + velocity_umap + cluster + colors, no problematic uns) to this script, which
draws dynamo's streamline_plot -- the plotter the user's reference notebooks use.

    .venv-dyn/bin/python analyses/reports/dyn_streamline.py \
        --input handoff.h5ad --cluster clusters --out fig.png --title "..." --basis umap
"""
import argparse
import warnings

warnings.filterwarnings("ignore")
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dynamo as dyn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--basis", default="umap")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.input)
    ax = dyn.pl.streamline_plot(
        adata, color=args.cluster, basis=args.basis, show_arrowed_spines=True,
        show_legend="on data", save_show_or_return="return")
    axx = ax[0] if isinstance(ax, (list, tuple)) else ax
    try:
        axx.set_title(args.title)
    except Exception:
        pass
    fig = axx.get_figure()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
