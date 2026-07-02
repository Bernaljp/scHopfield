"""CLI wrapper to make a dataset scHopfield-ready.

The preprocessing logic now lives in the package
(:func:`scHopfield.preprocessing.prepare_dataset`); this script just loads a
dataset, calls it, and writes the result. Produces layers['Ms'],
layers['velocity_S'], layers['sigmoid'], var['gamma'], obsp['connectivities'],
var['scHopfield_used'].
"""
import argparse
import os

import scanpy as sc

import scHopfield as sch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-top", type=int, default=2000)
    args = ap.parse_args()

    a = sc.read_h5ad(args.inp)
    print(f"loaded {args.inp}: {a.shape}; layers={list(a.layers)}", flush=True)

    sch.pp.prepare_dataset(a, n_top_genes=args.n_top)
    print(f"usable genes: {a.n_vars}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    a.write(args.out)
    print(f"wrote {args.out}: {a.shape}", flush=True)


if __name__ == "__main__":
    main()
