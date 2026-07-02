"""Real-data validation of the bias term on MEF->iPSC reprogramming (Schiebinger 2019).

Reprogramming is driven by exogenous, dox-induced OSKM (Pou5f1, Sox2, Klf4, Myc):
an *external* input the endogenous network cannot produce. In v = W.sigma - gamma.x + bias_I,
a gene held high by such an input forces bias_I ~ gamma.x - W.sigma > 0. So the hypothesis
predicts the fitted bias bias_I should be large and localized on OSKM + the pluripotency
program, whereas in a natural differentiation dataset the bias should be flat.

Test: fit scHopfield (L1 vs L2 bias) on a stratified subsample of the reprogramming
time course (velocity estimated from the reprogramming day axis), rank genes by the
bias, and measure enrichment of OSKM/pluripotency markers in the high-bias genes.
A natural pancreas fit is the negative control.
"""
import json
import os

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.metrics import roc_auc_score

import scHopfield as sch

DEV = "cuda"
N_CELLS = 8000
N_GENES = 300
OUT = "benchmark_results/bias_penalty"

OSKM = ["Pou5f1", "Sox2", "Klf4", "Myc"]
PLURI = ["Nanog", "Zfp42", "Esrrb", "Sall4", "Lin28a", "Utf1", "Dppa3"]
PROGRAM = OSKM + PLURI  # the externally driven module


def prep_reprogramming():
    a = ad.read_h5ad("data/reprogramming/schiebinger_serum.h5ad", backed="r")
    day = a.obs["day"].astype(str).astype(float).values
    rng = np.random.default_rng(0)
    # stratified subsample across timepoints
    sel = []
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        take = min(len(idx), max(1, N_CELLS // len(np.unique(day))))
        sel.append(rng.choice(idx, take, replace=False))
    sel = np.sort(np.concatenate(sel))
    a = a[sel].to_memory()
    a.obs["day_num"] = a.obs["day"].astype(str).astype(float).values
    print(f"reprogramming subsample: {a.shape}", flush=True)

    # neighbours + genes (force-keep the program genes)
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=30)
    sc.pp.highly_variable_genes(a, n_top_genes=N_GENES)
    keep = a.var["highly_variable"].values.copy()
    for g in PROGRAM:
        if g in a.var_names:
            keep[a.var_names.get_loc(g)] = True
    a = a[:, keep].copy()
    a.var["scHopfield_used"] = True

    # Ms = log-normalized X; velocity from the reprogramming day axis
    import scipy.sparse as sp
    X = a.X
    a.layers["Ms"] = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    sch.pp.estimate_velocity_from_pseudotime(
        a, pseudotime_key="day_num", spliced_key="Ms",
        connectivity_key="connectivities", mode="forward", scale=1.0, store_key="velocity_S")
    a.var["gamma"] = np.float32(0.1)
    sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values, spliced_key="Ms")
    sch.pp.compute_sigmoid(a, spliced_key="Ms")
    a.obs["stage"] = "reprogramming"
    return a


def prep_natural():
    a = ad.read_h5ad("data/Pancreas/pancreas_scvelo_ready.h5ad")
    a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    vmag = np.abs(np.asarray(a.layers["velocity_S"])).mean(0).ravel()
    keep = np.argsort(vmag)[::-1][:N_GENES]
    a = a[:, keep].copy()
    a.var["scHopfield_used"] = True
    a.obs["stage"] = "natural"
    return a


def fit_bias(a, penalty, blam=0.1):
    b = a.copy()
    sch.inf.fit_interactions(
        b, cluster_key="stage", w_scaffold=None, skip_all=True, w_threshold=1e-12,
        spliced_key="Ms", velocity_key="velocity_S", degradation_key="gamma",
        bias_penalty=penalty, bias_regularization=blam, reconstruction_regularization=1.0,
        n_epochs=500, batch_size=256, learning_rate=0.05, device=DEV, seed=0,
        infer_I=True, refit_gamma=False)
    stage = b.obs["stage"].astype(str).unique()[0]
    used = b.var["scHopfield_used"].values
    bias_I = b.var[f"I_{stage}"].values[used]
    return np.asarray(bias_I).ravel(), np.asarray(b.var_names[used])


def enrichment(bias_I, genes, program):
    absI = np.abs(bias_I)
    y = np.array([1.0 if g in set(program) else 0.0 for g in genes])
    auroc = roc_auc_score(y, absI) if y.sum() > 0 else float("nan")
    order = np.argsort(absI)[::-1]
    ranks = {g: int(np.where(genes[order] == g)[0][0]) + 1 for g in program if g in set(genes)}
    prog_mean = float(absI[y == 1].mean()) if y.sum() else float("nan")
    rest_mean = float(absI[y == 0].mean())
    return {"auroc": float(auroc), "program_mean_absI": prog_mean, "rest_mean_absI": rest_mean,
            "contrast": prog_mean / (rest_mean + 1e-9), "ranks": ranks}


def main():
    os.makedirs(OUT, exist_ok=True)
    sch.set_seed(0)
    results = {}

    print("\n### REPROGRAMMING (forced OSKM) ###", flush=True)
    rep = prep_reprogramming()
    for pen in ["l1", "l2"]:
        bias_I, genes = fit_bias(rep, pen)
        e = enrichment(bias_I, genes, PROGRAM)
        results[f"reprogramming_{pen}"] = e
        print(f"  {pen}: program-vs-rest |bias_I| AUROC={e['auroc']:.3f} contrast={e['contrast']:.1f} "
              f"(program mean|bias_I|={e['program_mean_absI']:.3f}, rest={e['rest_mean_absI']:.3f})", flush=True)
        print(f"       OSKM/pluripotency ranks (of {len(genes)}): {e['ranks']}", flush=True)

    print("\n### NATURAL pancreas (negative control) ###", flush=True)
    nat = prep_natural()
    # use the same marker set intersect for a fair 'flat' check; else random markers
    for pen in ["l1"]:
        bias_I, genes = fit_bias(nat, pen)
        rng = np.random.default_rng(0)
        rand_prog = list(rng.choice(genes, len(PROGRAM), replace=False))
        e = enrichment(bias_I, genes, rand_prog)
        results[f"natural_{pen}_randmarkers"] = e
        print(f"  {pen}: random-markers-vs-rest |bias_I| AUROC={e['auroc']:.3f} contrast={e['contrast']:.1f} "
              f"(should be ~0.5 / ~1: no localization)", flush=True)

    json.dump(results, open(f"{OUT}/reprogramming_validation.json", "w"), indent=2)
    print(f"\nwrote {OUT}/reprogramming_validation.json", flush=True)


if __name__ == "__main__":
    main()
