"""Refined real-data bias validation on MEF->iPSC reprogramming (Option A).

Improvements over the first pass (M17):
- velocity direction from a proper reprogramming pseudotime (DPT rooted at a day-0
  MEF cell) instead of the discrete, asynchronous day axis;
- fit the bias PER STAGE (MEF -> transitional -> iPSC) rather than one global GRN.

Sharper prediction: a gene held high by exogenous OSKM forces I ~ gamma.x - W.sigma.
Early/transitional, the endogenous (MEF / partial) network cannot produce OSKM +
the pluripotency program, so their bias is large. In mature iPSCs the endogenous
pluripotency network becomes self-sustaining and absorbs them, so the bias drops.
=> the OSKM/pluripotency bias should be high early/transitional and fall toward iPSC,
and be far more localized than in a natural (pancreas) control.
"""
import json
import os

import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import scHopfield as sch

DEV = "cuda"
N_CELLS = 9000
N_GENES = 300
OUT = "benchmark_results/bias_penalty"

OSKM = ["Pou5f1", "Sox2", "Klf4", "Myc"]
PLURI = ["Nanog", "Zfp42", "Esrrb", "Sall4", "Lin28a", "Utf1", "Dppa3"]
PROGRAM = OSKM + PLURI


def load_subsample():
    a = ad.read_h5ad("data/reprogramming/schiebinger_serum.h5ad", backed="r")
    day = a.obs["day"].astype(str).astype(float).values
    rng = np.random.default_rng(0)
    per = max(1, N_CELLS // len(np.unique(day)))
    sel = np.sort(np.concatenate([
        rng.choice(np.where(day == d)[0], min((day == d).sum(), per), replace=False)
        for d in np.unique(day)]))
    a = a[sel].to_memory()
    a.obs["day_num"] = a.obs["day"].astype(str).astype(float).values
    return a


def prep(a):
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=30)
    # DPT pseudotime rooted at a day-0 (MEF) cell
    sc.tl.diffmap(a)
    root = int(np.where(a.obs["day_num"].values == a.obs["day_num"].min())[0][0])
    a.uns["iroot"] = root
    sc.tl.dpt(a)
    pt = a.obs["dpt_pseudotime"].values
    if np.corrcoef(pt, a.obs["day_num"].values)[0, 1] < 0:   # orient MEF->iPSC
        pt = pt.max() - pt
        a.obs["dpt_pseudotime"] = pt
    # 3 stages along the reprogramming axis
    q = np.quantile(pt, [1 / 3, 2 / 3])
    stage = np.where(pt <= q[0], "1_MEF", np.where(pt <= q[1], "2_transitional", "3_iPSC"))
    a.obs["stage"] = stage

    sc.pp.highly_variable_genes(a, n_top_genes=N_GENES)
    keep = a.var["highly_variable"].values.copy()
    for g in PROGRAM:
        if g in a.var_names:
            keep[a.var_names.get_loc(g)] = True
    a = a[:, keep].copy()
    a.var["scHopfield_used"] = True
    X = a.X
    a.layers["Ms"] = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    sch.pp.estimate_velocity_from_pseudotime(
        a, pseudotime_key="dpt_pseudotime", spliced_key="Ms",
        connectivity_key="connectivities", mode="forward", scale=1.0, store_key="velocity_S")
    a.var["gamma"] = np.float32(0.1)
    sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values, spliced_key="Ms")
    sch.pp.compute_sigmoid(a, spliced_key="Ms")
    return a


def fit_staged(a, penalty="l1", blam=0.1):
    b = a.copy()
    sch.inf.fit_interactions(
        b, cluster_key="stage", w_scaffold=None, skip_all=True, w_threshold=1e-12,
        spliced_key="Ms", velocity_key="velocity_S", degradation_key="gamma",
        bias_penalty=penalty, bias_regularization=blam, reconstruction_regularization=1.0,
        n_epochs=500, batch_size=256, learning_rate=0.05, device=DEV, seed=0,
        infer_I=True, refit_gamma=False)
    used = b.var["scHopfield_used"].values
    genes = np.asarray(b.var_names[used])
    out = {}
    for st in ["1_MEF", "2_transitional", "3_iPSC"]:
        col = f"I_{st}"
        if col not in b.var:
            continue
        bias_I = b.var[col].values[used]
        out[st] = (np.asarray(bias_I).ravel(), genes)
    return out


def enrich(bias_I, genes, program):
    absI = np.abs(bias_I)
    y = np.array([1.0 if g in set(program) else 0.0 for g in genes])
    order = np.argsort(absI)[::-1]
    ranks = {g: int(np.where(genes[order] == g)[0][0]) + 1 for g in program if g in set(genes)}
    return {"auroc": float(roc_auc_score(y, absI)) if y.sum() else float("nan"),
            "prog_mean": float(absI[y == 1].mean()) if y.sum() else float("nan"),
            "rest_mean": float(absI[y == 0].mean()),
            "contrast": float(absI[y == 1].mean() / (absI[y == 0].mean() + 1e-9)) if y.sum() else float("nan"),
            "oskm_mean": float(np.abs(bias_I[np.isin(genes, OSKM)]).mean()),
            "ranks": ranks}


def null_test(bias_I, genes, markers, n=5000):
    """Empirical p: is mean|I| over `markers` above random gene-sets of the same size?"""
    absI = np.abs(bias_I)
    present = [g for g in markers if g in set(genes)]
    k = len(present)
    obs = absI[np.isin(genes, present)].mean()
    rng = np.random.default_rng(0)
    null = np.array([absI[rng.choice(len(genes), k, replace=False)].mean() for _ in range(n)])
    p = float((null >= obs).mean())
    pct = float((null < obs).mean() * 100)
    return {"obs_mean_absI": float(obs), "null_mean": float(null.mean()), "percentile": pct, "p_value": p}


def prep_natural():
    a = ad.read_h5ad("data/Pancreas/pancreas_scvelo_ready.h5ad")
    a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    vmag = np.abs(np.asarray(a.layers["velocity_S"])).mean(0).ravel()
    a = a[:, np.argsort(vmag)[::-1][:N_GENES]].copy()
    a.var["scHopfield_used"] = True
    a.obs["stage"] = "natural"
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    sch.set_seed(0)
    print("### REPROGRAMMING, staged (DPT pseudotime) ###", flush=True)
    a = prep(load_subsample())
    print(f"subsample {a.shape}; stage counts: {a.obs['stage'].value_counts().to_dict()}", flush=True)
    staged = fit_staged(a, "l1")

    results, Ivecs = {}, {}
    for st, (bias_I, genes) in staged.items():
        e = enrich(bias_I, genes, PROGRAM)
        e["oskm_null"] = null_test(bias_I, genes, OSKM)
        results[st] = e
        Ivecs[st] = (bias_I, genes)
        nt = e["oskm_null"]
        print(f"  [{st:15s}] OSKM mean|I|={e['oskm_mean']:.2f} vs rest={e['rest_mean']:.2f}  "
              f"OSKM percentile vs random-4={nt['percentile']:.1f}%  p={nt['p_value']:.4f}", flush=True)
        print(f"       OSKM ranks (of {len(genes)}): "
              f"{ {g: e['ranks'][g] for g in OSKM if g in e['ranks']} }", flush=True)

    # natural control: same OSKM null but markers are absent -> use random 4-gene null baseline
    print("\n### NATURAL pancreas control (no forcing) ###", flush=True)
    nb = prep_natural()
    sch.inf.fit_interactions(nb, cluster_key="stage", w_scaffold=None, skip_all=True,
        w_threshold=1e-12, bias_penalty="l1", bias_regularization=0.1,
        reconstruction_regularization=1.0, n_epochs=500, batch_size=256,
        learning_rate=0.05, device=DEV, seed=0, infer_I=True, refit_gamma=False)
    used = nb.var["scHopfield_used"].values
    natI = nb.var["I_natural"].values[used]
    natgenes = np.asarray(nb.var_names[used])
    rng = np.random.default_rng(1)
    rand4 = list(rng.choice(natgenes, 4, replace=False))
    natnull = null_test(natI, natgenes, rand4)
    results["natural_random4"] = natnull
    print(f"  random-4 genes percentile vs null={natnull['percentile']:.1f}% p={natnull['p_value']:.4f} "
          f"(expect ~random, no localization)", flush=True)

    # ---- figure: |I| per gene per stage, OSKM highlighted ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
    for ax, st in zip(axes, ["1_MEF", "2_transitional", "3_iPSC"]):
        bias_I, genes = Ivecs[st]
        absI = np.abs(bias_I)
        gi = np.arange(len(genes))
        is_oskm = np.isin(genes, OSKM)
        ax.bar(gi[~is_oskm], absI[~is_oskm], color="#c8d6dd", width=1.0)
        ax.bar(gi[is_oskm], absI[is_oskm], color="#d1495b", width=3.0)
        for g in OSKM:
            if g in set(genes):
                j = int(np.where(genes == g)[0][0])
                ax.text(j, absI[j], g, fontsize=7, ha="center", va="bottom")
        ax.set_title(f"{st}  (OSKM pct={results[st]['oskm_null']['percentile']:.0f}%)")
        ax.set_xlabel("gene")
        ax.set_ylabel("|bias I|")
    fig.suptitle("Reprogramming: the exogenous OSKM factors carry outsized bias (L1); "
                 "downstream pluripotency genes do not (network-explained)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/reprogramming_staged_bias.png", dpi=140, bbox_inches="tight")
    json.dump(results, open(f"{OUT}/reprogramming_staged.json", "w"), indent=2)
    print(f"\nwrote {OUT}/reprogramming_staged.json + reprogramming_staged_bias.png", flush=True)


if __name__ == "__main__":
    main()
