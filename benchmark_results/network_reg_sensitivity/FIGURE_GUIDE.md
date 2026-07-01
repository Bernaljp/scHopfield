# Network × Regularization Sensitivity — Figure Guide

**Experiment.** On the Paul et al. (2015) mouse hematopoiesis dataset, scHopfield was
fit under **6 configurations** = 2 CellOracle mouse base GRNs (`scATAC_atlas`,
`promoter`) × 3 scaffold-regularization regimes (**free** = 0, **low** = 0.01,
**high** = 1.0 elastic-net), all with a fixed seed. A 7th fit uses **no scaffold**
(Moore–Penrose pseudoinverse: a dense W with no TF restriction). For every
configuration genes are ranked two ways:

- **driver score** — `score_driver_tfs` (W-norm + out-degree + energy–gene correlation),
- **perturbation** — in-silico single-KO ranked by |lineage bias| over a fixed candidate set.

**Headline.** Perturbation-based drivers are *robust* to network and regularization
(mean pairwise Jaccard **0.67**) and are the *canonical* regulators (Gata1, Spi1, Klf1,
E2f4, Stat3, Irf8). The static driver score is *unstable* (Jaccard **0.20**) and
dominated by ribosomal/housekeeping genes. The two rankings are **almost disjoint**
(0–1 shared genes). Dropping the scaffold entirely is an outlier that loses Gata1/Klf1.

---

## 1. Summary figure

![summary](sensitivity.png)

Left: mean pairwise Jaccard of the top-15 lists. Perturbation within-scaffold = 0.67
(green) vs the no-scaffold contrast = 0.36 (red); both score rankings ≈ 0.20 (grey).
Right: presence of each top-10 perturbation driver per setting; the scaffold columns
agree, the red no-scaffold column loses Gata1/Klf1.

---

## 2. Stability heatmaps (Jaccard of top-15 lists)

Raw (fixed order) and **clustered** (rows/cols reordered by similarity) versions.

| perturbation | score (erythroid) | score (myeloid) |
|---|---|---|
| ![](plots/1_jaccard_top_pert.png) | ![](plots/1_jaccard_top_score_ery.png) | ![](plots/1_jaccard_top_score_mye.png) |
| ![](plots/C_jaccard_top_pert.png) | ![](plots/C_jaccard_top_score_ery.png) | ![](plots/C_jaccard_top_score_mye.png) |

The perturbation matrix is uniformly bright (all scaffold settings agree ~0.6–0.8);
the lone dark row/column is `NO_SCAFFOLD`. The two score matrices are mostly dark
(0.0–0.4): static scores share few top genes across settings. In the **clustered**
versions the six scaffold settings form one tight block and `NO_SCAFFOLD` splits off
as its own branch; the two `free` settings are identical (Jaccard 1.0).

---

## 3. Gene recurrence maps (presence in top-10)

| perturbation | score-ery | score-mye |
|---|---|---|
| ![](plots/2_recurrence_top_pert.png) | ![](plots/2_recurrence_top_score_ery.png) | ![](plots/2_recurrence_top_score_mye.png) |

Rows are genes, columns are settings; a filled cell means the gene is in that setting's
top-10. Perturbation: a solid block of canonical drivers is present in every column
(except no-scaffold, which drops Gata1/Klf1). Score: sparse and scattered, i.e. the
"top" score genes change from setting to setting.

**Biclustered perturbation recurrence** (genes and settings both reordered):

![](plots/C_recurrence_biclustered.png)

Groups genes that co-occur; the always-present core (Spi1, Gata1, Klf1, Stat3, Irf8, E2f4)
clusters at the top, and no-scaffold's idiosyncratic genes (Cebpe, Gfi1, Myb, Fli1, Meis1)
separate out.

---

## 4. KO lineage-bias values

![](plots/3_bias_value_heatmap.png)

Signed KO lineage bias per gene per setting (red = erythroid-biasing, blue =
myeloid-biasing; blank = not in that setting's top-15). Gata1 and Klf1 are consistently
blue (KO pushes myeloid), Spi1/Irf8 consistently red (KO pushes erythroid), with stable
magnitudes across settings.

**Biclustered** (NaN filled to 0 for clustering):

![](plots/C_bias_biclustered.png)

Genes with similar bias profiles group together; the erythroid-master block (Gata1,
Klf1, negative) and myeloid-master block (Spi1, Irf8, positive) are clearly separated,
and the no-scaffold column stands apart.

### Bias trajectories (canonical genes)

![](plots/4_bias_trajectories.png)

Each line is one canonical gene's KO bias across the 6 settings. Signs never flip;
Gata1 (≈ −0.21) and Spi1 (≈ +0.25) are the strongest and most stable; E2f4 strengthens
as regularization increases.

---

## 5. Structure and comparisons

### Settings dendrogram
![](plots/5_settings_dendrogram.png)

Settings clustered by 1 − Jaccard of their perturbation lists. The two `free` settings
merge first (identical); the six scaffold settings form one clade.

### Static-score vs perturbation overlap
![](plots/6_score_vs_pert_overlap.png)

Number of genes shared between the top-15 score list and the top-15 perturbation list,
per setting. It is **0–1 in every setting**: the two "driver" definitions pick almost
entirely different genes. Given that perturbation genes are the canonical regulators,
this argues the static score is not a reliable driver readout.

### Perturbation-driver recurrence frequency
![](plots/7_consensus_frequency.png)

How many of the 6 settings each gene appears in (top-10). A core set (Spi1, Gata1, Klf1,
Stat3, Irf8, E2f4) hits all 6 (green); the rest are setting-specific.

### Stability by comparison type
![](plots/8_stability_by_type.png)

Perturbation Jaccard split by comparison: within-network ≈ cross-network (both ~0.6–0.7),
but vs no-scaffold drops to 0.36. Network choice barely matters; having a scaffold does.

### Effect of regularization within each network
![](plots/9_reg_effect.png)

Pairwise Jaccard of free/low/high within each network. Adjacent regimes (free↔low,
low↔high) are more similar than the extremes (free↔high), i.e. rankings drift smoothly
with regularization but never collapse.

### Per-setting top-10 perturbation bars
![](plots/10_pert_bars_per_setting.png)

The actual top-10 KO drivers and their bias values for each setting (red = erythroid,
blue = myeloid). Visual confirmation that the same faces recur, and that no-scaffold
looks different.

### Rank bump chart
![](plots/11_rank_bump.png)

Each line tracks one gene's rank in the perturbation list across settings. Canonical
drivers stay near the top; secondary genes cross and reshuffle.

### No-scaffold vs each setting
![](plots/12_no_scaffold_vs_each.png)

Jaccard of the no-scaffold lists against each scaffold setting, for all three rankings.
Perturbation overlap tops out ~0.4; score overlap is near zero.

### Erythroid vs myeloid score overlap
![](plots/13_score_ery_mye_overlap.png)

How many genes appear in *both* the top-15 erythroid and top-15 myeloid score lists per
setting: a few generic high-expression genes score highly for both lineages, another
sign the static score is not lineage-specific.

---

## Takeaways

1. **Use perturbation, not the static score, for driver discovery.** The two are nearly
   disjoint; only perturbation surfaces the canonical regulators.
2. **Perturbation drivers are robust** to network choice and regularization strength.
3. **A TF scaffold matters; which one does not.** No-scaffold dilutes regulatory
   influence and loses Gata1/Klf1.
