#!/usr/bin/env Rscript
# Simulate dyngen datasets with several backbones, each with a KNOWN ground-truth GRN
# (model$feature_network: from -> to, effect +-1) and spliced/unspliced counts (for RNA
# velocity). Export to reproducibility/data/dyngen/<name>/ as MatrixMarket + CSV; the Python
# side (02_build_h5ad.py) assembles AnnData and runs the benchmarks.
#
# Run:  bash reproducibility/dyngen/run.sh reproducibility/dyngen/01_simulate.R
suppressMessages(library(dyngen))

OUTROOT <- "reproducibility/data/dyngen"
CACHE   <- "~/.cache/dyngen"
NCELLS  <- 500
NTARGET <- 70
NHK     <- 20
NSIM    <- 24

simulate_one <- function(bb, name, seed = 1) {
  message(sprintf("=== simulating %s ===", name))
  set.seed(seed)
  ntf <- nrow(bb$module_info)
  config <- initialise_model(
    backbone = bb, num_cells = NCELLS,
    num_tfs = ntf, num_targets = NTARGET, num_hks = NHK,
    simulation_params = simulation_default(
      census_interval = 10, ssa_algorithm = ssa_etl(tau = 300 / 3600),
      experiment_params = simulation_type_wild_type(num_simulations = NSIM)),
    verbose = FALSE, download_cache_dir = CACHE)
  out <- generate_dataset(config, make_plots = FALSE)
  model <- out$model; dataset <- out$dataset

  d <- file.path(OUTROOT, name); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  Matrix::writeMM(dataset$counts_spliced,   file.path(d, "spliced.mtx"))
  Matrix::writeMM(dataset$counts_unspliced, file.path(d, "unspliced.mtx"))
  writeLines(colnames(dataset$counts_spliced), file.path(d, "genes.txt"))
  writeLines(rownames(dataset$counts_spliced), file.path(d, "cells.txt"))

  fn <- as.data.frame(model$feature_network)[, c("from", "to", "effect")]
  write.csv(fn, file.path(d, "feature_network.csv"), row.names = FALSE)
  fi <- as.data.frame(model$feature_info)
  keep <- intersect(c("feature_id", "module_id", "is_tf", "is_hk", "basal"), colnames(fi))
  write.csv(fi[, keep], file.path(d, "feature_info.csv"), row.names = FALSE)

  ci <- as.data.frame(dataset$cell_info)
  write.csv(ci, file.path(d, "cell_info.csv"), row.names = FALSE)
  if (!is.null(dataset$progressions))
    write.csv(as.data.frame(dataset$progressions), file.path(d, "progressions.csv"), row.names = FALSE)

  cat(sprintf("[DONE %s] %d cells x %d genes | %d GT edges | %d TFs | effects: %s\n",
      name, nrow(dataset$counts_spliced), ncol(dataset$counts_spliced),
      nrow(model$feature_network), sum(fi$is_tf),
      paste(names(table(fn$effect)), table(fn$effect), sep = ":", collapse = " ")))
}

backbones <- list(
  linear       = backbone_linear(),
  bifurcating  = backbone_bifurcating(),
  trifurcating = backbone_trifurcating(),
  cycle        = backbone_cycle(),
  converging   = backbone_converging(),
  binary_tree  = backbone_binary_tree()
)

for (nm in names(backbones)) {
  tryCatch(simulate_one(backbones[[nm]], nm),
           error = function(e) cat(sprintf("[FAIL %s] %s\n", nm, conditionMessage(e))))
}
cat("ALL SIMULATIONS DONE\n")
