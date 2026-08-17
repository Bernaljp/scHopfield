#!/usr/bin/env Rscript
# Tiny dyngen run to learn the model/dataset structure (field names for spliced/unspliced,
# ground-truth network, feature/cell info) before writing the full export.
suppressMessages(library(dyngen))
set.seed(1)
bb <- backbone_linear()
config <- initialise_model(
  backbone = bb, num_cells = 100, num_tfs = 12, num_targets = 20, num_hks = 8,
  simulation_params = simulation_default(
    census_interval = 5, ssa_algorithm = ssa_etl(tau = 300/3600),
    experiment_params = simulation_type_wild_type(num_simulations = 10)),
  verbose = TRUE, download_cache_dir = "~/.cache/dyngen")
out <- generate_dataset(config, make_plots = FALSE)
model <- out$model; dataset <- out$dataset
cat("\n=== names(model) ===\n"); print(names(model))
cat("\n=== names(model$experiment) ===\n"); print(names(model$experiment))
for (nm in names(model$experiment)) {
  x <- model$experiment[[nm]]
  if (is.matrix(x) || inherits(x, "Matrix")) cat(sprintf("  experiment$%s : %s  dim %s\n", nm, class(x)[1], paste(dim(x), collapse="x")))
}
cat("\n=== names(dataset) ===\n"); print(names(dataset))
for (nm in names(dataset)) {
  x <- dataset[[nm]]
  if (is.matrix(x) || inherits(x, "Matrix")) cat(sprintf("  dataset$%s : %s  dim %s\n", nm, class(x)[1], paste(dim(x), collapse="x")))
}
cat("\n=== feature_network head ===\n"); print(head(as.data.frame(model$feature_network)))
cat("\n=== feature_info cols ===\n"); print(colnames(model$feature_info))
cat("\n=== feature_info head ===\n"); print(head(as.data.frame(model$feature_info)))
cat("\n=== cell_info / progressions ===\n")
if (!is.null(dataset$cell_info)) print(head(as.data.frame(dataset$cell_info)))
if (!is.null(dataset$progressions)) print(head(as.data.frame(dataset$progressions)))
cat("\nINTROSPECT DONE\n")
