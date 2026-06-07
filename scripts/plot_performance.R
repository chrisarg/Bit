# install dependencies ggplot2, data.table, this.path, bit64 if not already installed
sapply(c("ggplot2", "data.table", "this.path", "bit64"), function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
})
library(ggplot2)
library(this.path)
library(data.table)

# Load the data

script_dir <- this.dir()
data_dir <- file.path(dirname(script_dir), "benchmark_GPU_params")

csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(csv_files) == 0) {
  stop("No CSV files found in ", data_dir)
}

data <- rbindlist(lapply(csv_files, fread), use.names = TRUE, fill = TRUE)

data[, `:=`(
  QxR = `number of queries` * `number of reference sequences`,
  data_workload = `number of queries` * `number of reference sequences` * (`number of bits in bitsets` / 8)^2
)]

ggplot(data[`timing type` == "total"],
  aes(
    x = as.factor(TILE_J),
    y = `iterations per second`*(`number of bits in bitsets`/65536)^2,
  fill = as.factor(ILP))
) +
  facet_grid(`number of queries` ~ `number of reference sequences`, labeller = label_both,scales="free_y") +
  geom_boxplot() +
  labs(title = "Total Time", x = "Tile J", y = "Iterations per Second /(Q x R in Billions)") +
  theme_minimal() +
  theme(legend.position = "bottom")
