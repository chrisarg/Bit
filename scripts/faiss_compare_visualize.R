#!/usr/bin/env Rscript
# faiss_compare_visualize.R -- Distribution report for the FAISS comparison.
#
# Reads the long-format per-iteration CSV produced by scripts/faiss_compare.pl
# (benchmark_FAISS/faiss_compare_results.csv), and produces a multipage PDF of
# boxplots plus a per-cell summarized CSV. Runs agnostically of the caller's
# working directory.

# --- 1. Dependencies: check and install before loading -----------------------
required_pkgs <- c("data.table", "ggplot2", "this.path", "bit64")
invisible(sapply(required_pkgs, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}))
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(this.path)
  library(bit64)
})

# --- 2. Path resolution and data ingestion ----------------------------------
script_dir <- this.path::this.dir()
repo_root  <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
out_dir    <- file.path(repo_root, "benchmark_FAISS")

csv_files <- list.files(out_dir,
                        pattern = "^faiss_compare_results.*\\.csv$",
                        full.names = TRUE)
if (length(csv_files) == 0L) {
  stop("No faiss_compare_results*.csv files found in ", out_dir,
       ". Run scripts/faiss_compare.pl first.")
}

dt <- rbindlist(lapply(csv_files, fread), use.names = TRUE, fill = TRUE)

# --- 3. Defensive validation --------------------------------------------------
required_cols <- c("Backend", "Build", "Device", "Bitset_Bits", "Top_K",
                   "Num_Queries", "Num_Refs", "Threads", "Iteration",
                   "Timing_ns")
missing_cols <- setdiff(required_cols, names(dt))
if (length(missing_cols) > 0L) {
  stop("Missing required column(s) in the results CSV: ",
       paste(missing_cols, collapse = ", "))
}

# --- 4. data.table transformations -------------------------------------------
dt[, Timing_ns := as.numeric(Timing_ns)]
dt[, Throughput := 1e9 / Timing_ns]

factor_cols <- c("Backend", "Build", "Device", "Bitset_Bits", "Top_K",
                 "Num_Refs")
dt[, (factor_cols) := lapply(.SD, as.factor), .SDcols = factor_cols]
# Keep the numeric dimensions in ascending order on axes/facets.
dt[, Bitset_Bits := factor(Bitset_Bits,
                           levels = sort(as.numeric(levels(Bitset_Bits))))]
dt[, Top_K := factor(Top_K, levels = sort(as.numeric(levels(Top_K))))]
dt[, Num_Refs := factor(Num_Refs,
                        levels = sort(as.numeric(levels(Num_Refs))))]

# --- 5. Per-cell summary (mean/median/sd/min/max) -----------------------------
summary_dt <- dt[, .(
  N               = .N,
  Mean_ns         = mean(Timing_ns, na.rm = TRUE),
  Median_ns       = median(Timing_ns, na.rm = TRUE),
  Sd_ns           = sd(Timing_ns, na.rm = TRUE),
  Min_ns          = min(Timing_ns, na.rm = TRUE),
  Max_ns          = max(Timing_ns, na.rm = TRUE),
  Mean_Throughput = mean(Throughput, na.rm = TRUE)
), by = .(Backend, Build, Device, Bitset_Bits, Top_K, Num_Refs)]
setorder(summary_dt, Backend, Build, Bitset_Bits, Top_K, Num_Refs)

summary_csv <- file.path(out_dir, "faiss_compare_summary.csv")
fwrite(summary_dt, summary_csv)

# --- 6. Visualization ----------------------------------------------------------
pdf_path <- file.path(out_dir, "faiss_compare_report.pdf")
pdf(pdf_path, width = 11, height = 8.5, onefile = TRUE)

backend_levels <- levels(dt$Backend)
pal <- scales::hue_pal()(length(backend_levels))
names(pal) <- backend_levels

# Plot 1: per-iteration time distribution. x = bitset size, rows = database
# size, columns = top_k; the four builds are dodged at each x tick.
p1 <- ggplot(dt,
             aes(x = Bitset_Bits, y = Timing_ns, fill = Build,
                 color = Build)) +
  geom_boxplot(position = position_dodge2(preserve = "single", padding = 0.1),
               outlier.alpha = 0.3, na.rm = TRUE) +
  facet_grid(Num_Refs ~ Top_K, scales = "free_y", labeller = label_both) +
  scale_y_continuous(trans = "log10") +
  labs(title = "FAISS vs Bit: per-iteration time distribution",
       subtitle = "Boxplots of per-iteration end-to-end time (ns), log scale",
       x = "Bitset size (bits)", y = "Iteration time (ns, log10)") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")
print(p1)
ggsave(file.path(out_dir, "faiss_compare_time_distribution.png"),
       p1, width = 11, height = 8.5, dpi = 120)

# Plot 2: throughput distribution, same layout as Plot 1.
p2 <- ggplot(dt,
             aes(x = Bitset_Bits, y = Throughput, fill = Build,
                 color = Build)) +
  geom_boxplot(position = position_dodge2(preserve = "single", padding = 0.1),
               outlier.alpha = 0.3, na.rm = TRUE) +
  facet_grid(Num_Refs ~ Top_K, scales = "free_y", labeller = label_both) +
  labs(title = "FAISS vs Bit: per-iteration throughput distribution",
       subtitle = "Boxplots of per-iteration searches/sec",
       x = "Bitset size (bits)", y = "Throughput (searches/sec)") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")
print(p2)
ggsave(file.path(out_dir, "faiss_compare_throughput_distribution.png"),
       p2, width = 11, height = 8.5, dpi = 120)

# Plot 3: median time trend across bitset sizes. Points, lines, and error
# bars share one dodge so each build is offset around its x tick instead of
# overlapping.
trend_dodge <- position_dodge(width = 0.35)
p3 <- ggplot(summary_dt,
             aes(x = Bitset_Bits, y = Median_ns, color = Build,
                 linetype = Build, group = Build)) +
  geom_point(size = 2, position = trend_dodge, na.rm = TRUE) +
  geom_line(position = trend_dodge, na.rm = TRUE) +
  geom_errorbar(aes(ymin = Median_ns - Sd_ns, ymax = Median_ns + Sd_ns),
                width = 0.1, position = trend_dodge, na.rm = TRUE) +
  facet_grid(Num_Refs ~ Top_K, scales = "free_y", labeller = label_both) +
  scale_y_continuous(trans = "log10") +
  labs(title = "Median per-iteration time vs bitset size",
       subtitle = "Points show median; error bars show +/-1 sd",
       x = "Bitset size (bits)", y = "Median iteration time (ns, log10)") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")
print(p3)
ggsave(file.path(out_dir, "faiss_compare_median_trend.png"),
       p3, width = 11, height = 8.5, dpi = 120)

dev.off()

cat("Wrote", nrow(dt), "per-iteration rows across", length(csv_files),
    "CSV file(s)\n")
cat("Summary CSV :", summary_csv, "\n")
cat("Report PDF  :", pdf_path, "\n")
