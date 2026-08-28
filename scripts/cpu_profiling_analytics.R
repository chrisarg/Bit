library(data.table)
library(ggplot2)

# 1. Identify and Load Data
script_args <- commandArgs(trailingOnly = FALSE)
script_arg <- script_args[grepl("^--file=", script_args)]
script_path <- if (length(script_arg) > 0) sub("^--file=", "", script_arg[[1]]) else ""
script_dir <- if (nzchar(script_path)) dirname(script_path) else getwd()
repo_root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)

config_candidates <- unique(c(
  file.path(getwd(), "scripts", "benchmark_config_cpu.json"),
  file.path(getwd(), "benchmark_config_cpu.json"),
  file.path(script_dir, "benchmark_config_cpu.json"),
  file.path(script_dir, "..", "benchmark_config_cpu.json"),
  file.path(repo_root, "scripts", "benchmark_config_cpu.json"),
  file.path(repo_root, "benchmark_config_cpu.json")
))
config_path <- config_candidates[file.exists(config_candidates)][1]

config_out_dir <- NULL
if (!is.null(config_path) && !is.na(config_path) && nzchar(config_path)) {
  config_text <- paste(readLines(config_path, warn = FALSE), collapse = "\n")
  out_dir_match <- regexpr('"out_dir"\\s*:\\s*"([^"]+)"', config_text, perl = TRUE)
  if (out_dir_match[[1]] > 0) {
    config_out_dir <- regmatches(config_text, out_dir_match)[[1]]
    config_out_dir <- sub('.*"out_dir"\\s*:\\s*"', '', config_out_dir)
    config_out_dir <- sub('"$', '', config_out_dir)
  }
}

candidate_dirs <- unique(c(
  if (!is.null(config_out_dir) && nzchar(config_out_dir) && !grepl("^/", config_out_dir)) file.path(getwd(), config_out_dir) else NULL,
  if (!is.null(config_out_dir) && nzchar(config_out_dir) && grepl("^/", config_out_dir)) config_out_dir else NULL,
  file.path(getwd(), "benchmark_CPU_params"),
  file.path(getwd(), "scripts", "benchmark_CPU_params"),
  file.path(script_dir, "benchmark_CPU_params"),
  file.path(script_dir, "..", "benchmark_CPU_params"),
  file.path(repo_root, "benchmark_CPU_params")
))

output_dir <- NULL
for (candidate in candidate_dirs) {
  if (!is.null(candidate) && !is.na(candidate) && nzchar(candidate) && dir.exists(candidate)) {
    output_dir <- candidate
    break
  }
}

if (is.null(output_dir) || is.na(output_dir) || !nzchar(output_dir)) {
  output_dir <- if (!is.null(config_out_dir) && nzchar(config_out_dir)) {
    if (grepl("^/", config_out_dir)) config_out_dir else file.path(getwd(), config_out_dir)
  } else {
    file.path(repo_root, "benchmark_CPU_params")
  }
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

file_list <- list.files(
  path = output_dir,
  pattern = "\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)
file_list <- file_list[!basename(file_list) %in% c("optimal_cpu_parameters.csv")]

if (length(file_list) == 0) {
  stop(sprintf("No benchmark CSV files found in '%s'.", output_dir))
}

# Robust column-name based reading
data <- rbindlist(lapply(file_list, fread), fill = TRUE)

# 2. Normalize the current Perl producer schema to the names expected by this analysis
rename_map <- c(
  cc = "Compiler",
  opt_level = "Opt_Level",
  libpopcnt = "LIBPOPCNT",
  outer_row_num = "OUTER_ROW_NUM",
  outer_col_num = "OUTER_COL_NUM",
  outer_vec_blk = "OUTER_VEC_BLK",
  num_bits = "Bitset_Size",
  dim_left = "Dim_Left",
  dim_right = "Dim_Right",
  threads = "Threads",
  omp_bind = "OMP_Bind",
  numa_policy = "NUMA_Policy",
  buffer_size = "BUFFER_SIZE",
  cpu_tile = "CPU_TILE",
  bitvector_tile = "BITVECTOR_TILE",
  lto = "LTO",
  march = "MARCH"
)

for (old_name in names(rename_map)) {
  if (old_name %in% names(data) && !(rename_map[[old_name]] %in% names(data))) {
    names(data)[names(data) == old_name] <- rename_map[[old_name]]
  }
}

# 3. Strict Name-Based Data Mutation & Factor Casting
# Check required named columns exist
required_cols <- c("Timing_ns", "OUTER_ROW_NUM", "OUTER_COL_NUM", "Bitset_Size",
                   "LIBPOPCNT", "vpopcountHW", "Benchmark_Type", "Compiler", "Opt_Level")
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop(paste("Missing expected columns in CSV:", paste(missing_cols, collapse = ", ")))
}

# Compute Metrics by Column Name
data[, Throughput := 1e9 / Timing_ns]
data[, Registers  := (OUTER_ROW_NUM * OUTER_COL_NUM) + OUTER_ROW_NUM + OUTER_COL_NUM]

# Convert Categoricals dynamically by Name
factor_cols <- c("Bitset_Size", "LIBPOPCNT", "vpopcountHW", "Benchmark_Type", 
                 "Compiler", "Opt_Level", "OMP_Bind", "NUMA_Policy", "Cache_State")
valid_factors <- intersect(factor_cols, names(data))
data[, (valid_factors) := lapply(.SD, as.factor), .SDcols = valid_factors]

data[, vpopcountHW := as.character(vpopcountHW)]
data[, vpopcountHW := ifelse(vpopcountHW == "asimd", "vcntq_u8", vpopcountHW)]

# 3. Aggregate Repetitions over Run Matrix
# Grouping factors for repetition aggregation
group_cols <- intersect(c("Processor", "SIMD", "vpopcountHW", "Compiler", "Opt_Level", "LTO", "MARCH",
                          "LIBPOPCNT", "CPU_TILE", "BITVECTOR_TILE", "BUFFER_SIZE", 
                          "OUTER_ROW_NUM", "OUTER_COL_NUM", "OUTER_VEC_BLK", 
                          "Benchmark_Type", "Bitset_Size", "Dim_Left", "Dim_Right", 
                          "Threads", "OMP_Bind", "NUMA_Policy", "Cache_State", "Hugepages"), names(data))

aggregated_data <- data[, .(
  Mean_Throughput   = mean(Throughput, na.rm = TRUE),
  Median_Throughput = median(Throughput, na.rm = TRUE),
  SD_Throughput     = sd(Throughput, na.rm = TRUE),
  Rep_Count         = .N
), by = group_cols]

# 4. Optimal Configuration Extractor (Name-Grounded)
optimal_configs <- aggregated_data[, .SD[which.max(Median_Throughput)],
  by = .(Processor, SIMD, vpopcountHW, Compiler, Benchmark_Type, Bitset_Size, Threads)
]

fwrite(optimal_configs, file.path(output_dir, "optimal_cpu_parameters.csv"))

max_thread_val <- max(aggregated_data$Threads, na.rm = TRUE)

# =====================================================================
# REPORT GENERATION: CONTAINERIZED BENCHMARKS
# =====================================================================
pdf(file.path(output_dir, "benchmark_CPU_analytics_Containerized.pdf"), width = 12, height = 8)

opt_cont  <- optimal_configs[Benchmark_Type == "Containerized"]
data_cont <- aggregated_data[Benchmark_Type == "Containerized"]

# Plot 1: Optimization Frontier (Throughput vs Threads by Compiler)
p1 <- ggplot(opt_cont, aes(x = Threads, y = Median_Throughput, color = Bitset_Size, linetype = Compiler)) +
  geom_line(linewidth = 1) + 
  geom_point(size = 2) +
  facet_grid(LIBPOPCNT ~ SIMD, labeller = label_both) +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Optimization Frontier: Median Throughput vs Threads", 
       y = "Searches / Sec")
print(p1)

# Plot 2: Register Pressure vs Throughput
reg_data <- data_cont[Threads == max_thread_val, .SD[which.max(Median_Throughput)], 
                      by = .(SIMD, LIBPOPCNT, OUTER_ROW_NUM, OUTER_COL_NUM)]
reg_data[, Registers := (OUTER_ROW_NUM * OUTER_COL_NUM) + OUTER_ROW_NUM + OUTER_COL_NUM]

p2 <- ggplot(reg_data, aes(x = Registers, y = Median_Throughput, color = LIBPOPCNT)) +
  geom_line(linewidth = 1) + 
  geom_point(size = 3) +
  geom_vline(xintercept = 16, linetype = "dashed", color = "darkred", alpha = 0.6) +
  geom_vline(xintercept = 32, linetype = "dashed", color = "darkblue", alpha = 0.6) +
  facet_wrap(~SIMD, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Register Pressure vs Throughput (Max Threads)", 
       subtitle = "Lines at 16 (AVX2/NEON) and 32 (AVX-512) architectural register limits")
print(p2)

# Plot 3: Cache Saturation & Variance Range
cache_data <- data_cont[Threads == max_thread_val, .SD[which.max(Median_Throughput)], 
                        by = .(SIMD, Bitset_Size, LIBPOPCNT, vpopcountHW, BUFFER_SIZE)]

p3 <- ggplot(cache_data, aes(x = BUFFER_SIZE, y = Median_Throughput, color = LIBPOPCNT, shape = vpopcountHW)) +
  geom_line(linewidth = 1) + 
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Median_Throughput - SD_Throughput, ymax = Median_Throughput + SD_Throughput), width = 0.1) +
  scale_x_continuous(trans = "log2") +
  facet_grid(Bitset_Size ~ SIMD, labeller = label_both, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Cache Saturation Curve with Variance (Log2 Memory Wall)", 
       x = "SETOP_BUFFER_SIZE (Words)", y = "Median Throughput")
print(p3)

dev.off()