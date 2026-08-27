library(data.table)
library(ggplot2)

# 1. Identify and Load Data
file_list <- list.files(
  path = "benchmark_CPU_params",
  pattern = "^cpu_sweep_.*\\.csv$",
  full.names = TRUE
)

if (length(file_list) == 0) {
  stop("No benchmark CSV files found matching 'cpu_sweep_*.csv' in the target directory.")
}

# Robust column-name based reading
data <- rbindlist(lapply(file_list, fread), fill = TRUE)

# 2. Strict Name-Based Data Mutation & Factor Casting
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

data[, vpopcountHW := fcase(vpopcountHW == "asimd", "vcntq_u8", default = vpopcountHW)]

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

fwrite(optimal_configs, "benchmark_CPU_params/optimal_cpu_parameters.csv")

max_thread_val <- max(aggregated_data$Threads, na.rm = TRUE)

# =====================================================================
# REPORT GENERATION: CONTAINERIZED BENCHMARKS
# =====================================================================
pdf("benchmark_CPU_params/benchmark_CPU_analytics_Containerized.pdf", width = 12, height = 8)

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