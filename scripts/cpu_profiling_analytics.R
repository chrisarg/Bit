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

data <- rbindlist(lapply(file_list, fread), fill = TRUE)

# 2. Data Mutation
data[, c("Throughput", "Registers", "Bitset_Size", "LIBPOPCNT", "vpopcountHW", "Benchmark_Type") := .(
  1e9 / Timing_ns,
  (OUTER_ROW_NUM * OUTER_COL_NUM) + OUTER_ROW_NUM + OUTER_COL_NUM,
  as.factor(Bitset_Size),
  as.factor(LIBPOPCNT),
  as.factor(vpopcountHW),
  as.factor(Benchmark_Type)
)]

max_thread_val <- max(data$Threads, na.rm = TRUE)

# 3. Optimal Configuration Extractor (Handles both benchmark types dynamically)
optimal_configs <- data[, .SD[which.max(Throughput)],
  by = .(Processor, SIMD, vpopcountHW, LIBPOPCNT, Benchmark_Type, Bitset_Size, Threads)
]

fwrite(optimal_configs, "benchmark_CPU_params/optimal_cpu_parameters.csv")

# =====================================================================
# REPORT 1: CONTAINERIZED BENCHMARKS
# =====================================================================
pdf("benchmark_CPU_params/benchmark_CPU_analytics_Containerized.pdf", width = 12, height = 8)

opt_cont <- optimal_configs[Benchmark_Type == "Containerized"]
data_cont <- data[Benchmark_Type == "Containerized"]

# Plot 1.1: Optimization Frontier
p1 <- ggplot(opt_cont, aes(x = Threads, y = Throughput, color = Bitset_Size, linetype = vpopcountHW)) +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_grid(LIBPOPCNT ~ SIMD, labeller = label_both) +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Optimization Frontier: Throughput vs Threads", y = "Searches / Sec")
print(p1)

# Plot 1.2: Register Pressure Analysis (Filtered to max threads)
reg_data <- data_cont[Threads == max_thread_val, .SD[which.max(Throughput)], by = .(SIMD, LIBPOPCNT, Registers)]

p2 <- ggplot(reg_data, aes(x = Registers, y = Throughput, color = LIBPOPCNT)) +
  geom_line(linewidth = 1) + geom_point(size = 3) +
  geom_vline(xintercept = 16, linetype = "dashed", color = "darkred", alpha = 0.6) +
  geom_vline(xintercept = 32, linetype = "dashed", color = "darkblue", alpha = 0.6) +
  facet_wrap(~SIMD, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Register Pressure vs Throughput (Max Threads)", subtitle = "Lines at 16 (AVX2/NEON) and 32 (AVX-512) architectural limits")
print(p2)

# Plot 1.3: Microkernel Tiling Heatmaps
heat_data <- data_cont[Threads == max_thread_val, .SD[which.max(Throughput)], by = .(SIMD, LIBPOPCNT, OUTER_ROW_NUM, OUTER_COL_NUM)]

p3 <- ggplot(heat_data, aes(x = as.factor(OUTER_COL_NUM), y = as.factor(OUTER_ROW_NUM), fill = Throughput)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c() +
  facet_grid(LIBPOPCNT ~ SIMD, labeller = label_both) +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Microkernel Tiling Heatmap", x = "Outer Col Num", y = "Outer Row Num")
print(p3)

# Plot 1.4: Cache Saturation Curve
cache_data <- data_cont[Threads == max_thread_val, .SD[which.max(Throughput)], by = .(SIMD, Bitset_Size, LIBPOPCNT, vpopcountHW, BUFFER_SIZE)]

p4 <- ggplot(cache_data, aes(x = BUFFER_SIZE, y = Throughput, color = LIBPOPCNT, shape = vpopcountHW)) +
  geom_line(linewidth = 1) + geom_point(size = 3) +
  scale_x_continuous(trans = "log2") +
  facet_grid(Bitset_Size ~ SIMD, labeller = label_both, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Cache Saturation Curve (Log2 Memory Wall)", x = "SETOP_BUFFER_SIZE (Words)")
print(p4)

# Plot 1.5: Amdahl's Efficiency
baseline_cont <- opt_cont[Threads == min(Threads), .(Processor, SIMD, LIBPOPCNT, Bitset_Size, Base_Throughput = Throughput)]
amdahl_data_cont <- opt_cont[baseline_cont, on = .(Processor, SIMD, LIBPOPCNT, Bitset_Size)]
amdahl_data_cont[, Speedup := Throughput / Base_Throughput]

p5 <- ggplot(amdahl_data_cont, aes(x = Threads, y = Speedup, color = SIMD, linetype = LIBPOPCNT)) +
  geom_line(linewidth = 1) + 
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dotted") +
  facet_wrap(~Bitset_Size) +
  theme_minimal(base_size = 14) +
  labs(title = "[Containerized] Amdahl's Efficiency (Speedup Factor)", subtitle = "Dotted line represents theoretical ideal scaling")
print(p5)
dev.off()

# =====================================================================
# REPORT 2: NON-CONTAINERIZED BENCHMARKS
# =====================================================================
pdf("benchmark_CPU_params/benchmark_CPU_analytics_NonContainerized.pdf", width = 12, height = 8)

opt_non <- optimal_configs[Benchmark_Type == "Non-Containerized"]
data_non <- data[Benchmark_Type == "Non-Containerized"]

# Plot 2.1: Optimization Frontier
p6 <- ggplot(opt_non, aes(x = Threads, y = Throughput, color = Bitset_Size, linetype = vpopcountHW)) +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_grid(LIBPOPCNT ~ SIMD, labeller = label_both) +
  theme_minimal(base_size = 14) +
  labs(title = "[Non-Containerized] Optimization Frontier: Throughput vs Threads", y = "Searches / Sec")
print(p6)

# Plot 2.2: Cache Saturation Curve
cache_data_non <- data_non[Threads == max_thread_val, .SD[which.max(Throughput)], by = .(SIMD, Bitset_Size, LIBPOPCNT, vpopcountHW, BUFFER_SIZE)]

p7 <- ggplot(cache_data_non, aes(x = BUFFER_SIZE, y = Throughput, color = LIBPOPCNT, shape = vpopcountHW)) +
  geom_line(linewidth = 1) + geom_point(size = 3) +
  scale_x_continuous(trans = "log2") +
  facet_grid(Bitset_Size ~ SIMD, labeller = label_both, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(title = "[Non-Containerized] Cache Saturation Curve (Log2 Memory Wall)", x = "SETOP_BUFFER_SIZE (Words)")
print(p7)

# Plot 2.3: Amdahl's Efficiency
baseline_non <- opt_non[Threads == min(Threads), .(Processor, SIMD, LIBPOPCNT, Bitset_Size, Base_Throughput = Throughput)]
amdahl_data_non <- opt_non[baseline_non, on = .(Processor, SIMD, LIBPOPCNT, Bitset_Size)]
amdahl_data_non[, Speedup := Throughput / Base_Throughput]

p8 <- ggplot(amdahl_data_non, aes(x = Threads, y = Speedup, color = SIMD, linetype = LIBPOPCNT)) +
  geom_line(linewidth = 1) + 
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dotted") +
  facet_wrap(~Bitset_Size) +
  theme_minimal(base_size = 14) +
  labs(title = "[Non-Containerized] Amdahl's Efficiency (Speedup Factor)", subtitle = "Dotted line represents theoretical ideal scaling")
print(p8)
dev.off()