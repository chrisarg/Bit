import faiss
import numpy as np
import time

def run_multi_gpu_benchmark():
    # --- 1. Define Workload Parameters ---
    d_bits = 8192                # Vector size in bits
    d_bytes = d_bits // 8         # 8,192 bytes per vector
    num_a = 2048                  # Matrix A (Queries)
    num_b = 2048                  # Matrix B (Database)
    k = num_a                     # Retrieve all pairs
    iterations = 10              # Number of times to run the full search

    print(f"--- FAISS Multi-GPU Binary Search Benchmark ---")
    print(f"Matrix A: {num_a} vectors, {d_bits}-bit ({d_bytes} bytes)")
    print(f"Matrix B: {num_b} vectors, {d_bits}-bit ({d_bytes} bytes)")
    print(f"Task: Full {num_a} x {num_b} pairwise intersection matrix")
    if(k!=num_b):
        print(f"Return top {k} results per query vector")
    else:
        print(f"Return all {num_a} results per query vector")  
    print(f"Iterations: {iterations}\n")

    # --- 2. Generate Random Binary Data ---
    print("Generating binary matrices...")
    matrix_A = np.random.randint(256, size=(num_a, d_bytes), dtype=np.uint8)
    matrix_B = np.random.randint(256, size=(num_b, d_bytes), dtype=np.uint8)

    results = {}

    # --- 3. CPU prestaging ---
    print("\nInitializing CPU IndexBinaryFlat...")
    cpu_index = faiss.IndexBinaryFlat(d_bits)
    cpu_index.add(matrix_B)

    # --- 4. GPU Benchmarks ---
    try:
        num_gpus = faiss.get_num_gpus()
        print(f"\nDetected {num_gpus} GPU(s) via FAISS.")

        for gpu_id in range(num_gpus):
            print(f"\nInitializing GPU {gpu_id}...")
            res = faiss.StandardGpuResources()
            
            # Transfer the CPU index to the specific GPU ID
            gpu_index = faiss.index_binary_cpu_to_gpu(res, gpu_id, cpu_index)
            
            # Warmup
            _ = gpu_index.search(matrix_A, k)

            print(f"Running Benchmark on GPU {gpu_id}...")
            gpu_start = time.perf_counter()
            for _ in range(iterations):
                _ = gpu_index.search(matrix_A, k)
            gpu_end = time.perf_counter()

            gpu_searches_per_sec = iterations / (gpu_end - gpu_start)
            results[f"GPU {gpu_id}"] = gpu_searches_per_sec

    except AttributeError:
        print("\nERROR: FAISS-GPU is not installed or CUDA is not available.")

    # --- 5. Final Results Table ---
    print("\n=========================================")
    print("             FINAL RESULTS               ")
    print("=========================================")
    
    cpu_base = results["GPU 0"]
    print(f"{'Device':<10} | {'Throughput (searches/sec)':<25} | {'Speedup vs GPU 0'}")
    print("-" * 55)
    
    for device, throughput in results.items():
        speedup = throughput / cpu_base
        print(f"{device:<10} | {throughput:<25.2f} | {speedup:.2f}x")
    print("=========================================")

if __name__ == '__main__':
    run_multi_gpu_benchmark()