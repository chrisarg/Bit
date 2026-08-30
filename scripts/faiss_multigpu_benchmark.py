# Run this script from the Conda environment containing the FAISS installation.
# First identify the environment and confirm that it contains FAISS:
#
#   conda env list
#   conda list -n <environment-name> faiss
#
# Activate it and verify that this Python interpreter sees the GPU build:
#
#   conda activate <environment-name>
#   which python
#   python -c "import faiss; print('FAISS GPUs:', faiss.get_num_gpus())"
#
# If `conda activate` reports that the shell is not initialized, run:
#
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate <environment-name>
#
# Then launch the benchmark from the repository root:
#
#   python scripts/faiss_multigpu_benchmark.py \
#     <size> <number-of-bitsets> <number-of-reference-bitsets> \
#     <top-k> <gpu-iterations> [gpu-id]
#
# For example:
#
#   python scripts/faiss_multigpu_benchmark.py 65536 1024 65536 256 10 0
#
# A GPU-enabled installation should report at least one device. A missing
# `faiss` module usually means that the script is using the wrong environment.

import argparse
import sys
import time

import faiss
import numpy as np


MIN_SIZE = 128
MAX_GPU_ITERATIONS = 1024


def positive_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark FAISS IndexBinaryFlat on CPU and one GPU."
    )
    parser.add_argument("size", type=positive_integer, help="bit-vector size")
    parser.add_argument(
        "num_bitsets", type=positive_integer, help="number of query bitsets"
    )
    parser.add_argument(
        "num_ref_bitsets",
        type=positive_integer,
        help="number of reference bitsets",
    )
    parser.add_argument("top_k", type=positive_integer, help="results per query")
    parser.add_argument(
        "gpu_iterations", type=positive_integer, help="measured search iterations"
    )
    parser.add_argument(
        "gpu_id", nargs="?", type=nonnegative_integer, default=0, help="GPU ID"
    )
    args = parser.parse_args()

    if args.size < MIN_SIZE:
        print(f"Warning: size increased to {MIN_SIZE}", file=sys.stderr)
        args.size = MIN_SIZE
    if args.size % 8 != 0:
        parser.error("size must be divisible by 8 for FAISS binary indexes")
    if args.gpu_iterations > MAX_GPU_ITERATIONS:
        print(
            f"Warning: gpu iterations capped to {MAX_GPU_ITERATIONS}",
            file=sys.stderr,
        )
        args.gpu_iterations = MAX_GPU_ITERATIONS
    if args.top_k > args.num_ref_bitsets:
        print(
            f"Warning: top-k decreased to {args.num_ref_bitsets}", file=sys.stderr
        )
        args.top_k = args.num_ref_bitsets

    return args

def generate_lcg_sequence(num_words: int, seed: int = 0xDEADBEEF):
    """
    Generates `num_words` uint64 values using the same LCG as the C code.
    Returns the numpy array and the final seed.
    """
    a = 6364136223846793005
    c = 1442695040888963407
    MASK64 = 0xFFFFFFFFFFFFFFFF

    seed = int(seed) & MASK64
    result = np.empty(num_words, dtype=np.uint64)
    
    for i in range(num_words):
        seed = (seed * a + c) & MASK64
        result[i] = seed
        
    return result, np.uint64(seed)

def benchmark_search(index, queries, top_k, iterations):
    _ = index.search(queries, top_k)
    records = []

    for iteration in range(1, iterations + 1):
        start_ns = time.perf_counter_ns()
        distances, ids = index.search(queries, top_k)
        elapsed_ns = time.perf_counter_ns() - start_ns

        records.append(
            {
                "iteration": iteration,
                "elapsed_ns": elapsed_ns,
                "best_distance": int(distances.min()),
                "distance_checksum": int(np.sum(distances, dtype=np.int64)),
                "id_checksum": int(np.sum(ids, dtype=np.int64)),
            }
        )

    elapsed = np.asarray([record["elapsed_ns"] for record in records], dtype=np.float64)
    mean_ns = float(elapsed.mean())
    return {
        "records": records,
        "mean_ns": mean_ns,
        "stddev_ns": float(elapsed.std()),
        "searches_per_sec": 1.0e9 / mean_ns if mean_ns > 0 else 0.0,
        "best_distance": min(record["best_distance"] for record in records),
        "distance_checksum": records[-1]["distance_checksum"],
        "id_checksum": records[-1]["id_checksum"],
    }


def print_iteration_results(device, summary):
    print(f"\nEnd-to-End FAISS Search Timings [{device}]:")
    print(
        f"{'Iteration':>9} | {'Time (ns)':>15} | "
        f"{'Searches/sec':>14} | {'Best distance':>13} | {'vs first':>9}"
    )
    print("-" * 76)

    first_ns = summary["records"][0]["elapsed_ns"]
    for record in summary["records"]:
        elapsed_ns = record["elapsed_ns"]
        searches_per_sec = 1.0e9 / elapsed_ns if elapsed_ns > 0 else 0.0
        speedup = first_ns / elapsed_ns if elapsed_ns > 0 else 0.0
        print(
            f"{record['iteration']:>9} | {elapsed_ns:>15d} | "
            f"{searches_per_sec:>14.2f} | "
            f"{record['best_distance']:>13d} | {speedup:>8.2f}x"
        )


def print_final_results(summaries, baseline_device, workload):
    print("\n" + "=" * 94)
    print("FAISS END-TO-END SEARCH SUMMARY")
    print("=" * 94)
    print(
        f"{'Device':<10} | {'Mean (ns)':>15} | {'Stddev (ns)':>15} | "
        f"{'Searches/sec':>14} | {'Speedup':>9} | {'Best distance':>13}"
    )
    print("-" * 94)

    baseline_ns = summaries[baseline_device]["mean_ns"]
    for device, summary in summaries.items():
        speedup = baseline_ns / summary["mean_ns"]
        print(
            f"{device:<10} | {summary['mean_ns']:>15.3f} | "
            f"{summary['stddev_ns']:>15.3f} | "
            f"{summary['searches_per_sec']:>14.2f} | "
            f"{speedup:>8.2f}x | {summary['best_distance']:>13d}"
        )

    print("=" * 94)
    for device, summary in summaries.items():
        machine_device = device.replace(" ", "_")
print(
            "\n"
            "================ SEARCH SUMMARY ================\n"
            "Backend                    : FAISS\n"
            "Method                     : IndexBinaryFlat\n"
            f"Device                     : {machine_device}\n"
            "Score Type                 : hamming_distance\n"
            "Score Order                : min\n"
            "Selection                  : FAISS_topk\n"
            "Timing Scope               : end_to_end_search_call\n"
            f"Bitset Bits                : {workload['bitset_bits']}\n"
            f"Num Queries                : {workload['num_queries']}\n"
            f"Num Refs                   : {workload['num_refs']}\n"
            f"Top K                      : {workload['top_k']}\n"
            f"Iterations                 : {workload['iterations']}\n"
            f"E2E Avg Time (ns)          : {summary['mean_ns']:.3f}\n"
            f"E2E StdDev Time (ns)       : {summary['stddev_ns']:.3f}\n"
            f"E2E Searches/sec           : {summary['searches_per_sec']:.6f}\n"
            f"Best Score                 : {summary['best_distance']}\n"
            f"Distance Checksum          : {summary['distance_checksum']}\n"
            f"ID Checksum                : {summary['id_checksum']}\n"
            "================================================"
        )


def run_multi_gpu_benchmark(args):
    # --- 1. Define Workload Parameters ---
    d_bits = args.size
    d_bytes = d_bits // 8
    num_a = args.num_bitsets
    num_b = args.num_ref_bitsets
    k = args.top_k
    iterations = args.gpu_iterations
    gpu_id = args.gpu_id

    print("--- FAISS Multi-GPU Binary Search Benchmark ---")
    print(f"Matrix A: {num_a} vectors, {d_bits}-bit ({d_bytes} bytes)")
    print(f"Matrix B: {num_b} vectors, {d_bits}-bit ({d_bytes} bytes)")
    print(f"Task: {num_a} x {num_b} binary Hamming-distance search")
    print(f"Return top {k} reference candidates per query")
    print(f"Iterations: {iterations}\n")

    # --- 2. Generate Random Binary Data ---
    print("Generating binary matrices...")
    queries_words = (num_a * d_bits + 63) // 64
    refs_words = (num_b * d_bits + 63) // 64
    h_queries, seed = generate_lcg_sequence(queries_words, seed=0xDEADBEEF)
    h_refs,    seed = generate_lcg_sequence(refs_words,    seed=seed) 

    matrix_A = h_queries.view(np.uint8).reshape(num_a, d_bytes)
    matrix_B = h_refs.view(np.uint8).reshape(num_b, d_bytes)

    workload = {
        "bitset_bits": d_bits,
        "num_queries": num_a,
        "num_refs": num_b,
        "top_k": k,
        "iterations": iterations,
    }
    summaries = {}

    # --- 3. CPU Benchmark ---
    print("\nInitializing CPU IndexBinaryFlat...")
    cpu_index = faiss.IndexBinaryFlat(d_bits)
    cpu_index.add(matrix_B)

    print("Running CPU Benchmark...")
    summaries["CPU"] = benchmark_search(cpu_index, matrix_A, k, iterations)
    print_iteration_results("CPU", summaries["CPU"])

    # --- 4. GPU Benchmarks ---
    try:
        num_gpus = faiss.get_num_gpus()
        print(f"\nDetected {num_gpus} GPU(s) via FAISS.")

        if gpu_id >= num_gpus:
            print(
                f"\nERROR: requested GPU {gpu_id}, but FAISS detected "
                f"{num_gpus} GPU(s)."
            )
        else:
            print(f"\nInitializing GPU {gpu_id}...")
            res = faiss.StandardGpuResources()

            # Transfer the CPU index to the specific GPU ID
            gpu_index = faiss.index_binary_cpu_to_gpu(res, gpu_id, cpu_index)

            print(f"Running Benchmark on GPU {gpu_id}...")
            device = f"GPU {gpu_id}"
            summaries[device] = benchmark_search(gpu_index, matrix_A, k, iterations)
            print_iteration_results(device, summaries[device])

    except AttributeError:
        print("\nERROR: FAISS-GPU is not installed or CUDA is not available.")

    # `search` returns host NumPy arrays, so each interval includes completion
    # of distance calculation, top-k selection, and result transfer.
    print_final_results(summaries, "CPU", workload)


if __name__ == "__main__":
    run_multi_gpu_benchmark(parse_arguments())
