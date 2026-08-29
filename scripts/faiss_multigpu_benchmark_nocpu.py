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
#   python scripts/faiss_multigpu_benchmark_nocpu.py \
#     <size> <number-of-bitsets> <number-of-reference-bitsets> \
#     <top-k> <gpu-iterations> [gpu-id]
#
# For example:
#
#   python scripts/faiss_multigpu_benchmark_nocpu.py 65536 1024 65536 256 10 0
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
        description="Benchmark FAISS IndexBinaryFlat on one GPU without a CPU run."
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
            "SEARCH_SUMMARY,backend=FAISS,method=IndexBinaryFlat,"
            f"device={machine_device},score=hamming_distance,score_order=min,"
            "selection=FAISS_topk,timing_scope=end_to_end_search_call,"
            f"bitset_bits={workload['bitset_bits']},"
            f"num_queries={workload['num_queries']},"
            f"num_refs={workload['num_refs']},top_k={workload['top_k']},"
            f"iterations={workload['iterations']},"
            f"end_to_end_avg_ns={summary['mean_ns']:.3f},"
            f"end_to_end_stddev_ns={summary['stddev_ns']:.3f},"
            f"end_to_end_searches_per_sec={summary['searches_per_sec']:.6f},"
            f"best_score={summary['best_distance']},"
            f"distance_checksum={summary['distance_checksum']},"
            f"id_checksum={summary['id_checksum']}"
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
    if k != num_b:
        print(f"Return top {k} reference candidates per query")
    else:
        print(f"Return all {num_b} reference candidates per query")
    print(f"Iterations: {iterations}\n")

    # --- 2. Generate Random Binary Data ---
    print("Generating binary matrices...")
    matrix_A = np.random.randint(256, size=(num_a, d_bytes), dtype=np.uint8)
    matrix_B = np.random.randint(256, size=(num_b, d_bytes), dtype=np.uint8)

    workload = {
        "bitset_bits": d_bits,
        "num_queries": num_a,
        "num_refs": num_b,
        "top_k": k,
        "iterations": iterations,
    }
    summaries = {}

    # --- 3. CPU prestaging ---
    print("\nInitializing CPU IndexBinaryFlat...")
    cpu_index = faiss.IndexBinaryFlat(d_bits)
    cpu_index.add(matrix_B)

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

    if not summaries:
        print("\nNo FAISS GPU results were collected.")
        return

    # `search` returns host NumPy arrays, so each interval includes completion
    # of distance calculation, top-k selection, and result transfer.
    baseline_device = "GPU 0" if "GPU 0" in summaries else next(iter(summaries))
    print_final_results(summaries, baseline_device, workload)


if __name__ == "__main__":
    run_multi_gpu_benchmark(parse_arguments())
