# Bit CPU tuning sweep

## Measurement setup

- Architecture tag: `x86-64-intel-r-xeon-r-cpu-e5-2697-v4-2-30ghz`
- Run label: `none`
- CPU affinity: `0-35`
- NUMA policy: `default OS policy`
- OpenMP places: `runtime default`; thread binding: `runtime default`
- Priority: `nice`; elevated execution: no
- Benchmark: `openmp_bit_container 65536 1000 1000 36 1`
- Perf repetitions per configuration: 1
- Config order seed: `12345` (configuration order is shuffled; set seed to reproduce)
- Perf profiles: `summary`
  - `summary`: Timing ranking, IPC, branch behavior, and generic cache miss rate. Events: `cycles,instructions,branches,branch-misses,cache-references,cache-misses`
- Configurations requested: 2
- Successful configurations: 0

## Ranked successful configurations

| Rank | Mode | CPU tile | K block | Shape | Unroll | Buffer | Average ns | Gqword-pairs/s | IPC | Cache miss % | Branch miss % |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|

## Failed configurations

- `001-lib0-t8-k512-r1c1-u1-b-`: build=ok, run=failed
- `002-lib0-t8-k768-r2c2-u1-b-`: build=ok, run=failed

## Profile collection status

| Configuration | summary |
|:---|:---|
| `001-lib0-t8-k512-r1c1-u1-b-` | failed |
| `002-lib0-t8-k768-r2c2-u1-b-` | failed |

A failed non-summary profile normally means that its events are unavailable on this CPU/kernel or require additional perf permissions; it does not invalidate the timing result. See that profile's benchmark log for the exact perf error.

## Raw data

- CSV: `summary-x86-64-intel-r-xeon-r-cpu-e5-2697-v4-2-30ghz-20260905-150204.csv` (includes every collected counter, prefixed with `perf_<profile>_`)
- Local intermediate artifacts: `.work/x86-64-intel-r-xeon-r-cpu-e5-2697-v4-2-30ghz-20260905-150204/` by default; override with `OUT_DIR`.
- Per-configuration build logs: `*.build.log`
- Per-profile benchmark output: `*.<profile>.benchmark.log`
- Per-profile perf output: `*.<profile>.perf.csv`
