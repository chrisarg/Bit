# Bit CPU tuning sweep

## Measurement setup

- CPU affinity: `0-9`
- Priority: `nice`; elevated execution: no
- Benchmark: `openmp_bit_container 65536 10240 1024 10 1`
- Perf repetitions per configuration: 1
- Perf events: `cycles,instructions,branches,branch-misses,cache-references,cache-misses`
- Configurations requested: 1
- Successful configurations: 0

## Ranked successful configurations

| Rank | Mode | CPU tile | K block | Shape | Unroll | Buffer | Average ns | Gqword-pairs/s | IPC | Cache miss % | Branch miss % |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|

## Failed configurations

- `001-lib0-t4-k512-r2c2-u1-b-`: build=ok, run=failed

## Raw data

- CSV: `summary.csv`
- Per-configuration build logs: `*.build.log`
- Per-configuration benchmark output: `*.benchmark.log`
- Per-configuration perf output: `*.perf.csv`
