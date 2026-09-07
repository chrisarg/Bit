#!/usr/bin/env bash
# run_numa_sweeps.sh -- generic dual-socket CPU tuning experiment.
#
# Runs four comparable configurations of scripts/sweep_cpu_tuning.pl:
#   1. socket0-local          : one socket's cores, allocation bound to its node
#   2. socket1-local          : the other socket's cores, allocation local
#   3. dual-first-touch-spread: all cores, default Linux first-touch placement
#   4. dual-interleave        : all cores, memory interleaved across both nodes
#
# The single-socket runs provide local-memory baselines; comparing the two
# dual-socket runs separates an asymmetric first-touch placement effect from
# the effect of explicit interleaving.
#
# Topology is AUTO-DISCOVERED from `lscpu -e=CPU,NODE,SOCKET,CORE` (fallback:
# /sys/devices/system/cpu/*/topology). Physical cores only by default (one
# OpenMP thread per physical core, matching the original Xeon semantics);
# pass --smt to include SMT siblings. All values can be overridden with the
# flags below; --dry-run prints the four resolved experiments without building.
#
# WHERE RESULTS ARE SAVED (anchored to the repository root, regardless of the
# directory this script is invoked from -- sweep_cpu_tuning.pl enforces CWD=root):
#   tuning-results/summary-<arch>-<label>-<timestamp>.csv    raw per-config rows
#   tuning-results/llm-summary-<arch>-<label>-<timestamp>.md ranked table/report
#   tuning-results/.work/<arch>-<label>-<timestamp>/         per-config build/perf logs
#   tuning-results/numa-compare-<timestamp>.md               cross-experiment table
# RESULTS_DIR/OUT_DIR (passed through to the tuner) override those roots.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

# ---------------------------------------------------------------------------
# Defaults (env-overridable pass-through to sweep_cpu_tuning.pl)
# ---------------------------------------------------------------------------
PROFILES_DEFAULT="summary,cache-l1,cache-l2,cache-l3-dram,cache-stalls,buffers-pending,buffers-store,execution-uops,execution-ports,frontend,frequency,vectorization,tlb,uncore-numa,power-rapl"
LIBPOPCNT_MODES="${LIBPOPCNT_MODES:-0,1}"
REPS="${REPS:-5}"
PERF_REPS="${PERF_REPS:-3}"
PERF_PROFILES="${PERF_PROFILES:-$PROFILES_DEFAULT}"
ELEVATE="${ELEVATE:-always}"

# ---------------------------------------------------------------------------
# CLI flags (override auto-discovery)
# ---------------------------------------------------------------------------
SOCKET0_CPUS="" ; SOCKET1_CPUS="" ; DUAL_CPUS=""
SOCKET0_THREADS="" ; SOCKET1_THREADS="" ; DUAL_THREADS=""
NODES="" ; ARCH_TAG="" ; USE_SMT=0 ; DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: run_numa_sweeps.sh [options]

Generic dual-socket CPU tuning experiment (four comparable sweeps through
scripts/sweep_cpu_tuning.pl). Topology is auto-discovered from lscpu; every
value below overrides discovery.

Options:
  --socket0-cpus LIST   CPU list for the socket0-local run (e.g. 0-17)
  --socket1-cpus LIST   CPU list for the socket1-local run (e.g. 18-35)
  --dual-cpus LIST      CPU list for the two dual-socket runs (e.g. 0-35)
  --socket0-threads N   OpenMP threads for socket0-local (default: list size)
  --socket1-threads N   OpenMP threads for socket1-local (default: list size)
  --dual-threads N      OpenMP threads for the dual runs (default: list size)
  --nodes N0,N1         NUMA node IDs for the two sockets (default: first node
                        seen on each socket's CPUs; override on sub-NUMA hosts)
  --arch-tag TAG        ARCH_TAG passed to the tuner (default: auto-detect)
  --smt                 Use all logical CPUs (incl. SMT siblings) instead of
                        one thread per physical core
  --dry-run             Print the four resolved experiments and exit (no build)
  -h, --help            Show this help

Environment passed through to the tuner (with defaults):
  LIBPOPCNT_MODES=0,1 REPS=5 PERF_REPS=3 ELEVATE=always PERF_PROFILES=<full mesh>
  Also honored: CC SEED MAX_CONFIGS RESULTS_DIR OUT_DIR OMP_PLACES OMP_PROC_BIND

Results are written under <repo-root>/tuning-results/ (see the header comment).
Testing hook: NUMA_LSCPU_FILE=<file> reads canned `lscpu -e` output.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --socket0-cpus)    SOCKET0_CPUS="$2"; shift 2;;
    --socket1-cpus)    SOCKET1_CPUS="$2"; shift 2;;
    --dual-cpus)       DUAL_CPUS="$2"; shift 2;;
    --socket0-threads) SOCKET0_THREADS="$2"; shift 2;;
    --socket1-threads) SOCKET1_THREADS="$2"; shift 2;;
    --dual-threads)    DUAL_THREADS="$2"; shift 2;;
    --nodes)           NODES="$2"; shift 2;;
    --arch-tag)        ARCH_TAG="$2"; shift 2;;
    --smt)             USE_SMT=1; shift;;
    --dry-run)         DRY_RUN=1; shift;;
    -h|--help)         usage; exit 0;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2;;
  esac
done

command -v numactl >/dev/null || {
  echo "ERROR: numactl is required but was not found in PATH." >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Expand "0-3,8,10-11" -> one CPU per line.
expand_cpus() {
  local spec=$1 part lo hi
  echo "$spec" | tr ',' '\n' | while IFS= read -r part; do
    case "$part" in
      *-*)
        lo=${part%-*}; hi=${part#*-}
        seq "$lo" "$hi"
        ;;
      *) [ -n "$part" ] && echo "$part";;
    esac
  done
}

# Compact a sorted-unique CPU list (stdin, one/line) into "0-3,8,10-11".
compact_cpus() {
  awk '
    { cpus[NR] = $1 }
    END {
      if (NR == 0) exit 1
      start = cpus[1]; prev = cpus[1]
      for (i = 2; i <= NR; i++) {
        if (cpus[i] == prev + 1) { prev = cpus[i]; continue }
        printf "%s", (start == prev ? start : start "-" prev)
        if (i < NR) printf ","
        start = cpus[i]; prev = cpus[i]
      }
      printf "%s\n", (start == prev ? start : start "-" prev)
    }'
}

# ---------------------------------------------------------------------------
# Topology discovery
# ---------------------------------------------------------------------------
# Emit "cpu socket node core" lines (comments stripped) from lscpu or fixture.
lscpu_table() {
  if [ -n "${NUMA_LSCPU_FILE:-}" ]; then
    cat "$NUMA_LSCPU_FILE"
  else
    lscpu -p=CPU,NODE,SOCKET,CORE
  fi | grep -v '^#'
}

# Args: socket_id ; prints compacted physical-core CPU list (or all logical with --smt).
socket_cpus() {
  local sock=$1
  lscpu_table | awk -F',' -v s="$sock" -v smt="$USE_SMT" '
    { cpu=$1; node=$2; socket=$3; core=$4
      if (socket != s) next
      if (smt == 1) { print cpu; next }          # all logical CPUs
      key = socket ":" core
      if (!(key in seen)) { seen[key]=1; print cpu }  # first CPU per physical core
    }' | sort -n | uniq | compact_cpus
}

# Args: socket_id ; prints the first NUMA node seen on that socket's CPUs.
socket_node() {
  local sock=$1
  lscpu_table | awk -F',' -v s="$sock" '$3 == s { print $2; exit }'
}

list_size() { expand_cpus "$1" | wc -l; }

# ---------------------------------------------------------------------------
# Discover / override the two sockets
# ---------------------------------------------------------------------------
mapfile -t SOCKETS < <(lscpu_table | awk -F',' '{print $3}' | sort -n | uniq)

if [ ${#SOCKETS[@]} -lt 2 ]; then
  cat >&2 <<'ERR'
ERROR: fewer than 2 CPU sockets detected -- this is a dual-socket experiment.
For a single-socket host, call the tuner directly, e.g.:
  LIBPOPCNT_MODES=0,1 CORES=auto THREADS=auto REPS=5 PERF_REPS=3 \
    PERF_PROFILES=summary ELEVATE=always ./scripts/sweep_cpu_tuning.pl
ERR
  exit 1
fi

S0=${SOCKETS[0]}
S1=${SOCKETS[1]}

[ -z "$SOCKET0_CPUS" ] && SOCKET0_CPUS=$(socket_cpus "$S0")
[ -z "$SOCKET1_CPUS" ] && SOCKET1_CPUS=$(socket_cpus "$S1")
if [ -z "$DUAL_CPUS" ]; then
  DUAL_CPUS=$( { expand_cpus "$SOCKET0_CPUS"; expand_cpus "$SOCKET1_CPUS"; } \
               | sort -n | uniq | compact_cpus )
fi

# NUMA nodes: explicit --nodes N0,N1 wins; else first node per socket.
if [ -n "$NODES" ]; then
  NODE0=${NODES%%,*}
  NODE1=${NODES##*,}
else
  NODE0=$(socket_node "$S0")
  NODE1=$(socket_node "$S1")
fi

# Thread counts default to the size of their CPU list.
[ -z "$SOCKET0_THREADS" ] && SOCKET0_THREADS=$(list_size "$SOCKET0_CPUS")
[ -z "$SOCKET1_THREADS" ] && SOCKET1_THREADS=$(list_size "$SOCKET1_CPUS")
[ -z "$DUAL_THREADS" ]    && DUAL_THREADS=$(list_size "$DUAL_CPUS")

# ---------------------------------------------------------------------------
# Validation (before any build)
# ---------------------------------------------------------------------------
fail() { echo "ERROR: $*" >&2; exit 1; }

# Online CPUs: /sys on a real run; the fixture's own CPU set under the test hook
# (so canned topologies are validated against themselves, not this host).
if [ -n "${NUMA_LSCPU_FILE:-}" ]; then
  ONLINE=$(lscpu_table | awk -F',' '{print $1}' | sort -n | uniq | compact_cpus || true)
elif [ -r /sys/devices/system/cpu/online ]; then
  ONLINE=$(cat /sys/devices/system/cpu/online)
else
  ONLINE=$(lscpu_table | awk -F',' '{print $1}' | sort -n | uniq | compact_cpus || true)
fi

cpu_subset() { # $1 = candidate list, $2 = superset list; returns 0 if subset
  awk -v a="$1" -v b="$2" 'BEGIN{
    nb=split(b, B, ","); for (i=1;i<=nb;i++){ split(B[i],r,"-");
      lo=r[1]; hi=(r[2]==""?r[1]:r[2]); for(c=lo;c<=hi;c++) ok[c]=1 }
    na=split(a, A, ","); for (i=1;i<=na;i++){ split(A[i],r,"-");
      lo=r[1]; hi=(r[2]==""?r[1]:r[2]); for(c=lo;c<=hi;c++) if(!(c in ok)) exit 1 }
  }'
}

[ -n "$SOCKET0_CPUS" ] || fail "socket $S0 has no CPUs (discovery or --socket0-cpus)"
[ -n "$SOCKET1_CPUS" ] || fail "socket $S1 has no CPUs (discovery or --socket1-cpus)"

# Non-overlapping socket lists.
overlap=$( { expand_cpus "$SOCKET0_CPUS"; expand_cpus "$SOCKET1_CPUS"; } \
           | sort -n | uniq -d | head -1)
[ -z "$overlap" ] || fail "socket CPU lists overlap at CPU $overlap"

cpu_subset "$SOCKET0_CPUS" "$ONLINE" || fail "--socket0-cpus ($SOCKET0_CPUS) contains offline/unknown CPUs (online: $ONLINE)"
cpu_subset "$SOCKET1_CPUS" "$ONLINE" || fail "--socket1-cpus ($SOCKET1_CPUS) contains offline/unknown CPUs (online: $ONLINE)"
cpu_subset "$DUAL_CPUS"    "$ONLINE" || fail "--dual-cpus ($DUAL_CPUS) contains offline/unknown CPUs (online: $ONLINE)"

# Nodes must exist per numactl --hardware (skip under the fixture test hook).
if [ -z "${NUMA_LSCPU_FILE:-}" ]; then
  avail_nodes=$(numactl --hardware | awk '/^available:/ {print $2}')
  for n in "$NODE0" "$NODE1"; do
    [ -n "$n" ] || fail "could not determine a NUMA node for a socket (use --nodes N0,N1)"
    [ "$n" -lt "$avail_nodes" ] 2>/dev/null || \
      fail "NUMA node $n not available (numactl reports $avail_nodes nodes); use --nodes"
  done
fi

# Threads must be positive and <= list size.
for pair in "socket0:$SOCKET0_THREADS:$SOCKET0_CPUS" \
            "socket1:$SOCKET1_THREADS:$SOCKET1_CPUS" \
            "dual:$DUAL_THREADS:$DUAL_CPUS"; do
  name=${pair%%:*}; rest=${pair#*:}; thr=${rest%%:*}; lst=${rest#*:}
  [[ "$thr" =~ ^[0-9]+$ ]] && [ "$thr" -gt 0 ] || fail "$name thread count '$thr' is not a positive integer"
  [ "$thr" -le "$(list_size "$lst")" ] || \
    fail "$name threads ($thr) exceed CPU list size ($(list_size "$lst"))"
done

# ---------------------------------------------------------------------------
# Resolved experiment summary (shared by --dry-run and the real run)
# ---------------------------------------------------------------------------
print_plan() {
  cat <<EOF
Resolved dual-socket experiment:
  Sockets          : $S0, $S1
  NUMA nodes       : $NODE0, $NODE1 $([ -n "$NODES" ] && echo "(override)")
  SMT siblings     : $([ $USE_SMT -eq 1 ] && echo "included (--smt)" || echo "excluded (physical cores only)")
  ARCH_TAG         : ${ARCH_TAG:-<auto-detect>}
  Tuner env        : LIBPOPCNT_MODES=$LIBPOPCNT_MODES REPS=$REPS PERF_REPS=$PERF_REPS ELEVATE=$ELEVATE

  1. socket0-local          CORES=$SOCKET0_CPUS THREADS=$SOCKET0_THREADS  bind=close  NUMA: cpunodebind=$NODE0 membind=$NODE0
  2. socket1-local          CORES=$SOCKET1_CPUS THREADS=$SOCKET1_THREADS  bind=close  NUMA: cpunodebind=$NODE1 membind=$NODE1
  3. dual-first-touch-spread CORES=$DUAL_CPUS THREADS=$DUAL_THREADS  bind=spread  NUMA: default (first-touch)
  4. dual-interleave         CORES=$DUAL_CPUS THREADS=$DUAL_THREADS  bind=spread  NUMA: interleave=$NODE0,$NODE1
EOF
}

if [ $DRY_RUN -eq 1 ]; then
  print_plan
  exit 0
fi

# ---------------------------------------------------------------------------
# Run the four experiments
# ---------------------------------------------------------------------------
common_env=(
  "LIBPOPCNT_MODES=$LIBPOPCNT_MODES"
  "REPS=$REPS"
  "PERF_REPS=$PERF_REPS"
  "PERF_PROFILES=$PERF_PROFILES"
  "ELEVATE=$ELEVATE"
)
[ -n "$ARCH_TAG" ] && common_env+=( "ARCH_TAG=$ARCH_TAG" )

run_sweep() {
  local label=$1 numa_policy=$2 numa_cmd=$3
  shift 3
  echo "=== Starting $label ==="
  env "${common_env[@]}" \
    "RUN_LABEL=$label" \
    "NUMA_POLICY=$numa_policy" \
    "NUMA_CMD=$numa_cmd" \
    "$@" \
    ./scripts/sweep_cpu_tuning.pl
}

print_plan
echo

run_sweep socket0-local "cpunodebind=$NODE0, membind=$NODE0" \
  "numactl --cpunodebind=$NODE0 --membind=$NODE0" \
  OMP_PLACES=cores OMP_PROC_BIND=close CORES="$SOCKET0_CPUS" THREADS="$SOCKET0_THREADS"

run_sweep socket1-local "cpunodebind=$NODE1, membind=$NODE1" \
  "numactl --cpunodebind=$NODE1 --membind=$NODE1" \
  OMP_PLACES=cores OMP_PROC_BIND=close CORES="$SOCKET1_CPUS" THREADS="$SOCKET1_THREADS"

run_sweep dual-first-touch-spread 'default first-touch policy' '' \
  OMP_PLACES=cores OMP_PROC_BIND=spread CORES="$DUAL_CPUS" THREADS="$DUAL_THREADS"

run_sweep dual-interleave "interleave=$NODE0,$NODE1" \
  "numactl --interleave=$NODE0,$NODE1" \
  OMP_PLACES=cores OMP_PROC_BIND=spread CORES="$DUAL_CPUS" THREADS="$DUAL_THREADS"

echo
echo "=== All four sweeps complete ==="
echo "Per-experiment reports are under tuning-results/ (llm-summary-*-<label>-*.md)."

# ---------------------------------------------------------------------------
# Cross-experiment comparison table
# ---------------------------------------------------------------------------
# Pull the best (lowest-avg_ns) successful config from each experiment's
# summary-*.csv and tabulate experiment x best-config x avg_ns x Gqps, so the
# socket-local / first-touch / interleave comparison is visible in one place.
write_compare_table() {
  local results_root=${RESULTS_DIR:-"$root/tuning-results"}
  local out
  out="$results_root/numa-compare-$(date +%Y%m%d-%H%M%S).md"
  local labels=(socket0-local socket1-local dual-first-touch-spread dual-interleave)
  {
    echo "# Dual-socket NUMA comparison"
    echo
    echo "- Sockets: \`$S0, $S1\`; NUMA nodes: \`$NODE0, $NODE1\`; SMT: \`$([ $USE_SMT -eq 1 ] && echo included || echo physical-cores-only)\`"
    echo "- Best = lowest mean per-iteration time (\`avg_ns\`) among successful configs."
    echo
    echo "| Experiment | Best config | Avg time (ns) | Gqword-pairs/s | Summary CSV |"
    echo "| --- | --- | --- | --- | --- |"
    local label csv best
    for label in "${labels[@]}"; do
      csv=$(ls -t "$results_root"/summary-*-"$label"-*.csv 2>/dev/null | head -1)
      if [ -z "$csv" ]; then
        echo "| \`$label\` | — | — | — | (no summary CSV found) |"
        continue
      fi
      best=$(awk -F',' '
        NR==1 { for (i=1;i<=NF;i++) h[$i]=i; next }
        $h["run_status"]=="ok" && $h["avg_ns"]+0>0 {
          if (min=="" || $h["avg_ns"]+0 < min) {
            min=$h["avg_ns"]+0; tag=$h["tag"]; gqps=$h["gqps"]
          }
        }
        END { if (min!="") printf "%s|%s|%s", tag, min, gqps }' "$csv")
      if [ -n "$best" ]; then
        IFS='|' read -r btag bavg bgqps <<<"$best"
        echo "| \`$label\` | \`$btag\` | $bavg | $bgqps | \`$(basename "$csv")\` |"
      else
        echo "| \`$label\` | (no valid config) | — | — | \`$(basename "$csv")\` |"
      fi
    done
  } > "$out"
  echo "Cross-experiment comparison written to: $out"
}

write_compare_table
