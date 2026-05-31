#!/usr/bin/env bash
set -euo pipefail

: "${REPORT_PATH:?REPORT_PATH is required}"
: "${CC:?CC is required}"
: "${GPU:?GPU is required}"
: "${BUG_TARGET:?BUG_TARGET is required}"
: "${BUILD_DIR:?BUILD_DIR is required}"
: "${MAKE:?MAKE is required}"
: "${OPENMP_FLAG:?OPENMP_FLAG is required}"
: "${OFFLOAD_FL:?OFFLOAD_FL is required}"
: "${CFLAGS0:?CFLAGS0 is required}"
: "${CFLAGS:?CFLAGS is required}"
: "${DEFINES:=}"
: "${CC_ENV:=}"
: "${CUDA_PATH:=}"
: "${AMD_ARCH:=}"
: "${NVIDIA_ARCH:=}"
: "${NVIDIA_SYSTEM_ARCHES:=}"
: "${NVIDIA_EFFECTIVE_ARCHES:=}"

cc_basename=$(basename "$CC")
preprocessed_file="$REPORT_PATH/src-bit.preprocessed.i"
backtrace_file="$REPORT_PATH/backtrace.txt"
config_file="$REPORT_PATH/config.txt"
build_log="$REPORT_PATH/build.log"

action_with_optional_env() {
  if [[ -n "$CC_ENV" ]]; then
    eval "$CC_ENV $*"
  else
    eval "$*"
  fi
}

mkdir -p "$REPORT_PATH"
{
  echo "Compiler bug report generation"
  echo "Timestamp: $(date -Is)"
  echo "CC=$CC"
  echo "GPU=$GPU"
  echo "NVIDIA_ARCH=$NVIDIA_ARCH"
  echo "NVIDIA_SYSTEM_ARCHES=$NVIDIA_SYSTEM_ARCHES"
  echo "NVIDIA_EFFECTIVE_ARCHES=$NVIDIA_EFFECTIVE_ARCHES"
  echo "AMD_ARCH=$AMD_ARCH"
  echo "CUDA_PATH=$CUDA_PATH"
  echo "OPENMP_FLAG=$OPENMP_FLAG"
  echo "OFFLOAD_FL=$OFFLOAD_FL"
  echo "BUG_TARGET=$BUG_TARGET"
  echo "CFLAGS=$CFLAGS"
  echo "Compiler version:"
  "$CC" --version | head -n 1
} > "$config_file"

"$MAKE" clean >/dev/null

build_status=0
"$MAKE" BUG_REPORT=1 BUG_REPORT_OUT="$REPORT_PATH" "$BUG_TARGET" > "$build_log" 2>&1 || build_status=$?

if [[ "$cc_basename" == "gcc" ]]; then
  "$CC" -v > "$REPORT_PATH/gcc-v.txt" 2>&1 || true
  printf '%s\n' "$CC -v -save-temps=obj $CFLAGS -c src/bit.c -o $BUILD_DIR/gcc-bug-report.o" > "$REPORT_PATH/gcc-repro-command.txt"
  "$CC" -v -save-temps=obj $CFLAGS -c src/bit.c -o "$BUILD_DIR/gcc-bug-report.o" > "$REPORT_PATH/gcc-save-temps.log" 2>&1 || true
  if [[ -f "$BUILD_DIR/gcc-bug-report.i" ]]; then
    cp -f "$BUILD_DIR/gcc-bug-report.i" "$preprocessed_file"
  else
    echo "Failed to generate GCC preprocessed source with -save-temps." > "$preprocessed_file"
  fi
  rm -f "$BUILD_DIR/gcc-bug-report.o" "$BUILD_DIR/gcc-bug-report.i" "$BUILD_DIR/gcc-bug-report.s"
else
  action_with_optional_env "\"$CC\" $DEFINES $OPENMP_FLAG $OFFLOAD_FL $CFLAGS0 -E src/bit.c -o \"$preprocessed_file\" >/dev/null 2>&1" || true
fi

if [[ "$build_status" -ne 0 ]]; then
  if command -v gdb >/dev/null 2>&1; then
    gdb --batch \
      -ex "set pagination off" \
      -ex "set follow-fork-mode child" \
      -ex "set detach-on-fork off" \
      -ex "set print thread-events off" \
      -ex "run" \
      -ex "thread apply all bt full" \
      --args "$MAKE" BUG_REPORT=1 BUG_REPORT_OUT="$REPORT_PATH" "$BUG_TARGET" \
      > "$backtrace_file" 2>&1 || true
  else
    {
      echo "gdb not found; unable to collect full backtrace automatically."
      echo "Install gdb and rerun make bug_report to include full backtrace."
    } > "$backtrace_file"
  fi
else
  {
    echo "Build completed successfully; no compiler crash occurred."
    echo "No backtrace available because there was no failing process."
  } > "$backtrace_file"
fi

find "$BUILD_DIR" -maxdepth 1 \( -name '*.i' -o -name '*.ii' -o -name '*.s' -o -name '*.bc' -o -name '*.cui' \) -delete 2>/dev/null || true

echo "Saved bug report files under $REPORT_PATH"
echo "- Build log: $build_log"
echo "- Config: $config_file"
if [[ "$cc_basename" == "gcc" ]]; then
  echo "- GCC -v details: $REPORT_PATH/gcc-v.txt"
  echo "- GCC repro command: $REPORT_PATH/gcc-repro-command.txt"
  echo "- GCC save-temps log: $REPORT_PATH/gcc-save-temps.log"
fi
echo "- Backtrace: $backtrace_file"
echo "- Preprocessed source: $preprocessed_file"

exit "$build_status"
