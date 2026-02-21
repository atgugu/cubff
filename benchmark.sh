#!/bin/bash
# benchmark.sh — Finds the fastest CuBFF execution strategy
#
# Builds multiple variants with different compile-time constants,
# runs each configuration multiple times, and collects MOps/s.
#
# Usage: bash benchmark.sh [--no-head] [--gpu-only] [--cpu-only] [--runs N]

set -euo pipefail

SEED=42
EPOCHS=512
LANG=bff_noheads
PRINT_INTERVAL=512
RUNS=3
RESULTS_DIR=bench_results
SKIP_HEAD=0
GPU_ONLY=0
CPU_ONLY=0

# Parse arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-head) SKIP_HEAD=1; shift ;;
    --gpu-only) GPU_ONLY=1; shift ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --runs) RUNS=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Detect CUDA availability.
HAS_CUDA=0
if command -v nvcc &>/dev/null && command -v nvidia-smi &>/dev/null; then
  HAS_CUDA=1
fi

if [[ $GPU_ONLY -eq 1 && $HAS_CUDA -eq 0 ]]; then
  echo "ERROR: --gpu-only requested but no CUDA toolkit found."
  exit 1
fi

echo "=== CuBFF Benchmark ==="
echo "Epochs: $EPOCHS | Runs per config: $RUNS | Language: $LANG"
echo "CUDA available: $HAS_CUDA"
echo ""

# --- Phase 0: Setup ---
mkdir -p "$RESULTS_DIR" bin build
CSV="$RESULTS_DIR/results.csv"
echo "config,run,mops_s,wall_s" > "$CSV"

# Record hardware info.
{
  echo "=== CPU ==="
  lscpu | head -20
  echo ""
  echo "=== GPU ==="
  nvidia-smi 2>/dev/null || echo "(no GPU)"
} > "$RESULTS_DIR/hardware.txt"

# --- Helper functions ---

build_variant() {
  local name=$1 cuda=$2 extra_flags=${3:-}
  echo "  Building $name (CUDA=$cuda, flags: ${extra_flags:-none})..."
  make clean -s 2>/dev/null || true
  make CUDA="$cuda" EXTRA_LDFLAGS="$extra_flags" -j -s 2>&1 | tail -5
  cp bin/main "$RESULTS_DIR/main_$name"
}

run_bench() {
  local binary=$1 label=$2 extra_flags=${3:-} env_prefix=${4:-}
  echo "  Running $label..."
  for run in $(seq 1 "$RUNS"); do
    local outfile="$RESULTS_DIR/out_${label}_run${run}.txt"
    local start end wall_s mops
    start=$(date +%s%N)
    eval $env_prefix "$binary" --lang "$LANG" --seed "$SEED" \
      --max_epochs "$EPOCHS" --print_interval "$PRINT_INTERVAL" \
      --save_interval "$PRINT_INTERVAL" --clear_interval "$PRINT_INTERVAL" \
      $extra_flags > "$outfile" 2>&1 || true
    end=$(date +%s%N)
    wall_s=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    # MOps/s is printed to stdout; extract last occurrence.
    # Strip ANSI escape codes before extracting MOps/s.
    mops=$(sed 's/\x1b\[[0-9;:]*[a-zA-Z]//g' "$outfile" 2>/dev/null | grep -oP 'MOps/s:\s+\K[0-9.]+' | tail -1 || echo "0")
    echo "    Run $run: ${mops} MOps/s (${wall_s}s)"
    echo "${label},${run},${mops},${wall_s}" >> "$CSV"
  done
}

# --- Phase 1: Build all variants ---
echo ""
echo "=== Phase 1: Building variants ==="

declare -a BUILDS=()

# GPU builds (only if CUDA available and not CPU-only mode).
if [[ $HAS_CUDA -eq 1 && $CPU_ONLY -eq 0 ]]; then
  build_variant gpu_shmem_256 1 "-DCUBFF_NUM_THREADS=256 -DCUBFF_USE_SHMEM=1"
  BUILDS+=(gpu_shmem_256)

  build_variant gpu_shmem_128 1 "-DCUBFF_NUM_THREADS=128 -DCUBFF_USE_SHMEM=1"
  BUILDS+=(gpu_shmem_128)

  build_variant gpu_shmem_32 1 "-DCUBFF_NUM_THREADS=32 -DCUBFF_USE_SHMEM=1"
  BUILDS+=(gpu_shmem_32)

  build_variant gpu_noshmem_256 1 "-DCUBFF_NUM_THREADS=256 -DCUBFF_USE_SHMEM=0"
  BUILDS+=(gpu_noshmem_256)
fi

# CPU builds (only if not GPU-only mode).
if [[ $GPU_ONLY -eq 0 ]]; then
  build_variant cpu_32 0 "-DCUBFF_NUM_THREADS=32"
  BUILDS+=(cpu_32)

  build_variant cpu_128 0 "-DCUBFF_NUM_THREADS=128"
  BUILDS+=(cpu_128)

  build_variant cpu_256 0 "-DCUBFF_NUM_THREADS=256"
  BUILDS+=(cpu_256)
fi

# HEAD baseline builds.
if [[ $SKIP_HEAD -eq 0 ]]; then
  echo ""
  echo "  Building HEAD baselines (stashing local changes)..."
  git stash -q
  trap 'git stash pop -q 2>/dev/null || true' EXIT

  if [[ $HAS_CUDA -eq 1 && $CPU_ONLY -eq 0 ]]; then
    build_variant head_gpu 1
    BUILDS+=(head_gpu)
  fi
  if [[ $GPU_ONLY -eq 0 ]]; then
    build_variant head_cpu 0
    BUILDS+=(head_cpu)
  fi

  git stash pop -q
  trap - EXIT
fi

# --- Phase 2: Run benchmarks ---
echo ""
echo "=== Phase 2: Running benchmarks ==="

NPROC=$(nproc)

for build in "${BUILDS[@]}"; do
  binary="$RESULTS_DIR/main_$build"

  case $build in
    gpu_*)
      # GPU builds: sweep cpu_fraction values.
      for cf in 0.0 0.1 0.3 0.5; do
        run_bench "$binary" "${build}_cf${cf}" "--cpu_fraction $cf"
      done
      ;;
    head_gpu)
      # HEAD GPU: run without cpu_fraction (flag may not exist in HEAD).
      run_bench "$binary" "${build}" ""
      ;;
    cpu_*|head_cpu)
      # CPU builds: sweep OMP_NUM_THREADS.
      for threads in 2 4 8 $NPROC; do
        run_bench "$binary" "${build}_omp${threads}" "" "OMP_NUM_THREADS=$threads"
      done
      ;;
  esac
done

# --- Phase 3: Summary report ---
echo ""
echo "=== Phase 3: Generating summary ==="

awk -F',' '
NR == 1 { next }
{
  config = $1
  mops = $3 + 0
  wall = $4 + 0
  sum_mops[config] += mops
  sum_wall[config] += wall
  sum_mops2[config] += mops * mops
  count[config]++
}
END {
  n = 0
  for (c in count) {
    mean_mops = sum_mops[c] / count[c]
    mean_wall = sum_wall[c] / count[c]
    variance = (sum_mops2[c] / count[c]) - (mean_mops * mean_mops)
    stddev = (variance > 0) ? sqrt(variance) : 0
    results[n] = sprintf("%s\t%.1f\t%.1f\t%.1f\t%d", c, mean_mops, stddev, mean_wall, count[c])
    sort_key[n] = mean_mops
    n++
  }
  # Bubble sort by mean MOps/s descending.
  for (i = 0; i < n - 1; i++) {
    for (j = i + 1; j < n; j++) {
      if (sort_key[j] > sort_key[i]) {
        tmp = results[i]; results[i] = results[j]; results[j] = tmp
        tmp = sort_key[i]; sort_key[i] = sort_key[j]; sort_key[j] = tmp
      }
    }
  }
  printf "%-4s  %-40s  %10s  %6s  %8s  %s\n", "Rank", "Configuration", "Mean MOps/s", "StdDev", "Wall(s)", "Runs"
  printf "%-4s  %-40s  %10s  %6s  %8s  %s\n", "----", "-------------", "-----------", "------", "-------", "----"
  for (i = 0; i < n; i++) {
    split(results[i], f, "\t")
    printf "%-4d  %-40s  %10s  %6s  %8s  %4s\n", i+1, f[1], f[2], f[3], f[4], f[5]
  }
}
' "$CSV" | tee "$RESULTS_DIR/summary.txt"

echo ""
echo "Raw data:  $CSV"
echo "Summary:   $RESULTS_DIR/summary.txt"
echo "Hardware:  $RESULTS_DIR/hardware.txt"
echo "Done."
