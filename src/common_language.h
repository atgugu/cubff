/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#ifdef __CUDACC__
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CUCHECK(op)                                                           \
  {                                                                           \
    cudaError_t cudaerr = op;                                                 \
    if (cudaerr != cudaSuccess) {                                             \
      printf("%s failed with error: %s\n", #op, cudaGetErrorString(cudaerr)); \
      exit(1);                                                                \
    }                                                                         \
  }

constexpr size_t kWarpSize = 32;

__inline__ __device__ uint64_t warpReduceSum(uint64_t val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(~0, val, offset);
  return val;
}

__inline__ __device__ size_t GetIndex() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

__inline__ __device__ void IncreaseInsnCount(unsigned long long count,
                                             unsigned long long *storage) {
  size_t index = GetIndex();
  size_t warp_ops = warpReduceSum(count);
  if (index % kWarpSize == 0) {
    atomicAdd(storage, warp_ops);
  }
}

inline void Synchronize() { CUCHECK(cudaDeviceSynchronize()); }

// Phase 3b: Host-side execution helpers for CPU+GPU pair split.
inline size_t &HostIndexThreadLocal() {
  thread_local size_t index;
  return index;
}

inline void HostIncreaseInsnCount(unsigned long long count,
                                  unsigned long long *storage) {
  __atomic_add_fetch(storage, count, __ATOMIC_RELAXED);
}

template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size) { CUCHECK(cudaMalloc(&data, size * sizeof(T))); }
  void Write(const T *host, size_t count) {
    CUCHECK(cudaMemcpy(data, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void Read(T *host, size_t count) {
    CUCHECK(cudaMemcpy(host, data, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  void ReadAsync(T *host, size_t count, cudaStream_t stream) {
    CUCHECK(cudaMemcpyAsync(host, data, count * sizeof(T),
                            cudaMemcpyDeviceToHost, stream));
  }
  T *Get() { return data; }
  ~DeviceMemory() { CUCHECK(cudaFree(data)); }
  DeviceMemory(DeviceMemory &) = delete;
};

#define RUN(grid, block, fun, ...) fun<<<grid, block>>>(__VA_ARGS__)
#define RUN_SHMEM(grid, block, shmem, fun, ...) \
  fun<<<grid, block, shmem>>>(__VA_ARGS__)
#define RUN_SHMEM_STREAM(grid, block, shmem, stream, fun, ...) \
  fun<<<grid, block, shmem, stream>>>(__VA_ARGS__)

#else
#define __device__
#define __host__
#define __global__

inline size_t &IndexThreadLocal() {
  thread_local size_t index;
  return index;
}

inline size_t GetIndex() { return IndexThreadLocal(); }

inline void IncreaseInsnCount(unsigned long long count,
                              unsigned long long *storage) {
  __atomic_add_fetch(storage, count, __ATOMIC_RELAXED);
}

inline void Synchronize() {}

template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size) { data = (T *)malloc(size * sizeof(T)); }
  void Write(const T *host, size_t count) {
    memcpy(data, host, count * sizeof(T));
  }
  void Read(T *host, size_t count) { memcpy(host, data, count * sizeof(T)); }
  T *Get() { return data; }
  ~DeviceMemory() { free(data); }
  DeviceMemory(DeviceMemory &) = delete;
};

#define RUN(grid, block, fun, ...)                                            \
  _Pragma("omp parallel for") for (size_t _threadcnt = 0;                     \
                                   _threadcnt < grid * block; _threadcnt++) { \
    IndexThreadLocal() = _threadcnt;                                          \
    fun(__VA_ARGS__);                                                         \
  }

#define RUN_SHMEM(grid, block, shmem, fun, ...)                               \
  _Pragma("omp parallel for") for (size_t _threadcnt = 0;                     \
                                   _threadcnt < grid * block; _threadcnt++) { \
    IndexThreadLocal() = _threadcnt;                                          \
    fun(__VA_ARGS__);                                                         \
  }

#define RUN_SHMEM_STREAM(grid, block, shmem, stream, fun, ...)                \
  _Pragma("omp parallel for") for (size_t _threadcnt = 0;                     \
                                   _threadcnt < grid * block; _threadcnt++) { \
    IndexThreadLocal() = _threadcnt;                                          \
    fun(__VA_ARGS__);                                                         \
  }

#endif

#define CHECK(op)                 \
  if (!(op)) {                    \
    printf("%s is false\n", #op); \
    exit(1);                      \
  }

inline __device__ __host__ uint64_t SplitMix64(uint64_t seed) {
  uint64_t z = seed + 0x9e3779b97f4a7c15;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

// Bijective pseudo-random permutation of [0, n) using a balanced Feistel cipher
// with cycle-walking. Replaces Fisher-Yates so each element can compute its
// permuted index independently in O(1), enabling parallel GPU execution.
inline __device__ __host__ uint32_t FeistelPermute(uint32_t idx, uint32_t n,
                                                   uint64_t key) {
  // Find smallest power of 4 >= n so we get balanced halves.
  uint32_t bits = 1;
  while ((1u << (2 * bits)) < n) bits++;
  uint32_t half_bits = bits;
  uint32_t half_mask = (1u << half_bits) - 1;
  // full_n = (1 << (2 * half_bits)), the Feistel domain size.

  uint32_t val = idx;
  for (;;) {
    // 6-round balanced Feistel network.
    uint32_t left = val >> half_bits;
    uint32_t right = val & half_mask;
    for (uint32_t round = 0; round < 6; round++) {
      uint32_t f =
          (uint32_t)(SplitMix64(key ^ SplitMix64(round * 0x9e3779b9u +
                                                  right)) &
                     half_mask);
      uint32_t new_right = left ^ f;
      left = right;
      right = new_right;
    }
    val = (left << half_bits) | right;
    // Cycle-walk: if result is outside [0, n), re-apply the permutation.
    if (val < n) return val;
  }
}

template <typename Language>
__global__ void InitPrograms(size_t seed, size_t num_programs,
                             uint8_t *programs, bool zero_init) {
  size_t index = GetIndex();
  auto prog = programs + index * kSingleTapeSize;
  if (index >= num_programs) return;
  if (zero_init) {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      prog[i] = 0;
    }
  } else {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      prog[i] = SplitMix64(kSingleTapeSize * num_programs * seed +
                           kSingleTapeSize * index + i) %
                256;
    }
  }
}

template <typename Language>
__global__ void MutateAndRunPrograms(uint8_t *programs,
                                     const uint32_t *shuf_idx, size_t seed,
                                     uint32_t mutation_prob,
                                     unsigned long long *insn_count,
                                     size_t num_programs, size_t num_indices,
                                     int shuffle_mode, uint64_t epoch_seed,
                                     size_t shuffle_offset,
                                     uint32_t shuffle_flip) {
  size_t index = GetIndex();
#ifdef __CUDACC__
#ifndef CUBFF_USE_SHMEM
#define CUBFF_USE_SHMEM 1
#endif
#if CUBFF_USE_SHMEM
  extern __shared__ uint8_t shared_tape_mem[];
  uint8_t *tape = &shared_tape_mem[threadIdx.x * 2 * kSingleTapeSize];
  for (size_t i = 0; i < 2 * kSingleTapeSize; i++) tape[i] = 0;
#else
  uint8_t tape[2 * kSingleTapeSize] = {};
#endif
#else
  uint8_t tape[2 * kSingleTapeSize] = {};
#endif
  if (2 * index >= num_programs) return;
  uint32_t p1, p2;
  if (shuf_idx) {
    p1 = shuf_idx[2 * index];
    p2 = shuf_idx[2 * index + 1];
  } else {
    size_t i1 = 2 * index;
    size_t i2 = 2 * index + 1;
    switch (shuffle_mode) {
      case 0:  // Feistel permutation
        p1 = FeistelPermute((uint32_t)i1, (uint32_t)num_programs, epoch_seed);
        p2 = FeistelPermute((uint32_t)i2, (uint32_t)num_programs, epoch_seed);
        break;
      case 1:  // fixed_shuffle
        p1 = (uint32_t)(((i1 * shuffle_offset) % num_programs) ^ shuffle_flip);
        p2 = (uint32_t)(((i2 * shuffle_offset) % num_programs) ^ shuffle_flip);
        break;
      case 2:  // shift
        p1 = (uint32_t)(i1 == 0 ? num_programs - 1 : i1 - 1);
        p2 = (uint32_t)(i2 == 0 ? num_programs - 1 : i2 - 1);
        break;
      default:  // identity
        p1 = (uint32_t)i1;
        p2 = (uint32_t)i2;
        break;
    }
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    tape[i] = programs[p1 * kSingleTapeSize + i];
    tape[i + kSingleTapeSize] = programs[p2 * kSingleTapeSize + i];
  }
  for (size_t i = 0; i < 2 * kSingleTapeSize; i++) {
    uint64_t rng =
        SplitMix64((num_programs * seed + index) * kSingleTapeSize * 2 + i);
    uint8_t repl = rng & 0xFF;
    uint64_t prob_rng = (rng >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
    }
  }
  bool debug = false;
  size_t ops;
  if (index < num_indices) {
    ops = Language::Evaluate(tape, 8 * 1024, debug);
  } else {
    ops = 0;
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    programs[p1 * kSingleTapeSize + i] = tape[i];
    programs[p2 * kSingleTapeSize + i] = tape[i + kSingleTapeSize];
  }
  IncreaseInsnCount(ops, insn_count);
}

template <typename Language>
__global__ void RunOneProgram(uint8_t *program, size_t stepcount, bool debug) {
  size_t ops = Language::Evaluate(program, stepcount, debug);
  printf("%s", ResetColors());
  printf("ops: %d\n", (int)ops);
  printf("\n");
}

template <typename Language>
__global__ void CheckSelfRep(uint8_t *programs, size_t seed,
                             size_t num_programs, size_t *result, bool debug,
                             size_t num_iters, size_t num_gens,
                             size_t sample_pct) {
  size_t index = GetIndex();
  constexpr size_t kMaxIters = 13;
  if (index > num_programs) return;
  if (num_iters > kMaxIters) num_iters = kMaxIters;
  if (num_gens < 1) num_gens = 1;

  // Sampling: skip non-sampled programs.
  if (sample_pct < 100) {
    uint64_t sample_rng = SplitMix64(seed ^ SplitMix64(index + 0x12345));
    if ((sample_rng % 100) >= sample_pct) {
      result[index] = 0;
      return;
    }
  }

  uint8_t tapes[kMaxIters][2 * kSingleTapeSize] = {};
  uint8_t orig_match_count[kSingleTapeSize] = {};
  uint64_t local_seed = SplitMix64(num_programs * seed + index);
  size_t num_extra_gens = num_gens - 1;

  for (size_t i = 0; i < num_iters; i++) {
    bool eval_debug = false;
    uint8_t noise[kSingleTapeSize];
    for (int j = 0; j < kSingleTapeSize; j++) {
      noise[j] =
          SplitMix64(local_seed ^ SplitMix64((i + 1) * kSingleTapeSize + j)) %
          256;
    }
    uint8_t *tape = &tapes[i][0];
    for (int j = 0; j < kSingleTapeSize; j++) {
      tape[j] = programs[index * kSingleTapeSize + j];
      tape[j + kSingleTapeSize] = noise[j];
    }
    if (debug) {
      size_t separators[1] = {kSingleTapeSize};
      printf("Iteration %lu: before first step\n", i);
      Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                             separators, 1);
    }
    Language::Evaluate(tape, 8 * 1024, eval_debug);
    if (debug) {
      size_t separators[1] = {kSingleTapeSize};
      printf("Iteration %lu: after first step\n", i);
      Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                             separators, 1);
    }

    for (size_t g = 0; g < num_extra_gens; g++) {
      for (int j = 0; j < kSingleTapeSize; j++) {
        tape[j] = tape[j + kSingleTapeSize];
        tape[j + kSingleTapeSize] = noise[j];
      }
      if (debug) {
        size_t separators[1] = {kSingleTapeSize};
        printf("Iteration %lu: before step %lu\n", i, g + 2);
        Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                               separators, 1);
      }
      Language::Evaluate(tape, 8 * 1024, eval_debug);
      if (debug) {
        size_t separators[1] = {kSingleTapeSize};
        printf("Iteration %lu: after step %lu\n", i, g + 2);
        Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                               separators, 1);
      }
    }

    // Update orig_match_count for first-half bytes.
    for (int j = 0; j < kSingleTapeSize; j++) {
      if (tape[j] == programs[index * kSingleTapeSize + j]) {
        orig_match_count[j]++;
      }
    }

    // Heuristic pre-filter: after 3 iterations, if no first-half byte matched
    // the original in ANY iteration, bail out.
    if (i == 2) {
      bool any_match = false;
      for (int j = 0; j < kSingleTapeSize; j++) {
        if (orig_match_count[j] > 0) {
          any_match = true;
          break;
        }
      }
      if (!any_match) {
        result[index] = 0;
        return;
      }
    }

    // Formal early termination: when few iterations remain, check if it's
    // mathematically impossible to reach kSelfrepThreshold.
    if (i >= 3 && i + 4 >= num_iters) {
      size_t remaining = num_iters - i - 1;
      size_t possible_count = 0;
      for (int j = 0; j < kSingleTapeSize; j++) {
        if (orig_match_count[j] + remaining > num_iters / 4) {
          possible_count++;
        }
      }
      if (possible_count < kSelfrepThreshold) {
        result[index] = 0;
        return;
      }
    }
  }
  size_t res[2] = {};
  for (int i = 0; i < 2 * kSingleTapeSize; ++i) {
    for (size_t a = 0; a < num_iters; a++) {
      size_t count = 1;
      if (i < kSingleTapeSize &&
          tapes[a][i] != programs[index * kSingleTapeSize + i]) {
        continue;
      }
      for (size_t b = a + 1; b < num_iters; b++) {
        if (tapes[a][i] == tapes[b][i]) count++;
      }
      if (count > num_iters / 4) {
        res[i / kSingleTapeSize]++;
        break;
      }
    }
  }
  result[index] = res[0] < res[1] ? res[0] : res[1];
}

// GPU kernel: each thread computes its own shuffle index, eliminating the
// CPU-side Fisher-Yates + synchronous H2D transfer.
//   mode 0: Feistel permutation (replaces Fisher-Yates)
//   mode 1: fixed_shuffle: ((i * offset) % n) ^ flip
//   mode 2: shift: i == 0 ? n-1 : i-1
//   mode 3: identity: i
inline __global__ void GenerateShuffleIndices(uint32_t *shuf_idx, size_t n,
                                       uint64_t epoch_seed, int mode,
                                       size_t offset, uint32_t flip) {
  size_t i = GetIndex();
  if (i >= n) return;
  switch (mode) {
    case 0:
      shuf_idx[i] = FeistelPermute((uint32_t)i, (uint32_t)n, epoch_seed);
      break;
    case 1:
      shuf_idx[i] = (uint32_t)(((i * offset) % n) ^ flip);
      break;
    case 2:
      shuf_idx[i] = (uint32_t)(i == 0 ? n - 1 : i - 1);
      break;
    case 3:
      shuf_idx[i] = (uint32_t)i;
      break;
  }
}

// Phase 3c: Host-callable version of MutateAndRunPrograms kernel body.
#ifdef __CUDACC__
template <typename Language>
void MutateAndRunProgramsHost(uint8_t *programs, const uint32_t *shuf_idx,
                              size_t seed, uint32_t mutation_prob,
                              unsigned long long *insn_count,
                              size_t num_programs, size_t num_indices,
                              int shuffle_mode, uint64_t epoch_seed,
                              size_t shuffle_offset, uint32_t shuffle_flip,
                              size_t index) {
  uint8_t tape[2 * kSingleTapeSize] = {};
  if (2 * index >= num_programs) return;
  uint32_t p1, p2;
  if (shuf_idx) {
    p1 = shuf_idx[2 * index];
    p2 = shuf_idx[2 * index + 1];
  } else {
    size_t i1 = 2 * index;
    size_t i2 = 2 * index + 1;
    switch (shuffle_mode) {
      case 0:
        p1 = FeistelPermute((uint32_t)i1, (uint32_t)num_programs, epoch_seed);
        p2 = FeistelPermute((uint32_t)i2, (uint32_t)num_programs, epoch_seed);
        break;
      case 1:
        p1 = (uint32_t)(((i1 * shuffle_offset) % num_programs) ^ shuffle_flip);
        p2 = (uint32_t)(((i2 * shuffle_offset) % num_programs) ^ shuffle_flip);
        break;
      case 2:
        p1 = (uint32_t)(i1 == 0 ? num_programs - 1 : i1 - 1);
        p2 = (uint32_t)(i2 == 0 ? num_programs - 1 : i2 - 1);
        break;
      default:
        p1 = (uint32_t)i1;
        p2 = (uint32_t)i2;
        break;
    }
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    tape[i] = programs[p1 * kSingleTapeSize + i];
    tape[i + kSingleTapeSize] = programs[p2 * kSingleTapeSize + i];
  }
  for (size_t i = 0; i < 2 * kSingleTapeSize; i++) {
    uint64_t rng =
        SplitMix64((num_programs * seed + index) * kSingleTapeSize * 2 + i);
    uint8_t repl = rng & 0xFF;
    uint64_t prob_rng = (rng >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
    }
  }
  bool debug = false;
  size_t ops;
  if (index < num_indices) {
    ops = Language::Evaluate(tape, 8 * 1024, debug);
  } else {
    ops = 0;
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    programs[p1 * kSingleTapeSize + i] = tape[i];
    programs[p2 * kSingleTapeSize + i] = tape[i + kSingleTapeSize];
  }
  HostIncreaseInsnCount(ops, insn_count);
}
#endif

template <typename Language>
void Simulation<Language>::RunSingleParsedProgram(
    const std::vector<uint8_t> &parsed, size_t stepcount, bool debug) const {
  DeviceMemory<uint8_t> mem(kSingleTapeSize * 2);
  uint8_t zero[2 * kSingleTapeSize] = {};
  memcpy(zero, parsed.data(), parsed.size());
  mem.Write(zero, 2 * kSingleTapeSize);
  Language::PrintProgram(2 * kSingleTapeSize, zero, 2 * kSingleTapeSize,
                         nullptr, 0);

  RUN(1, 1, RunOneProgram<Language>, mem.Get(), stepcount, debug);

  uint8_t final_state[2 * kSingleTapeSize];
  Synchronize();
  mem.Read(final_state, 2 * kSingleTapeSize);
  Language::PrintProgram(2 * kSingleTapeSize, final_state, 2 * kSingleTapeSize,
                         nullptr, 0);
}

template <typename Language>
void Simulation<Language>::RunSingleProgram(std::string program,
                                            size_t stepcount,
                                            bool debug) const {
  RunSingleParsedProgram(Language::Parse(program), stepcount, debug);
}

template <typename Language>
void Simulation<Language>::PrintProgram(size_t pc_pos, const uint8_t *mem,
                                        size_t len, const size_t *separators,
                                        size_t num_separators) const {
  Language::PrintProgram(pc_pos, mem, len, separators, num_separators);
}

template <typename Language>
std::vector<uint8_t> Simulation<Language>::Parse(const std::string& program) {
  return Language::Parse(program);
}


template <typename Language>
size_t Simulation<Language>::EvalSelfrep(std::string program, size_t epoch,
                                         size_t seed, bool debug) {
  std::vector<uint8_t> parsed = Language::Parse(program);
  return EvalParsedSelfrep(parsed, epoch, seed, debug);
}

template <typename Language>
size_t Simulation<Language>::EvalParsedSelfrep(std::vector<uint8_t> &parsed,
                                               size_t epoch, size_t seed,
                                               bool debug) {
  DeviceMemory<uint8_t> mem(kSingleTapeSize);
  uint8_t zero[kSingleTapeSize] = {};
  memcpy(zero, parsed.data(), parsed.size());
  mem.Write(zero, kSingleTapeSize);
  DeviceMemory<size_t> result(1);
  size_t epoch_seed = SplitMix64(SplitMix64(seed) ^ SplitMix64(epoch));
  RUN(1, 1, CheckSelfRep<Language>, mem.Get(), epoch_seed, 1, result.Get(),
      debug, 13, 5, 100);

  Synchronize();
  std::vector<size_t> res(1);
  result.Read(res.data(), 1);
  return res[0];
}

template <typename Language>
void Simulation<Language>::RunSimulation(
    const SimulationParams &params, std::optional<std::string> initial_program,
    std::function<bool(const SimulationState &)> callback) const {
#ifndef CUBFF_NUM_THREADS
#define CUBFF_NUM_THREADS 256
#endif
  constexpr size_t kNumThreads = CUBFF_NUM_THREADS;
  size_t num_programs = params.num_programs;

  size_t reset_index = 1;
  size_t epoch = 0;

  FILE *load_file = nullptr;
  if (params.load_from.has_value()) {
    load_file = CheckFopen(params.load_from->c_str(), "r");
    CHECK(fread(&reset_index, sizeof(reset_index), 1, load_file) == 1);
    CHECK(fread(&num_programs, sizeof(num_programs), 1, load_file) == 1);
    CHECK(fread(&epoch, sizeof(epoch), 1, load_file) == 1);
  }

  DeviceMemory<uint8_t> programs(kSingleTapeSize * num_programs);
  DeviceMemory<unsigned long long> insn_count(1);

  CHECK(num_programs % 2 == 0);

#ifdef __CUDACC__
  cudaStream_t compute_stream, transfer_stream;
  cudaEvent_t compute_done;
  CUCHECK(cudaStreamCreate(&compute_stream));
  CUCHECK(cudaStreamCreate(&transfer_stream));
  CUCHECK(cudaEventCreate(&compute_done));
  // Pinned host memory for async D2H transfer of the soup.
  uint8_t *pinned_soup;
  CUCHECK(cudaMallocHost(&pinned_soup,
                         num_programs * kSingleTapeSize * sizeof(uint8_t)));
#endif

  auto seed = [&](size_t seed2) {
    return SplitMix64(SplitMix64(params.seed) ^ SplitMix64(seed2));
  };

  RUN((num_programs + kNumThreads - 1) / kNumThreads, kNumThreads,
      InitPrograms<Language>, seed(0), num_programs, programs.Get(),
      params.zero_init);

  if (initial_program.has_value()) {
    std::vector<uint8_t> parsed = Language::Parse(*initial_program);
    programs.Write((const unsigned char *)parsed.data(), parsed.size());
  }

  unsigned long long zero = 0;
  insn_count.Write(&zero, 1);

  unsigned long long total_ops = 0;

  SimulationState state;
  state.soup.reserve(num_programs * kSingleTapeSize + 16);
  state.soup.resize(num_programs * kSingleTapeSize);
  state.replication_per_prog.resize(num_programs);
  state.shuffle_idx.resize(num_programs);
  Language::InitByteColors(state.byte_colors);

  if (params.save_to.has_value()) {
    CHECK(mkdir(params.save_to->c_str(),
                S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != -1 ||
          errno == EEXIST);
  }

  if (load_file) {
    CHECK(fread(state.soup.data(), 1, num_programs * kSingleTapeSize,
                load_file) == num_programs * kSingleTapeSize);
    fclose(load_file);
    programs.Write(state.soup.data(), num_programs * kSingleTapeSize);
  }

  // Only allocate device shuffle buffer when allowed_interactions needs it.
  bool use_allowed_interactions = !params.allowed_interactions.empty();
  DeviceMemory<uint32_t> shuf_idx(use_allowed_interactions ? num_programs : 1);

  std::vector<uint32_t> &s = state.shuffle_idx;

  for (size_t i = 0; i < num_programs; i++) {
    s[i] = i;
  }

  std::vector<uint32_t> shuffle_tmp_buf(num_programs);
  std::vector<char> used_program(num_programs);

  Synchronize();

  auto do_shuffle = [&](uint32_t *begin, uint32_t *end, uint64_t base_seed) {
    size_t len = end - begin;
    for (size_t i = len; i-- > 0;) {
      size_t j = SplitMix64(seed(epoch * len + i)) % (i + 1);
      std::swap(begin[i], begin[j]);
    }
  };

  std::vector<uint8_t> brotlified_data(
      BrotliEncoderMaxCompressedSize(num_programs * kSingleTapeSize));

#ifdef __CUDACC__
  // Phase 1: Background callback thread state.
  SimulationState bg_state;
  bg_state.soup.reserve(num_programs * kSingleTapeSize + 16);
  bg_state.soup.resize(num_programs * kSingleTapeSize);
  bg_state.replication_per_prog.resize(num_programs);
  bg_state.shuffle_idx.resize(num_programs);
  Language::InitByteColors(bg_state.byte_colors);

  std::vector<uint8_t> bg_brotlified_data(
      BrotliEncoderMaxCompressedSize(num_programs * kSingleTapeSize));

  std::mutex bg_mutex;
  std::condition_variable bg_cv;
  bool bg_work_ready = false;
  bool bg_shutdown = false;
  bool bg_done = true;
  std::atomic<bool> should_stop{false};

  // Data communicated to bg thread at each callback point.
  size_t bg_raw_epoch = 0;
  size_t bg_reset_index = 0;

  // Phase 2: Persistent selfrep device memory and event.
  DeviceMemory<size_t> selfrep_result(params.eval_selfrep ? num_programs : 1);
  cudaEvent_t selfrep_done;
  CUCHECK(cudaEventCreate(&selfrep_done));
  size_t *pinned_selfrep = nullptr;
  if (params.eval_selfrep) {
    CUCHECK(
        cudaMallocHost(&pinned_selfrep, num_programs * sizeof(size_t)));
  }
  cudaStream_t bg_stream;
  CUCHECK(cudaStreamCreate(&bg_stream));

  auto bg_worker = [&]() {
    while (true) {
      std::unique_lock<std::mutex> lock(bg_mutex);
      bg_cv.wait(lock, [&] { return bg_work_ready || bg_shutdown; });
      if (bg_shutdown) break;
      bg_work_ready = false;
      lock.unlock();

      // Brotli compression (overlaps with CheckSelfRep on GPU).
      size_t brotli_size = bg_brotlified_data.size();
      BrotliEncoderCompress(2, 24, BROTLI_MODE_GENERIC, bg_state.soup.size(),
                            bg_state.soup.data(), &brotli_size,
                            bg_brotlified_data.data());

      // Phase 2: Wait for CheckSelfRep and read results.
      if (params.eval_selfrep) {
        CUCHECK(cudaEventSynchronize(selfrep_done));
        CUCHECK(cudaMemcpyAsync(pinned_selfrep, selfrep_result.Get(),
                                num_programs * sizeof(size_t),
                                cudaMemcpyDeviceToHost, bg_stream));
        CUCHECK(cudaStreamSynchronize(bg_stream));
        memcpy(bg_state.replication_per_prog.data(), pinned_selfrep,
               num_programs * sizeof(size_t));
      }

      // Byte frequency counting and entropy.
      size_t counts[256] = {};
      for (auto c : bg_state.soup) counts[c]++;

      uint8_t sorted[256];
      double h0 = 0;
      for (size_t i = 0; i < 256; i++) {
        sorted[i] = i;
        double frac = counts[i] * 1.0 / bg_state.soup.size();
        h0 -= counts[i] ? frac * std::log2(frac) : 0.0;
      }
      std::sort(sorted, sorted + 256, [&](uint8_t a, uint8_t b) {
        return std::make_pair(counts[b], b) < std::make_pair(counts[a], a);
      });

      double brotli_bpb =
          brotli_size * 8.0 / (num_programs * kSingleTapeSize);

      bg_state.brotli_size = brotli_size;
      bg_state.brotli_bpb = brotli_bpb;
      bg_state.bytes_per_prog = brotli_size * 1.0 / num_programs;
      bg_state.h0 = h0;
      bg_state.higher_entropy = h0 - brotli_bpb;

      for (size_t i = 0; i < bg_state.frequent_bytes.size(); i++) {
        uint8_t c = sorted[i];
        char chmem[32];
        bg_state.frequent_bytes[i].first = Language::MapChar(c, chmem);
        bg_state.frequent_bytes[i].second =
            counts[(int)c] * 1.0 / bg_state.soup.size();
      }
      for (size_t i = 0; i < bg_state.uncommon_bytes.size(); i++) {
        uint8_t c = sorted[256 - bg_state.uncommon_bytes.size() + i];
        char chmem[32];
        bg_state.uncommon_bytes[i].first = Language::MapChar(c, chmem);
        bg_state.uncommon_bytes[i].second =
            counts[(int)c] * 1.0 / bg_state.soup.size();
      }

      // Save checkpoint.
      if (params.save_to.has_value() &&
          (bg_raw_epoch % params.save_interval == 0)) {
        std::vector<char> save_path(params.save_to->size() + 20);
        snprintf(save_path.data(), save_path.size(), "%s/%010zu.dat",
                 params.save_to->c_str(), bg_raw_epoch);
        FILE *f = CheckFopen(save_path.data(), "w");
        size_t epoch_to_save = bg_raw_epoch + 1;
        fwrite(&bg_reset_index, sizeof(bg_reset_index), 1, f);
        fwrite(&num_programs, sizeof(num_programs), 1, f);
        fwrite(&epoch_to_save, sizeof(epoch_to_save), 1, f);
        fwrite(bg_state.soup.data(), 1, bg_state.soup.size(), f);
        fclose(f);
      }

      // Call the user callback.
      if (callback(bg_state)) {
        should_stop.store(true, std::memory_order_relaxed);
      }

      lock.lock();
      bg_done = true;
      lock.unlock();
      bg_cv.notify_one();
    }
  };

  std::thread bg_thread(bg_worker);

  // Phase 3d: Host programs buffer for CPU+GPU pair split.
  uint8_t *host_programs = nullptr;
  unsigned long long host_insn_count = 0;
  if (params.cpu_fraction > 0.0f) {
    CUCHECK(cudaMallocHost(&host_programs,
                           num_programs * kSingleTapeSize * sizeof(uint8_t)));
    programs.Read(host_programs, num_programs * kSingleTapeSize);
  }
#endif

  size_t num_runs = 0;
  auto start = std::chrono::high_resolution_clock::now();
  auto simulation_start = std::chrono::high_resolution_clock::now();
  for (;; epoch++) {
    size_t num_indices = num_programs / 2;
    // Shuffle parameters for the fused kernel.
    int shuffle_mode = 3;  // identity by default
    uint64_t epoch_seed = seed(epoch);
    size_t shuffle_offset = 0;
    uint32_t shuffle_flip = 0;
    const uint32_t *shuf_idx_ptr = nullptr;

    if (use_allowed_interactions) {
      // allowed_interactions requires complex CPU-side neighbor selection.
      for (size_t i = 0; i < num_programs; i++) {
        shuffle_tmp_buf[i] = i;
        used_program[i] = false;
      }
      do_shuffle(shuffle_tmp_buf.data(),
                 shuffle_tmp_buf.data() + shuffle_tmp_buf.size(), epoch);
      num_indices = 0;
      for (size_t i : shuffle_tmp_buf) {
        auto &interact = params.allowed_interactions;
        if (interact.size() <= i || interact[i].empty()) {
          continue;
        }
        size_t idx = seed(seed(epoch) ^ seed(i)) % interact[i].size();
        size_t neigh = interact[i][idx];
        if (used_program[i] || used_program[neigh]) {
          continue;
        }
        used_program[i] = used_program[neigh] = true;
        s[num_indices * 2] = i;
        s[num_indices * 2 + 1] = neigh;
        num_indices++;
      }
      size_t idx = num_indices * 2;
      for (size_t i = 0; i < num_programs; i++) {
        if (!used_program[i]) {
          s[idx++] = i;
        }
      }
      shuf_idx.Write(s.data(), num_programs);
      shuf_idx_ptr = shuf_idx.Get();
    } else {
      // Compute shuffle mode — kernel will inline the permutation.
      if (params.permute_programs) {
        if (params.fixed_shuffle) {
          shuffle_mode = 1;  // fixed_shuffle
          shuffle_flip = epoch & 1;
          size_t max_pow2 = 31 - __builtin_clz(num_programs);
          shuffle_offset = (1 << (epoch % max_pow2 + 1)) - 1;
        } else {
          shuffle_mode = 0;  // Feistel permutation
        }
      } else if (epoch % 2 == 1) {
        shuffle_mode = 3;  // identity
      } else {
        shuffle_mode = 2;  // shift
      }
    }

#ifdef __CUDACC__
    // Phase 3d: Split work between GPU and CPU.
    size_t gpu_pairs = num_indices;
    size_t cpu_start_pair = num_indices;
    if (host_programs && params.cpu_fraction > 0.0f) {
      gpu_pairs = (size_t)(num_indices * (1.0f - params.cpu_fraction));
      cpu_start_pair = gpu_pairs;
    }

    // GPU kernel processes pairs [0, gpu_pairs).
#if CUBFF_USE_SHMEM
    constexpr size_t kMutateRunShmem = kNumThreads * 2 * kSingleTapeSize;
#else
    constexpr size_t kMutateRunShmem = 0;
#endif
    RUN_SHMEM_STREAM(
        (num_programs + 2 * kNumThreads - 1) / (2 * kNumThreads), kNumThreads,
        kMutateRunShmem, compute_stream,
        MutateAndRunPrograms<Language>, programs.Get(), shuf_idx_ptr,
        seed(epoch), params.mutation_prob, insn_count.Get(), num_programs,
        gpu_pairs, shuffle_mode, epoch_seed, shuffle_offset, shuffle_flip);

    // CPU processes pairs [cpu_start_pair, num_indices) concurrently.
    if (host_programs && cpu_start_pair < num_indices) {
      const uint32_t *host_shuf_ptr =
          use_allowed_interactions ? s.data() : nullptr;
      _Pragma("omp parallel for")
      for (size_t i = cpu_start_pair; i < num_indices; i++) {
        MutateAndRunProgramsHost<Language>(
            host_programs, host_shuf_ptr, seed(epoch), params.mutation_prob,
            &host_insn_count, num_programs, num_indices, shuffle_mode,
            epoch_seed, shuffle_offset, shuffle_flip, i);
      }

      // Merge: D2H GPU results, overlay CPU results, H2D merged buffer.
      CUCHECK(cudaEventRecord(compute_done, compute_stream));
      CUCHECK(cudaStreamWaitEvent(transfer_stream, compute_done, 0));
      programs.ReadAsync(pinned_soup, num_programs * kSingleTapeSize,
                         transfer_stream);
      CUCHECK(cudaStreamSynchronize(transfer_stream));

      // Overlay CPU-modified programs into the staging buffer.
      for (size_t i = cpu_start_pair; i < num_indices; i++) {
        uint32_t p1, p2;
        if (use_allowed_interactions) {
          p1 = s[2 * i];
          p2 = s[2 * i + 1];
        } else {
          size_t i1 = 2 * i, i2 = 2 * i + 1;
          switch (shuffle_mode) {
            case 0:
              p1 = FeistelPermute((uint32_t)i1, (uint32_t)num_programs,
                                  epoch_seed);
              p2 = FeistelPermute((uint32_t)i2, (uint32_t)num_programs,
                                  epoch_seed);
              break;
            case 1:
              p1 = (uint32_t)(((i1 * shuffle_offset) % num_programs) ^
                              shuffle_flip);
              p2 = (uint32_t)(((i2 * shuffle_offset) % num_programs) ^
                              shuffle_flip);
              break;
            case 2:
              p1 = (uint32_t)(i1 == 0 ? num_programs - 1 : i1 - 1);
              p2 = (uint32_t)(i2 == 0 ? num_programs - 1 : i2 - 1);
              break;
            default:
              p1 = (uint32_t)i1;
              p2 = (uint32_t)i2;
              break;
          }
        }
        memcpy(&pinned_soup[p1 * kSingleTapeSize],
               &host_programs[p1 * kSingleTapeSize], kSingleTapeSize);
        memcpy(&pinned_soup[p2 * kSingleTapeSize],
               &host_programs[p2 * kSingleTapeSize], kSingleTapeSize);
      }

      // H2D merged buffer back to device (async on compute_stream).
      CUCHECK(cudaMemcpyAsync(programs.Get(), pinned_soup,
                              num_programs * kSingleTapeSize,
                              cudaMemcpyHostToDevice, compute_stream));
      // Update host_programs mirror for next epoch.
      memcpy(host_programs, pinned_soup, num_programs * kSingleTapeSize);
    }
#else
    RUN_SHMEM_STREAM(
        (num_programs + 2 * kNumThreads - 1) / (2 * kNumThreads), kNumThreads,
        0, 0,
        MutateAndRunPrograms<Language>, programs.Get(), shuf_idx_ptr,
        seed(epoch), params.mutation_prob, insn_count.Get(), num_programs,
        num_indices, shuffle_mode, epoch_seed, shuffle_offset, shuffle_flip);
#endif
    num_runs += num_indices;

    if (epoch % params.callback_interval == 0) {
      auto stop = std::chrono::high_resolution_clock::now();
#ifdef __CUDACC__
      // Wait for any previous background callback to finish.
      {
        std::unique_lock<std::mutex> lock(bg_mutex);
        bg_cv.wait(lock, [&] { return bg_done; });
      }
      if (should_stop.load(std::memory_order_relaxed)) break;

      // D2H transfer.
      CUCHECK(cudaEventRecord(compute_done, compute_stream));
      CUCHECK(cudaStreamWaitEvent(transfer_stream, compute_done, 0));
      unsigned long long insn;
      insn_count.ReadAsync(&insn, 1, transfer_stream);
      programs.ReadAsync(pinned_soup, num_programs * kSingleTapeSize,
                         transfer_stream);
      CUCHECK(cudaStreamSynchronize(transfer_stream));

      // Add CPU contribution to instruction count.
      insn += host_insn_count;
      host_insn_count = 0;

      // Copy soup and shuffle_idx to bg_state.
      memcpy(bg_state.soup.data(), pinned_soup,
             num_programs * kSingleTapeSize);
      total_ops += insn;
      if (use_allowed_interactions) {
        shuf_idx.Read(bg_state.shuffle_idx.data(), num_programs);
      } else {
        for (size_t i = 0; i < num_programs; i++) {
          switch (shuffle_mode) {
            case 0:
              bg_state.shuffle_idx[i] =
                  FeistelPermute((uint32_t)i, (uint32_t)num_programs,
                                 epoch_seed);
              break;
            case 1:
              bg_state.shuffle_idx[i] =
                  (uint32_t)(((i * shuffle_offset) % num_programs) ^
                             shuffle_flip);
              break;
            case 2:
              bg_state.shuffle_idx[i] =
                  (uint32_t)(i == 0 ? num_programs - 1 : i - 1);
              break;
            default:
              bg_state.shuffle_idx[i] = (uint32_t)i;
              break;
          }
        }
      }

      // Compute scalar metrics for bg_state.
      float elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
              .count() *
          1e-6;
      float mops_s = insn * 1.0 / elapsed_s * 1e-6;
      float sim_elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(
              stop - simulation_start)
              .count() *
          1e-6;
      bg_state.elapsed_s = sim_elapsed_s;
      bg_state.total_ops = total_ops;
      bg_state.mops_s = mops_s;
      bg_state.epoch = epoch + 1;
      bg_state.ops_per_run = insn * 1.0 / num_runs;
      bg_raw_epoch = epoch;
      bg_reset_index = reset_index;

      // Phase 2: Launch CheckSelfRep asynchronously on compute_stream.
      if (params.eval_selfrep) {
        RUN_SHMEM_STREAM(num_programs / kNumThreads, kNumThreads, 0,
                         compute_stream, CheckSelfRep<Language>,
                         programs.Get(), seed(epoch), num_programs,
                         selfrep_result.Get(), false,
                         params.selfrep_iters, params.selfrep_gens,
                         params.selfrep_sample_pct);
        CUCHECK(cudaEventRecord(selfrep_done, compute_stream));
      }

      // Signal background thread to do Brotli/stats/callback.
      {
        std::unique_lock<std::mutex> lock(bg_mutex);
        bg_done = false;
        bg_work_ready = true;
      }
      bg_cv.notify_one();

      // Reset counters for next interval (async to avoid blocking).
      num_runs = 0;
      start = std::chrono::high_resolution_clock::now();
      CUCHECK(cudaMemsetAsync(insn_count.Get(), 0,
                              sizeof(unsigned long long), compute_stream));
#else
      // CPU path: synchronous callback.
      Synchronize();
      unsigned long long insn;
      insn_count.Read(&insn, 1);
      programs.Read(state.soup.data(), num_programs * kSingleTapeSize);
      total_ops += insn;
      if (use_allowed_interactions) {
        shuf_idx.Read(state.shuffle_idx.data(), num_programs);
      } else {
        for (size_t i = 0; i < num_programs; i++) {
          switch (shuffle_mode) {
            case 0:
              state.shuffle_idx[i] =
                  FeistelPermute((uint32_t)i, (uint32_t)num_programs,
                                 epoch_seed);
              break;
            case 1:
              state.shuffle_idx[i] =
                  (uint32_t)(((i * shuffle_offset) % num_programs) ^
                             shuffle_flip);
              break;
            case 2:
              state.shuffle_idx[i] =
                  (uint32_t)(i == 0 ? num_programs - 1 : i - 1);
              break;
            default:
              state.shuffle_idx[i] = (uint32_t)i;
              break;
          }
        }
      }
      size_t brotli_size = brotlified_data.size();
      BrotliEncoderCompress(2, 24, BROTLI_MODE_GENERIC, state.soup.size(),
                            state.soup.data(), &brotli_size,
                            brotlified_data.data());
      float elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
              .count() *
          1e-6;
      float mops_s = insn * 1.0 / elapsed_s * 1e-6;
      float sim_elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(
              stop - simulation_start)
              .count() *
          1e-6;

      size_t counts[256] = {};
      for (auto c : state.soup) {
        counts[c]++;
      }

      std::vector<uint8_t> sorted(256);
      double h0 = 0;
      for (size_t i = 0; i < 256; i++) {
        sorted[i] = i;
        double frac = counts[i] * 1.0 / state.soup.size();
        h0 -= counts[i] ? frac * std::log2(frac) : 0.0;
      }
      std::sort(sorted.begin(), sorted.end(), [&](uint8_t a, uint8_t b) {
        return std::make_pair(counts[b], b) < std::make_pair(counts[a], a);
      });

      double brotli_bpb = brotli_size * 8.0 / (num_programs * kSingleTapeSize);

      state.elapsed_s = sim_elapsed_s;
      state.total_ops = total_ops;
      state.mops_s = mops_s;
      state.epoch = epoch + 1;
      state.ops_per_run = insn * 1.0 / num_runs;
      state.brotli_size = brotli_size;
      state.brotli_bpb = brotli_bpb;
      state.bytes_per_prog = brotli_size * 1.0 / num_programs;
      state.h0 = h0;
      state.higher_entropy = h0 - brotli_bpb;

      for (size_t i = 0; i < state.frequent_bytes.size(); i++) {
        uint8_t c = sorted[i];
        char chmem[32];
        state.frequent_bytes[i].first = Language::MapChar(c, chmem);
        state.frequent_bytes[i].second =
            counts[(int)c] * 1.0 / state.soup.size();
      }
      for (size_t i = 0; i < state.uncommon_bytes.size(); i++) {
        uint8_t c = sorted[256 - state.uncommon_bytes.size() + i];
        char chmem[32];
        state.uncommon_bytes[i].first = Language::MapChar(c, chmem);
        state.uncommon_bytes[i].second =
            counts[(int)c] * 1.0 / state.soup.size();
      }

      if (params.eval_selfrep) {
        DeviceMemory<size_t> result(num_programs);
        RUN(num_programs / kNumThreads, kNumThreads, CheckSelfRep<Language>,
            programs.Get(), seed(epoch), num_programs, result.Get(), false,
            params.selfrep_iters, params.selfrep_gens,
            params.selfrep_sample_pct);
        Synchronize();
        result.Read(state.replication_per_prog.data(), num_programs);
      }
      if (params.save_to.has_value() && (epoch % params.save_interval == 0)) {
        std::vector<char> save_path(params.save_to->size() + 20);
        snprintf(save_path.data(), save_path.size(), "%s/%010zu.dat",
                 params.save_to->c_str(), epoch);
        FILE *f = CheckFopen(save_path.data(), "w");
        size_t epoch_to_save = epoch + 1;
        fwrite(&reset_index, sizeof(reset_index), 1, f);
        fwrite(&num_programs, sizeof(num_programs), 1, f);
        fwrite(&epoch_to_save, sizeof(epoch), 1, f);
        fwrite(state.soup.data(), 1, state.soup.size(), f);
        fclose(f);
      }
      if (callback(state)) {
        break;
      }
      num_runs = 0;
      start = std::chrono::high_resolution_clock::now();
      insn_count.Write(&zero, 1);
#endif
    }

    if (params.reset_interval.has_value() &&
        epoch % *params.reset_interval == 0) {
      RUN(num_programs / kNumThreads, kNumThreads, InitPrograms<Language>,
          seed(reset_index), num_programs, programs.Get(), params.zero_init);
      reset_index++;
    }
  }

#ifdef __CUDACC__
  // Wait for final background callback and shut down the thread.
  {
    std::unique_lock<std::mutex> lock(bg_mutex);
    bg_cv.wait(lock, [&] { return bg_done; });
    bg_shutdown = true;
  }
  bg_cv.notify_one();
  bg_thread.join();

  if (host_programs) CUCHECK(cudaFreeHost(host_programs));
  if (pinned_selfrep) CUCHECK(cudaFreeHost(pinned_selfrep));
  CUCHECK(cudaEventDestroy(selfrep_done));
  CUCHECK(cudaStreamDestroy(bg_stream));
  CUCHECK(cudaFreeHost(pinned_soup));
  CUCHECK(cudaEventDestroy(compute_done));
  CUCHECK(cudaStreamDestroy(transfer_stream));
  CUCHECK(cudaStreamDestroy(compute_stream));
#endif
}
