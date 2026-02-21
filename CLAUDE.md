# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CuBFF is a CUDA-accelerated simulation of self-modifying program soups that exhibit emergent self-replication. It supports multiple minimal programming languages (BFF variants, Forth variants, SUBLEQ) and tracks metrics like Brotli compression to measure structural emergence.

## Build Commands

```bash
make                        # CUDA-enabled build (default on Linux)
make CUDA=0                 # CPU-only build (uses OpenMP)
make CUDA=0 PYTHON=1        # CPU build with Python bindings
make -j CUDA=0 PYTHON=1     # Parallel build (used in CI)
make clean                  # Remove build artifacts
```

Dependencies: `build-essential`, `libbrotli-dev`, optionally CUDA toolkit and `python3-pybind11`.

## Testing

```bash
./tests/test.sh <language>        # Test a single language against reference output
./tests/test.sh bff_noheads       # Example
```

Tests run the simulation with seed 10248 for 256 epochs and diff against ground truth in `tests/testdata/`. CI tests all 13 languages. To regenerate test data:
```bash
./bin/main --lang <language> --max_epochs 256 --disable_output --log tests/testdata/<language>.txt --seed 10248
```

## Running

```bash
bin/main --lang bff_noheads                    # Basic run
bin/main --lang bff_noheads --eval_selfrep     # With self-replication detection
bin/main --lang bff_noheads --seed 42 --max_epochs 1024
```

## Architecture

### Simulation Pipeline

Three GPU/CPU kernels run each epoch:
1. **InitPrograms** — initializes random 64-byte programs (`kSingleTapeSize`)
2. **MutateAndRunPrograms** — pairs programs, applies mutation, executes the language interpreter (up to 8192 steps)
3. **CheckSelfRep** — tests programs for consistent self-replication across 13 noisy iterations

### Core Files

- **`src/common.h`** — `SimulationParams`, `SimulationState`, `LanguageInterface` base class, and the `REGISTER()` macro
- **`src/common.cc`** — Language registry (global map from name string to `LanguageInterface`)
- **`src/common_language.h`** — CUDA/OpenMP abstraction layer, kernel templates (`InitPrograms`, `MutateAndRunPrograms`, `CheckSelfRep`), `DeviceMemory` wrapper. Contains the `Simulation<Language>` template that implements `LanguageInterface`
- **`src/main.cc`** — CLI entry point, flag parsing, simulation orchestration, checkpoint I/O, visualization output

### Adding a New Language

Each language is a struct with static methods, registered via the `REGISTER()` macro (constructor-time registration). The required interface:

```cpp
struct MyLang {
  static const char *name();
  static OpKind GetOpKind(uint8_t byte);
  static size_t Evaluate(uint8_t *tape, size_t steps, bool debug);
  static void PrintProgram(...);
  static std::vector<uint8_t> Parse(const std::string &s);
  static void InitByteColors(...);
};
REGISTER(MyLang);
```

Language implementations live in `src/*.cu` files (compiled as CUDA or C++ depending on build flags). BFF variants include `bff.inc.h`, Forth variants include `forth.inc.h`. A minimal language file (e.g., `src/bff_noheads.cu`) just includes the shared header and defines `name()`.

### Python Bindings

`src/cubff_py.cc` exposes `SimulationParams`, `SimulationState`, `LanguageInterface`, and `GetLanguage()` via pybind11. Python analysis scripts live in `python/`.

## Code Style

- C++17 with Google style (`.clang-format: BasedOnStyle: Google`)
- `.cu` files are compiled as either CUDA or C++ depending on the `CUDA` flag
- `__device__`/`__host__` macros are no-ops in CPU-only builds
