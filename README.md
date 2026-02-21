# CuBFF

CUDA-accelerated simulation of self-modifying program soups that exhibit emergent self-replication.

Based on the paper [**"Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction"**](https://arxiv.org/abs/2406.19108).

## What It Does

CuBFF evolves populations of 64-byte programs that interact in pairs — one program executes using the other as memory, then they swap roles. Over thousands of epochs, natural selection-like dynamics emerge: programs that can copy themselves into their partner persist, and self-replicators spontaneously arise from random initial conditions. The simulation tracks Brotli compression ratio as a proxy for structural order and Shannon entropy to measure emergent complexity.

## Quick Start

```bash
sudo apt install build-essential libbrotli-dev    # dependencies
make                                               # CUDA build (or: make CUDA=0)
bin/main --lang bff_noheads                        # run simulation
bin/main --lang bff_noheads --eval_selfrep         # with self-replication detection
```

## Realtime Visualizer

Build with Python bindings, then launch the server:

```bash
make CUDA=0 PYTHON=1
pip install numpy websockets
python3 python/realtime_server.py --lang bff_noheads --eval_selfrep
```

Open `http://localhost:8766/realtime-visualizer.html` in your browser. The server runs the simulation and streams state over WebSocket (port 8765) to the dashboard.

Server flags: `--lang`, `--seed`, `--num`, `--port` (WebSocket, HTTP is port+1), `--eval_selfrep`, `--callback_interval`, `--max_visible`, `--max_epochs`.

### Dashboard Panels

<p align="center">
  <img src="docs/images/dashboard_soup.png" width="49%" alt="Soup Heatmap — each column is a 64-byte program, colored by opcode type">
  <img src="docs/images/dashboard_kymograph.png" width="49%" alt="Kymograph — evolutionary trajectories over 512 epochs">
</p>
<p align="center">
  <img src="docs/images/dashboard_spectrogram.png" width="49%" alt="Spectrogram — byte frequency distribution over time showing selection pressure">
  <img src="docs/images/dashboard_entropy.png" width="49%" alt="Entropy Map — per-position Shannon entropy drops as structure emerges">
</p>
<p align="center"><em>Live dashboard views of a <code>bff_noheads</code> simulation at epoch 9,536. Clockwise from top-left: Soup Heatmap, Kymograph, Entropy Map, Spectrogram. The metrics chart (right) tracks the phase transition — note the sharp Brotli BPB drop and entropy rise as self-replicating structure emerges.</em></p>

| Panel | Description |
|-------|-------------|
| **Soup Heatmap** | Programs x byte positions, colored by language-defined byte colors. Green border = self-replicator. |
| **Kymograph** | Time-series stacked over 512 epochs showing mean byte, dominant byte, or rep score per program. |
| **Spectrogram** | Byte value (0-255) frequency distribution over time. Shows which bytes are selected for/against. |
| **Entropy Map** | Per-position Shannon entropy over time. Low entropy = structured/meaningful positions. |
| **Rep Score** | Soup heatmap modulated by self-replication score (dim=none, red=partial, green=confirmed). |
| **Metrics Chart** | Multi-line graph tracking phase transition indicators: Brotli BPB (drops when structure emerges), H0 (Shannon entropy), higher-order entropy, and replicator count (spikes when self-replicators appear). |
| **Byte Frequency** | Most and least common bytes in the current population. |
| **Inspector** | Click any program to see its 64 bytes with hex codes and colors. |

The phase transition is visible when Brotli compression drops, entropy stabilizes, and self-replicators suddenly appear — the kymograph and spectrogram panels show this as the soup goes from random noise to structured, self-replicating programs.

### Other Visualizations

- **1D/2D frame output** — `bin/main --draw_to <dir>` and `--draw_to_2d <dir>` write PPM image sequences of the soup state each epoch
- **BFF trace visualizer** (`python/bff-visualizer.html`) — step-by-step execution debugger for individual BFF programs
- **Python analysis scripts** — `python/time_to_sr.py` (time-to-self-replication statistics), `python/selfrep_spawning.py` (spawn rate analysis), `python/cond_prob.py` / `python/cond_exp.py` (conditional entropy analysis)

## Supported Languages

| Language | Family | Description |
|----------|--------|-------------|
| `bff` | BFF | Brainfuck variant with two movable read/write heads |
| `bff_noheads` | BFF | BFF without movable heads (fixed position read/write) |
| `bff_noheads_4bit` | BFF | BFF without heads, opcodes determined by low 4 bits only |
| `bff8` | BFF | BFF with 8-bit head positions (heads can wrap) |
| `bff8_noheads` | BFF | 8-bit BFF without movable heads |
| `bff_perm` | BFF | BFF with permuted opcode assignments (byte values 0-255 mapped to ops) |
| `bff_selfmove` | BFF | BFF where head movement ops use byte values 0-6 instead of ASCII |
| `forth` | Forth | Stack-based language with read/write, arithmetic, and control flow (12 ops) |
| `forthtrivial` | Forth | Simplified Forth with reduced opcode set |
| `forthtrivial_reset` | Forth | Forth trivial with stack reset on underflow |
| `forthcopy` | Forth | Forth variant with explicit copy, XOR, and const operations |
| `subleq` | SUBLEQ | One-instruction ISA: subtract-and-branch-if-less-or-equal |
| `rsubleq4` | SUBLEQ | SUBLEQ with 4-bit relative addressing |

All 13 languages use 64-byte program tapes and are executed for up to 8192 steps per epoch.

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lang` | string | (required) | Language to simulate |
| `--seed` | int | 0 | Random seed |
| `--num` | int | 131072 | Number of programs in the soup |
| `--max_epochs` | int | unlimited | Stop after N epochs |
| `--mutation_prob` | float | 1/(256*16) | Per-byte mutation probability |
| `--eval_selfrep` | bool | false | Enable self-replication detection |
| `--selfrep_iters` | int | 13 | Noise iterations for self-rep check |
| `--selfrep_gens` | int | 5 | Generations per iteration for self-rep check |
| `--selfrep_sample_pct` | int | 100 | Percentage of programs to check for self-rep |
| `--cpu_fraction` | float | 0.0 | Fraction of work to run on CPU (0.0 = GPU only) |
| `--permute_programs` | bool | true | Shuffle program pairings each epoch |
| `--fixed_shuffle` | bool | false | Use deterministic Feistel-based shuffle |
| `--zero_init` | bool | false | Initialize all programs to zero instead of random |
| `--log` | string | | Write CSV metrics to file |
| `--checkpoint_dir` | string | | Directory for periodic checkpoints |
| `--save_interval` | int | 256 | Epochs between checkpoints |
| `--print_interval` | int | 64 | Epochs between terminal output |
| `--load` | string | | Resume from a checkpoint file |
| `--run` | string | | Run a single program (debug mode) |
| `--stopping_bpb` | float | | Stop when Brotli BPB drops below this |
| `--stopping_selfrep_count` | int | | Stop when this many self-replicators exist |
| `--initial_program` | string | | Seed the soup with a specific program |

Run `bin/main --help` for the full list.

## Performance Optimizations

- **Shared memory tapes** — Program tapes loaded into CUDA shared memory for fast interpreter access
- **Warp-level reductions** — Entropy and statistics computed with warp shuffle intrinsics
- **Async CUDA streams** — Overlapped kernel execution and host-device transfers
- **Background compression** — Brotli compression runs in a separate CPU thread
- **CPU+GPU hybrid** — Configurable fraction of work offloaded to CPU via OpenMP (`--cpu_fraction`)
- **Feistel permutation** — O(1) per-thread deterministic shuffle with no host-to-device transfer (`--fixed_shuffle`)
- **CheckSelfRep early-exit** — Programs that diverge early skip remaining iterations
- **Sampling** — Check only a percentage of programs for self-replication (`--selfrep_sample_pct`)

## Python Bindings

Build with pybind11 support:

```bash
sudo apt install python3-pybind11
make CUDA=0 PYTHON=1
```

Example usage:

```python
from bin import cubff

language = cubff.GetLanguage("bff_noheads")

def callback(state):
    print(state.epoch, state.brotli_bpb)
    return state.epoch > 1024

params = cubff.SimulationParams()
params.num_programs = 131072
language.RunSimulation(params, None, callback)
```

See `python/cubff_example.py` for a complete example.

## Testing

Run a single language against reference output:

```bash
./test.sh bff_noheads
```

CI tests all languages. To regenerate test data:

```bash
bin/main --lang <language> --max_epochs 256 --disable_output --log testdata/<language>.txt --seed 10248
```

## Dependencies

**Linux (Debian/Ubuntu):**
```bash
sudo apt install build-essential libbrotli-dev
```

**Arch Linux:**
```bash
pacman -S base-devel brotli
```

**Nix:**
```bash
nix develop
```

CUDA toolkit is optional — use `make CUDA=0` for CPU-only builds with OpenMP.

## License

Apache 2.0 — see [LICENSE](LICENSE).
