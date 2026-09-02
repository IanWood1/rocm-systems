# Benchmarking rocjitsu

The in-tree benchmark suite runs a fixed workload matrix through rocjitsu on
`gfx950` and `gfx1250`. It is intended for nightly performance tracking and
manual evaluation of changes. The first version writes local JSON artifacts;
CI scheduling, historical storage, comparison policy, and dashboard publishing
are follow-on work.

## Suite

`benchmarks/suites/nightly.toml` contains the targets, ordered case IDs, and
suite-wide measurement settings. Commands and filesystem paths are derived from
the source and build layouts, so the manifest stays short and portable.

The nightly suite has 12 cases per target:

| Provider | Cases | Coverage |
|---|---:|---|
| Native HIP | 3 | Launch overhead, a vectorized 32 MiB model-weight copy, and tail handling |
| Triton | 6 | Softmax, BF16 RMSNorm, BF16 GEMM, and attention shapes |
| hipBLASLt/TensileLite | 3 | FP16 GEMM, batched BF16 GEMM, and scaled FP8 GEMM |

This produces 24 case/target cells in a full run. Inputs and launch
configurations are fixed, and measured runs do not autotune.

## Measurement

Each cell runs in a fresh process. The workload performs allocation,
initialization, compilation or descriptor setup, and algorithm selection before
measurement. It then runs three untimed warmups followed by 21 timed samples.
Every warmup and sample launches the operation and synchronizes the device; a
sample is the host elapsed time around that launch-and-synchronize pair.

Only the 21 samples contribute to the reported minimum, median, and maximum.
Process startup, setup, warmups, JSON output, and teardown are excluded. For a
native or Triton case, one sample contains one kernel launch. For a hipBLASLt
case, one sample contains one `hipblasLtMatmul` call, which may launch more than
one internal kernel.

This first version does not establish numerical correctness for the benchmark
adapters. It validates execution and timing integrity only: the runner rejects
failed processes, timeouts, malformed workload JSON, identity or target
mismatches, and timing vectors that do not contain the requested number of
positive samples. Separate correctness coverage can be added later without
putting reference computation or output copies inside the timing boundary.

## Install and build

Use Python 3.12 and install the pinned binary dependencies:

```bash
src=/path/to/rocm-systems/emulation/rocjitsu
python=/path/to/python3.12
"$python" -m venv /path/to/rocjitsu-benchmark-env
python=/path/to/rocjitsu-benchmark-env/bin/python
"$python" -m pip install -r "$src/benchmarks/requirements.txt"
```

The requirements use AMD's multi-architecture wheel index and provide the ROCm
SDK, PyTorch, Triton, hipBLASLt, and target-specific gfx950/gfx1250 device
libraries. The benchmark build consumes those installed packages; it does not
fetch or build hipBLASLt, TensileLite, or Python requirements from source. The
requirements file rejects source distributions if a matching binary package is
unavailable.

Configure a Release build against the SDK installed in the environment:

```bash
build=/path/to/rocjitsu-build-release
rocm=$("$python" -c \
  'import pathlib, _rocm_sdk_devel; print(pathlib.Path(_rocm_sdk_devel.__file__).parent)')
export LD_LIBRARY_PATH="$rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DRJ_BUILD_BENCHMARKS=ON \
  -DLTO=OFF \
  -DROCM_PATH="$rocm" \
  -DPython3_EXECUTABLE="$python"

cmake --build "$build" --target rocjitsu_benchmark_workloads
```

`ROCM_PATH` is authoritative: CMake clears cached package locations and finds
HIP, hipBLAS-common, and hipBLASLt package configurations only beneath that
SDK. The build produces one native executable for each target and one host
hipBLASLt adapter. Normal rocjitsu builds are unchanged because
`RJ_BUILD_BENCHMARKS` defaults to `OFF`. Keeping the SDK library directory on
`LD_LIBRARY_PATH` also lets PyTorch resolve the shared libraries shipped in the
same environment.

Build the selected checkout immediately before every official run. The runner
verifies that the build tree was configured from the current worktree, but it
does not try to infer whether binaries became stale after a source or revision
change.

## Run

Run the module from the rocjitsu source directory:

```bash
cd "$src"

"$python" -m benchmarks.runner \
  --build-dir "$build" \
  --output /path/to/results/run-001
```

The checked-in nightly manifest is the default. `--manifest` selects another
manifest; repeated `--target` and `--case` flags select a development subset.
`--warmups` and `--samples` override the manifest for short experiments, and
`--list` prints the selected matrix without running it. The runner refuses to
overwrite an existing output directory.

For a quick end-to-end check, `smoke.toml` runs only the native launch-overhead
case on `gfx950` with one warmup and three samples:

```bash
"$python" -m benchmarks.runner \
  --manifest benchmarks/suites/smoke.toml \
  --build-dir "$build" \
  --output /path/to/results/smoke
```

Each cell has a 300-second default timeout. A cell failure is recorded and does
not discard results from other cells. The command returns failure when any
selected cell fails.

## Artifacts

The output root contains `run.json` with schema
`rocjitsu.benchmark.run.v1`. It records:

- overall status, timestamps, and wall time;
- the git revision and dirty state;
- host, Python, build type, and installed dependency versions;
- the selected matrix and effective warmup/sample policy;
- each cell's provider, parameters, raw timings, minimum, median, maximum,
  status, artifact paths, and failure message.

Per-cell files are retained under `cases/<case>/<target>/`:

```text
workload.json
stdout.txt
stderr.txt
```

`workload.json` uses `rocjitsu.benchmark.workload.v1` and contains the case,
target, parameters, and raw synchronized timings emitted by the workload. The
stdout and stderr files make failures diagnosable without expanding the root
artifact. If a workload fails before emitting JSON, its artifact entry is null;
the logs and partial `run.json` remain available.

The runner intentionally has no built-in comparison, throughput pass, or
profiling collector. Historical analysis can consume `run.json` after enough
nightly data exists to define a useful regression policy.

## Optional manual profiling

Use a `RelWithDebInfo` build and invoke `perf` directly when investigating a
specific case. For example, this records one copy operation on `gfx950`:

```bash
perf record -g -- "$build/tools/rocjitsu/rocjitsu" \
  --config "$src/configs/gfx950_mi355x_kmd.json" -- \
  "$build/benchmarks/rocjitsu-benchmark-native-gfx950" \
  --case hip.copy_fp32_32m \
  --target gfx950 \
  --warmups 0 \
  --samples 1 \
  --output /tmp/rocjitsu-copy-workload.json
```

`perf` observes the complete process, including setup and teardown, so its wall
time is not comparable to the synchronized samples in a Release nightly run.

## Adding a case

Implement the fixed case in the corresponding provider and add its ID to the
suite manifest. A workload must use the public runtime path, check the reported
target, emit the workload schema, return exactly the requested positive timing
samples, and avoid autotuning or network access during a run.
