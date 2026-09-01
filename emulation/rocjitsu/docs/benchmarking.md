# Benchmarking rocjitsu

The in-tree benchmark system runs a fixed workload matrix through the public
HIP runtime on `gfx950` and `gfx1250`. It is intended for nightly performance
tracking and manual change evaluation. Version 1 produces immutable local JSON
artifacts; CI scheduling, historical storage, and dashboard publication are
separate follow-on work.

## Suite and case catalog

`benchmarks/suites/nightly.toml` is deliberately a small composition file. It
contains the suite identity, target list, time budget, ordered case IDs, and
the suite-wide sampling policy. It contains no commands or filesystem paths.

The trusted definitions behind those IDs live in `benchmarks/catalog.py`.
Each definition owns its provider, complete parameters, timeout, source files,
and any stable diagnostic dispatch count. Provider code resolves the Rocjitsu
launcher, target configs, build products, Python module, and dependency lock
from the source/build layout. Copying a suite manifest to another directory
therefore cannot change what it runs.

The nightly matrix contains 12 cases per target:

| Provider | Cases | Coverage |
|---|---:|---|
| Native HIP | 3 | launch overhead, a vectorized 32 MiB model-weight-style copy, and tail handling |
| Triton | 6 | aligned/boundary softmax, BF16 RMSNorm, aligned/ragged BF16 GEMM, and noncausal attention |
| hipBLASLt/TensileLite | 3 | FP16 GEMM, batched BF16 GEMM, and scaled FP8 GEMM |

Every case runs on both suite targets. Inputs are deterministic, launch
configurations are fixed, and autotuning is prohibited during measurement.
Generated kernels, caches, and HSCO files are build artifacts and must not be
committed. The dependency cohort is pinned in
`benchmarks/dependencies.lock.toml`; source attribution is in
`benchmarks/SOURCES.md`.

## What is measured

The primary metric is `synchronized_dispatch_ns`. One primary process is
created for each case/target pair. That process performs:

1. allocation, deterministic initialization, compilation or descriptor setup,
   and algorithm selection;
2. three untimed `{ launch; device synchronize; }` warmups;
3. twenty-one individually host-timed
   `{ start; launch; device synchronize; stop; }` observations.

Only step 3 contributes to the primary statistics. Process startup,
allocation, input transfers, Triton compilation, hipBLASLt heuristic selection,
warmups, JSON output, and teardown are excluded. The runner derives median,
median absolute deviation, minimum, maximum, and range from the 21 raw values.

For a native or Triton case, one observation contains one kernel launch. For a
hipBLASLt case, it contains one `hipblasLtMatmul` API call; the selected
implementation may launch multiple device kernels. The separate throughput
diagnostic records those internal dispatches.

GNU `time` and `perf stat` wrap the entire primary process and are retained as
diagnostics. They are not used as the performance trend metric. A second
instrumented process runs zero warmups and one operation with the Rocjitsu
throughput plugin; observer overhead from that pass cannot affect primary
timings.

The throughput pass records every dispatch made by that process, including a
runtime or library initialization dispatch when one is required. The catalog
enforces an exact count only when it is target-independent; otherwise the
runner requires a positive count and comparison requires that count to remain
stable between runs.

### Correctness policy

Nightly timing does not perform numerical correctness checks. Native workloads
do not copy outputs to the host, Triton does not construct CPU references, and
the hipBLASLt workload does not request verification. This keeps reference
work and output comparison out of both timing and process diagnostics.

The runner still enforces measurement integrity. A result fails for a nonzero
exit status, timeout, malformed JSON, wrong schema/case/target/parameters,
wrong dependency identity, mismatched sampling counts, or a timing vector that
does not contain exactly the requested number of positive integer values.
These checks can detect a broken execution path, but they cannot detect a
kernel that runs successfully and computes the wrong result.

## Prerequisites and build

Use a dedicated, quiet Linux host with stable BIOS, firmware, governor, CPU
placement, SDK, and environment. The initial cohort requires:

- a ROCm SDK with `amdclang++` and HIP support for both targets;
- Python 3.12 with the pinned ROCm SDK, PyTorch, and Triton packages;
- the locked `rocm-sdk-libraries` package, including hipBLASLt headers,
  shared libraries, and device libraries for both targets;
- GNU `time` and permission to use userspace `perf` counters.

Configure the benchmark build against that SDK and Python environment:

```bash
src=/path/to/rocm-systems/emulation/rocjitsu
build=/path/to/rocjitsu-build-release
rocm=/path/to/rocm-sdk
python=/path/to/pinned/bin/python

cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DRJ_BUILD_BENCHMARKS=ON \
  -DLTO=OFF \
  -DROCM_PATH="$rocm" \
  -DPython3_EXECUTABLE="$python"

cmake --build "$build" --target \
  rocjitsu_bin rocjitsu_shared rocjitsu_plugin_throughput_so \
  rocjitsu_benchmark_workloads
```

hipBLASLt is never built from source by this workflow; the in-tree adapter uses
the installed SDK package. If the host loader does not honor the executable's
build RPATH, add `$rocm/lib` to `LD_LIBRARY_PATH`.
Official symmetric runs use the SDK's standard device-library layout and
leave `HIPBLASLT_TENSILE_LIBPATH` unset. For a one-target development run with
a nonstandard layout, that variable must name the exact target directory (for
example, `/alternate/hipblaslt/library/gfx950`), not its parent.

The build produces target-specific native HIP workloads and one host
hipBLASLt workload linked to the selected package. Preflight runs the latter's
probe through Rocjitsu for every selected target, checks the public package
version against the lock, and requires files in each target's installed
device-library directory. Hashes of the loaded library and selected target's
device libraries are retained as artifact provenance. Normal Rocjitsu builds
are unchanged because `RJ_BUILD_BENCHMARKS` defaults to `OFF`.

## Preflight and run

Run the package from the Rocjitsu source directory:

```bash
cd "$src"

"$python" -m benchmarks.runner preflight \
  --manifest benchmarks/suites/nightly.toml \
  --build-dir "$build"

"$python" -m benchmarks.runner run \
  --manifest benchmarks/suites/nightly.toml \
  --build-dir "$build" \
  --output /path/to/results/run-001
```

The runner derives the Rocjitsu executable from `--build-dir`; `--rocjitsu`
can override it. `--targets` and `--cases` select development subsets.
`--warmups` and `--samples` are recorded noncanonical overrides for short
development runs. `--config`, `--collectors`, and `--env` are likewise recorded
in provenance. Official historical results must use the checked-in defaults.

`preflight --list` prints the ordered case/target matrix without requiring
the benchmark executables. A normal preflight validates the build type,
commands, configs, source files, package cohort, hipBLASLt package, and
collector permissions. The runner refuses to overwrite an output directory.

## Artifacts

The root `run.json` uses `rocjitsu.benchmark.run.v1`. It contains the effective
suite and sampling policy, catalog and manifest digests, host/config/build
provenance, the path-independent dependency-verifier report and its source
digest, one primary result per case/target, throughput diagnostics, summaries,
and failures.

The v1 identifiers describe the final layout documented here. Artifacts made
from earlier, unpublished development drafts are not compatible; the comparer
rejects them when the required result matrix or execution fields are absent.

Each primary result is stored below:

```text
cases/<case>/<target>/timing/
  case-result.json
  command.json
  workload.json
  stdout.txt
  stderr.txt
  time.txt
  perf-stat.csv
```

`case-result.json` uses `rocjitsu.benchmark.case-result.v1` and retains the 21
raw timings plus their statistics. Process wall time and normalized GNU time/
perf data appear under `diagnostics`. The corresponding `throughput/`
directory holds its derived config and plugin JSONL. Failed commands preserve
the same normalized artifacts whenever possible, including sanitized external
JSON with non-finite values replaced by `null`.

Source provenance uses a path-independent catalog digest plus content digests
for each case's implementation files. Absolute paths remain informational, so
identical builds/configs in different worktrees can still compare.

## Compare and profile

Comparison requires the same matrix, manifest/catalog, sampling policy, host,
build type, configs, dependencies, payloads, sources, collectors, and effective
environment. It pairs every internal timing by ordinal and reports ratio,
delta, and percentage-change statistics. Process-wall diagnostics are never
ratio inputs.

```bash
"$python" -m benchmarks.runner compare \
  --baseline /path/to/baseline/run.json \
  --candidate /path/to/candidate/run.json \
  --output /path/to/comparison.json
```

Profiles use a separate `RelWithDebInfo` build. Triton cases first run one
unprofiled operation to populate their disk cache; native and hipBLASLt cases
do not need that priming pass. `perf record` then wraps the entire zero-warmup,
one-operation process, including startup, setup, the synchronized dispatch,
and teardown:

```bash
"$python" -m benchmarks.runner profile \
  --manifest benchmarks/suites/nightly.toml \
  --target gfx950 \
  --case hip.copy_fp32_32m \
  --build-dir /path/to/rocjitsu-build-relwithdebinfo \
  --output /path/to/profiles/copy-gfx950
```

Do not compare profile timings with the Release series.

## Adding a case

Add the case once to `benchmarks/catalog.py`, implement it in the corresponding
built-in provider, and add its ID to the desired suite manifest. New cases
must use the public runtime path, verify the reported target, emit the exact
catalog parameter object, return the requested synchronized timing vector,
avoid autotuning and network access, and pass catalog parity plus both-target
smoke coverage. Arbitrary manifest commands are intentionally unsupported.
