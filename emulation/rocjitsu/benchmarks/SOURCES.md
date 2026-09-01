# Benchmark source attribution

The rocjitsu benchmark adapters are original implementations derived from the
fixed workload definitions below. They intentionally use public HIP, Triton,
and hipBLASLt interfaces; none load code objects or call `rj_vm` directly.

The exact executable dependency cohort is recorded in `dependencies.lock.toml`.
Generated kernels and HSCO files are build/cache artifacts and must not be
committed.

## Triton workloads

The adapter uses the public language and launch interfaces from
<https://github.com/ROCm/triton>. Its package build is pinned by the complete
distribution version in `dependencies.lock.toml`; no unverifiable upstream
source revision is inferred from that wheel version.

The kernels in `workloads/triton_workloads.py` are original fixed
implementations. They use fixed launch configurations and never autotune during
a measured run. The aligned/boundary pairs are deliberate local coverage rather
than copies of an upstream parameter sweep.

## hipBLASLt/TensileLite workloads

The adapter consumes hipBLASLt from the `rocm-sdk-libraries` distribution in
the locked ROCm SDK cohort. No rocm-libraries checkout, device-library
generation, or source build is part of benchmark setup. The implementation is
maintained in <https://github.com/ROCm/rocm-libraries>.

Preflight verifies the public hipBLASLt package version. Each run hashes both
the loaded shared object and the selected target's installed Tensile
device-library directory. This identifies the actual package artifacts without
imposing a separate source dependency.

The in-tree adapter expresses three fixed problems through the public
hipBLASLt API and requests exactly one heuristic solution from the installed
library. It does not run CPU verification, an exhaustive problem set, or any
tuning/search loop. Allocation, initialization, descriptor construction, and
heuristic selection occur before synchronized timing begins.
