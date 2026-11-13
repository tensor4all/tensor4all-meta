# ROCm (AMD GPU) backend, built together with CUDA

**Status:** Detailed design, proposed (2026-11). Implementation that runs HIP on a device waits for AMD MI-series hardware (maintainer decision of 2026-11-08 on tenferro-rs#2489). Work marked **(no AMD HW)** can start before then.
**Scope:** t4a-gpu-backend, t4a-cuda-kernels (to be renamed t4a-gpu-kernels), tlinalg-rs (`tlinalg-gpu`), tenferro-rs, and the planned C API and language bindings.
**Tracking issue:** [tenferro-rs#2489](https://github.com/tensor4all/tenferro-rs/issues/2489). Related: [tlinalg-rs#102](https://github.com/tensor4all/tlinalg-rs/issues/102).
**Checked against:** tenferro-rs main `be356ad1`, t4a-cuda-kernels main `b32744d`, t4a-gpu-backend main (2026-11-14).

This document fixes how the AMD backend is built, loaded, layered and tested.
It supersedes the design comment of 2026-11-14 on tenferro-rs#2489 where the two
disagree. The main change is that a **build with both vendors is the primary
artifact**, not just a combination that is allowed to compile.

## 1. Requirements

1. **Features are additive.** `cuda` and `rocm` are independent Cargo features
   in every crate that has them. Enabling one never disables, replaces or
   changes the behaviour of the other. Nothing in the stack is selected with
   `cfg(not(feature = "…"))` between vendors.
2. **One binary serves both vendors.** The artifact that FFI consumers use is
   built with `cuda,rocm`: the C API shared library, the Python wheel and the
   Julia JLL. It must start on a machine with NVIDIA only, AMD only, both, or
   neither, and report each missing backend as *unavailable*.
3. **Vendor libraries are loaded at run time only.** No crate links against a
   vendor library or needs a vendor toolkit at build time. This is what makes
   requirement 2 possible: a build host has neither toolkit installed.
4. **One kernel source.** Custom kernels are written once and compiled at run
   time with NVRTC for CUDA and hiprtc for HIP.
5. **Vendor libraries first.** The first phase uses AMD's own libraries where
   the CUDA backend uses NVIDIA's, and adds no AMD-specific hand-written
   kernels.
6. **C FFI coexists too.** Every C interface in the stack (the kernel ABI of
   t4a-gpu-kernels and the tenferro C API) can be used for both vendors at
   once: from one process, one shared library and one C translation unit.
   Vendor-specific C names therefore carry a vendor prefix
   (`t4a_cuda_*` / `t4a_hip_*`, `tfe_cuda_*` / `tfe_rocm_*`). No C name
   space is shared between vendors (§6.2, §8).
7. The M0 policy of tenferro-rs#2489 still holds: both runtimes may be
   initialised in one process, device ordinals are per vendor, there is no
   implicit transfer between vendors, there is no HIP-on-NVIDIA, and vendor
   libraries are opened with `RTLD_LOCAL`.

## 2. Why the dual-vendor build is the primary artifact

FFI consumers cannot pick Cargo features. A Julia or Python user gets the one
shared library that was published, and an HPC site has NVIDIA and AMD
partitions behind the same file system. If the two backends could only be
built separately, every binding would need two packages and a selection rule
for the user to follow. Building one library with both backends, and choosing
the backend at run time, keeps the binding to a single package and keeps vendor
selection where the user's data already says it: in the `DeviceId` of the
tensor.

So the dual build is not an edge case to test once. It is the configuration
that ships, and the single-vendor builds are the reduced configurations.

## 3. Rules that keep features additive

These rules apply to every crate in the scope above.

| Rule | Concretely |
| --- | --- |
| No exclusive selection | No `cfg(not(feature = "cuda"))` that means "therefore ROCm", and no `compile_error!` for `cuda` together with `rocm`. A `cfg(not(…))` is allowed only for "this vendor is absent", for example returning `false` from a capability probe. |
| Vendor code in sibling modules | `src/cuda/` and `src/rocm/` (or `vendor/cuda`, `vendor/rocm`) each sit behind their own `cfg(feature)`. Shared code never contains `#[cfg(feature = "cuda")]` branches that pick a vendor. It takes the vendor as a run-time value (`GpuBackendKind`) or a type (`CudaEngine` / `RocmEngine`). |
| Probes enumerate, not select | Capability checks such as `is_cuda_engine` in `tenferro-einsum/src/extension.rs:1150-1158` and `tenferro-fft/src/lib.rs:1515-1527` become "is any enabled GPU engine": one arm per enabled vendor, each behind its own `cfg`, combined with `||`. |
| Tests do not switch off | `#[cfg(all(test, not(feature = "cuda")))] mod tests` (`tenferro-linalg/src/extension.rs:23`) must not grow a ROCm twin. CPU tests run in every feature combination, and GPU tests are gated on their own test features. |
| No vendor version features | The bindings target one minimum API, and newer entry points are optional symbols resolved at run time (§5). There is no feature like cudarc's `cuda-12080` on the ROCm side. Such features are mutually exclusive in practice: two dependents that ask for different versions break the build. |
| The kernel crate has no vendor feature | `t4a-gpu-kernels` ships sources, planning and the fusion emitter for both targets unconditionally. They are data and host code with no vendor dependency. The vendor is a field of the planning `Target` (§6). Its C ABI is vendor-prefixed, so both sets of names exist at once. |
| Test-only features are additive too | `gpu-tests-cuda` and `gpu-tests-hip`, never one `gpu-tests` whose meaning depends on another feature. |

The existing `cfg(not(feature = "cuda"))` sites in tenferro (five today) are
"vendor absent" probes or test gates. They are changed to the forms above in
the same pull request that adds the first `rocm` code path in that crate.

## 4. Layering

| Piece | CUDA | ROCm | Repository |
| --- | --- | --- | --- |
| Raw vendor bindings | cudarc (`dynamic-loading`) plus libloading tables | **`t4a-hip-sys`** (new, §5) | t4a-gpu-backend |
| Runtime: streams, pools, transfers, modules, launch | `t4a-cuda-runtime` | **`t4a-hip-runtime`** (new, §5) | t4a-gpu-backend |
| Custom kernels, planning, fusion emitter | `t4a-cuda-kernels` | the same crate, renamed **`t4a-gpu-kernels`** (§6) | t4a-gpu-kernels |
| Dense linalg drivers | `tlinalg-gpu` `vendor/cuda` (cuSOLVER, cuBLAS) | `tlinalg-gpu` `vendor/rocm` (rocSOLVER, rocBLAS) | tlinalg-rs |
| GEMM | cuBLAS | rocBLAS | tenferro-gpu |
| Contraction | cuTENSOR | hipTensor; rocBLAS GEMM fallback when hipTensor cannot express the contraction | tenferro-gpu / tenferro-einsum |
| FFT | cuFFT | hipFFT | tenferro-fft |
| Engine and tensor glue | `tenferro-gpu::cuda::CudaEngine` | `tenferro-gpu::rocm::RocmEngine` | tenferro-gpu |
| Tensor semantics, fusion planning, AD | tenferro, vendor-independent | unchanged | tenferro-rs |

Library choice rule: use the `roc*` library where one exists (rocBLAS,
rocSOLVER). Use a `hip*` library only when it is the only AMD implementation
(hipTensor) or when its API mirrors the CUDA library closely enough that a
driver can be shared (hipFFT mirrors cuFFT's plan API, so the plan cache in
`tenferro-fft` is shared). hipSOLVER is not used: on AMD it is a layer over
rocSOLVER, so it would mean loading one more library for an API similarity
that the tlinalg-gpu vendor trait already provides.

Out of scope for the first phase: Composable Kernel, rocPRIM/hipCUB,
hipBLASLt, RCCL, AMD peer-to-peer, and hand-tuned AMD kernels.

## 5. HIP bindings and runtime

### 5.1 Why our own crates

No crate on crates.io does for HIP what cudarc does for CUDA, which is to load
the runtime and the vendor libraries at run time (survey of 2026-11-14):

- `cubecl-hip-sys` is widely used, but it binds at build time and is versioned by HIP release.
- `rocmrc` (modelled after cudarc, with a `dynamic-loading` feature) is the closest match. It is young and has one maintainer. It covers HIP, hiprtc, rocBLAS and hipBLASLt, but not rocSOLVER, hipTensor or hipFFT. Its ROCm-version features (`rocm-06002` … `rocm-07020`) break the "no vendor version features" rule.
- `rocm-rs`, `hip-sys` and others link at build time or are small.

The decision is therefore to write our own. `rocmrc` may replace `t4a-hip-sys`
only if it loads everything at run time including hiprtc, exposes
stream-ordered memory pools, accepts rocSOLVER, hipTensor and hipFFT tables,
and drops version features. Without all four, we keep our own crate.

### 5.2 `t4a-hip-sys`

- One lazily loaded function table per library: `libamdhip64` (device,
  stream, event, `hipMallocAsync` and memory pools, memcpy, module, launch),
  `libhiprtc`, `librocblas`, `librocsolver`, `libhiptensor` and `libhipfft`.
  A consumer pays only for the libraries it opens.
- Bindings are generated once, offline, with bindgen against the minimum
  supported ROCm (6.2, to be confirmed while generating them), and committed as
  source. There is no `build.rs`. A script in the repository regenerates them
  and CI checks that the committed file matches the script's output for the
  pinned headers.
- Library discovery: `ROCM_PATH/lib` when `ROCM_PATH` is set, then the
  dynamic loader's search path. Each library has a fixed soname candidate list
  covering ROCm 6.x and 7.x (for example `libamdhip64.so.7`, `libamdhip64.so.6`,
  `libamdhip64.so`). Every library is opened with `RTLD_LOCAL | RTLD_NOW`.
- Symbols are classified as required or optional for each table. A missing
  required symbol makes that library unavailable. A missing optional symbol
  makes only the entry point that uses it unavailable. This follows the
  optional-symbol contract on the CUDA side (tenferro-rs#2413).
- Errors name the library, the candidates that were probed, and the version
  that was found. The crate has no safe API and no policy.

### 5.3 `t4a-hip-runtime`

The same ownership contracts as `t4a-cuda-runtime`
(t4a-gpu-backend `docs/design/cuda-runtime.md`):

- one real stream and one explicit memory pool per runtime, including when two runtimes open the same device;
- stream-ordered allocation, bounded pinned staging, and explicit ordered transfers;
- completion, witnesses and retirement;
- hiprtc artifacts and per-runtime modules;
- checked launches with scalar, device-pointer and by-value parameter blocks, plus the raw consumer seam.

The public API mirrors `t4a-cuda-runtime` name for name wherever the semantics
match. It is a sibling crate, not an implementation of a shared trait, which
follows t4a-gpu-backend's "no universal abstraction first" rule. Host-only
pieces that end up duplicated, such as retirement bookkeeping or artifact-key
hashing, are extracted into a shared crate only once both runtimes exist and
the duplication can be seen.

Compile artifacts: hiprtc produces a code object for one `gfx` target. The
artifact key contains the vendor, the compiler identity (hiprtc version), the
target `gfx` name with its feature flags (for example `gfx90a:xnack-`), the
wave size and the options.

## 6. One kernel source: `t4a-gpu-kernels`

The repository `tensor4all/t4a-cuda-kernels` is renamed to
`tensor4all/t4a-gpu-kernels` on GitHub, so old links redirect. The crate is
published as `t4a-gpu-kernels`. The last `t4a-cuda-kernels` release
re-exports it and is deprecated. The crate keeps its contract: no runtime
dependency, and the caller owns compilation, module loading, streams and
scratch.

### 6.1 Source changes

1. **Portability header** `kernels/include/t4a/port.h`. It is selected by
   `__HIP_PLATFORM_AMD__` or `__CUDACC__` and holds only the differences:
   warp shuffles, ballot and active mask, the lane-mask type (32-bit or
   64-bit), and the `__launch_bounds__` spelling. Every shuffle in
   `cuda/include/t4a/abi.cuh:713-739` and `cuda/src/rrqr.cu` goes through
   `T4A_SHFL_XOR` / `T4A_SHFL`. On HIP the header maps to the non-`_sync`
   shuffles, so the kernels do not depend on when HIP made its `*_sync`
   intrinsics the default. Atomics, including the complex `atomicAdd` pairs in
   `gather_scatter.cu`, compile under HIP unchanged.
2. **Wave size is a planning input.** Today the planner hard-codes
   `WARP_WIDTH: u32 = 32` (`src/lib.rs:3034`) and `rrqr.cu:35` uses
   `kWarpMask = 0xFFFFFFFF`. CDNA (gfx90a, gfx942) runs 64-wide wavefronts and
   RDNA runs 32-wide ones. Planning takes
   `Target { vendor: Vendor::{Cuda, Hip}, arch, wave_size }`. Kernels read the
   wave size as a compile-time constant (`-DT4A_WAVE_SIZE=…`), and the
   multi-plane reduction path (`PARTIALS_*` in `src/lib.rs`) is re-derived for
   a wave size of 64. This is the main correctness risk of the port.
3. **Vendor-prefixed C ABI, generated from one source.** The C ABI keeps a
   vendor prefix so that both vendors' ABIs can be used together, in one
   process and in one translation unit (§6.2). The existing `t4a_cuda_*`
   names stay unchanged: no rename, and no ABI revision bump for CUDA. HIP gets
   the parallel set `t4a_hip_*` (`t4a_hip_add`, `t4a_hip_dtype`,
   `t4a_hip_c64`, …). Sources spell each name through a prefix macro, for
   example `T4A_SYM(add)`, which `port.h` expands to `t4a_cuda_add` or
   `t4a_hip_add`. The kernel bodies are therefore written once, and every
   kernel exists twice in the ABI with identical meaning. The revision number
   (`T4A_*_ABI_REVISION`) is shared, and a revision always changes both
   vendors together.
4. **Fusion emitter.** `fusion.rs` emits the same source for both vendors.
   The artifact key gains the vendor and the wave size.
5. **Host planning is shared.** Launch geometry, stride normalisation and
   validation are computed by the same Rust code for both vendors, with the
   `Target` as an input. A plan is valid only for the target it was computed
   for.

### 6.2 Why the C ABI is not renamed to `t4a_gpu_*`

A single vendor-neutral `t4a_gpu_*` name space cannot hold two vendors at once:

- **Headers.** The CUDA and HIP headers are generated from one template into
  `t4a/cuda/abi.h` and `t4a/hip/abi.h`. Their include guards
  (`T4A_CUDA_ABI_H_` / `T4A_HIP_ABI_H_`) and every type, enum constant and
  function name are vendor-prefixed. A C or C++ consumer that drives both
  vendors includes both headers in one file. With shared `t4a_gpu_*` names,
  the second include would redefine the first.
- **Host code compiled into a library.** A consumer that compiles host-side
  shims, launch wrappers or the kernels themselves into one shared library
  for both vendors would get duplicate `t4a_gpu_*` symbols at link time. With
  vendor prefixes it gets two disjoint sets.
- **Device modules.** NVRTC and hiprtc modules are separate, so lookup by
  name would work even with shared names. However, a vendor prefix makes a
  wrong-vendor lookup fail by name instead of by misbehaving, and keeps the
  module names the same as the C ABI.

The Rust facade mirrors this. `t4a_gpu_kernels::cuda` and
`t4a_gpu_kernels::hip` are both always compiled, and neither sits behind a
feature. They expose the same planning API over their own ABI constants. The
shared planning code underneath takes a `Target` (step 2 above).

### 6.3 CI for the shared source (no AMD HW)

- New job `kernel-compile-hip`: `hipcc --offload-arch=gfx90a,gfx942,gfx1100 --genco`
  over every source, run in the official `rocm/dev-ubuntu` container on a
  GitHub-hosted runner. Compiling needs no GPU.
- The CUDA `kernel-compile` job and the A100 `gpu-tests-cuda` suites are
  unchanged. They are the oracle that steps 1–5 changed nothing for CUDA.
- Host planning tests run for both a 32-wide and a 64-wide `Target`.

## 7. tenferro

- **Engines.** `tenferro-gpu` gets `src/rocm/` with `RocmEngine`, a sibling of
  `src/cuda/` and `CudaEngine`. Its runtime adapter, event domains,
  publication, pinned staging and structural dispatch mirror the CUDA modules.
  The op-family dispatch tables that only map a tensor op to a kernel
  catalogue entry are written once against `t4a-gpu-kernels` and the `Target`.
- **Device discovery.** Discovery lists CUDA and ROCm devices independently.
  If `libamdhip64` is absent, no ROCm devices are listed. Asking for
  `Gpu(Rocm)` explicitly then returns a typed *backend unavailable* error that
  names the missing library and the probed paths, with the remedy (install
  ROCm or set `ROCM_PATH`).
- **Cross-vendor operands.** An op whose operands are on devices of different
  vendors refuses with an error. The remedy is an explicit copy through host
  memory.
- **Feature wiring.** The `rocm` feature is added to `tenferro-gpu`,
  `tenferro-einsum`, `tenferro-linalg` and `tenferro-fft`, each in the pull
  request that gives that crate a working ROCm path. That way no feature name
  exists without an implementation (tensor4all-meta#72).
- **Existing probes.** The capability probes and test gates listed in §3 are
  rewritten in the additive form in those same pull requests.

## 8. FFI

- The planned `tenferro-capi` shared library, the Python wheel and the Julia
  JLL are built with `cuda,rocm`. A release that ships only one vendor is not
  produced.
- **The C header does not depend on features.** Every GPU entry point exists
  in every build. In a build or on a machine without the vendor, the entry
  point returns `TFE_BACKEND_UNAVAILABLE` (a new status code) with a message
  naming the vendor. Host languages therefore bind one header with no
  conditional symbols.
- **Device kinds.** The C device enum carries `CUDA` and `ROCM` with
  per-vendor ordinals, mirroring `DeviceId`. A query function reports, for
  each vendor, whether it is available and how many devices it has.
- **DLPack.** Export maps ROCm memory to `kDLROCM` (10) and pinned host
  memory to `kDLROCM_HOST` (11). The mapping table in tenferro-rs
  `docs/design/capi.md` already lists both. Import stays CPU-only until GPU
  import is designed for both vendors together.
- **Two kinds of C entry point.** Operations on tensors (`tfe_einsum_*`,
  `tfe_linalg_*` and so on) have one vendor-neutral symbol each, because the
  vendor comes from the tensor's device at run time. Entry points that expose
  vendor objects are vendor-prefixed and exist side by side: `tfe_cuda_*` and
  `tfe_rocm_*`. Both sets are present in every build, and each returns
  `TFE_BACKEND_UNAVAILABLE` when its vendor is missing.
- **Raw interop** (the counterpart of tenferro-rs#1940, which covers direct
  cuBLAS and cuSOLVER calls on tenferro buffers) uses the vendor-prefixed
  set. `tfe_cuda_tensor_stream` returns a `CUstream`, and
  `tfe_rocm_tensor_stream` returns a `hipStream_t`. They are declared as
  opaque pointer typedefs in the header, so the header does not include any
  vendor header. No `void*` with a vendor tag needs to be checked by the
  caller. A consumer that drives both vendors calls both sets from one
  translation unit.
- **Symbol hygiene.** Everything vendor-specific is resolved through
  `RTLD_LOCAL` handles inside the shared library. Neither NVIDIA's nor AMD's
  symbols, and none of the LLVM copies bundled in NVRTC or comgr, are
  re-exported. The shared library exports only `tfe_*` symbols, and a CI check
  of `nm -D --defined-only` enforces it.

## 9. Validation

| Lane | Where | What it proves |
| --- | --- | --- |
| `--features cuda,rocm` build, clippy, CPU tests | GitHub-hosted, every PR (no AMD HW) | Additivity: the dual build compiles with no toolkit installed, and the CPU behaviour is identical |
| `--features cuda`, `--features rocm`, no GPU features | GitHub-hosted, every PR | The reduced builds still compile |
| Absent-library tests | GitHub-hosted | With neither vendor installed, the dual build starts, lists no GPU devices, and returns *unavailable* errors that name the library |
| `kernel-compile-hip` | GitHub-hosted, ROCm container | The single kernel source compiles for gfx90a, gfx942 and gfx1100 |
| `gpu-tests-cuda` | Local A100 | CUDA unchanged by the shared-source refactor |
| `gpu-tests-hip` | AMD MI-series, once available | Runtime, kernels (golden tests at wave sizes 64 and 32), linalg, einsum, FFT |
| Dual-stack test | A host with both vendors, or one vendor plus a stub library for the other | Both runtimes initialise in one process; NVRTC and hiprtc each compile one kernel without an LLVM symbol clash |
| Exported-symbol check | GitHub-hosted, C API crate | Only `tfe_*` symbols are exported |

## 10. Milestones

These replace M1–M5 of tenferro-rs#2489. M0 stays as approved, and M6 stays
as written.

- **M1a — shared kernel source (no AMD HW).** Rename the repository and crate
  to t4a-gpu-kernels. Add the portability header and the `T4A_SYM` prefix
  macro, so that the CUDA names stay byte-identical and `t4a_hip_*` is added.
  Make wave size a planning input, generate both headers from one template
  together with a test that includes both in one C translation unit, and add
  `kernel-compile-hip`. Done when the A100 suites are unchanged and every
  source compiles for the three AMD targets.
- **M1b — `t4a-hip-sys` (no AMD HW).** Function tables, discovery, optional
  symbols, and the regeneration script. Done when the absent-library tests
  pass and the committed bindings match the script.
- **M1c — additivity in tenferro (no AMD HW).** Rewrite the five
  `cfg(not(feature = "cuda"))` sites into the forms of §3, and add the
  `cuda,rocm` CI lane as soon as the first crate has a `rocm` feature. Until
  then, add a lint script that rejects new exclusive vendor `cfg` patterns.
- **M2 — `t4a-hip-runtime` (AMD HW).** The same standalone validation as
  `t4a-cuda-runtime`, plus the dual-stack test.
- **M3 — kernel catalogue on HIP (AMD HW).** The golden tests at wave sizes 64
  and 32. `RocmEngine` gets the structural, elementwise and reduction families.
- **M4 — linalg.** `tlinalg-gpu` `vendor/rocm` on rocSOLVER and rocBLAS;
  `tenferro-linalg` gains `rocm`.
- **M5 — einsum and FFT.** hipTensor with the rocBLAS fallback, and hipFFT.
  `tenferro-einsum` and `tenferro-fft` gain `rocm`.
- **M6 — parity.** AD and fusion parity across CPU, CUDA and ROCm, and the
  FFI build with `cuda,rocm` (§8).

## 11. Open questions

- Validation hardware: local MI-series access or a cloud runner. M1a–M1c do not need it.
- The minimum ROCm (6.2 proposed). It is fixed when M1b generates the bindings.
- Whether hipTensor covers tenferro's contraction set (mixed strides, complex
  types, batch modes) well enough to be the primary path. The rocBLAS fallback
  is designed in either way.
- Whether a host with both vendors is available for the dual-stack test, or
  whether a stub `libamdhip64` is enough to exercise symbol isolation.
