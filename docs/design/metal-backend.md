# Metal (Apple GPU) backend over a shared allocation domain

**Status:** Detailed design, proposed (2026-11). Prerequisite: tenferro-rs#2609 (vendor BLAS/LAPACK providers removed) is merged; nothing here assumes Accelerate.
**Scope:** t4a-gpu-backend (`t4a-metal-sys`, `t4a-metal-runtime`), t4a-gpu-kernels (MSL sources and the Metal target), tenferro-rs (tensor placement, `tenferro-cpu`, `tenferro-gpu`, operation crates).
**Tracking issue:** [tenferro-rs#2402](https://github.com/tensor4all/tenferro-rs/issues/2402). Sibling design: [rocm-backend.md](./rocm-backend.md); this document follows its rules wherever they apply.
**Checked against:** tenferro-rs main `b77d4f00`, tensor4all-meta main `72b4f85`.

This document supersedes the parts of the tenferro-rs#2402 body that disagree
with it: f64 is never routed to the CPU automatically, and host buffers are not
wrapped with `newBufferWithBytesNoCopy`; a shared allocation domain replaces
both (§4).

## 1. Intent

tenferro lost Apple GPU execution when the CUDA backend moved to
`t4a-cuda-runtime` and the kernel catalogue (tenferro-rs#2208) and the
CubeCL-era code was retired (#2312; docs swept in #2375 and #2403). The Metal
path and the Apple shared CPU/Metal path were removed together with CubeCL. Nobody
rejected their design.

Many people who would try tenferro develop on a Mac and run production on CUDA
clusters, and every Rust ML stack they compare against (candle, burn, MLX) runs
on Metal. Without Metal, tenferro reads as "no GPU on a Mac". The goal is to
remove that reason not to choose tenferro without bending its explicit
execution contract.

## 2. Requirements

1. **Explicit execution.** No implicit CPU-GPU transfer, no silent CPU
   fallback, no implicit copy. An op/dtype the Metal engine does not serve
   returns a typed `Unsupported` error (tenferro-rs `REPOSITORY_RULES.md`,
   "Device Transfer And Backend Buffer Errors").
2. **f32 and Complex32 on the GPU, never f64.** Apple GPUs have no double
   precision (measured in tenferro-rs#2402: the MSL compiler rejects
   `double`). f64 and Complex64 operands of a Metal op return `Unsupported`
   whose message says to run the op on the paired `CpuEngine`, which reads the
   same tensors without a transfer (§4).
3. **A vendor-neutral shared allocation domain.** Buffers in it are readable
   and writable by the CPU engine and a GPU engine at the same address. Metal
   is its first provider. AMD APUs (MI300A, Strix Halo) and NVIDIA coherent
   systems (Grace Hopper, GB10) are later providers of the same concept, so
   nothing in the domain contract names Metal.
4. **Readers wait automatically.** Touching a shared allocation from the CPU
   waits for pending GPU writes to it. Waiting moves no data, so it is not an
   implicit transfer.
5. **No build-time linking** (tenferro-rs#2402 §1). Metal,
   MetalPerformanceShaders and MetalPerformanceShadersGraph are opened with
   `dlopen`, classes resolved with `objc_getClass`, and `objc_msgSend` cast per
   signature. No `build.rs`, no `-framework`, no `links =`.
   `otool -L <binary> | grep -Ei 'Metal|MetalPerformanceShaders|Accelerate'`
   prints nothing.
6. **Additive features** ([rocm-backend.md §3](./rocm-backend.md)). `metal`
   is an independent feature that can be enabled together with `cuda` and
   `rocm`. Nothing selects a vendor through `cfg(not(feature = …))`. On
   non-Apple targets the `metal` code compiles out and discovery lists no
   Metal device.
7. **One ownership model.** The shared domain uses the ownership kernel of
   tenferro-rs `docs/design/storage-ownership-contracts.md`; it adds no second
   ownership model.

Non-goals for the first version: emulated f64 (double-float), bf16/f16,
WebGPU/wgpu (out of scope by the 2026-11-08 decision on tenferro-rs#2402),
device-private (`storageModePrivate`) Metal tensors, multi-GPU, and the AMD or
NVIDIA providers of the shared domain.

## 3. Approaches considered

| Approach | Trade-off | Decision |
| --- | --- | --- |
| **A. Own sys/runtime crates mirroring CUDA/HIP**: `t4a-metal-sys` (dlopen function and class tables), `t4a-metal-runtime`, MSL sources in t4a-gpu-kernels compiled at run time with `newLibraryWithSource:`, MPS for vendor GEMM | Full control of storage mode, ownership and completion, which the shared domain and automatic waiting need; same structure as CUDA and ROCm | **Chosen** |
| B. wgpu/WGSL | One portable backend, but wgpu abstracts memory so the shared domain cannot be controlled; no f64 anywhere; the road CubeCL already took; WebGPU is out of scope | Rejected |
| C. MLX through mlx-c | Broad op coverage quickly, but MLX's lazy graph and allocator conflict with caller-held engines and the ownership kernel; heavy C++ dependency | Rejected |
| objc2-metal for bindings | Mature, but links the frameworks at build time (violates requirement 5) | Rejected |

Also rejected during the design discussion:

- **Device-local Metal tensors with copy-free "move" uploads** (consume a host
  tensor and wrap it with `newBufferWithBytesNoCopy`). It pushes
  page-alignment and ownership special cases into the transfer API. A shared
  placement removes the transfer instead.
- **Routing f64 to the CPU automatically** (the tenferro-rs#2402 body).
  Violates requirement 1. The caller routes f64 work to the CPU engine
  explicitly.

## 4. The shared allocation domain (tenferro-rs)

### 4.1 What exists

The CPU half survived the CubeCL retirement and is reused, not rebuilt:
`tenferro_tensor::SharedTensorAllocationDomain` (`id`, `allocate`),
`AllocationDomainId`, `CpuEngine::with_allocation_domain` (the engine retains
the domain owner, not only its id, so CPU extension ops can allocate
same-domain outputs), the domain checks in `tenferro-cpu/src/runtime_adapter.rs`
and `tenferro-cpu/src/lib.rs`, `HostWriteGuard`, and the managed FFT and
Cholesky paths with their `FakeDomain` tests. The earlier record is tenferro-rs
`docs/worklogs/2026-07-19-apple-shared-allocation-domain.md`.

### 4.2 Placement

- Add `GpuBackendKind::Metal` (`tenferro-tensor/src/types.rs`).
- A shared domain provider allocates host-visible, device-visible buffers. For
  Metal these are `MTLBuffer`s with `MTLResourceStorageModeShared`.
- Every Metal tensor in the first version lives in its context's shared
  domain. There is no device-local Metal placement.
- A `MetalContext` owns one device, one command queue, one allocation domain,
  and hands out a paired `CpuEngine` (built with `with_allocation_domain`) and
  a `MetalEngine` bound to that domain. Independently created contexts get
  distinct domains.

### 4.3 CPU engine over shared tensors: one boundary change

Most CPU kernels reach storage through `typed_host_data` / `typed_view`
(`tenferro-cpu-basic/src/lib.rs`), which reject backend buffers today. The
change is made once, at that boundary: a buffer in the engine's own shared
domain is admitted by waiting for its pending GPU writes (§5) and then taking a
host slice of the shared allocation. Every existing CPU kernel (elementwise,
reductions, einsum, linalg, FFT) then runs on shared tensors without per-kernel
work. Strided views of shared buffers go through the same admission.

- Buffers of another domain, or device-local buffers, keep today's
  runtime-state error.
- CPU outputs computed from shared-domain inputs are allocated in the same
  domain (the existing `with_allocation_domain` behaviour), so results chain
  back into the Metal engine without an upload.
- `HostWriteGuard` was write-only because WGPU mappings are. A Metal shared
  buffer can be written in place, so the shared-domain path may hand out
  `&mut [T]` under the ownership kernel's exclusive capability.

### 4.4 Transfers and host inspection

- `MetalContext::upload(&Tensor)` copies a host tensor into a new shared
  allocation and counts upload bytes. `MetalContext::download(&Tensor)` copies
  a shared tensor into a plain host tensor and counts download bytes. Both are
  explicit copies; nothing else counts as a transfer.
- Host inspection (`host_data`, `host_data_mut`) of a shared-domain tensor is
  allowed. It waits (§5) and maps; it does not copy. `REPOSITORY_RULES.md`
  gains this case next to the existing "backend buffers are runtime-state
  failures" rule.
- A CPU tensor passed to a Metal op, or a Metal tensor passed to a CUDA or
  ROCm op, keeps the existing errors: no cross-vendor or cross-domain implicit
  movement ([rocm-backend.md §7](./rocm-backend.md)).

## 5. Synchronization

- Every shared allocation records the completion of the last GPU submission
  that writes it and of submissions that read it.
- **CPU readers and writers wait.** CPU-engine admission (§4.3), host
  inspection, and `download` wait for the pending GPU writes before
  proceeding; a CPU write also waits for pending GPU reads. This replaces the
  `HostAccessError::GpuAccessInProgress` result on the shared-domain path. The
  managed-Cholesky tests that assert that error are updated, not deleted, to
  assert the wait.
- **GPU submissions order after CPU work.** CPU ops complete before they
  publish their outputs, so a GPU op that reads a CPU-produced shared buffer
  needs no extra fence. A GPU submission that meets a live host guard on its
  operand returns `MappedForHost`, as today: a guard is caller-scoped, and
  waiting on it from inside a submission could deadlock.
- The completion source is the command buffer's completion (`MTLSharedEvent`
  or completed-handler signalling; builder's choice). It plugs into the
  runtime's `EventDomainDriver` / `EventToken` vocabulary, as
  `tenferro-gpu/src/cuda/event_domain.rs` does.
- **Traced programs do not mix engines in the first version.** One traced
  program runs on one engine. Eager code mixes Metal and CPU ops freely, with
  the waiting above. Cross-engine event ordering inside the traced runtime is
  later work.

## 6. t4a-gpu-backend: `t4a-metal-sys` and `t4a-metal-runtime`

- **`t4a-metal-sys`**: lazily loaded tables for `libobjc` (`objc_getClass`,
  `sel_registerName`, `objc_msgSend`), Metal (`MTLCreateSystemDefaultDevice`
  and the selectors used), MetalPerformanceShaders and
  MetalPerformanceShadersGraph. MPSGraph lives in its own framework, so its
  classes are absent until that framework is opened; the "MPSGraph absent"
  probe result in tenferro-rs#2402 must be re-checked with it loaded. Required
  and optional entries follow the CUDA/HIP optional-symbol contract
  (tenferro-rs#2413). No safe API, no policy.
- **`t4a-metal-runtime`**: a sibling of `t4a-cuda-runtime` and
  `t4a-hip-runtime`, name for name where the semantics match ("no universal
  abstraction first"). One device, one command queue, shared-mode allocation
  with stable allocation identity, command-buffer submission and completion,
  `newLibraryWithSource:` artifacts with a per-runtime module cache, checked
  launches with by-value parameter blocks, and the raw consumer seam.
- **Probe before constructing** (tenferro-rs#2402 §2): `dlopen` →
  `MTLCreateSystemDefaultDevice` (nil means absent) → class presence →
  per-dtype kernel compile → one small real call. A failed probe makes the
  backend *unavailable* with a typed error naming what failed; nothing panics.

## 7. t4a-gpu-kernels: the Metal target

- MSL is not CUDA-source compatible, so Metal kernels are a separate source
  set in t4a-gpu-kernels (for example `kernels/metal/`), shipped as data with
  no vendor feature, like the CUDA/HIP sources
  ([rocm-backend.md §6](./rocm-backend.md)).
- The planning `Target` gains a Metal vendor with its SIMD-group width (32) as
  a planning input, the same way wave size is for HIP.
- The C ABI adds a `t4a_metal_*` set beside `t4a_cuda_*` and `t4a_hip_*`, with
  its own header, generated through the same prefix macro.
- Complex32 has no MSL type: kernels carry it as an f32 pair.
- The fusion emitter gains an MSL backend for the existing planner
  (`segment_exec_program`, `build_elementwise_fusion_plan`,
  `TensorFusion::execute_elementwise_fusion`), carrying over the CPU lessons in
  tenferro-rs#2402 §5: fusion can decline, short regions are not fused, and
  reordering fusion is opt-in.

## 8. tenferro-gpu and operation crates

- `tenferro-gpu/src/metal/` with `MetalContext` and `MetalEngine`, a sibling of
  `src/cuda/` (and the planned `src/rocm/`), on the backend-neutral `gpu/`
  seam. Op-family dispatch tables written against t4a-gpu-kernels and the
  `Target` are shared with the other vendors.
- **dtypes**: f32, Complex32, and the integer/bool families CUDA serves where
  MSL supports them (MSL 64-bit integer and atomic support decides i64
  reductions). Everything else: `Unsupported`.
- **GEMM and contraction**: real f32 GEMM through `MPSMatrixMultiplication`.
  Complex32 GEMM is four real MPS GEMMs (three with Karatsuba once profiling
  asks for it), as decided on tenferro-rs#2402 on 2026-11-03; it is the
  accepted implementation, not a fallback. Integer contractions return
  `Unsupported`. A failing or missing MPS call is a typed provider error,
  never a native-kernel fallback. No handwritten contraction kernels
  (tenferro-rs AGENTS.md).
- **FFT and linalg on Metal** are later units. Until they land, they return
  `Unsupported` on the Metal engine; the paired CPU engine serves them on the
  same shared tensors.
- **Feature wiring**: `metal` is added to `tenferro-gpu` and to each operation
  crate in the pull request that gives it a working Metal path (no name-only
  features, tensor4all-meta#72).
- Coverage is published as a table in tenferro-rs
  `docs/guides/devices-and-gpu.md`, as CUDA's limitations are.

## 9. Validation

| Lane | Where | What it proves |
| --- | --- | --- |
| `--features cuda,rocm,metal` build, clippy, CPU tests | GitHub-hosted Linux, every PR | Additivity; the Metal code compiles out off Apple |
| `macOS workspace tests` with Metal targets, `TENFERRO_REQUIRE_METAL=1` | GitHub-hosted `macos-15` (Apple Silicon), required gate | Probe, runtime, kernels, shared-domain waiting, CPU admission; an unavailable device fails the gate, as it did before #2312 |
| No-link check | macOS lane | `otool -L` shows no Metal, MPS or Accelerate |
| Absent-framework tests | macOS lane, framework path forced to fail | Discovery lists no Metal device and errors name what failed |
| Shared-domain tests | macOS lane | f32 on Metal and f64 on the CPU engine over the same tensors with no transfer counted; CPU read after a GPU write waits; GPU submit against a live host guard returns `MappedForHost` |

Before the CI lane is relied on, run one Metal smoke test on the hosted
`macos-15` runner; if Metal is unavailable there, the required gate needs a
self-hosted Apple Silicon runner instead.

## 10. Milestones

- **U1 — shared domain in tenferro, no Metal yet.** `GpuBackendKind::Metal`;
  the CPU-boundary admission of §4.3 with the wait hook of §5, tested with a
  fake domain provider; `REPOSITORY_RULES.md` update for shared-domain host
  inspection.
- **U2 — `t4a-metal-sys` and `t4a-metal-runtime`** in t4a-gpu-backend, with the
  probe and standalone validation.
- **U3 — Metal kernel sources and target** in t4a-gpu-kernels (after its
  rename, rocm-backend.md M1a).
- **U4 — `MetalContext` / `MetalEngine`** in tenferro-gpu: shared allocation
  provider, upload/download, structural, elementwise and reduction families,
  macOS CI lane.
- **U5 — GEMM and einsum** through MPS (real and four-GEMM Complex32).
- **U6 — fusion emitter for MSL.**
- **Later — FFT and linalg on Metal; AMD/NVIDIA shared-domain providers.**

## 11. Left to the builder

- Exact public names within the storage-ownership contracts' naming rules.
- `MTLSharedEvent` versus completed handlers for completion.
- MSL kernel structure, threadgroup sizing, and the module cache layout.
- The minimum macOS version, set by what the probe and MPS calls need; record
  it in the guide.
- Issue decomposition per unit, following the tenferro-rs#2208 pattern.
