# Adopting pencil-rs, and building it on tenferro

**Status:** Decision record (2026-11-18), revised the same day: the dependency direction is now pencil-rs → tenferro, and section 5 plans how to remove the implementations pencil-rs and tenferro duplicate.
**Scope:** [pencil-rs](https://github.com/tensor4all/pencil-rs), [tenferro-rs](https://github.com/tensor4all/tenferro-rs) (`tenferro-tensor-core`, `tenferro-tensor`, `tenferro-fft`, `tenferro-mpi`, `tenferro-hdf5`), and later distributed work on top of [colmat-rs](https://github.com/tensor4all/colmat-rs).
**Checked against:** pencil-rs `dev` `0c192c7`, tenferro-rs `main` `856db9f` (0.12.0), colmat-rs `main` `d1015b0`, on 2026-11-18.

## 1. Decision

1. **pencil-rs joins tensor4all as it is.** Done on 2026-11-18: the repository moved from `nogaki/pencil-rs` to `tensor4all/pencil-rs` by a GitHub transfer, keeping its name, history and issues. Kosuke Nogaki (GitHub: `nogaki`) stays its maintainer with admin rights.
2. **The dependency direction is pencil-rs → tenferro.** A distributed array is a layer above local tensors, as PyTorch's `DTensor` is above `Tensor` and PencilArrays.jl is above Julia's `Array`. tenferro never depends on pencil-rs.
3. **pencil-rs is tensor4all's distributed layer.** No separate `tenferro-distributed` crate is built inside tenferro-rs. What such a crate would have held (distributed tensors, their AD rules, the layout abstraction) goes into pencil-rs.
4. **The dependency is introduced in stages** (section 6). Today pencil-rs does not depend on tenferro. The first stage adds only `tenferro-tensor-core`; heavier crates follow once their APIs settle and the gates in section 6 pass.
5. **Duplicate implementations are removed by moving them into tenferro, not by keeping two.** Where pencil-rs has a local capability tenferro lacks (R2R and DHT transforms, the FFTW backend, allocation-free execution, HDF5 hyperslabs, MPI collectives on tensors), that code moves into the tenferro crate that owns the topic, and pencil-rs calls it.

An earlier version of this record (#117) set the opposite direction (tenferro → pencil-rs) and deferred a tenferro-side layer. That choice favoured isolation from tenferro's API churn over layering; the layering argument and the duplication it would leave decided the revision.

## 2. What pencil-rs is

A row-major, MPI-distributed multidimensional array library with distributed FFTs, inspired by PencilArrays.jl and PencilFFTs.jl (MIT).

| Crate | Role | Code lines |
| --- | --- | --- |
| `pencil-array` | Pencil layouts, checked Alltoallv and point-to-point transposes, spatial remap, collectives, pointwise operations | 8.7k |
| `pencil-fft` | Local FFTs (RustFFT/RealFFT, MPI-free by default); the `distributed` feature adds distributed C2C, R2C/C2R, R2R (DCT/DST I-IV), DHT and mixed per-axis transforms | 20.6k (about 17k distributed) |
| `pencil-io` | MPI-IO, serial and parallel HDF5 (`hdf5-metno`) | 17.1k |
| `pencil-fftw` | Optional FFTW backend loaded at run time (FFTW is GPL; see its NOTICE) | 1.5k |

Code lines exclude blank lines, comment lines and inline test modules. `src/` holds about 6k more lines of test-only files; production code is about 42k lines, and `tests/` adds 42k. The size comes from the distributed FFT API, not the array container: `pencil-fft` has 391 `pub fn`, 76 of them `_with_timing`/`_with_overlap` variants, and each transform family (C2C, R2C, R2R, mixed, DHT) repeats the same constructor × direction × in-place × timing structure.

Strengths: explicit MPI collective contracts, preflight checks before any payload communication, checked arithmetic on counts and displacements, verification against FFTW reference values generated in Julia, allocation-free tests, and CI checks that keep the dependency layering intact. Gaps: no thread parallelism, SIMD or GPU in local work (the local transpose is a scalar per-element loop), and no published performance numbers. Its main downstream user is irflex-rs (Nogaki's physics solver), which stays outside the org and uses column-major arrays.

## 3. Immediate steps

1. ~~Transfer the repository.~~ Done; admin rights for the maintainer are set.
2. Update the four references to the old URL: [tensor4all/pencil-rs#152](https://github.com/tensor4all/pencil-rs/pull/152).
3. Apply tensor4all-agent-rules and the org CI conventions.
4. The scheduled audit workflow (`.github/workflows/audit.yml`) runs a third-party agent from npm and sends code to DeepSeek models through the OpenCode Go and Command Code relays, on personal API keys. Whether to keep, move or stop it is the maintainer's call; if kept, it should follow the org's AI rules (provenance on filed issues, reviewed providers).
5. **Unify on column-major** (stage 0 in section 6). `pencil-array` is row-major, while `pencil-fft`'s `LocalC2cNdPlan` (added for irflex-rs, pencil-rs#35) is column-major. The git history shows row-major was a premise of the first design spec (2026-09-11) with no stated reason. tenferro, colmat, the Julia packages, PencilArrays.jl and irflex-rs are column-major, and every later stage assumes it.

## 4. Division of responsibility

pencil-rs keeps only what is about distribution:

* pencils, topology and decomposition, the per-axis layout description;
* redistribution: Alltoallv and point-to-point transposes, spatial remap;
* MPI preflight and descriptor agreement;
* the stage composition of distributed FFTs;
* coordination of parallel I/O and the distributed catalog.

Everything local is done once, in tenferro: FFTs, permutation and pack/unpack copies, pointwise operations, local reductions, tensor ↔ HDF5 dataset conversion, and generic MPI transport of tensor payloads. Each rank's local block is a `TypedTensor` / `TypedTensorView`.

## 5. Removing the duplicates

### 5.1 pencil-array

| pencil-rs part | Destination |
| --- | --- |
| `geometry`, `axis`, `checked`, `extra_shape`, the sealed scalar set | `tenferro-tensor-core` (`TensorLayout`, `Rank`/`DynRank`, `ScalarSet`, `col_major_strides`); deleted |
| storage in `PencilArray`, `view`, `many` | `TypedTensor` / views; pencil types become thin wrappers holding the distribution metadata |
| `local_transpose`, pack/unpack copies | tenferro permutation (strided-perm underneath); deleted |
| local parts of `pointwise`, `map_reduce`, reductions | tenferro elementwise operations and reductions; only the MPI reduction step stays |
| `pencil`, `topology`, `decomposition`, transposes, `spatial_remap` | stay: the core of pencil-rs |

### 5.2 Local FFTs and the tenferro-fft extensions

pencil-fft's local C2C, R2C, C2R and `LocalC2cNdPlan` are deleted in favour of tenferro-fft, which already offers `fft`/`ifft`/`rfft`/`irfft` and the `n`-dimensional forms over selected axes, three normalization conventions, CPU (RustFFT) and CUDA (cuFFT) backends, plan caches, and AD rules. pencil-rs's raw positive-sign `backward` is `ifft` with `FftNorm::Forward`.

What tenferro-fft lacks moves into it from pencil-rs, as tenferro-rs pull requests under tenferro's review rules:

1. **R2R (DCT/DST I-IV, eight kinds) and DHT** ([tenferro-rs#2651](https://github.com/tensor4all/tenferro-rs/issues/2651)) as new `FftOperation` variants, with pencil's implementation and its FFTW reference tests. They are linear, with adjoints expressible by other R2R kinds, so AD rules fit the existing pattern. cuFFT has no DCT; the CUDA backend reports them as unsupported until a kernel exists.
2. **An FFTW backend** ([tenferro-rs#2652](https://github.com/tensor4all/tenferro-rs/issues/2652)): `pencil-fftw` becomes an `fftw` feature of tenferro-fft, loaded at run time, with its GPL notice. Every tenferro user gains it.
3. **Allocation-free execution** ([tenferro-rs#2653](https://github.com/tensor4all/tenferro-rs/issues/2653)): `_into` variants writing caller-provided outputs, and a guarantee of no allocation on repeated calls with a warmed cache. pencil-rs tests this today; the same tests become tenferro-fft acceptance tests. The FFT registry entries currently declare no `_into` modes, and the CPU provider gathers non-compact operands into a compact buffer first, which costs a full copy per distributed stage.

### 5.3 Distributed FFTs

The distributed FFT (about 17k lines) is rebuilt rather than ported:

* **One implementation.** The mixed transform, which takes a transform kind per axis, becomes the only engine; C2C, R2C/C2R, R2R and DHT are special cases of it. Each stage is "local tenferro-fft transform over the locally complete axes, then pencil redistribution".
* **One entry point.** Direction, in-place or not, timing and overlap instrumentation, and the plan's source (pencil, array or shape) become fields of an options value instead of separate functions.
* **AD** by registering the distributed transform as a tenferro `ExtensionOp`. Its adjoint is the reverse redistribution plus the adjoints of the local transforms, which tenferro-fft already has.

### 5.4 I/O and MPI

* pencil-io moves from `hdf5-metno` to `hdf5-rt`, which tenferro-hdf5 uses.
* The generic HDF5 parts move to **tenferro-hdf5**: tensor ↔ dataset, dtype and complex handling, the attribute schema, chunking and filters, and **hyperslab reads and writes** ([tenferro-rs#2654](https://github.com/tensor4all/tenferro-rs/issues/2654), a design discussion because tenferro-rs#2611 kept parallel HDF5 and partitioning out of tenferro-hdf5's v1 scope). With hyperslabs there, parallel HDF5 in pencil-io reduces to each rank writing its own range in collective mode.
* pencil-io keeps raw MPI-IO, the coordination of parallel HDF5 and the distributed catalog.
* pencil-rs uses `rsmpi-rt`, as `tenferro-mpi` does. Generic collectives on tensor payloads (alltoallv, allreduce, gather) are added to **tenferro-mpi**, which today offers only `send`, `recv` and `recv_into` ([tenferro-rs#2655](https://github.com/tensor4all/tenferro-rs/issues/2655), a design discussion because tenferro-rs#2612 listed global tensor collectives as out of scope; the proposal limits them to per-rank payloads with explicit counts). pencil-specific descriptor agreement stays in pencil-rs.

### 5.5 Expected size

Rough estimates, to be replaced by measurements as the stages land:

| | Now | After |
| --- | --- | --- |
| `pencil-array` | 8.7k | 5-6k |
| local part of `pencil-fft` | about 5k | about 0 (R2R and DHT move to tenferro-fft) |
| distributed part of `pencil-fft` | about 17k | 5-8k |
| `pencil-fftw` | 1.5k | 0 (moves to tenferro-fft) |
| `pencil-io` | 17.1k | 10-12k |
| **pencil-rs production code** | **about 42k** | **20-26k** |

tenferro grows by an estimated 5-7k lines (R2R, DHT, FFTW, hyperslabs, collectives). These are not duplicates: every tenferro user gets them. The size of the rebuilt distributed FFT is the least certain number until a prototype exists.

## 6. Stages and gates

| Stage | Work | Gate |
| --- | --- | --- |
| 0 | Unify pencil-rs on column-major (0.x breaking release); irflex-rs follows | existing tests and FFTW reference checks pass |
| 1 | Depend on `tenferro-tensor-core`; move to `rsmpi-rt` and `hdf5-rt` | no behaviour change |
| 2 | Extend tenferro-fft: R2R, DHT, FFTW backend, `_into` and allocation-free execution (tenferro-rs#2651, tenferro-rs#2652, tenferro-rs#2653) | pencil-rs's FFTW reference tests and allocation tests pass in tenferro-fft |
| 3 | Local blocks become `TypedTensor`; local FFTs call tenferro-fft; pencil's local FFT code is deleted | all pencil-rs tests pass; local FFT time no worse than today |
| 4 | Rebuild the distributed FFT as one mixed engine with one entry point | benchmarks on campaign-toolkit (`pencil-benchmark`) show no regression against stage 3 |
| 5 | tenferro-mpi collectives (tenferro-rs#2655), tenferro-hdf5 hyperslabs (tenferro-rs#2654), AD rules, GPU | per feature |

Stage 0 comes first because every later stage assumes column-major layouts. Stages 2 and 4 are the large ones and are shared between the pencil-rs maintainer and the tenferro maintainers.

## 7. Risks

* **tenferro API churn.** tenferro is at 0.12. pencil-rs pins tenferro versions and takes updates at planned points.
* **Performance.** tenferro-fft draws memory from pools, while pencil-rs guarantees allocation-free execution. Stage 2 makes that guarantee a tenferro-fft requirement; stages 3 and 4 have benchmark gates.
* **Weight for downstream users.** irflex-rs would pull in `tenferro-cpu` and its dependencies from stage 3. Agree with the maintainer before stage 3; stage 1 (`tenferro-tensor-core` only) is light.
* **Review load.** Stage 2 lands in tenferro-rs and follows its review rules.

## 8. Longer term: layouts beyond pencils, and colmat

The layout abstraction lives in pencil-rs: each axis is either undistributed or mapped to a process-grid axis with a block size (block or block-cyclic). A pencil leaves at least one axis undistributed; a tile layout distributes every axis. This is the idea behind JAX's `PartitionSpec` and PyTorch's `DTensor` placements. Generalising `pencil-array` to it, with general redistribution between layouts (as in COSTA), is the base for distributed dense linear algebra.

The goal there is one stack: colmat inside a node and tensor4all's own distributed layer between nodes, rather than wrapping ScaLAPACK or ELPA permanently. colmat fits because each MPI rank can own its `threadteam` team (`Par::Team`) with no global pool, because it is column-major, and because it is validated against LAPACK's test suites. colmat-rs has no distributed plan today. When the work starts:

* a tensor-free distributed linear algebra crate on the generalised layouts, using colmat for each rank's tiles, kept outside colmat so colmat gains no MPI dependency;
* staged by difficulty: distributed GEMM (SUMMA) first, then Cholesky and QR on block-cyclic layouts, and eigensolvers and SVD last (ELPA-level performance is a research-scale effort);
* ScaLAPACK and ELPA kept as optional backends until then, and as test oracles afterwards, in the way LAPACK serves colmat.

## 9. Constraints to keep

* tenferro never depends on pencil-rs.
* Keep the borrowed external-buffer views (`PencilArrayView::from_slice`, `PencilArrayViewMut::from_slice_mut`) until local blocks are tenferro views.
* Keep `pencil-array` free of FFT dependencies, as its CI already enforces.
* Measure before optimising local work; benchmarks go in a `pencil-benchmark` repository on campaign-toolkit.
