# Adopting pencil-rs, and deferring tenferro-distributed

**Status:** Decision record (2026-11-18).
**Scope:** [pencil-rs](https://github.com/nogaki/pencil-rs) (to move to `tensor4all/pencil-rs`), [tenferro-rs](https://github.com/tensor4all/tenferro-rs), and later distributed work on top of [colmat-rs](https://github.com/tensor4all/colmat-rs).
**Checked against:** pencil-rs `main` `c48decc`, tenferro-rs `main` `856db9f` (0.12.0), colmat-rs `main` `d1015b0`, on 2026-11-18.

## 1. Decision

1. **pencil-rs joins tensor4all as it is.** The repository moves from `nogaki/pencil-rs` to `tensor4all/pencil-rs` by a GitHub transfer, keeping its name, history, issues and crate layout. Kosuke Nogaki (GitHub: `nogaki`) stays its maintainer.
2. **No `tenferro-distributed` crate yet.** A tenferro-side layer for distributed tensors is built only when one of the triggers in section 4 occurs. Until then, nothing in tenferro-rs depends on pencil-rs.

The reason is risk. A general distributed-tensor layer needs an abstraction for layouts (pencil, tile, block-cyclic) and redistribution between them. Designing it now would rest on one user (distributed FFT) and would most likely produce a pencil-specific design under a general name. Moving pencil-rs costs little, gives it the org's infrastructure at once, and keeps every later option open.

## 2. What pencil-rs is

A row-major, MPI-distributed multidimensional array library with distributed FFTs, inspired by PencilArrays.jl and PencilFFTs.jl (MIT). It has four crates:

| Crate | Role |
| --- | --- |
| `pencil-array` | Pencil layouts, checked Alltoallv and point-to-point transposes, collectives, pointwise operations |
| `pencil-fft` | Local FFTs (RustFFT/RealFFT, MPI-free by default); the `distributed` feature adds distributed C2C, R2C/C2R, R2R (DCT/DST I-IV) and DHT |
| `pencil-io` | MPI-IO, serial and parallel HDF5 |
| `pencil-fftw` | Optional FFTW backend loaded at run time (FFTW is GPL; see its NOTICE) |

About 67k lines of source and 42k lines of tests, written between 2026-09-11 and 2026-11-16. Strengths: explicit MPI collective contracts, preflight checks before any payload communication, checked arithmetic on counts and displacements, verification against FFTW reference values generated in Julia, allocation-free tests, and CI checks that keep the dependency layering intact. Its main downstream user is irflex-rs (Nogaki's physics solver), which stays outside the org.

What it does not have yet: thread parallelism, SIMD or GPU in local work (the local transpose is a scalar per-element loop), and published performance numbers.

## 3. What happens now

Low-risk steps, none of which changes behaviour:

1. Transfer the repository to tensor4all.
2. Apply tensor4all-agent-rules and the org CI conventions. Nogaki's AI audit workflow (`.github/audit`) is kept, and is a candidate to share across the org.
3. Align dependencies with the stack where it costs little: MPI through `rsmpi-rt` (CI already exercises it), HDF5 through `hdf5-rt`.
4. **Settle the memory order early.** pencil-rs mixes conventions: `pencil-array` is row-major, while `pencil-fft`'s `LocalC2cNdPlan` (added for irflex-rs, issue #35) is column-major. The git history shows row-major was set as a premise of the first design spec (2026-09-11) with no stated reason. tenferro, colmat, the Julia packages, PencilArrays.jl and irflex-rs are all column-major. The direction to agree on with the maintainer is to unify on column-major in a 0.x breaking release. Implementing it is not urgent, but the decision gets more expensive with every new user.

## 4. When to build tenferro-distributed

Build a tenferro-side distributed layer when at least one of these occurs:

* a tenferro user needs distributed tensors (not only distributed FFT on raw buffers);
* work starts on distributed dense linear algebra (section 5);
* a second distribution scheme besides pencils (tile or block-cyclic) is needed by a real user.

When it is built:

* Name it by function (`tenferro-distributed`), following `tenferro-linalg`, `tenferro-einsum` and `tenferro-fft`, which do not carry backend names. pencil-rs keeps its own name as the tensor-free component underneath.
* Describe a layout as a per-axis assignment: each axis is either undistributed or mapped to a process-grid axis with a block size (block or block-cyclic). A pencil is the case where at least one axis is undistributed; a tile layout distributes every axis. This is the same idea as JAX's `PartitionSpec` and PyTorch's `DTensor` placements.
* Keep the split with `tenferro-mpi`: `tenferro-mpi` is transport (send/receive); `tenferro-distributed` owns distributed data and operations on it.
* Put AD rules on the tenferro side. Redistribution and FFTs are linear, so their adjoints are the reverse redistribution and the backward FFT; pencil-rs needs no AD.
* The dependency direction is tenferro → pencil-rs, never the reverse, so users without tenferro (such as irflex-rs) stay light.

## 5. Longer-term direction: colmat for distributed linear algebra

The goal is one stack: colmat for work inside a node and tensor4all's own distributed layer between nodes, rather than wrapping ScaLAPACK or ELPA permanently. colmat fits this because each MPI rank can own its `threadteam` team (`Par::Team`) with no global pool, because it is column-major, and because it is validated against LAPACK's test suites. colmat-rs has no distributed plan today.

When this starts, the likely shape is:

* the per-axis layout abstraction and general redistribution (as in COSTA), generalised from `pencil-array`, shared by distributed FFT and distributed linear algebra;
* a tensor-free distributed linear algebra crate on top of it, using colmat for each rank's tiles, kept outside colmat so colmat gains no MPI dependency;
* staged by difficulty: distributed GEMM (SUMMA) first, then Cholesky and QR on block-cyclic layouts, and eigensolvers and SVD last (ELPA-level performance is a research-scale effort);
* ScaLAPACK and ELPA kept as optional backends until then, and as test oracles afterwards, in the way LAPACK serves colmat.

## 6. Keep the door open

Things to preserve in pencil-rs so that sections 4 and 5 stay cheap:

* Keep the borrowed external-buffer views (`PencilArrayView::from_slice` and `PencilArrayViewMut::from_slice_mut`) as public API. They let tenferro tensors enter without copies.
* Keep `pencil-array` free of FFT dependencies and the local `pencil-fft` path free of MPI, as its CI already enforces.
* Measure before optimising local work. Candidates when numbers justify them: strided-rs for local transposes and pack/unpack, threadteam (and hataori-rs) for hybrid MPI + threads, tenferro-fft's cuFFT path and t4a-gpu-backend for GPUs. Benchmarks go in a `pencil-benchmark` repository on campaign-toolkit.
