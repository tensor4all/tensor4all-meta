# Symmetric tensors: symtn, tenferro and tensor4all-rs

**Status:** Decision record (2026-11-19).
**Scope:** [symtn-rs](https://github.com/Ryo-wtnb11/symtn-rs) (downstream, outside the org), [tenferro-rs](https://github.com/tensor4all/tenferro-rs) and the crates around it (`cpueinsum`, `tcontract`, `tlinalg`, strided-rs), [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs), and lessons from the [ITensor](https://github.com/ITensor) rewrite.
**Checked against:** symtn-rs `main` `a01b11b` (2026-11-17, pins tenferro 0.11.0); tenferro-rs `main` `d48982d3` (2026-11-19, release 0.12.0); ITensor GradedArrays.jl `eceb349`, BlockSparseArrays.jl `d5ff336`, TensorAlgebra.jl `f25bff9`, ITensorBase.jl `bcc8f21`, ITensorActions `f809c30`. Findings come from reading code; nothing was built or benchmarked.

## 1. Decision

1. **tensor4all-rs does not implement quantum numbers.** No symmetry sectors, graded axes or block-sparse QN tensors. Symmetric tensors are left to symtn or other tensor network libraries. The `t4a-blocksparse` plan was removed from tensor4all-rs's design document in [tensor4all/tensor4all-rs#1402](https://github.com/tensor4all/tensor4all-rs/pull/1402).
2. **symtn owns the representation of symmetry.** Sectors on axes, fusion trees and recoupling, the abelian fast path, and the split of a block into structure and reduced data all stay in symtn.
3. **tensor4all provides symmetry-agnostic high-performance kernels.** Their API speaks only of flat buffers, offsets, strides, shapes, coefficients and plain integer group IDs; no sector type crosses it.
4. **These kernels live outside the tenferro core.** The core already excludes irregular batches ([core boundary](./core-boundary-and-execution-ownership.md), "Irregular batches (different shapes per item, as with symmetry blocks) are out of scope for the core"), and grouped GEMM was removed from the core in [tensor4all/tenferro-rs#2198](https://github.com/tensor4all/tenferro-rs/issues/2198). The natural home is the `cpueinsum`/`tcontract` layer, where `cpueinsum::GroupedPlan` already lives, or a separate crate.
5. **Payloads are dense tenferro tensors, so tenferro's AD applies directly** (section 4).

The reason for a single owner: the ITensor rewrite ended up with three symmetric-tensor implementations (BlockSparseArrays.jl, GradedArrays.jl, FusionTensors.jl), duplicated interfaces to avoid dependencies between them, and finally delegated non-abelian recoupling to TensorKit (section 5). Two implementations inside one ecosystem are not maintainable.

## 2. Division of responsibility

| Layer | Owns | Does not own |
| --- | --- | --- |
| symtn (or another TN library) | Sectors, graded axes, dual flags, fusion trees, recoupling coefficients, fermion signs, the abelian fast path, which block jobs to issue, quantum-dimension weights | Dense kernels, thread and stream control |
| Kernel crates (`cpueinsum`/`tcontract` layer or new crate) | Irregular-batch strided GEMM, batched small-block permute/copy/axpy, batched per-block factorization, weighted truncation across groups, their plan caches, CPU and GPU versions | Any notion of sector or fusion rule |
| tenferro core | Dense tensors, regular batches (`dot_general`), linalg, AD engine, engines and providers | Irregular batches |
| tensor4all-rs | Named-index tensors without quantum numbers, tensor networks, TCI | Symmetric tensors |

## 3. Kernel candidates from symtn

symtn already separates a symmetry-free half from its symmetric compile layer. The symmetry-free half is the source for the kernel crates.

| Candidate | symtn source | What must change |
| --- | --- | --- |
| Irregular-batch GEMM replay: offset/stride jobs, plan-time run partitions, `Rank2Gemm::matmul_rank2_batch` | `symtn-operations/src/fusion_replay/` (the module header says "nothing here consumes fusion rules") | It still imports `symtn_core` types (`BlockStructure`, `SectorId`, `CoupledSectorRegion`); replace the coupled `SectorId` with a generic group label |
| Strided GEMM batch and the geometry-keyed GEMM plan cache | `symtn-dense/src/executor.rs` (`DenseGemmBatchJob`, `strided_batch_runs`), `symtn-dense/src/block_kernel/gemm.rs` (`Geometry`, `GemmPlans`, `gemm_batch_into`), CUDA `symtn-dense/src/cuda_adapter/gemm.rs` | Little; it is already dense-only |
| Batched per-block factorization | `symtn-dense/src/block_kernel/factor.rs` (one `tlinalg` call per block, spread over lanes) | Output contracts (leg roles, compact plans) stay in symtn |
| Weighted truncation across blocks | `symtn-matrixalgebra/src/truncation.rs` (`WeightedSpectrum`, `select_truncation`) | Make the sector key and the tie-break order generic; weights arrive as plain `f64` per group |
| Lane scheduler for independent block jobs | `symtn-dense/src/lanes/` (LPT placement with a cost model) | Decouple from `symtn_core` cache registration |
| Byte-weighted structure cache with single-flight misses | `symtn-core/src/cache.rs` (`StructureCache<K, V>`) | Decouple from the `StructureCacheKind` enum |
| Contraction-order planning over labels and sizes | `symtn-network/src/{ir,cost,labels,optimizer,pathopt,plan,slice,cotengra_python}.rs` | None found; compare with `tenferro-einsum`'s `omeco` path before moving anything |

Not a candidate: symtn has no symmetry-free block-sparse contraction. `symtn-tensors/src/contract/structure.rs` rejects any operand with more than one block ("block-sparse contraction enumeration is not implemented yet"); the only block-sparse contraction is the fusion-tree path. What can move is the execution half, not a block-sparse tensor type.

symtn's `docs/layering.md` still says "Tenferro owns dense tensors, GEMM and grouped GEMM"; after tenferro-rs#2198 that is out of date and should point at the kernel layer chosen in decision 4.

## 4. Dense payloads and AD

If the payload of a symmetric tensor is one flat dense tenferro tensor, and the block structure is static metadata held by the op, tenferro's AD engine needs nothing symmetry-specific:

- tenferro has an extension-op mechanism (`tenferro_ops::ext_op::ExtensionOp`). An extension op can supply AD rules through `tenferro-ad/src/semantic_extension.rs` (`linearize`), or get them for free by lowering to standard ops (`lower_to_standard_ops`).
- The kernels are closed under AD. The VJP of an irregular-batch GEMM is another irregular-batch GEMM with transposed jobs; the VJP of a block permute is the inverse permute. Both are the same extension op with different metadata.
- symtn decides which jobs to build; tenferro differentiates the payload.
- Per-block SVD/QR use tenferro-linalg's existing rules while degeneracies stay inside one block. Which singular values a cross-block truncation keeps is a discrete choice and is treated as non-differentiable, as ITensor marks its bookkeeping zero-derivative (`TensorAlgebraMooncakeExt`).

Not yet tried: the extension-op path was confirmed to exist in the code, not exercised with a batched GEMM.

## 5. Lessons from the ITensor rewrite

### 5.1 Symmetric arrays (symtn's scope)

- **Non-abelian recoupling is delegated to TensorKit.** GradedArrays.jl depends on TensorKit and TensorKitSectors directly. Its permute (`bipermutedimsopadd!`) and fermion twist wrap the shared buffer as a TensorKit `TensorMap` and call TensorKit, without copying (`src/gradedarray.jl`). The explicit fusion-tree package FusionTensors.jl has stalled (last commit 2026-06, compat bounds far behind). symtn reimplements TensorKit and QSpace in Rust instead; both projects nevertheless arrived at the same storage: block-diagonal matrices over coupled sectors in one contiguous buffer, with per-sector offset and size.
- **Symmetry is a property of the axis.** `GradedOneTo` carries sectors, degeneracies and one dual flag; the tensor and the naming layer do not know about symmetry.
- **Abelian sectors get their own type.** `UniqueSectorArray` holds dense data and one sector per axis and never touches fusion-tree machinery. In symtn abelian symmetries go through the fusion-tree path; a dedicated fast path is worth considering.
- **Structure and reduced data are separated.** `KroneckerArrays.jl` represents a block lazily as `structure ⊗ data`; GEMM runs only on the reduced data, and quantum-dimension factors stay outside the GEMM.
- **What not to copy.** BlockSparseArrays.jl multiplies blocks by scanning every stored pair (O(n₁·n₂), `SparseArraysBase/src/abstractsparsearraystyle.jl`), with no threading, batching or GPU kernels. symtn is far ahead on performance.

### 5.2 Interfaces (tenferro and tensor4all-rs)

- **Three layers.** TensorAlgebra.jl does positional contraction, permutation, matricization and factorization; NamedDimsArrays.jl and ITensorBase.jl only attach names to any array (`ITensor = NamedTensor{IndexName}`); ITensorNetworksNext.jl reads only index names and sizes.
- **Small override surface.** Contraction defaults to matricize-then-GEMM; a new storage type joins by overriding about five functions (`matricize`, `unmatricize`, `allocate_output`, `check_input`, `dual`) and a type-keyed `default_algorithm`. In Rust this is a trait with default methods; expect it to be heavier than in Julia.
- **Factorizations go through one matrix-level layer.** SVD, QR and eigendecompositions share one loop: matricize, call MatrixAlgebraKit, reattach axes. Truncation is a strategy value; `svd_trunc` also returns the discarded error. `TruncationDegenerate` keeps degenerate multiplets together.
- **Contraction ordering is pluggable.** Greedy is built in; TreeSA and KaHyPar come from an optional `OMEinsumContractionOrders` extension.
- **AD and GPU care.** Label arithmetic and algorithm selection are marked zero-derivative; `norm`, `dot` and scaling avoid scalar indexing.
- **What went wrong.** `AbstractAlgorithm` is defined separately in TensorAlgebra.jl and ITensorNetworksNext.jl, and BackendSelection.jl is barely used. One algorithm-value type for the whole ecosystem is the lesson.

### 5.3 Running many repositories (T4ARegistry and the org)

| Practice | ITensor implementation |
| --- | --- |
| Shared reusable workflows, called through thin per-repository callers pinned to a moving major tag (`@v2`); a release script snapshots internal refs to a SHA before tagging | `ITensorActions/.scripts/release.sh`, `ITensorActions/README.md` |
| New repositories generated from one template | `ITensorPkgSkeleton.jl/template/` |
| Downstream tests: develop the upstream PR, then run the tests of the dependents listed in `pkgs`; `/integrationtest <pkg>` for fork PRs | `ITensorActions/.github/workflows/IntegrationTest.yml`, `IntegrationTestRequest.yml` |
| Canary packages that test the CI machinery itself | `ITensorTestPackage.jl`, `ITensorTestPackageDependent.jl` |
| Bot-driven registration into the local registry, auto-merged after 10 minutes, with strict +1 version bumps | `register-version` action, `ITensorRegistry/.github/workflows/automerge.yml` |
| Path classification so documentation-only PRs skip heavy jobs | `ITensorActions/.github/actions/classify-pr` |
| Idempotent mass PRs across repositories | `MassApplyPatch.jl` |

Avoid two things: `IntegrationTest.yml` treats a resolver error as success ("Not compatible with this release. No problem."), so an incompatible upstream change passes silently; and template changes are not propagated to existing repositories.

## 6. symtn workarounds for tenferro issues

symtn's source cites eight tenferro-rs issues. All are closed. Every fix except #2587 is in tenferro 0.11.0, the version symtn pins.

| Issue | Problem | Fix | symtn site | Action |
| --- | --- | --- | --- | --- |
| [#1922](https://github.com/tensor4all/tenferro-rs/issues/1922) | Complex division overflows through \|b\|² | GPU #1970; CPU regression tests at 2^±600 in `tenferro-runtime/tests/integration/complex_division.rs` since v0.10.0 | `symtn/src/typed/numeric.rs` (port of Julia's `inv`), `symtn-dense/src/cuda_hermitian.rs` (power-of-two normalizer) | Remove after `symtn/tests/scaled_complex64_reciprocal.rs` passes on tenferro's division (it pins Julia's inf/NaN behaviour). The power-of-two normalizer is exact and may stay on its own merits |
| [#2133](https://github.com/tensor4all/tenferro-rs/issues/2133) | CUDA complex `reduce_sum` failed NVRTC compilation | 0.11 pin move (#2132) | `symtn-dense/src/cuda_adapter/finite.rs` (`abs` before the sum) | Remove |
| [#2134](https://github.com/tensor4all/tenferro-rs/issues/2134) | Full reduction ran on one warp | #2152 (1e7 f64: 38 ms to 0.1 ms) | `finite.rs` cost comment | Update the comment |
| [#1833](https://github.com/tensor4all/tenferro-rs/issues/1833) | CUDA complex zero value failed NVRTC compilation | #1837 | `symtn-operations/src/cuda_transform.rs` (zeros uploaded from the host); `cuda_adapter/factorization.rs` (history note) | Replace the upload with `alloc_zero_output`; the history note can stay |
| [#1834](https://github.com/tensor4all/tenferro-rs/issues/1834) | No fill of an existing device buffer | `Session::fill_zero_write` (#1836) | `symtn-dense/src/cuda_adapter/context.rs` (resident zero template) | Remove; the comment itself says #1834 would remove it |
| [#1836](https://github.com/tensor4all/tenferro-rs/pull/1836) | Offset and strided region copies | Merged for 0.6.0 | `cuda_adapter/elementwise.rs` | Already used; strided sources could remove some packing (optimisation) |
| [#2349](https://github.com/tensor4all/tenferro-rs/issues/2349) | One plan slot rebuilt on every shape change | Closed without a tenferro change: the fix belongs to the caller | `symtn-dense/src/tenferro_adapter.rs` (`descriptor_slot`) | Keep; this is the intended fix |
| [#2587](https://github.com/tensor4all/tenferro-rs/issues/2587) | Extension cache store not covered by cache controls; `take` allocates | #2588, **in 0.12.0 only** | `symtn/src/scratch.rs`, `symtn-network/tests/intermediate_allocations.rs` | Remove after moving to 0.12; revisit `trim` and allocation counts |

## 7. Open questions

1. Where the kernel crates live: inside the `cpueinsum`/`tcontract` repositories, or a new crate. Who maintains them, and whether symtn's code moves or is rewritten against its design.
2. The generic job API: how group labels, weights and tie-break order for truncation are expressed without a sector type.
3. Whether the extension-op AD path (section 4) carries the irregular-batch GEMM without per-call overhead that would hurt many small blocks.
4. The rest of tensor4all-rs's `docs/design/t4a_unified_tensor_backend.md` (the `t4a-*` crate plan) predates the tenferro-based backend and is out of date beyond the block-sparse part.
