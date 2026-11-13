# A next generation of dense linear algebra in colmat: prior work and design direction

Status: Design under review (2026-11). Scope: [colmat](https://github.com/tensor4all/colmat-rs), the dense linear algebra library of the stack. The first target is double-double support with generic fallbacks, a per-scalar cost model, and caller-owned threads and workspace. Review the contracts and resolve the [open questions for the first design](#11-open-questions) here before implementing the redesign. Batched operations are deferred (section 9). See [core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md) for the stack's ownership model.

The design discussion is in [colmat-rs#171](https://github.com/tensor4all/colmat-rs/issues/171). This document records what already exists elsewhere, in papers and in code, so that colmat reuses it and does not reinvent it.

## 1. Why

Hiroshi Shinaoka and Markus Wallerberger (2026-11) discussed a common set of primitives for colmat's linear algebra. The design has these requirements:

1. **Generic primitives with optional specialisations.** LU, QR, EVD and SVD have generic implementations. The boundaries for independently specialised kernels are under review.
2. **A per-scalar cost model.** Parallel width and block sizes come from a cost model that depends on the scalar. Today a non-native scalar borrows the `f64` constants.
3. **Extended precision.** Support [xprec-rs](https://github.com/tuwien-cms/xprec-rs) (`Df64`, double-double).
4. **A caller-owned thread token.** Operations that schedule parallel work borrow the caller's team (`Par`). Sequential leaves run within that schedule.
5. **Caller-owned workspace.** Low-level routines borrow temporary storage from the caller, including storage used by packing and parallel subcalls.
6. **Batched operations.** Many independent small or medium problems in one call. Deferred (Hiroshi, 2026-11-13).

Findings from the survey and the subsequent review:

* **Inconsistent eps for double-double.** colmat's `fx128` (`qd::Quad`), xprec's `Df64` and qd itself define three different values for the same format: `8·Quad::EPSILON`, `f64::EPSILON²/2` and `f64::EPSILON²`.
* **Hard-coded eps in a downstream caller.** sparse-ir-rs selects eps by downcasting through `Any`, with `1e-15` as the fallback for other types (`sparse-ir-basis/src/tsvd.rs` at `SpM-lab/sparse-ir-rs@2ea2f23`).
* **`fx128` has raw and converting SIMD hooks.** The probe in [colmat-rs#173](https://github.com/tensor4all/colmat-rs/issues/173#issuecomment-6836382872) called the raw hooks. The normal path calls the converting hooks. [colmat-rs#177](https://github.com/tensor4all/colmat-rs/pull/177) adds coverage with nonzero low words.
* **The former `fx128` matmul panic outside `cfg(test)` was fixed** in [colmat-rs#177](https://github.com/tensor4all/colmat-rs/pull/177), with an integration test exercising the downstream build path.

## 2. Summary: what exists, what colmat adds

| Requirement | Closest prior work | What colmat takes | What is new in colmat |
| --- | --- | --- | --- |
| Generic scalars over one code base | \<T\>LAPACK (generic C++ templates with an optional optimised backend), BLAS++/LAPACK++, C++ `std::linalg` (P1673) | A generic reference body for every routine, plus optional fast paths | Independently selectable kernel specialisations in Rust. Selection and trait boundaries are open |
| Machine constants of non-IEEE types | MPLAPACK's per-type `Rlamch` (double-double: eps ≈ 2⁻¹⁰⁴, emin = −968); \<T\>LAPACK derives its constants from `numeric_limits` | Per-type precision and safe range | `eps`, `safe_min` and `safe_max` are the initial candidates. Check scaling and precision users before fixing the required set |
| Double-double kernels | XBLAS, QD, Dot2 (Ogita–Rump–Oishi), Ozaki scheme, DD-AVX, sparse-ir-rs `ddgemm` | SoA high/low arrays for SIMD; Dot2-style accumulation; room for an Ozaki-type gemm | AoS storage with conversion at load and at packing, as one contract for every non-native scalar |
| Coarse-grained primitives | BLIS (microkernel plus packing, five-loop structure); LAPACK on top of BLAS | Packing as the place where precision and layout are converted (BLIS mixed precision) | Primitives that take a team and return a cost estimate |
| Thread token | BLIS `rntm_t` per call and `thrinfo_t`/`thrcomm_t` trees; P1673 execution policies; Kokkos serial/team/procedure | The token is passed per call and split per loop | Already in colmat: `Par::Team` is exclusive and borrowed, and regions run at the widest width that pays |
| Workspace | colmat's `StackReq` / `MemStack` interface for factorisations | Query size and alignment before execution and lend disjoint scratch to concurrent work | Extend the contract through packing, reductions and nested kernels |
| Cost model | OpenBLAS m·n·k thresholds, BLIS `sup` thresholds, analytical BLIS blocking, ML thread-count models | Shape-based gates; analytical block sizes | Rates per scalar and per kernel family against the dispatch cost of the token (colmat `cost.rs`) |
| Many threads in factorizations | PLASMA tiles, SLATE, static look-ahead and malleable BLIS (Catalán et al.), CALU/TSQR | Look-ahead, worker sharing, reduction-tree panels | Already planned in colmat (tensor4all/t4a-faer#168): equal-area partitions, SPMD sections, look-ahead |
| Batched operations | Batched BLAS standard (group API, `info` modes), oneMKL group/strided/compact, KokkosBatched compact kernels, LIBXSMM, BLASFEO, MAGMA batched | Group API with fixed-size and strided fast paths; per-problem error report; an interleaved family only for tiny sizes | Batches split with the caller's token (already the stack's rule) |

Not found: a Rust library that runs SIMD across a batch on CPU, a published paper on \<T\>LAPACK itself, and a reference implementation of the 2017 Next-Generation BLAS proposal beyond the batched part.

## 3. Prior proposals for a next-generation BLAS/LAPACK

* **A Proposal for a Next-Generation BLAS** (Riedy, Henry, Demmel, Gates, Li, Tang; SC17 BoF, 2017).
  * A new naming scheme that covers the current BLAS and extends it with mixed and extended precision, reproducibility, batched operations, fixed point and error handling.
  * It says a reference implementation will follow. Only the batched part has one (BBLAS, section 4).
* **BALLISTIC**, LAWN 297 (Demmel, Dongarra, Julie Langou, Julien Langou, Luszczek, Mahoney; 2020).
  * The plan for the next LAPACK and ScaLAPACK: layered software, new lower and higher precisions, mixed precision, refining XBLAS, Batched BLAS, user-controllable autotuning.
  * It names "C++ prototypes of a new templated LAPACK library", which is \<T\>LAPACK.
* **Proposed Consistent Exception Handling for the BLAS and LAPACK** (Demmel, Dongarra, Gates, Henry, Langou, Li, Luszczek, Pereira, Riedy, Rubio-González; arXiv:2207.09281, SC22 Correctness workshop).
  * Consistent NaN and Inf propagation, and new error reporting beyond `INFO`.
  * colmat's upstream reports faer-007, -010, -011, -014, -020 and -039 are defects of exactly this class.
* **C++ API for BLAS and LAPACK**, SLATE Working Note 2 (Gates, Luszczek, Kurzak, Dongarra et al.; 2017, revised 2018).
  * Argues that type-in-name interfaces (`sgemm`, `dgemm`) do not fit templated code, and proposes one generic name per routine. This became BLAS++ and LAPACK++.
  * SLATE Working Note 4 does the same for batched BLAS.
* **C++ `std::linalg`**, P1673 (Hoemmen, Hollman, Trott et al.).
  * A free-function BLAS interface over `mdspan`, with support for user-defined value types.
  * The optional execution policy provides a per-call hook for parallel execution, including hierarchical parallelism.
  * LAPACK-like functionality is excluded.

## 4. Test and reference implementations

Checked on 2026-11-13 with the GitHub API: language, licence and last commit.

| Implementation | What it demonstrates | Language, licence | Last commit | State |
| --- | --- | --- | --- | --- |
| [tlapack/tlapack](https://github.com/tlapack/tlapack) (\<T\>LAPACK) | Generic LAPACK over scalar and layout plugins (mdspan, Eigen, MPFR, `__float128`, half, bfloat16, StarPU); BLAS++ dispatch for native types | C++, BSD-3 | 2026-11-09 | Active, "work in progress"; no paper found |
| [icl-utk-edu/blaspp](https://github.com/icl-utk-edu/blaspp), [lapackpp](https://github.com/icl-utk-edu/lapackpp) | One generic C++ name per routine over CPU and GPU vendor libraries | C++, BSD-3 | 2026-11-12, 2026-11-11 | Active (SLATE) |
| [kokkos/stdBLAS](https://github.com/kokkos/stdBLAS) | Reference implementation of P1673 | C++, Kokkos/NTESS | 2026-07-01 | Maintained |
| [NLAFET/BBLAS](https://github.com/NLAFET/BBLAS) | Reference implementation of the Batched BLAS group API | C, no licence file | 2018-10-27 | Dormant |
| [ReproBLAS](https://bebop.cs.berkeley.edu/reproblas) | Reproducible summation and BLAS (Ahrens, Demmel, Nguyen, ACM TOMS 46(3), 2020) | C | — | Not checked on GitHub |
| [michael-lehn/cxxRefBLAS](https://github.com/michael-lehn/cxxRefBLAS) | Named "reference implementation for the next generation of BLAS" | — (almost empty) | 2014-07-11 | Dormant stub |
| [nakatamaho/mplapack](https://github.com/nakatamaho/mplapack) | All of LAPACK 3.9.1 (FABLE translation) in binary128, FP80, double-double, quad-double, GMP, MPFR; per-type `Rlamch` | C++, BSD-2 + LAPACK | 2026-11-13 | Very active |
| [flame/blis](https://github.com/flame/blis) | Microkernel plus packing; mixed domain and precision at packing; per-call `rntm_t`; `sup` path for skinny shapes | C, BSD-3 | 2026-10-06 | Mature |
| [kokkos/kokkos-kernels](https://github.com/kokkos/kokkos-kernels) (KokkosBatched) | Compact (SIMD-across-the-batch) kernels with serial/team/procedure variants | C++, Apache-2.0 + LLVM exception | 2026-11-12 | Mature |
| [libxsmm/libxsmm](https://github.com/libxsmm/libxsmm) | JIT kernels specialised per small shape, one problem per call | C, BSD-3 | 2026-11-10 | Active |
| [giaf/blasfeo](https://github.com/giaf/blasfeo) | Panel-major native format for small matrices, single-threaded | C/asm, BSD-2 | 2026-11-12 | Active |
| [icl-utk-edu/plasma](https://github.com/icl-utk-edu/plasma), [slate](https://github.com/icl-utk-edu/slate) | Tile algorithms on OpenMP tasks; distributed tiles with look-ahead | C, C++; BSD-3 | 2026-05-11, 2026-08-27 | Low activity; active |
| [icl-utk-edu/magma](https://github.com/icl-utk-edu/magma) | GPU batched factorisations and batched SVD | C/C++/CUDA/HIP, BSD-3 | 2026-10-28 | Active |
| [JuliaLinearAlgebra/GenericLinearAlgebra.jl](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl) | Generic SVD, eigensolvers, blocked Cholesky/QR for `BigFloat` etc. | Julia, MIT | 2026-10-26 | A "playground" (README) |
| [t-hishinuma/DD-AVX_v3](https://github.com/t-hishinuma/DD-AVX_v3) | AVX/AVX-512 double-double level-1 BLAS and SpMV with separate `hi`/`lo` arrays | C++, Apache-2.0 | 2021-06-18 | Dormant |
| [enp1s0/ozIMMU](https://github.com/enp1s0/ozIMMU) | DGEMM on integer matrix units by the Ozaki scheme | CUDA, MIT | 2025-12-02 | Research code |
| [sarah-quinones/faer-rs](https://github.com/sarah-quinones/faer-rs) (upstream faer, JOSS 2026) | Decompositions built on matmul; generic `ComplexField` with SIMD hooks; `fx128` | Rust, MIT | 2026-10-22 | Active |
| sparse-ir-rs `sparse-ir-basis/src/ddgemm.rs` ([SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs)) | Dot2-style `Df64` gemm on separate high/low arrays, split over columns, independent of thread count; `AcaScalar` overrides `matmul` and an operation cost per scalar | Rust | `2ea2f23` | In production in sparse-ir |

The closest in spirit is \<T\>LAPACK: one generic code base whose routines a type can redirect to an optimised backend. MPLAPACK is the closest for double-double, and KokkosBatched for batched kernels under a team abstraction.

## 5. The scalar contract and machine constants

### Prior work

* **\<T\>LAPACK** (`include/tlapack/base/constants.hpp`) derives every constant from `std::numeric_limits`:
  * `ulp()` is `epsilon`;
  * `safe_min()` is `radix^max(emin−1, 1−emax)`, and `safe_max()` is its reciprocal;
  * Blue's scaling constants come from radix, digits and the exponent range.

  A custom type must therefore specialise `numeric_limits`.
* **MPLAPACK** declares double-double's constants by hand (`include/mplapack_arithmetic_params_dd.h`):
  * eps is fixed at `0x1.fffffffffffffp-105` (about 2⁻¹⁰⁴), deliberately not QD's `_eps`;
  * emin is −968, not −1021, because the low word must stay normal;
  * the header warns that QD's `numeric_limits<dd_real>` is incomplete and that `min_exponent` may return 0.
* **Proven double-word error bounds**, with u = 2⁻⁵³ (Joldes, Muller, Popescu, ACM TOMS 44(2), 2017; Muller and Rideau 2022, Coq-verified):

  | Operation | Relative error bound |
  | --- | --- |
  | `DWPlusFP` | ≤ 2u² |
  | accurate `DW + DW` | ≤ 3u²/(1−4u) |
  | `DW × DW` (FMA) | < 5u² |
  | sloppy `DW + DW` (QD's default) | can have order-one relative error under cancellation |

### Current state in the stack

| Source | eps of double-double | Smallest value |
| --- | --- | --- |
| qd 0.8.0 `Quad::EPSILON` | `f64::EPSILON²` ≈ 4.9e-32 | `Quad::MIN_POSITIVE` |
| xprec 0.2.3 `Df64::EPSILON` | `f64::EPSILON²/2` ≈ 2.5e-32 | `f64::MIN_POSITIVE / f64::EPSILON` (both words normal) |
| colmat `fx128::epsilon_impl` | `8·Quad::EPSILON` ≈ 3.9e-31 | `max_positive = 1/min_positive` (not the true maximum) |
| sparse-ir-rs `get_epsilon_for_svd` | downcasts through `Any`; `1e-15` for other types | — |

colmat's production code already queries eps generically (about 56 call sites of `eps::<T>()` and 54 of `min_positive`). What is hard-coded is elsewhere:

* `nbits = 100` for `fx128`, which feeds `max(30, nbits/2)` iteration caps;
* `maxit = 30·n` in QZ;
* literal tolerance multipliers such as `128·eps`.

### Direction

* **Keep a minimal precision contract.** The working set is `eps`, `safe_min` and `safe_max`. Define `eps` as the type's precision parameter, safe bounds in terms of the operations they protect, and algorithm tolerances in the routine's parameters. Verify whether all existing users of `nbits` and scaled norms can derive their inputs from those three values before replacing the six `RealField` constants.
* **Specify arithmetic error per operation.** `Df64::EPSILON = 2u²`. The [xprec README](https://github.com/tuwien-cms/xprec-rs/blob/mainline/README.md#L10-L36) gives rounded figures of 3u² for addition and 4u² for multiplication, and reports 6u² observed for division against a bound of 10u². The [addition implementation](https://github.com/tuwien-cms/xprec-rs/blob/mainline/src/arith.rs#L258-L267) documents the higher-order term. Raising `eps` to the multiplication bound still leaves sloppy addition with order-one relative error under cancellation. An optimized kernel must document its accuracy and valid range against the generic operation it replaces.

## 6. Memory layout: AoS storage, SoA at load and at packing

Prior work:

* **BLIS** converts precision during packing, with no extra microkernels (Van Zee, Parikh, van de Geijn, ACM TOMS 47(2), 2021).
* **DD-AVX** keeps separate `hi` and `lo` arrays for AVX.
* **sparse-ir-rs [`ddgemm`](https://github.com/SpM-lab/sparse-ir-rs/blob/2ea2f23e10cdf1a936e969ac76615fd699953d61/sparse-ir-basis/src/ddgemm.rs#L27-L75)** splits the entire left operand into high and low arrays, reads the right operand by column, and allocates its output and column slice table. Its accumulation is a useful reference for colmat's packed kernel.
* **GPU double-double BLAS** stores interleaved `double2` (Mukunoki and Takahashi, PARA 2010).

Storage agreed in [colmat-rs#171](https://github.com/tensor4all/colmat-rs/issues/171#issuecomment-6848850641). The following load, store and packing contracts are for the redesign.

* **Storage.** Each stored matrix element is one `T`, so a double-double matrix is AoS. Views describe the elements through strides. A raw word access path needs an explicit layout guarantee from the scalar type. A transparent wrapper around `xprec::Df64` has the same layout as its inner `Compensated`, whose Rust representation leaves field order unspecified. The initial design can load through `hi()` and `lo()` and store through a constructor with the normalization precondition satisfied.
* **SIMD hooks.** The converting interface reads and writes scalar elements, using slices or element pointers with element-based masks. It deinterleaves on load and interleaves on store with fixed-size temporary storage. Only active elements may be accessed. The current raw hooks and their register-pointer interface are backend details.
* **Packing.** A packed kernel can write high and low words of its panels into caller-provided workspace. The packed representation belongs to the selected kernel. Pack only the operands and tiles it needs.
* **Lane order.** AArch64 has `vld2q_f64` for interleaved memory. The AVX2 two-register unpack can produce physical lane order `(0, 2, 1, 3)`. Values, masks and index vectors must agree on the mapping. A pivot search returns logical indices and applies its tie rule in that order. A numerical reduction also specifies whether reassociation is allowed.

### Caller-owned workspace

Use the existing [`StackReq` / `MemStack` convention](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/mod.rs#L6-L28) as the starting point. The low-level query returns the temporary size and alignment before execution. Query and execution use the same kernel selection, including any choice that depends on shape, layout, CPU capability or team width. A query may conservatively cover multiple paths. Execution borrows the destination and scratch, and borrows the caller's `Par` for work that can split the team.

* **Ownership covers the full call tree.** Packing panels, partial reductions, job tables and factorisation temporaries use the supplied scratch. The caller or an explicit convenience layer may own and grow the buffer. Low-level execution performs no heap allocation or implicit thread-local allocation, growth or fallback. Fixed-size leaf temporaries can live in registers or on the stack. The guarantee applies to colmat and the kernel paths it calls. If a scalar's arithmetic itself allocates, that capability must be documented separately before claiming zero allocations for that scalar.
* **Capacity and lifetime.** Validate size and alignment before changing outputs. A valid exact-size buffer must suffice. Return a capacity error if the selected path requires more storage than the caller supplied. Reuse space for sequential phases and give simultaneous workers disjoint scratch. Include alignment padding, nested calls and per-worker slices in the query.
* **Current gaps.** [`matmul` has no scratch parameter](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/matmul/mod.rs#L1986-L2000). Its [parallel matvec reduction](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/matmul/mod.rs#L886-L932) and the [x86 GEMM job tables](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/utils/team.rs#L563-L596) use thread-local storage that can allocate. [Householder factor formation](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/householder.rs#L284-L303) can allocate a vector of partials. Include the packing and scratch behaviour of dependency kernels when selecting a backend for this contract.

## 7. Primitives and the choice of algorithm

Prior work:

* **BLIS** builds nearly all level-2 and level-3 BLAS from a few microkernels plus packing (Van Zee and van de Geijn, ACM TOMS 41(3), 2015).
* **\<T\>LAPACK** has a generic body for every routine and a trait-gated redirect to BLAS++ for native types.
* **faer** builds its decompositions on matmul (JOSS 2026).
* **The Ozaki scheme** emulates high-precision gemm with native gemms:
  * the original (Ozaki, Ogita, Oishi, Rump, Numer. Algorithms 59, 2012);
  * binary128 gemm from DGEMMs, faster than MPLAPACK's DD and binary128 gemm on x86 (Mukunoki et al., ICPP 2021);
  * Ozaki scheme II with the Chinese Remainder Theorem (arXiv:2504.08009).

### Prior work on fused level-2 operations

colmat's fused kernels are one pass over a matrix that applies a low-rank update and computes the products or reductions that the next step needs. This class has been standardised and implemented before:

* **The BLAS Technical Forum Standard** (2001), chapter 2, added operations that bundle two calls into one pass over the data:
  * `GEMVER`: a rank-2 update `A ← A + u1·v1ᵀ + u2·v2ᵀ` followed by two matrix-vector products. In `GEMVER` the second product uses the first result; in colmat's Hessenberg and bidiagonal kernels the two products are independent.
  * `GEMVT`, `TRMVT`, `AXPY_DOT` and `GE_SUM_MV`.

  The Forum's netlib page lists reference implementations only for the legacy BLAS, the sparse BLAS and the extended and mixed precision BLAS (XBLAS). The chapter-2 extensions are still marked "in preparation" there.
* **"BLAS 2.5"** (Howell, Demmel, Fulton, Hammarling, Marmol, ACM TOMS 34(3), 2008). Bidiagonalisation with the two GEMVs done in one pass of the trailing matrix through cache. It roughly halves the main-memory traffic and cuts the run time by up to 25 %.
* **libflame's fused reductions** (Van Zee, van de Geijn, Quintana-Ortí, Elizondo, ACM TOMS 39(1), 2012). The paper extends the same fusion to the Hessenberg and tridiagonal reductions, with a model that estimates the speed-up of fusing.
  * The implementations are `FLA_Bidiag_UT_u_fus_var2–4`, `FLA_Hess_UT_fus_var2–4` and `FLA_Tridiag_UT_l_fus_var2–3`.
  * They are built on nine fused vector kernels in `src/base/flamec/blis/fused/`: `dotaxpy`, `dotsv2`, `dotsv3`, `axpyv2b`, `axpyv3b`, `axpyv2bdotaxpy`, `dotv2axpyv2b`, `axmyv2` and `dotaxmyv2`.
  * libflame has had no commit since 2024-05.
* **BLIS level-1f kernels** (`axpy2v`, `dotaxpyv`, `axpyf`, `dotxf`, `dotxaxpyf`; [KernelsHowTo](https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md)).
  * The level-2 operations are built from them. For example, `hemv` uses `dotxaxpyf`, which reads a block of columns once and produces both `Aᵀy` and `Ax`.
  * A kernel works on a block of `nf` columns (the fusing factor) and is replaced per architecture; the loop around it stays generic.
* **Fused Givens rotations** (Van Zee, van de Geijn, Quintana-Ortí, ACM TOMS 40(3), 2014). Applying many sets of rotations at once, vectorised, makes the tridiagonal and bidiagonal QR algorithms competitive with divide and conquer and MRRR.
* **Delayed column-norm updates** in column-pivoted QR (Quintana-Ortí, Sun, Bischof, SISC 19(5), 1998; LAPACK `xLAQPS`).
* **Generated fusion.**
  * Build to Order BLAS (Belter, Jessup, Karlin, Siek, SC09; arXiv:1205.1098) compiles MATLAB-like sequences such as GEMVER, BiCGK and AXPYDOT into fused C.
  * Filipovič et al. (J. Supercomputing 71, 2015) fuse BLAS-1/2 map and reduce kernels on GPUs.

Not found: a standard primitive that fuses a low-rank update with a pivot search, as colmat's LU and Bunch–Kaufman kernels do. That part is colmat's own.

### Primitive kinds (decided 2026-11-13)

Before any API, the primitives are fixed as kinds: mathematical operations with a data-access pattern. The list comes from what colmat's algorithms call at `colmat-rs@4abe04d`.

| # | Kind | Operation | Used in colmat | BLAS/LAPACK counterpart |
| --- | --- | --- | --- | --- |
| [P1](#p1-level-3-products) | Level-3 products | `C ← αAB + βC`, including a triangular-output product (a generalised herk/syrk/her2k) and products with a triangular factor (trmm) | Almost every blocked factorisation (gemm 76 call sites, triangular product 52) | gemm, gemmt, herk, her2k, trmm |
| [P2](#p2-level-3-triangular-solve) | Level-3 triangular solve | `X ← T⁻¹B`, many right-hand sides | LU, Cholesky, solves (97 call sites) | trsm |
| [P3](#p3-block-reflector-apply) | Block-reflector apply | `(I − V·T·Vᴴ)` from the left or the right, including forming `T` and applying a sequence of blocks | QR, Hessenberg, tridiagonal and bidiagonal reductions, back-transformations (138 call sites) | larfb, larft |
| [P4](#p4-fused-sweep) | Fused sweep | One pass that applies a rank-0, -1 or -2 update to a full matrix or a triangle and computes any of: `A·x`, `yᴴ·A`, a symmetric product from a triangle, column-norm downdates, a per-column absolute maximum, a 2-D absolute maximum | LU leaf, full-pivoting LU, column-pivoted QR, Bunch–Kaufman, Hessenberg, tridiagonal, bidiagonal | ger/syr2 + gemv/symv; BLAS 2.5 (GEMVER, GEMVT); BLIS level-1f |
| [P5](#p5-level-2-triangular) | Level-2 triangular | `x ← T⁻¹x`, `x ← Tx`, one right-hand side; a recurrence, unlike P4 | Single right-hand-side solves, triangular inverse | trsv, trmv |
| [P6](#p6-sequences-of-small-transforms) | Sequences of small transforms | Rotations (Givens, Jacobi) on pairs of rows or columns, applied singly or as sequences; bulge chasing with 2- or 3-element reflectors | Jacobi, real and complex Schur (`move_bulge`), QZ (`chase_bulge_2x2`), QR iterations, LLT/LDLT rank updates | rot, lasr, the laqr5 family |
| [P7](#p7-permutations) | Permutations | Row and column swaps, applying a permutation | LU, Bunch–Kaufman, column pivoting | laswp, lapmt |
| [P8](#p8-level-1-and-reductions) | Level 1 and reductions | dot, sum, norms with scaling, absolute maximum with its index, scale, axpy, copy, conjugate, layout-changing copies | Everywhere (`reductions/*`) | dot, nrm2, iamax, scal, axpy, copy |
| [P9](#p9-scalar-numerical-kernels) | Scalar numerical kernels | Householder and rotation generation; 2×2 and 3×3 problems (shifts, 2×2 eigenvalues and singular values, shifted 2×2 solves); secular-equation roots; dqds steps; safe reciprocal and hypot | `make_householder`, `secular_root`, `solve_shifted_2x2_cplx`, `svd_triu_2x2`, `geigval_2x2`, dqds | larfg, lartg, lanv2, lasv2, laed4, lasq* |
| [P10](#p10-leaf-factorisations) | Leaf factorisations (optional) | Unblocked factorisation of a small block | LU leaf, `simd_cholesky`, QR panels | getf2, potf2, geqr2 |

What each kind does, in pictures. **Click a figure (or a kind in the table) for its precise definition in [Appendix A](#appendix-a-precise-definitions-of-the-primitive-kinds).** Purple is read only, green is written, amber is computed in the same pass, and grey is a small or scalar operand. The figures are generated by `figures/make_primitive_figures.py` (standard library only; rerun it after editing).

[![P1: gemm, triangular-output products and trmm](figures/prim_p1_level3_products.svg)](#p1-level-3-products)

[![P2: trsm, the solution overwrites B](figures/prim_p2_trsm.svg)](#p2-level-3-triangular-solve)

[![P3: block-reflector apply as three products](figures/prim_p3_block_reflector.svg)](#p3-block-reflector-apply)

[![P4: one sweep over A with a low-rank update and by-products](figures/prim_p4_fused_sweep.svg)](#p4-fused-sweep)

[![P5: trsv as a recurrence](figures/prim_p5_trsv.svg)](#p5-level-2-triangular)

[![P6: one rotation, a sequence of rotations, and bulge chasing](figures/prim_p6_small_transforms.svg)](#p6-sequences-of-small-transforms)

[![P7: row swaps in the order of the pivots](figures/prim_p7_permutations.svg)](#p7-permutations)

[![P8: dot, axpy, norms with scaling, and the first maximum](figures/prim_p8_level1.svg)](#p8-level-1-and-reductions)

[![P9: the scalar kernels that read the Precision contract](figures/prim_p9_scalar_kernels.svg)](#p9-scalar-numerical-kernels)

[![P10: an unblocked leaf composed from P4, P7, P8 and P9](figures/prim_p10_leaf.svg)](#p10-leaf-factorisations)

Each kind describes the mathematical operation and data it touches. The same kind can use different kernels by shape or layout. P1 includes both large GEMM and thin products with different data reuse. P4 has fused sweeps whose data traffic matters. P8 includes dot, which can benefit from a different accumulation algorithm. P9 supplies scaling-sensitive scalar kernels. These distinctions motivate the implementation inventory in §7a.

Still open (recommendation in brackets):

1. Whether P5 is a special case of P2 [separate: a recurrence, and memory-bound].
2. Whether P3 is a composition of P1 [separate: the most call sites, and colmat's own fused splits].
3. Whether the LLT/LDLT rank updates belong to P6 or P4 [P6, as a column recurrence; the kernels have not been read in detail].
4. Whether P10 is a primitive or a composition.

### Direction (colmat-rs#171)

* **Keep the generic implementations.** Factorisations can compose the kinds above, including fused sweeps and team splits, for a scalar that supplies only arithmetic and precision.
* **Expose estimates where the caller decides team width.** An estimate describes the selected path's serial work and sections. Register-level operations and sequential leaf kernels can run within a region that already has a team and an estimate.
* **Let a selected kernel change the accumulation algorithm.** Packed Dot2 and an Ozaki-type split have different preparation, scratch requirements and error behaviour. Compare each against the generic operation's contract.

## 7a. Trait boundaries and API sketch (from colmat-rs#171)

The [original issue](https://github.com/tensor4all/colmat-rs/issues/171) sketches a single `Kernels<T>` trait for the coarse operations. [Markus Wallerberger's review](https://github.com/tensor4all/colmat-rs/issues/171#issuecomment-6848850641) proposes independent hooks based on the prologues and dispatch paths already in BLAS/faer. He uses dot as an example: a general strided implementation with optional contiguous and alignment-specific paths. The public signatures, selection mechanism and crate boundaries are open for review.

### Existing dispatch paths

The following inventory is from `colmat-rs@71a0374`. Use these paths and representative downstream call sites to choose public hooks.

| Operation | What the implementation selects | Question for the design |
| --- | --- | --- |
| Dot (P8) | [`inner_prod`](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/matmul/mod.rs#L601-L684) uses SIMD for two contiguous inputs when the type supports it, with a scalar strided fallback. The SIMD path selects conjugation and dispatches the ISA. | Is a contiguous dot hook sufficient? Keep alignment handling internal unless a caller can supply a useful guarantee and measurements justify exposing it. |
| Matrix product (P1) | [`matmul_imp`](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/matmul/mod.rs#L1240-L1658) normalises strides, recognises vector and rank-one cases, and selects native or generic kernels. | Which hooks benefit from a different accumulation algorithm? Can the generic driver retain skinny and strided fallbacks when an adapter supplies only a packed GEMM? |
| Triangular solve (P2/P5) | The [lower-triangular driver](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/triangular_solve.rs#L734-L851) has small base cases and recursive GEMM updates. | Can it get the desired speedup through the GEMM hook? Measure the base case separately before exposing a solve hook. |
| Block reflector (P3) | [Factor formation](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/householder.rs#L227-L312) uses triangular multiplication, dot or parallel partial products by shape. [Application](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/householder.rs#L502-L643) has a narrow SIMD path. | Which gains come through dot and GEMM, and which require keeping the reflector work fused? |
| Fused sweep (P4) | [Full-pivot LU](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/lu/full_pivoting/factor.rs#L316-L348) fuses the rank-one update and pivot search for eligible SIMD layouts. [Bidiagonalisation](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer/src/linalg/svd/bidiag.rs#L684-L729) selects between a fused SIMD sweep and separate updates and product calls. | Check generic SIMD first for these sweeps. Preserve the one-pass path and specify the by-products and pivot rules before considering a separate hook. |

### Candidate boundary

Use dot and GEMM as contrasting cases for the review. A provider should be able to specialise either independently. Colmat owns operand validation, generic fallback selection and the team schedule. A hook's contract states the input views and valid layouts, destination update and aliasing rules, numerical behaviour, and workspace needs. The driver should select CPU features at a boundary that can be shared by repeated leaf calls. The driver lends scratch and, where the hook itself can partition work, the caller's team. Packed panels and ISA-specific pointers can stay within kernel implementations.

Use a generic path for each missing hook. A provider with a contiguous-dot implementation can leave strided dot, GEMM and factorisations to the generic paths. A provider with a packed GEMM can leave dot generic and still accelerate the GEMM updates inside the generic triangular solve. The query and execution must agree on whether that nested GEMM uses the packed path.

| Selection candidate | What it gives the caller | What to resolve |
| --- | --- | --- |
| Associated provider on the scalar or a newtype | Existing type-driven calls and operators can pick the same provider. | Establish how providers opt in per kernel on stable Rust, including a scalar-only default and the path for an external scalar. |
| Explicit provider supplied to the operation | The caller can select per invocation and use a provider type defined in an adapter crate. | Pass the provider through nested calls and define the default selection for operators without a provider argument. |

**Native non-regression is a requirement for either selection mechanism.** Preserve the existing default fast paths for `f64`, `c64`, `c32` and `f32`, including layout and shape special cases, ISA dispatch, and team-width decisions. Keep kernel selection out of the element and microkernel loops. Show that routing through the new boundary preserves numerical behaviour and performance against the current colmat implementation for each of the four types.

### Candidate crate layout and dependency direction (from the issue)

```
threadteam    colmat-traits (ComplexField)    gemm (fork), pulp
      \              |                          /
       colmat-core:  Mat/Col/Row/Diag/Perm, bound, zip, layout copies,
                     Par/ParShape, team skeletons, CostModel/HostClass/Rates,
                     generic operations and kernel selection,
                     operator impls (Mul -> matrix product)
          /                         \
   colmat (factorisations,        colmat-xprec (Df64 support + Df64 kernels,
     sparse, solvers, prelude)      e.g. Ozaki-type gemm on f64 GEMMs)
          \                         /
            users (tensor4all-rs, colmat-ffi, ...)
```

If the split is adopted, the dependencies would follow the issue:

* `xprec` stays a leaf scalar library. It does not depend on colmat.
* `colmat-xprec` depends on `xprec` and the crate that owns the hook contracts. Keep the adapter independent of the factorisation crate if those contracts can live in `colmat-core`.
* `colmat-xprec` gets `Par` and the team skeletons only through `colmat-core`'s re-exports. CI can deny a direct `rayon` or `threadteam` dependency, and `std::thread`, there.

What the issue checked about moving `mat/` into core:

* `mat/`, `col/`, `row/`, `diag/` and `perm/` depend on `linalg` only through `zip` and the layout copies.
* The `linalg` references in `utils/team.rs` and `utils/cost.rs` are test-only.
* The 208 operator impls of `linalg/mat_ops.rs` must move with `Mat` because of the orphan rule.
* The inherent decomposition methods become an extension trait. That is the main breaking change.

### What a new scalar type must implement (decided 2026-11-13)

**A new scalar type should get P1–P10 through the generic implementations.** Its required implementation is scalar arithmetic and precision. SIMD, kernel specialisations and cost overrides are optional (Hiroshi, 2026-11-13).

**Today** (`colmat-traits` at `colmat-rs@4abe04d`):

| Trait | Required items (no default) | Of which |
| --- | --- | --- |
| `ComplexField` | 56 functions, 9 associated types, 2 constants | 18 scalar functions and 38 SIMD functions, including `ctx_from_simd` |
| `RealField` | 6 functions | the precision constants (`epsilon_impl`, `nbits_impl`, `min_positive_impl`, ...) |

Two facts about this:

* **A type without SIMD still has to provide the required SIMD methods.** `SIMD_CAPABILITIES` (`None`, `Copy`, `Simd`) controls dispatch.
* **A generic complex implementation is built on `RealField`.** Preserve this path when testing scalar-only support.

| Layer | Contents | Required? | If absent |
| --- | --- | --- | --- |
| S0 arithmetic | scalar functions, operators and comparisons | **required** | — |
| S0 precision | precision and safe range, with the exact constants under review (section 5) | **required** | — |
| S1 SIMD | arithmetic and data movement on vector lanes | optional | generic scalar paths |
| S2 kernels | independently selected specialisations for individual kernels | optional | generic implementation for each missing hook |
| S3 cost | per-scalar rates | optional | a ratio to `f64`, or a one-time calibration |

### SIMD arithmetic and data movement

This classifies the required functions in the [existing trait](https://github.com/tensor4all/colmat-rs/blob/71a03747fa8e86ee0318f2e31157703f0e2e07aa/faer-traits/src/lib.rs#L1257-L1514). The counts describe the source inventory.

| Responsibility | Required today | Operations |
| --- | ---: | --- |
| Arithmetic and numerical reductions | 21 | `add`, `sub`, `neg`, `conj`, `abs1`, `abs_max`, `mul_real`, `mul_pow2`, `mul`, `conj_mul`, `mul_add`, `conj_mul_add`, `abs2`, `abs2_add`. Five value comparisons. `reduce_sum` and `reduce_max`. |
| Data movement, masks and indices | 17 | `simd_ctx` and `ctx_from_simd`. Two range-mask constructors, raw load and store, two splats, value select, index select/splat/add/less-than, three mask logic operations, and `first_true_mask`. The default load/store wrappers and `iota` also belong here. |

The data-movement layer can share implementations where the scalar or adapter supplies a valid layout. The arithmetic layer defines numeric comparisons and reductions over whole elements. A common reduction may combine lanes using those arithmetic operations with a stated reduction order. For indexed maxima, the comparison operates on scalar values, the mask carries the chosen indices, and the reduction specifies tie and NaN behaviour. A shared layout implementation must preserve the logical correspondence among all three.

### Attaching `Df64`

An adapter crate can implement a kernel trait parameterised by `xprec::Df64` for its own provider type. That satisfies the orphan rule for the kernel implementation. The generic scalar implementation still needs a legal home, such as a transparent newtype in the adapter or a `Df64` implementation in the crate that defines the scalar trait. Compare the user-facing type, feature and dependency costs of those options. Markus also suggested exposing SIMD operations over slices or strided slices directly from xprec, which a colmat provider could call through an adapter.

## 8. Threading: the token and the cost model

### Prior work

* **BLIS** (Smith et al., IPDPS 2014) chooses the loop to parallelise from the cache-sharing topology:
  * JC when L3 is private;
  * IC when L2 is private;
  * JR when L2 is shared;
  * the k loop (PC) is never split.

  In the code, a tree of `thrinfo_t` nodes with `thrcomm_t` communicators carries the team per loop level, and the expert API takes a per-call `rntm_t` instead of global state.
* **P1673** takes an execution policy per call.
* **KokkosBatched** offers each kernel as serial, team and procedure (whole execution space).
* **Width heuristics in use:**
  * OpenBLAS runs one thread while m·n·k ≤ 65536·4 (real types; 8192·4 for complex) and min(ncpu, m·n·k/(65536·4)) threads above that;
  * BLIS sends a gemm to `sup` when any of m, n, k is below a per-architecture threshold (201 on Haswell, double);
  * an ML model of runtime per thread count (Xia and Barca, arXiv:2406.19621) reports 1.5–3× over always using all threads.
* **Factorizations:**
  * PLASMA turns them into tile DAGs (Buttari et al., Parallel Computing 35, 2009; PLASMA 3 on OpenMP tasks, ACM TOMS 45(2), 2019);
  * static look-ahead on teams (Catalán et al., Parallel Computing 76, 2018);
  * malleable BLIS that lets panel threads join a running gemm (IEEE Access 7, 2019);
  * TSQR, CAQR and CALU turn the serial panel into a reduction (SISC 34, 2012; SIMAX 32, 2011).

### colmat today

* **The token.** `Par<'team>` is `Seq` or `Team(threadteam::TeamExec)`. It is exclusive, not `Clone`, lent with `reborrow`, and split into sub-teams. This is the BLIS team tree in token form.
* **The rule.** A region runs at the widest width at which `serial_estimate ≥ PAYS · sections · dispatch_cost`. The serial estimate is units of work times a per-family rate per host class (`faer/src/utils/cost.rs`), and the dispatch cost comes from the token.
* **The gap.** `Rate::of<T>` only distinguishes the native types, so every other scalar borrows the `f64` or `c64` rate.

### Direction

* **Rates per (host class × scalar × kernel family).** Native rows are as measured today. Non-native rows are measured, with a ratio to `f64` as the fallback.
* **Memory-bound and compute-bound families get separate ratios.**
  * Double-double level-1 work is about 2–3× `f64` because it is memory-bound (Mukunoki and Takahashi).
  * Double-double gemm is compute-bound. sparse-ir-rs weighs it at 12–16 `f64` operations per term (`sparse-ir-basis/src/cost.rs`).
* **This extends the stack's work unit.** [core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md) ("Work units and rates") counts one real multiply-add as 1 and one complex multiply-add as 4. A per-scalar weight extends the same unit.
* **Block sizes follow the same model**, not only the parallel gate. The level-2/level-3 crossover moves for a compute-bound scalar.
* **No primitive starts threads.** A scalar crate gets `Par` and the team skeletons only through colmat's re-exports, and CI can deny a direct `rayon` or `threadteam` dependency.

## 9. Batched operations (deferred)

Deferred on 2026-11-13: extended precision and the redefinition of the primitives come first. The survey below is kept so that the batched API can start from it later.

### Prior work

* **The Batched BLAS standard.**
  * Dongarra et al., Procedia CS 108, 2017; the API proposal, MIMS EPrint 2016.25; Abdelfattah et al., ACM TOMS 47(3), 2021; reference implementation NLAFET/BBLAS.
  * The **group API**: `group_count`, `group_sizes[]`, and every per-matrix parameter as a per-group array. A fixed-size batch is one group, and the implementation may route a call to a style-specific routine.
  * **Per-problem error reporting** through an `info` array, with modes chosen on input: `ReportAll`, `ReportGroup`, `ReportAny`, `ReportNone`.
* **Layouts** (Dongarra et al. 2017).
  * Pointer-to-pointer, strided, and interleaved.
  * Full interleaving collapses beyond 3×3. Block interleaving (k ≈ 72–168 matrices per block) beats per-matrix approaches only up to about 12×12 to 16×16.
  * Converting P2P to interleaved inside the routine roughly halves throughput.
* **oneMKL** (Knepper, SC24 BoF) ships four styles: group, strided, SYCL spans, and **compact** (CPU only). Compact covers gemm, trsm, getrfnp, getrinp, potrf and geqrf in an interleaved format with explicit pack/unpack helpers.
* **KokkosBatched compact kernels** (Kim et al., SC17).
  * Packs of V matrices interleaved, V = SIMD width.
  * On KNL with sizes 3 to 15, speedups over an OpenMP loop of MKL calls reach 13.8× (dgemm), 45.2× (dtrsm), and 1.7–27.4× (getrf without pivoting).
  * Each kernel comes as serial, team and procedure.
* **Compact LU is non-pivoting** in both MKL and KokkosBatched.
* **One problem per thread** for the sizes above that: LIBXSMM (Heinecke et al., SC16; JIT per shape) and BLASFEO (Frison et al., ACM TOMS 44(4), 2018; a native panel-major format, because repacking cannot be amortised at small n).
* **Batched SVD uses one-sided Jacobi** (Boukaram et al., Parallel Computing 74, 2018; Abdelfattah and Fasi, arXiv:2601.17979, 2026), because it parallelises uniformly across the batch.
* **In Rust,** no crate was found that runs SIMD across a batch on CPU.

### The stack's existing rule

[core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md) supports regular batches in one call and leaves irregular batches to the caller. A batched caller such as tlinalg splits the token: every slot goes to the batch with `Par::Seq` items when the batch is at least the budget; otherwise the token is forked over the items.

### A possible direction, for later (not discussed in colmat-rs#171)

* **One batch descriptor in colmat.** Either a uniform shape plus a stride, or a list of groups. Parameters are broadcast, per group or per item, as with oneMKL's spans.
* **Per-item results.** `Result` per item, or a single flag, chosen by the caller: the `info` modes in Rust form. Singular or indefinite items are expected in batched LU and Cholesky.
* **Two kernel families.**
  * Per-item calls on sub-teams, which is the existing rule and the default.
  * A compact family for tiny sizes, with explicit pack/unpack and an "already packed" batch type rather than silent conversion. It is non-pivoting where the interleaved layout forces it.
* **Batched SVD and EVD on one-sided Jacobi.** One-sided Jacobi is also built only from rotations and dot products, which suits double-double. That suitability is an inference, not a claim of the sources.

## 10. What colmat already has

The original inventory was taken at `tensor4all/colmat-rs@4abe04d`. Sections 6 and 7a link to the later source used for the workspace and dispatch review.

* **The token and the cost model:**
  * `Par<'team>`, the widest-width-that-pays rule;
  * `cost.rs`, with per-family rates per host class;
  * team skeletons (`run`, `for_each`, `tile_grid`, and equal-area triangle splits in Cholesky).
* **Scalars:**
  * `ComplexField`/`RealField`, with SIMD hooks per scalar;
  * `fx128` is an existing non-native scalar and can serve as a baseline for the generic and SIMD paths. See [colmat-rs#177](https://github.com/tensor4all/colmat-rs/pull/177) for the downstream matmul test and the converting-hook test.
* **Primitive calls in `linalg/` (excluding tests):** gemm 76, triangular product 52, triangular solve 97, block Householder apply 138. These counts describe call sites at the survey revision. The dispatch inventory in §7a identifies where individual hooks could be shared.
* **Column-pivoted QR with rank truncation** (`rel_tol`, `max_rank`; tensor4all/t4a-faer#303). It applies the same rule as sparse-ir-rs's TSVD front-end, so a truncated SVD for `Df64` can be composed from colmat's own parts once `Df64` is supported.

## 11. Open questions

1. **Kernel boundaries and selection.** Finish the dispatch inventory for dot, GEMM, triangular solve, Householder and the fused sweeps. Pick the first independently selectable hooks and their generic fallbacks. Resolve associated versus explicit selection and how operators and nested calls use it.
2. **Scalar and precision contracts.** Decide which crate owns the scalar implementation for `Df64`, how its provider attaches, and how scalar-only real and complex types reach the generic paths. Define the precision parameter, safe scaling bounds, and the error and reproducibility contracts for the selected kernels.
3. **Workspace API.** Agree on the query inputs, capacity error and the way nested and parallel calls borrow the buffer. Identify dependency routes that require an explicit scratch entry point or a compatible fallback.
4. **Exception handling.** Define the NaN and Inf contract for the first kernels and their fallbacks. Consider the conventions in arXiv:2207.09281 for the wider primitive set.

Batched API shape remains in section 9. Malleable teams remain a separate design question because the current token splits an existing team and has fixed membership.

### Design review before implementation

Review the proposed boundaries with Markus and the colmat maintainers in [colmat-rs#171](https://github.com/tensor4all/colmat-rs/issues/171). Record the agreed answers in this document before changing the implementation. Use dot and GEMM to walk through a scalar-only type, a type that specialises one kernel, and a nested factorisation that uses both generic and specialised paths. Include a strided view and a call with an exact-size workspace in the walkthrough.

The implementation plan should include the following acceptance checks:

* Numerical checks for nonzero low words, cancellation and scaling extremes. For SIMD, include full and partial lanes, logical indices and pivot ties. Check ordinary downstream entry points on AArch64 and x86-64.
* Workspace checks for exact size, alignment, insufficient capacity before output mutation, and concurrent subcalls. Count allocations across participating threads during execution with destination, team and scratch already supplied. Exercise first use, repeated calls, shape growth and nested calls.
* Native non-regression measurements against a pinned colmat baseline for `f64`, `c64`, `c32` and `f32`. Use the same hardware, build settings, inputs and thread placement. Cover dispatch-sensitive small dot and GEMM calls, strided and contiguous inputs, narrow and square products, complex conjugation, and representative factorisations. Exercise single-thread and team paths on AArch64 and x86-64. Check output equivalence against the baseline when the selected kernel and operation order are unchanged. Compare each type and workload, including packing and other preparation, so an aggregate result cannot hide a regression.

## 12. References

Papers (DOIs and arXiv identifiers were checked against Crossref or the publisher page on 2026-11-13):

* Riedy, Henry, Demmel, Gates, Li, Tang. A Proposal for a Next-Generation BLAS. SC17 BoF, 2017. [document](https://docs.google.com/document/d/1DY4ImZT1coqri2382GusXgBTTTVdBDvtD5I14QHp9OE/edit), [slides](https://icl.utk.edu/bblas/sc17/files/bblas-sc17-riedy.pdf)
* Demmel, Dongarra, Julie Langou, Julien Langou, Luszczek, Mahoney. Prospectus for the Next LAPACK and ScaLAPACK Libraries (BALLISTIC). LAWN 297, 2020. [pdf](https://www.stat.berkeley.edu/~mmahoney/pubs/lawn297.pdf)
* Demmel et al. Proposed Consistent Exception Handling for the BLAS and LAPACK. [arXiv:2207.09281](https://arxiv.org/abs/2207.09281), 2022.
* Gates, Luszczek, Kurzak, Dongarra et al. C++ API for BLAS and LAPACK. SLATE Working Note 2, 2017. [pdf](https://library.eecs.utk.edu/files/ut-eecs-17-753.pdf)
* Hoemmen, Hollman, Trott et al. P1673R13, A free function linear algebra interface based on the BLAS. [paper](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p1673r13.html).
* Van Zee, van de Geijn. BLIS: A Framework for Rapidly Instantiating BLAS Functionality. ACM TOMS 41(3), 2015. doi:10.1145/2764454
* Van Zee, Parikh, van de Geijn. Supporting Mixed-domain Mixed-precision Matrix Multiplication within the BLIS Framework. ACM TOMS 47(2), 2021. doi:10.1145/3402225
* Smith, van de Geijn, Smelyanskiy, Hammond, Van Zee. Anatomy of High-Performance Many-Threaded Matrix Multiplication. IPDPS 2014. doi:10.1109/IPDPS.2014.110
* Low, Igual, Smith, Quintana-Ortí. Analytical Modeling Is Enough for High-Performance BLIS. ACM TOMS 43(2), 2016. doi:10.1145/2925987
* El Kazdadi. faer: A linear algebra library for the Rust programming language. JOSS 11(123):6099, 2026. doi:10.21105/joss.06099
* Nakata. MPLAPACK version 2.0.1 user manual. [arXiv:2109.13406](https://arxiv.org/abs/2109.13406)
* Li, Demmel, Bailey, Henry et al. Design, implementation and testing of extended and mixed precision BLAS. ACM TOMS 28(2), 2002. doi:10.1145/567806.567808
* Hida, Li, Bailey. Algorithms for quad-double precision floating point arithmetic. ARITH-15, 2001. doi:10.1109/ARITH.2001.930115
* Joldes, Muller, Popescu. Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word Arithmetic. ACM TOMS 44(2), 2017. doi:10.1145/3121432. The bounds above were taken from the authors' slides ([RAIM 2025](https://raim2025.sciencesconf.org/data/program/slides_mioara_joldes.pdf), [NAC 2024](https://webusers.imj-prg.fr/~jean-claude.bajard/NAC2024/Supports/NAC2024_JMM.pdf)), because the full text was not reachable.
* Ogita, Rump, Oishi. Accurate Sum and Dot Product. SIAM J. Sci. Comput. 26(6), 2005. doi:10.1137/030601818
* BLAS Technical Forum. BLAS Technical Forum Standard, 2001. [netlib page](https://www.netlib.org/blas/blast-forum/); summary in [ICL report](https://hpcg.icl.utk.edu/files/publications/2001/icl-utk-77-2001.pdf).
* Howell, Demmel, Fulton, Hammarling, Marmol. Cache efficient bidiagonalization using BLAS 2.5 operators. ACM TOMS 34(3), 2008. doi:10.1145/1356052.1356055
* Van Zee, van de Geijn, Quintana-Ortí, Elizondo. Families of Algorithms for Reducing a Matrix to Condensed Form. ACM TOMS 39(1), 2012. [pdf](https://www.cs.utexas.edu/~flame/pubs/CondensedFormTOMS.pdf)
* Van Zee, van de Geijn, Quintana-Ortí. Restructuring the Tridiagonal and Bidiagonal QR Algorithms for Performance. ACM TOMS 40(3), 2014. doi:10.1145/2535371
* Quintana-Ortí, Sun, Bischof. A BLAS-3 Version of the QR Factorization with Column Pivoting. SIAM J. Sci. Comput. 19(5), 1998. doi:10.1137/S1064827595296732
* Belter, Jessup, Karlin, Siek. Automating the Generation of Composed Linear Algebra Kernels. SC09; Belter, Jessup, Nelson, Norris, Siek. Reliable Generation of High-Performance Matrix Algebra. [arXiv:1205.1098](https://arxiv.org/abs/1205.1098)
* Filipovič, Madzin, Fousek, Matyska. Optimizing CUDA code by kernel fusion: application on BLAS. J. Supercomputing 71(10), 2015. doi:10.1007/s11227-015-1483-z
* Ozaki, Ogita, Oishi, Rump. Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications. Numer. Algorithms 59, 2012. doi:10.1007/s11075-011-9478-1
* Mukunoki, Ozaki, Ogita, Imamura. Accurate Matrix Multiplication on Binary128 Format Accelerated by Ozaki Scheme. ICPP 2021. doi:10.1145/3472456.3472493
* Ootomo, Ozaki, Yokota. DGEMM on integer matrix multiplication unit. IJHPCA, 2024. doi:10.1177/10943420241239588
* Ozaki, Uchino, Imamura. Ozaki Scheme II. [arXiv:2504.08009](https://arxiv.org/abs/2504.08009), 2025.
* Mukunoki, Takahashi. Implementation and Evaluation of Quadruple Precision BLAS Functions on GPUs. PARA 2010, LNCS 7133. doi:10.1007/978-3-642-28151-8_25
* Ahrens, Demmel, Nguyen. Algorithms for Efficient Reproducible Floating Point Summation. ACM TOMS 46(3), 2020. doi:10.1145/3389360
* Dongarra, Hammarling, Higham, Relton, Valero-Lara, Zounon. The Design and Performance of Batched BLAS on Modern High-Performance Computing Systems. Procedia CS 108, 2017. doi:10.1016/j.procs.2017.05.138
* Dongarra et al. A Proposed API for Batched Basic Linear Algebra Subprograms. MIMS EPrint 2016.25. [record](https://eprints.maths.manchester.ac.uk/2464/)
* Abdelfattah et al. A Set of Batched Basic Linear Algebra Subprograms and LAPACK Routines. ACM TOMS 47(3), 2021. doi:10.1145/3431921
* Knepper. Evolution of Batched Linear Algebra in Intel oneMKL. SC24 BoF. [slides](https://icl.utk.edu/bblas/sc24/files/3-EvolutionOfBatchedLinearAlgebraInOneMKL.pdf)
* Kim, Costa, Deveci, Bradley, Hammond, Guney, Knepper, Story, Rajamanickam. Designing Vector-Friendly Compact BLAS and LAPACK Kernels. SC17. doi:10.1145/3126908.3126941
* Heinecke, Henry, Hutchinson, Pabst. LIBXSMM: Accelerating Small Matrix Multiplications by Runtime Code Generation. SC16. doi:10.1109/SC.2016.83
* Frison, Kouzoupis, Sartor, Zanelli, Diehl. BLASFEO: Basic Linear Algebra Subroutines for Embedded Optimization. ACM TOMS 44(4), 2018. doi:10.1145/3210754
* Jhurani, Mullowney. A GEMM interface and implementation on NVIDIA GPUs for multiple small matrices. JPDC 75, 2015. doi:10.1016/j.jpdc.2014.09.003
* Abdelfattah, Haidar, Tomov, Dongarra. Batched one-sided factorizations of tiny matrices using GPUs. J. Comput. Sci. 26, 2018. doi:10.1016/j.jocs.2018.01.005
* Boukaram, Turkiyyah, Ltaief, Keyes. Batched QR and SVD algorithms on GPUs with applications in hierarchical matrix compression. Parallel Computing 74, 2018. doi:10.1016/j.parco.2017.09.001
* Abdelfattah, Fasi. An Efficient Batch Solver for the SVD on GPUs. [arXiv:2601.17979](https://arxiv.org/abs/2601.17979), 2026 (preprint).
* Buttari, Langou, Kurzak, Dongarra. A class of parallel tiled linear algebra algorithms for multicore architectures. Parallel Computing 35(1), 2009. doi:10.1016/j.parco.2008.10.002
* Dongarra et al. PLASMA: Parallel Linear Algebra Software for Multicore Using OpenMP. ACM TOMS 45(2), 2019. doi:10.1145/3264491
* Gates, Kurzak, Charara, YarKhan, Dongarra. SLATE: Design of a Modern Distributed and Accelerated Linear Algebra Library. SC19. doi:10.1145/3295500.3356223
* Catalán et al. Static scheduling of the LU factorization with look-ahead on asymmetric multicore processors. Parallel Computing 76, 2018. doi:10.1016/j.parco.2018.04.006
* Catalán et al. A Case for Malleable Thread-Level Linear Algebra Libraries: The LU Factorization With Partial Pivoting. IEEE Access 7, 2019. doi:10.1109/ACCESS.2019.2895541
* Demmel, Grigori, Hoemmen, Langou. Communication-optimal parallel and sequential QR and LU factorizations. SISC 34(1), 2012. doi:10.1137/080731992
* Grigori, Demmel, Xiang. CALU: A Communication Optimal LU Factorization Algorithm. SIMAX 32, 2011. doi:10.1137/100788926
* Xia, Barca. Machine-Learning-Driven Runtime Optimization of BLAS Level 3 on Modern Multi-Core Systems. [arXiv:2406.19621](https://arxiv.org/abs/2406.19621), 2024.

Not verified, so not used above:

* a paper on \<T\>LAPACK;
* the full citations of the IAAT/IATF/LBBGEMM/IrGEMM small-GEMM papers;
* the DOIs of the 2019 OpenMP static look-ahead and the 2020 intra-routine malleability papers;
* a primary source for JAEA's QPBLAS.

## Appendix A. Precise definitions of the primitive kinds

Each figure in section 7 links here ([back to the figures](#primitive-kinds-decided-2026-11-13)). A definition fixes **what** is computed, not how: blocking, packing, fusion and parallel splits are implementation choices. "colmat today" points to the code at `colmat-rs@4abe04d` that the definition was read from.

### Conventions

* **Transposition.** `op(X)` is one of `X`, `X̄` (conjugate), `Xᵀ`, `Xᴴ`.
* **Accumulation.** `Accum::Replace` means β = 0, and the output is **not read**, so a NaN already in it does not propagate. `Accum::Add` reads it.
* **Triangular operands.**
  * `uplo` ∈ {lower, upper} selects the stored triangle.
  * `diag` = unit means the diagonal is **not read** and is taken to be 1.
  * `side` ∈ {left, right} says on which side a factor multiplies.
* **Aliasing.** An output aliases no input unless the definition says "in place".
* **Empty dimensions** are well defined. For example, a product with an inner dimension of 0 gives `C ← β·C`.
* **Rounding.** A definition gives the result in exact arithmetic. Implementations may reassociate sums. colmat promises bitwise identical results only at a fixed thread count (tensor4all/t4a-faer#327).
* **Magnitude for pivoting** is `abs1(z) = |Re z| + |Im z|`, as in BLAS `i?amax`.
* **NaN in a search** (the first maximum, a pivot) is not yet specified. arXiv:2207.09281 proposes "the first NaN, then the first Inf, then the first largest finite entry". See the exception handling question in section 11.

### P1. Level-3 products

* **gemm.** For `C` of size m×n, `op(A)` of size m×k and `op(B)` of size k×n:
  `C ← α·op(A)·op(B) + β·C`.
* **Triangular output** (gemmt; herk and syrk; her2k and syr2k). The same, but only the entries in the `uplo` triangle of the square `C` (diagonal included) are read and written; the other triangle is left untouched. Special cases:
  * `C ← α·A·Aᴴ + β·C` with α and β real (herk). The diagonal of `C` stays real.
  * `C ← α·A·Bᴴ + ᾱ·B·Aᴴ + β·C` (her2k).
* **trmm.** In place in `B`: `B ← α·op(T)·B` (left) or `B ← α·B·op(T)` (right), with `T` triangular (`uplo`, `diag`).
* **Precision used:** none.
* **colmat today:** `linalg::matmul::matmul` (`Accum`, conjugation flags) and `linalg::matmul::triangular`.

### P2. Level-3 triangular solve

* **trsm.** Solve `op(T)·X = α·B` (left) or `X·op(T) = α·B` (right). `X` overwrites `B`. `T` is triangular (`uplo`, `diag`).
* **Singularity is not checked.** A zero pivot gives IEEE Inf or NaN. Whether the primitive reports it is open; see arXiv:2207.09281.
* **colmat today:** `linalg::triangular_solve` (`solve_lower_triangular_in_place`, `solve_unit_lower_triangular_in_place`, and so on).

### P3. Block-reflector apply

* **The reflector.** `H = H₀·H₁⋯H_{k−1}` with `Hᵢ = I − τᵢ·vᵢ·vᵢᴴ`.
* **Compact WY form.** `H = I − V·T·Vᴴ` (forward, column-wise storage).
  * `V` is m×k and unit lower trapezoidal: `V[i,i] = 1` and `V[i,j] = 0` for i < j, neither stored nor read.
  * `T` is k×k upper triangular.
* **Forming T** (larft): `T[i,i] = τᵢ` and `T[0:i, i] = −τᵢ·T[0:i, 0:i]·V[:, 0:i]ᴴ·vᵢ`.
* **Apply** (larfb), in place in `C`, one of:
  * `C ← H·C`
  * `C ← Hᴴ·C`
  * `C ← C·H`
  * `C ← C·Hᴴ`

  A sequence of blocks applies their products in order.
* **colmat today:** `linalg::householder::apply_block_householder_{,transpose_}on_the_{left,right}_in_place_with_conj` and the `_sequence_` variants, plus `upgrade_householder_factor`.

### P4. Fused sweep

**Operands.**

* `A`, of size m×n, or the `uplo` triangle of a square Hermitian `A`. This is the *region*.
* Update vectors `uᵣ` (length m) and `wᵣ` (length n), for r = 0, 1, or 2 pairs. The rank r may be 0.
* Inputs of the by-products, as needed.

**Definition.** The result is as if the two phases below ran in this order. Fusion into one pass is an implementation property, not part of the meaning.

1. **Update.** `A ← A − Σᵣ uᵣ·wᵣᴴ`, applied to the region only. For a Hermitian triangle the rank-2 form is `A ← A − u·wᴴ − w·uᴴ`.
2. **By-products, computed from the updated A.** Any subset of:
   * `y ← A·x` (length m), and `zᴴ ← sᴴ·A` (length n);
   * for a Hermitian `A` stored as one triangle, `y ← A·x` with both triangles used;
   * **column-norm downdate.** `νⱼ ← √(νⱼ² − |a_{p,j}|²)` for the row `p` just eliminated. The norms are recomputed exactly from the columns when the largest downdated value falls below `√eps · ν_ref`, where `ν_ref` is the largest norm at the last recomputation.
   * **per-column first maximum.** For each column `j`, `iⱼ = min{ i : abs1(a_{ij}) = maxₖ abs1(a_{kj}) }`.
   * **2-D maximum.** `(i*, j*)` among the entries of the region with the largest `abs1`. For a real scalar, ties go to the smallest column, then the smallest row, of the matrix that is searched. For a complex scalar, the order among ties is not part of the contract (colmat's current contract, `lu/full_pivoting/factor.rs`, "Pivot ties").

**colmat's instances** (from the unfused fallbacks, which are the reference):

| Instance | Region and update | By-products |
| --- | --- | --- |
| Hessenberg (`hessenberg_fused_op_fallback`) | full; `A ← A − l₀·r₀ − l₁·r̄₁` (`l` columns, `r` rows) | `r_out ← A·r_in`, `l_out ← l̄_in·A` |
| Bidiagonal (`bidiag_fused_op`) | full; `A ← A − u·y − z·v` | the two products of the next step |
| Tridiagonal (`tridiag_fused_op_simd`) | lower triangle; Hermitian rank-2 | `y ← A·x`. The kernel returns two partial vectors (from the columns and from the rows of the triangle) that the caller sums. |
| LU leaf (`leaf_step_simd`) | full; rank 1 | per-column first maximum |
| Full-pivoting LU (`update_and_best_in_mat_simd`) | full; rank 1 | 2-D maximum |
| Column-pivoted QR (`update_mat_and_dot_simd`) | full; rank 1 (one reflector) | column-norm downdate, and the dots of the next reflector |
| Bunch–Kaufman (`rank_{1,2}_update_and_argmax`) | lower triangle; rank 1 or 2 | off-diagonal 2-D maximum |

**Relation to GEMVER.** The BLAS Technical Forum's `GEMVER` is the full-region rank-2 instance with a *dependent* second product (`w ← α·A·x` with `x ← β·Aᵀ·y + z`). colmat's products are independent.

**Precision used:** `eps` in the norm-downdate test.

### P5. Level-2 triangular

* **trsv:** `x ← op(T)⁻¹·x`. **trmv:** `x ← op(T)·x`.
* In place, one right-hand side, `T` triangular (`uplo`, `diag`).
* **The dependency order is part of the meaning of a recurrence.** For lower `T`, `xᵢ` depends on `x₀ … x_{i−1}`.
* **colmat today:** the one-column cases of `linalg::triangular_solve`, and `triangular_inverse`.

### P6. Sequences of small transforms

* **Rotation.** `G = [c s; −s̄ c]` with `c` real and `c² + |s|² = 1`.
  * On rows `i`, `j`: `[aᵢ; aⱼ] ← G·[aᵢ; aⱼ]`.
  * On columns: `[aᵢ aⱼ] ← [aᵢ aⱼ]·Gᴴ`.
  * A sequence `G_{k−1}⋯G₀` is applied in its stated order, as in LAPACK `xLASR`: the pivot pattern (variable, top, bottom) and the direction (forward, backward) are parameters.
* **Small reflector.** `H = I − τ·v·vᴴ` with `v` of length 2 or 3 and `v₀ = 1`.
  * It is applied to a range of rows from the left and/or a range of columns from the right.
  * The bulge chase itself (which reflector, where) belongs to the algorithm. The primitive is the application.
* **LLT/LDLT rank updates** are provisionally here, as a column recurrence of such transforms (open point 3 of section 7).
* **colmat today:** `linalg::jacobi` (`apply_on_the_left_in_place`, `apply_on_the_right_in_place`), and `move_bulge` in `evd/schur/{real,complex}_schur.rs` and `chase_bulge_2x2` in `gevd/qz_real`.

### P7. Permutations

* **laswp:** for k = k₀, …, k₁ in order (or in reverse, for the inverse), swap rows `k` and `piv[k]`.
* **Applying a permutation:** `B[p(i), :] = A[i, :]` (rows), or the same on columns.
* **No arithmetic.** NaN and Inf move with their entries.
* **colmat today:** `perm` and the row swaps inside the LU and Bunch–Kaufman drivers.

### P8. Level 1 and reductions

* **dot:** `s = Σᵢ x̄ᵢ·yᵢ` (conjugated) or `Σᵢ xᵢ·yᵢ`. **sum:** `Σᵢ xᵢ`.
* **Norms:**
  * `‖x‖₂ = √(Σᵢ |xᵢ|²)`, computed without spurious overflow or underflow whenever the result is representable. LAPACK `xNRM2` derives its scaling constants from radix, exponent bounds and precision. Section 5 leaves the source of the equivalent colmat parameters open.
  * `‖x‖₁ = Σᵢ |xᵢ|`;
  * `‖x‖_max = maxᵢ |xᵢ|`, which must propagate NaN (faer-039).
* **First maximum:** `i* = min{ i : abs1(xᵢ) = maxₖ abs1(xₖ) }`.
* **Updates:** `y ← α·x + y` (axpy), `x ← α·x` (scal), copy, and conjugate; layout-changing copies (between strides, and transposition).
* **Precision used:** `safe_min`, `safe_max` and `eps` for the scaled norm.
* **colmat today:** `linalg::reductions::{sum, norm_l1, norm_l2, norm_l2_sqr, norm_max}` and `zip!`.

### P9. Scalar numerical kernels

Each kernel is defined by its mathematical result, together with the range of inputs for which it must not overflow or underflow spuriously. That range is where `Precision` enters.

* **Householder generation** (larfg).
  * Given a scalar `α` and a vector `x`, find `β` real, `τ` and `v` such that `Hᴴ·[α; x] = [β; 0]`, with `H = I − τ·[1; v]·[1; v]ᴴ`.
  * When `x = 0` and `Im α = 0`, `τ = 0` (H = I).
  * When `|β|` falls below `safe_min / eps`, the input is rescaled first (LAPACK `xLARFG`).
  * colmat: `make_householder_in_place`.
* **Rotation generation** (lartg).
  * Given `f` and `g`, find `c` real, `s` and `r` such that `[c s; −s̄ c]·[f; g] = [r; 0]`. Use scaling to avoid spurious overflow or underflow when the outputs are representable.
  * The sign convention is that of the implementation; a wrong `r` for `g = 0` was faer-033.
  * colmat: `JacobiRotation::rotg`.
* **2×2 and 3×3 problems:**
  * the standardised real Schur form of a 2×2 block (lanv2);
  * the SVD of a 2×2 upper triangular matrix (lasv2);
  * shifted 2×2 solves with scaling (laln2);
  * the eigenvalues of a 2×2 pencil.

  colmat: `svd_triu_2x2`, `geigval_2x2`, `solve_shifted_2x2_cplx`, `solve_complex_shifted_2x2`, `check_shift_2x2`.
* **Secular equation** (laed4, lasd4).
  * The root `λ ∈ (dᵢ, dᵢ₊₁)` of `f(λ) = 1 + ρ·Σⱼ zⱼ²/(dⱼ − λ) = 0` (or its SVD form with `dⱼ² − σ²`), to relative accuracy `O(eps)`.
  * colmat: `secular_eq`, `secular_root` in `evd/tridiag_evd.rs`.
* **dqds.** One shifted dqds transform of the qd array, with its deflation tests (LAPACK `xLASQ*`). colmat: `svd/dqds.rs`.
* **Safe reciprocal and hypot.** `1/x` and `√(a² + b²)` without spurious overflow or underflow; a complex reciprocal of a subnormal value must not give NaN (faer-020).
* **Precision used:** all of `eps`, `safe_min` and `safe_max`.

### P10. Leaf factorisations

* **getf2.** `P·A = L·U` on an m×n_b panel, unblocked.
  * Pivot of column `k`: the per-column first maximum (P4) of the remaining rows.
  * Then the row swap (P7), a scaling by the safe reciprocal of the pivot (P9), and a rank-1 update of the rest of the panel (P4).
* **potf2.** `A = L·Lᴴ` (lower) on a small block. Reports the first non-positive pivot.
* **geqr2.** `A = Q·R`, with the Householder vectors stored below the diagonal and the `τ` in a separate vector.
* **colmat today:** the leaf of `lu/partial_pivoting` (`lu_in_place_unblocked_simd`), `simd_cholesky` in `cholesky/ldlt/factor.rs`, and the unblocked QR panels.
