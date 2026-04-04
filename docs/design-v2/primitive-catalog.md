# v2 Primitive Catalog

**Date:** 2026-04-04
**Status:** Draft
**Parent:** `README.md`
**Related:** `backend-architecture.md`, `tensor-design.md`, `stablehlo-primitives.md`, `jax-primitives.md`

---

## I. Purpose

This document answers the question:

> What exactly counts as a "primitive" or "instruction" in the v2 design at each
> level of the IR hierarchy, and what does each op mean?

**Normative status:** this file is the **source of truth** for the primitive
and instruction-set vocabulary that tenferro v2 is expected to implement at
all levels. If another design document has a shorter summary and the two
disagree, this file wins.

The design docs use "primitive" and "instruction" in several nearby but
different senses. For readability, this document separates them explicitly:

| Layer | Example | Meaning |
|-------|---------|---------|
| Surface API | `einsum`, `sum`, `mean`, `grad`, `svd()` | what users call |
| StableHLO-compatible IR (high-level) | `DotGeneral`, `ReduceSum`, `BroadcastInDim` | what may appear as `StdTensorOp` nodes in a `Fragment`; the single cut point between graph/AD and backends |
| Low-level IR | `BatchedGemm`, `ReduceSum`, `Permute`, `Reshape`, `CustomCall` | output of the optimizing compiler; input to faer / custom backends |
| Backend kernel | BLAS GEMM, cuSOLVER SVD, IREE module, faer routine | how an instruction is executed |

This document uses **three orthogonal classifications**:

1. **backend-facing execution architecture** (Section III): the 2-level IR,
   optimizing compiler, backend traits, and execution engine
2. **canonical graph primitive vocabulary** (Section IV): what the graph / AD
   stack talks about at the StableHLO IR level
3. **standard arithmetic extensions** (Section V): ops available only for
   ordinary dense numeric types

Important distinctions:

- `differentiate`, `transpose`, `resolve`, and `compile` are **transforms**, not
  primitives.
- `einsum` is **surface syntax**, not a final persistent graph primitive. It is
  lowered into graph primitives such as `DotGeneral`, `Mul`, `Transpose`,
  `BroadcastInDim`, and `ReduceSum`.
- High-level linalg ops such as `SVD` and `Solve` may remain explicit graph
  primitives because they are meaningful semantic units, even though their
  derivative rules emit lower-level primitives.
- `StdTensorOp` is **flat** (no `SemiringOpKind` wrapping). Most variants
  map 1:1 to a StableHLO op. Documented exceptions include composite
  lowerings (e.g., `Conj` -> 4 ops: `real` + `imag` + `negate` + `complex`)
  and multi-output linalg ops (e.g., `Svd` -> `custom_call` +
  `get_tuple_element` x N).
- `Tensor` allows **arbitrary strides** at the user level. Input
  pre-processing happens at **eval() time**: contiguous data (including
  permuted-contiguous views from `permute()` or `.t()`) is passed as-is
  with zero copy, preserving strides; only truly non-contiguous data
  (memory gaps from slicing) is physically copied to a contiguous buffer.
  No StableHLO ops are inserted for input normalization -- the StableHLO
  program is layout-independent. The execution engine is stride-aware and
  handles permuted inputs via BLAS trans flags at dispatch time.

Responsibility boundary:

- `chainrules-rs` owns the `PrimitiveOp` contract
- `tidu-rs` owns generic AD transforms that call `linearize` and
  `transpose_rule`
- tenferro owns the **concrete per-op derivative rules**

So this directory keeps the primitive vocabulary and cross-crate architecture,
but not a standalone per-op transpose-rule manual. Detailed formulas are a
downstream tenferro design/implementation concern.

---

## II. Reading the Tables

### Shape notation

- `x: [b, m, n]` means `x` is a rank-3 tensor with shape `(b, m, n)`.
- Scalars are rank-0 tensors and are written as `[]`.
- Batch dimensions are written explicitly; nothing is implied by position alone.

### No implicit broadcasting

Elementwise primitives such as `Add` and `Mul` do **not** silently broadcast.
If shapes differ, the graph must contain an explicit `BroadcastInDim`.

### `Transpose` vs AD transpose

`Transpose(perm)` is the tensor operation "permute axes".
It is unrelated to the AD transform `transpose(linear_fragment)`.

### Multi-output primitives

Some primitives produce multiple outputs:

- `SVD(A) -> (U, S, Vt)`
- `QR(A) -> (Q, R)`

The output ordering must be part of the primitive definition because
`GlobalValKey` includes `output_slot`.

### Column-major (Fortran) convention

Engine-produced intermediates and outputs use **column-major (Fortran)
ordering**. This is the convention for all data produced by the execution
engine. Input tensors may be contiguous with arbitrary axis ordering; the
engine inspects strides and adjusts dispatch accordingly (e.g., BLAS trans
flags for transposed inputs).

### What tenferro v2 is expected to implement

From this document's point of view, the implementation target is:

- implement the 2-level IR architecture and backend traits defined below
- implement the AD-closed graph core defined below
- implement the `Standard arithmetic only` primitives when tenferro claims
  standard dense numeric support
- treat control-flow primitives as future work, not part of the initial
  required set

---

## III. Backend-Facing Execution Architecture

This section defines the complete execution pipeline from graph-level
StableHLO-compatible IR down to backend kernels. The architecture has two IR
levels separated by an optimizing compiler.

### III.1 StableHLO-compatible IR (high-level) -- the single cut point

The StableHLO-compatible IR is the **single cut point** between the graph / AD
stack and all backends.

- An XLA/StableHLO backend can take this IR **directly** (serialize to
  StableHLO MLIR, hand to IREE or XLA).
- All other backends (faer, custom semiring) receive this IR and pass it through
  the optimizing compiler to produce low-level IR.

The ops in this IR are exactly the canonical graph primitive vocabulary defined
in Section IV. `StdTensorOp` is flat: most variants correspond 1:1 to a
StableHLO op. Documented exceptions include composite lowerings (e.g., `Conj`
-> 4 ops) and multi-output linalg ops (e.g., `Svd` -> `custom_call` +
`get_tuple_element` x N).

Input layout normalization is a pure runtime concern handled by the execution
engine, not an IR transformation. The StableHLO program is layout-independent.
The compile cache needs no layout signature in its key.

The input contract differs by backend:

**Low-level IR engine (faer, custom algebra):**

1. **Contiguous data** (including permuted-contiguous views from
   `tensor.permute()` or `.t()`, and contiguous slices): passed as-is with
   zero copy. The engine is stride-aware (BLAS trans flags, fusability checks).
2. **Non-contiguous data** (memory gaps from slicing): physically copied to
   a contiguous buffer before execution.

**XLA backend:**

XLA accepts only dense contiguous buffers (no stride concept). The XLA
backend always copies to column-major contiguous before uploading. The extra
host-side reorder is negligible because XLA is primarily for GPU, where
host→device transfer dominates.

No StableHLO ops are inserted for input normalization in either path.

The StableHLO-compatible IR is designed to be **actually serializable to
StableHLO MLIR**. It is not merely "inspired by" StableHLO -- the IR uses
the same op semantics, the same dimension numbering conventions, and the
same layout model (dense, no strides). Standard dtypes (f32, f64, c32, c64)
can be round-tripped through StableHLO serialization.

### III.2 Optimizing compiler (algebra-agnostic passes)

The optimizing compiler transforms StableHLO IR into low-level IR through a
sequence of algebra-agnostic passes:

| Pass | Purpose |
|------|---------|
| **TransposeFolding** | Fold chains of `Transpose` + `DotGeneral` into a single instruction with permuted dimension numbers |
| **DotDecomposer** | Break multi-contracting-dim `DotGeneral` into sequences that map to `BatchedGemm` |
| **LinalgCustomCallPassthrough** | Pass linalg `CustomCall` ops through to the low-level IR as-is |

Note: input contiguity checking is **not** a compiler pass. It happens at
eval() time as a runtime pre-processing step (see Section III.1). Only truly
non-contiguous data (memory gaps) is physically copied. Contiguous data with
arbitrary axis ordering is passed through as-is.

These passes are **algebra-agnostic**: they operate on shape metadata and
instruction structure, not on element values. They are shared by all non-XLA
backends.

### III.3 Low-level IR

The output of the optimizing compiler is a flat sequence of low-level
instructions. Input operands may be contiguous with arbitrary axis ordering.
The engine inspects strides and adjusts dispatch accordingly (e.g., BLAS
trans flags for transposed inputs, v1-style `prepare_one_operand` fusability
checks on dimension groups for BatchedGemm). Engine-produced intermediates
and outputs are column-major contiguous.

| Instruction | Signature | Definition |
|-------------|-----------|------------|
| `BatchedGemm` | `lhs: [b, m, k], rhs: [b, k, n] -> out: [b, m, n]` | Batched matrix multiply; the fundamental contraction instruction |
| `ReduceSum(axes)` | `x: [d0, ..., dn-1] -> y` | Sum over the listed axes |
| `Permute(perm)` | `x: [d0, ..., dn-1] -> y: [d_perm[0], ..., d_perm[n-1]]` | Axis permutation; output is a fresh contiguous allocation |
| `Reshape(shape)` | `x -> y: shape` | Reinterpret contiguous storage with a new shape |
| `CustomCall { target, config }` | `inputs -> outputs` | Dispatch to a registered kernel (e.g., LAPACK routine for SVD/QR/Eigh/Solve). The optimizing compiler passes linalg `custom_call` ops through to the low-level IR as-is. |

Structural ops (`Permute`, `Reshape`) are handled by **common
infrastructure** shared across all backends. They are not part of the custom
backend contract. Note that `Copy` is not a low-level IR instruction;
only truly non-contiguous input data (memory gaps) is physically copied at
eval() pre-processing before IR entry.

### III.4 Backend traits

Backend traits follow the v1 pattern of required core + optional fast paths.

**`SemiringCore`** (required):

| Method | Low-level instruction(s) covered |
|--------|----------------------------------|
| `batched_gemm` | `BatchedGemm` |
| `reduce_sum` | `ReduceSum` |

This is the **minimum contract** for a custom algebra backend. Implementing
only these two methods is sufficient to execute any einsum-derived program.

**`SemiringFastPath`** (optional):

| Method | Purpose |
|--------|---------|
| `contract` | Direct binary contraction; avoids decomposition to `BatchedGemm` + `ReduceSum` |
| `elementwise_mul` | Fast path for Hadamard products |
| `elementwise_add` | Fast path for semiring accumulation / fused patterns |

Fast-path methods can **absorb multiple low-level IR instructions** into a
single kernel call. The low-level IR and backend trait methods need **not** be
1:1; a fast-path method may pattern-match a subgraph of low-level instructions
and execute them as one fused operation.

### III.5 Custom algebra backend minimum

A custom algebra backend (e.g., tropical semiring) needs to implement only:

- `batched_gemm`
- `reduce_sum`

Everything else -- structural ops, optimizing passes, generic execution loop --
is provided by tenferro's common infrastructure.

### III.6 Generic execution engine

The generic execution engine is a simple interpreter that walks the low-level
IR instruction sequence and dispatches each instruction:

1. Structural ops (`Permute`, `Reshape`) are executed by common
   infrastructure.
2. `CustomCall` instructions are dispatched to a **registered kernel registry**.
   The faer/CPU backend registers LAPACK routines (e.g., `dgesvd`, `dgeqrf` +
   `dorgqr`, `dsyevd`); a GPU backend registers cuSOLVER equivalents. Unrecognized
   targets are a runtime error.
3. For compute ops, the engine's `prepare` step inspects input strides before
   dispatching. For `BatchedGemm`, this uses v1's `prepare_one_operand`
   approach: fusability check on dimension groups, BLAS trans flags for
   transposed inputs.
4. The engine first checks `SemiringFastPath` for an applicable pattern match.
5. If no fast path fires, the engine falls back to `SemiringCore` methods.

This design means a backend author can start with the minimum two-method
contract and add fast paths incrementally as performance needs arise.

---

## IV. Canonical Graph Primitive Vocabulary

This section is about the graph-level vocabulary that `computegraph-rs`,
`chainrules-rs`, `tidu-rs`, and tenferro's `StdTensorOp` layer talk about.

These ops define the **StableHLO-compatible IR** (high-level). Each op maps
to a StableHLO op. The XLA backend emits these directly; other backends
lower them through the optimizing compiler.

### AD-closed graph core

These are the minimal tensor primitives needed to express:

- scalar and tensor JVP/VJP rules
- explicit broadcasting and reshaping
- general contractions
- reverse-mode accumulation without hidden fan-out

Every op in this table is expected to implement `PrimitiveOp` directly.

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Add` | `x0: S, x1: S -> y: S` | `y[i] = x0[i] + x1[i]` | Same shape on both inputs; no hidden broadcasting |
| `Mul` | `x0: S, x1: S -> y: S` | `y[i] = x0[i] * x1[i]` | Elementwise multiply; same-shape contract |
| `Neg` | `x: S -> y: S` | `y[i] = -x[i]` | Unary elementwise |
| `Conj` | `x: S -> y: S` | `y[i] = conj(x[i])` | Identity on real dtypes, conjugation on complex dtypes |
| `DotGeneral(config)` | `lhs: A, rhs: B -> out: C` | General tensor contraction over explicit batch axes and contracting axes | Canonical contraction primitive; matrix multiply, batched matmul, and inner product are all special cases. Config defined below. |
| `Transpose(perm)` | `x: [d0, ..., dn-1] -> y: [d_perm[0], ..., d_perm[n-1]]` | Reorder axes according to `perm` | Pure axis permutation |
| `Reshape(shape)` | `x: [d0, ..., dn-1] -> y: shape` | Reinterpret the same element sequence with a new shape | Total element count must stay unchanged |
| `BroadcastInDim(shape, dims)` | `x: [a0, ..., ak-1] -> y: shape` | Place input axis `j` into output axis `dims[j]`, repeating along the others | Makes all broadcast semantics explicit |
| `ReduceSum(axes)` | `x: [d0, ..., dn-1] -> y` | `y` is formed by summing `x` over the listed axes | Rank drops unless a later op restores it |

**Structural ops** (`Transpose`, `Reshape`, `BroadcastInDim`) are meaningful at
the graph level for AD and shape inference, but they are handled by common
infrastructure at the backend level. They are **not** part of the custom backend
contract.

### DotGeneral config

```rust
struct DotGeneralConfig {
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
}
```

Contracting dims are summed over (inner product). Batch dims are preserved
in the output. Remaining dims appear in the output in lhs-then-rhs order.
This matches StableHLO's `dot_general` dimension numbers.

### Concrete examples

| Primitive | Example |
|-----------|---------|
| `DotGeneral` | `ij,jk->ik` (ordinary matrix multiply) |
| `BroadcastInDim` | `[n] -> [b, n]` with `dims=[1]` |
| `ReduceSum` | `[b, m, n] -> [b, n]` with `axes=[1]` |
| `Transpose` | `[b, m, n] -> [m, b, n]` with `perm=[1, 0, 2]` |

---

### Structured tensor graph ops and AD helpers

These ops are not part of the minimal AD-closed core, but they may still appear
in tenferro's graph vocabulary if v2 chooses to represent them explicitly
instead of lowering them away through tensor-layer views.

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Trace(paired_axes)` | `x: [..., n, ..., n, ...] -> y` | Sum entries where the listed axis pairs are equal | Example: matrix trace `[n, n] -> []`; not part of the strict semiring minimum if `diagonal` is a view |
| `Diag(mode, paired_axes)` | vector/tensor -> tensor/vector | Either embed values onto a diagonal or extract a diagonal slice | Think `i -> ii` or `ii -> i` in einsum notation |
| `AntiTrace` | `x -> y` | Scatter-add cotangent values back onto traced diagonal positions | Internal AD helper, not part of semiring core |
| `AntiDiag` | `x -> y` | Scatter or write cotangent values back through diagonal structure | Internal AD helper, not part of semiring core |

`Trace` and `Diag` matter because they are mathematically meaningful, while
`AntiTrace` and `AntiDiag` are primarily implementation helpers that may appear
inside transpose rules or decompositions.

---

## V. Standard Arithmetic Only

These primitives are available only for the ordinary dense numeric setting
(real/complex standard arithmetic). They are not assumed to exist for generic
semirings such as tropical algebra.

This section should be kept as close as practical to the official StableHLO op
set, so that tenferro's canonical graph primitives lower cleanly to StableHLO.
See `stablehlo-primitives.md` for the StableHLO-facing reference and
`jax-primitives.md` for the JAX-side reference point.

### Elementwise arithmetic, comparison, and selection

| Primitive | Definition | Notes |
|-----------|------------|-------|
| `Div` | `y[i] = x0[i] / x1[i]` | Canonical division op |
| `Abs` | `y[i] = abs(x[i])` | Real magnitude or complex modulus, depending on dtype contract |
| `Sign` | `y[i] = sign(x[i])` | Often used in stabilization logic |
| `Maximum` | `y[i] = max(x0[i], x1[i])` | Ordered real comparison |
| `Minimum` | `y[i] = min(x0[i], x1[i])` | Ordered real comparison |
| `Compare(dir)` | Produce a predicate/mask tensor from an elementwise comparison | `dir` is things like `eq`, `lt`, `le`, `gt`, `ge` |
| `Select` | `y[i] = pred[i] ? on_true[i] : on_false[i]` | Canonical conditional elementwise choice |
| `Clamp` | `y[i] = min(max(x[i], lower[i]), upper[i])` | Canonical clipping primitive |

### Analytic elementwise primitives

| Primitive | Definition |
|-----------|------------|
| `Exp` | `exp(x)` |
| `Log` | `log(x)` |
| `Sin` | `sin(x)` |
| `Cos` | `cos(x)` |
| `Tanh` | `tanh(x)` |
| `Sqrt` | `sqrt(x)` |
| `Rsqrt` | `1 / sqrt(x)` |
| `Pow` | `x^y` |
| `Expm1` | `exp(x) - 1` |
| `Log1p` | `log(1 + x)` |

The table above is the canonical v2 analytic seed set. Additional analytic ops
may be added later, but they are not part of the current required list unless
this document is updated.

### Indexing and structural data movement

| Primitive | Definition | Notes |
|-----------|------------|-------|
| `Gather` | Read values from `x` at positions specified by an index tensor | Shape-preserving indexed read pattern |
| `Scatter` | Write or accumulate values into `y` at positions specified by an index tensor | Indexed inverse of gather |
| `Slice` | Read a static rectangular subregion | Start/limit/stride known in the op |
| `DynamicSlice` | Read a slice whose start index is data-dependent | Dynamic counterpart of `Slice` |
| `Pad` | Extend a tensor with edge/interior padding values | Needed for transpose of slicing-like ops |
| `Concatenate` | Join tensors along one axis | Rank-preserving shape change |
| `Reverse` | Reverse the order of elements along selected axes | Useful for convolutions and sequence models |

### Additional reductions

| Primitive | Definition |
|-----------|------------|
| `ReduceProd` | Multiply values over the listed axes |
| `ReduceMax` | Max over the listed axes |
| `ReduceMin` | Min over the listed axes |

`ReduceSum` stays in the AD-closed graph core because it is essential both for primal tensor code
and for transpose rules.

### Linalg primitives

| Primitive | Outputs | Definition | StableHLO lowering |
|-----------|---------|------------|--------------------|
| `Cholesky` | `(L)` or `(U)` | Cholesky factorization of a positive-definite matrix | Direct StableHLO op (`stablehlo.cholesky`) |
| `SVD` | `(U, S, Vt)` | Thin singular value decomposition `A = U diag(S) Vt` | `stablehlo.custom_call` |
| `QR` | `(Q, R)` | Thin QR factorization `A = Q R` | `stablehlo.custom_call` |
| `Eigh` | `(eigenvalues, eigenvectors)` | Hermitian / symmetric eigendecomposition | `stablehlo.custom_call` |
| `Solve` | `(X)` | Solve `A X = B` for `X` | `stablehlo.custom_call` |

`Cholesky` has a direct StableHLO op. All other linalg primitives lower to
`stablehlo.custom_call` with appropriate target names (matching JAX/XLA
conventions for LAPACK/cuSOLVER dispatch).

Regardless of lowering path, derivative rules for all linalg ops emit graph
primitives that satisfy `PrimitiveOp` closure.

### Future control-flow primitives

| Primitive | Definition |
|-----------|------------|
| `Cond` | Branch between two subcomputations based on a predicate |
| `While` | Loop while a condition remains true |
| `Scan` | Structured loop with carried state and stacked outputs |

These are intentionally future-facing and are not required for the initial v2
vertical slice.

---

## VI. StableHLO Alignment

When there is a choice, the canonical graph vocabulary should prefer the
StableHLO-style name and semantics:

| Prefer in v2 | Instead of |
|--------------|------------|
| `DotGeneral` | `einsum` or `dot` as a primitive |
| `BroadcastInDim` | implicit broadcasting or generic `broadcast` primitive |
| `Compare(dir)` + `Select` | surface names like `greater`, `greater_equal`, `where` |
| `ReduceSum` / `ReduceMax` / ... | opaque reduction primitives whose combiner is not explicit |

The goal is not to copy StableHLO mechanically. The goal is to ensure that the
`Standard arithmetic only` part of tenferro's graph vocabulary has an obvious,
low-friction lowering path to StableHLO, because the StableHLO-compatible IR
is the single cut point for all backends.

See also `stablehlo-primitives.md` and `jax-primitives.md`.

---

## VII. Frontend Sugar and Canonical Lowering

Many familiar user-level ops are better treated as aliases or composites rather
than as distinct graph primitives.

| Surface op | Canonical graph form |
|------------|----------------------|
| `einsum(...)` | contraction planning + `DotGeneral`/`Mul`/`Transpose`/`Reshape`/`BroadcastInDim`/`ReduceSum` |
| `sum(x, axes)` | `ReduceSum(x, axes)` |
| `mean(x, axes)` | `ReduceSum(x, axes)` followed by scale by `1 / n` |
| `sub(x, y)` | `Add(x, Neg(y))` |
| `square(x)` | `Mul(x, x)` |
| `reciprocal(x)` | `Div(1, x)` |
| `where(pred, a, b)` | `Select(pred, a, b)` |
| `greater(x, y)` | `Compare(dir=gt)` |
| `greater_equal(x, y)` | `Compare(dir=ge)` |
| `clamp_min(x, lo)` / `clamp_max(x, hi)` | special cases of `Clamp` |
| `trace(x)` | `Trace(...)` |
| `diag(x)` / `extract_diag(x)` | `Diag(...)` |

This is useful for two reasons:

1. the canonical graph vocabulary stays smaller and easier to reason about
2. AD closure becomes easier to verify because fewer primitive rules are truly
   fundamental

---

## VIII. Implementation Note

This catalog defines the **semantic vocabulary at all IR levels** that the v2
graph, AD stack, and execution engine talk about.

- The **StableHLO-compatible IR** (Section IV) is the single interface between
  the graph/AD world and execution.
- The **low-level IR** (Section III.3) is the interface between the optimizing
  compiler and backend kernels.
- **Backend traits** (Section III.4) define what a custom backend must
  implement.

The boundary is deliberate: graph-level concerns (AD closure, shape inference)
live above the StableHLO IR cut point; execution concerns (memory layout,
kernel dispatch, fast paths) live below it.
