# v2 Primitive Catalog

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`
**Related:** `backend-architecture.md`, `tensor-design.md`, `stablehlo-primitives.md`, `jax-primitives.md`

---

## I. Purpose

This document answers the question:

> What exactly counts as a "primitive" in the v2 design, and what does each op
> mean?

**Normative status:** this file is the **source of truth** for the primitive
list that tenferro v2 is expected to implement. If another design document has
a shorter summary and the two disagree, this file wins.

The design docs use "primitive" in several nearby but different senses. For
readability, this document separates them explicitly:

| Layer | Example | Meaning |
|-------|---------|---------|
| Surface API | `einsum`, `sum`, `mean`, `grad`, `svd()` | what users call |
| Graph primitive | `DotGeneral`, `ReduceSum`, `BroadcastInDim` | what may appear as `StdTensorOp` nodes in a `Fragment` |
| Backend kernel | BLAS GEMM, cuSOLVER SVD, StableHLO op | how a primitive is executed |

This document uses **two orthogonal classifications**:

1. backend-facing execution subsets inside tenferro
2. canonical graph primitives that the v2 graph / AD stack talks about

Important distinctions:

- `differentiate`, `transpose`, `resolve`, and `compile` are **transforms**, not
  primitives.
- `einsum` is **surface syntax**, not a final persistent graph primitive. It is
  lowered into graph primitives such as `DotGeneral`, `Mul`, `Transpose`,
  `BroadcastInDim`, and `ReduceSum`.
- High-level linalg ops such as `SVD` and `Solve` may remain explicit graph
  primitives because they are meaningful semantic units, even though their
  derivative rules emit lower-level primitives.

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

### What tenferro v2 is expected to implement

From this document's point of view, the implementation target is:

- implement the backend-facing semiring execution subsets defined below
- implement the AD-closed graph core defined below
- implement the `Standard arithmetic only` primitives when tenferro claims
  standard dense numeric support
- treat control-flow primitives as future work, not part of the initial
  required set

---

## III. Backend-Facing Execution Subsets In tenferro

This section is about what tenferro backends need in order to execute einsum
and related tensor programs efficiently. These are **not** the same thing as
the canonical graph primitive vocabulary.

### Tensor-structural prerequisites

The strict semiring minimum below assumes that the tensor layer already provides
the following structural operations as views or cheap metadata transforms:

- `permute`
- `reshape`
- `broadcast`
- `diagonal`

With those available outside the backend prim vocabulary, repeated labels,
layout normalization, and singleton expansion do not need separate semiring
execution ops.

### Semiring core: strict minimum for einsum execution

This is the smallest backend-facing set needed to execute generic einsum after
contraction planning and structural-view normalization.

| Primitive | Why it is needed |
|-----------|------------------|
| `BatchedGemm` | binary contraction primitive for paired labels |
| `ReduceSum` | reduction of labels that survive planning but are not present in the final output |

If v2 keeps `diagonal` as a tensor-layer view, `Trace` is **not** part of this
strict minimum. If diagonal extraction/contraction is not available as a view,
`Trace` would have to move back into the semiring core.

### Semiring execution helper

| Primitive | Role |
|-----------|------|
| `MakeContiguous` | explicit materialization boundary for kernels that require contiguous operands |

`MakeContiguous` is useful in implementations, but it is not part of the
mathematical minimum needed to describe einsum semantics.

### Semiring fast-path extensions

Restrict this subset to the fast paths that are already present in the current
tenferro design:

- `Contract`
- `ElementwiseBinary { Add, Mul }`

These ops are not required for correctness, but they can avoid lowering all the
way to `BatchedGemm` + `ReduceSum` in common cases.

| Primitive | Role |
|-----------|------|
| `Contract` | direct binary einsum/contraction fast path |
| `ElementwiseBinary { Mul }` | fast path for Hadamard-style products |
| `ElementwiseBinary { Add }` | fast path for semiring accumulation / fused accumulation patterns |

---

## IV. Canonical Graph Primitive Vocabulary

This section is about the graph-level vocabulary that `computegraph-rs`,
`chainrules-rs`, `tidu-rs`, and tenferro's `StdTensorOp` layer talk about.

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

| Primitive | Outputs | Definition |
|-----------|---------|------------|
| `SVD` | `(U, S, Vt)` | Thin singular value decomposition `A = U diag(S) Vt` |
| `QR` | `(Q, R)` | Thin QR factorization `A = Q R` |
| `Cholesky` | `(L)` or `(U)` | Cholesky factorization of a positive-definite matrix |
| `Eigh` | `(eigenvalues, eigenvectors)` | Hermitian / symmetric eigendecomposition |
| `Solve` | `(X)` | Solve `A X = B` for `X` |

These ops are expected to lower to backend-specific linalg kernels
(`stablehlo.custom_call`, LAPACK, cuSOLVER, etc.), but their derivative rules
should still emit graph primitives that satisfy `PrimitiveOp` closure.

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
low-friction lowering path to StableHLO.

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

This catalog is about the **semantic tensor vocabulary** that the v2 graph and
AD stack talk about. It is intentionally independent of how tenferro chooses to
organize execution crates or backend traits internally.
