# v2 Primitive Catalog

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`
**Related:** `backend-architecture.md`, `tensor-design.md`

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
| Graph primitive | `DotGeneral`, `ReduceAdd`, `BroadcastInDim` | what may appear as `TensorOp` nodes in a `Fragment` |
| Backend kernel | BLAS GEMM, cuSOLVER SVD, StableHLO op | how a primitive is executed |

Important distinctions:

- `differentiate`, `transpose`, `resolve`, and `compile` are **transforms**, not
  primitives.
- `einsum` is **surface syntax**, not a final persistent graph primitive. It is
  lowered into graph primitives such as `DotGeneral`, `Mul`, `Transpose`,
  `BroadcastInDim`, and `ReduceAdd`.
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

- `Dup(x) -> (x0, x1)`
- `SVD(A) -> (U, S, Vt)`
- `QR(A) -> (Q, R)`

The output ordering must be part of the primitive definition because
`GlobalValKey` includes `output_slot`.

### What tenferro v2 is expected to implement

From this document's point of view, the implementation target is:

- implement the full Tier-1 core set
- implement the structured tensor and linalg primitives listed here when
  tenferro claims support for them
- implement the Tier-2 standard numeric set as the dense standard-arithmetic
  vocabulary
- treat control-flow primitives as future work, not part of the initial
  required set

---

## III. Tier 1: Core AD-Closed Primitive Set

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
| `Dup` | `x: S -> (y0: S, y1: S)` | `y0 = x`, `y1 = x` | Internal linear primitive that makes fan-out explicit |
| `DotGeneral(config)` | `lhs: A, rhs: B -> out: C` | General tensor contraction over explicit batch axes and contracting axes | Matrix multiply, batched matmul, and inner product are all special cases |
| `Transpose(perm)` | `x: [d0, ..., dn-1] -> y: [d_perm[0], ..., d_perm[n-1]]` | Reorder axes according to `perm` | Pure axis permutation |
| `Reshape(shape)` | `x: [d0, ..., dn-1] -> y: shape` | Reinterpret the same element sequence with a new shape | Total element count must stay unchanged |
| `BroadcastInDim(shape, dims)` | `x: [a0, ..., ak-1] -> y: shape` | Place input axis `j` into output axis `dims[j]`, repeating along the others | Makes all broadcast semantics explicit |
| `ReduceAdd(axes)` | `x: [d0, ..., dn-1] -> y` | `y` is formed by summing `x` over the listed axes | Rank drops unless a later op restores it |

### Concrete examples

| Primitive | Example |
|-----------|---------|
| `DotGeneral` | `ij,jk->ik` (ordinary matrix multiply) |
| `BroadcastInDim` | `[n] -> [b, n]` with `dims=[1]` |
| `ReduceAdd` | `[b, m, n] -> [b, n]` with `axes=[1]` |
| `Transpose` | `[b, m, n] -> [m, b, n]` with `perm=[1, 0, 2]` |

---

## IV. Structured Tensor Primitives and AD Helpers

These ops are not part of the minimal Tier-1 closure, but they are part of the
intended tensor vocabulary because they arise naturally in tensor-network and
linalg code.

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Trace(paired_axes)` | `x: [..., n, ..., n, ...] -> y` | Sum entries where the listed axis pairs are equal | Example: matrix trace `[n, n] -> []` |
| `Diag(mode, paired_axes)` | vector/tensor -> tensor/vector | Either embed values onto a diagonal or extract a diagonal slice | Think `i -> ii` or `ii -> i` in einsum notation |
| `AntiTrace` | `x -> y` | Scatter-add cotangent values back onto traced diagonal positions | Internal AD helper, not part of semiring core |
| `AntiDiag` | `x -> y` | Scatter or write cotangent values back through diagonal structure | Internal AD helper, not part of semiring core |

`Trace` and `Diag` matter because they are mathematically meaningful, while
`AntiTrace` and `AntiDiag` are primarily implementation helpers that may appear
inside transpose rules or decompositions.

---

## V. Tier 2: Standard Numeric Primitive Set

Tier 2 is the standard arithmetic vocabulary layered on top of the Tier-1 core.
These are the primitives needed for general-purpose differentiable programming
on dense tensors.

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

`ReduceAdd` stays in Tier 1 because it is essential both for primal tensor code
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

## VI. Frontend Sugar and Canonical Lowering

Many familiar user-level ops are better treated as aliases or composites rather
than as distinct graph primitives.

| Surface op | Canonical graph form |
|------------|----------------------|
| `einsum(...)` | contraction planning + `DotGeneral`/`Mul`/`Transpose`/`Reshape`/`BroadcastInDim`/`ReduceAdd` |
| `sum(x, axes)` | `ReduceAdd(x, axes)` |
| `mean(x, axes)` | `ReduceAdd(x, axes)` followed by scale by `1 / n` |
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

## VII. Implementation Note

This catalog is about the **semantic tensor vocabulary** that the v2 graph and
AD stack talk about. It is intentionally independent of how tenferro chooses to
organize execution crates or backend traits internally.
