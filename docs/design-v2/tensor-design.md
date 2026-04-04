# v2 Tensor Design: Structure and Einsum

**Date:** 2026-04-04
**Status:** Draft
**Parent:** `README.md`
**Related:** `tensor-api-pseudocode.md`, `ad-architecture.md`

---

## I. Principle: tenferro::Tensor is always dense

`Tensor` is a dense multi-dimensional array. It carries no structural metadata
(diagonal, symmetric, block-diagonal, sparse, etc.), but it may reside on CPU
or GPU. Runtime placement is described by `Placement`.

```rust
struct Placement {
    memory_kind: MemoryKind,
    resident_device: Option<ComputeDevice>,
}

enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

// Typed tensor (internal)
struct TensorData<T: Scalar> {
    buffer: Buffer<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    placement: Placement,
    preferred_compute_device: Option<ComputeDevice>,
}

// Type-erased tensor (user-facing)
enum Tensor {
    F32(TensorData<f32>),
    F64(TensorData<f64>),
    C32(TensorData<Complex<f32>>),
    C64(TensorData<Complex<f64>>),
}
```

```rust
enum Buffer<T> {
    Host(HostBuffer<T>),
    Backend(BufferHandle<T>),
}
```

`Tensor` allows **arbitrary strides**, enabling zero-copy views for permute
and slice operations. Strided views avoid data movement in the high-level
graph layer.

**Reshape semantics:** `Reshape` operates on the **logical element sequence**
(column-major traversal of the logical shape). It is zero-copy only when the
tensor's strides are column-major contiguous. For permuted-contiguous views
(e.g., from `.t()`, where physical memory is contiguous but axis order differs),
`Reshape` requires a physical copy to column-major before reinterpreting the
shape. This matches StableHLO's `Reshape`, which operates on dense arrays with
no stride concept.

At eval() time, input pre-processing checks memory contiguity:

1. **Contiguous data** (including permuted-contiguous views from
   `tensor.permute()` or `.t()`, and contiguous slices): passed as-is with
   zero copy. The strides are preserved.
2. **Non-contiguous data** (memory gaps from slicing): physically copied to
   a contiguous buffer before execution.

No StableHLO ops are inserted for input normalization -- the StableHLO
program is layout-independent. The execution engine is stride-aware: it
inspects strides at dispatch time and uses BLAS trans flags for transposed
inputs, v1-style fusability checks on dimension groups for BatchedGemm, etc.
Engine-produced intermediates and outputs use **column-major (Fortran)
layout** as the standard convention.

`TracedTensor` wraps `Tensor` with graph tracking for lazy evaluation
and AD (see `tensor-api-pseudocode.md`).

`Tensor` is the standard-algebra runtime value shared across CPU and GPU
backends. Methods such as `placement()`, `resident_device()`, `to_cpu()`, and
`to_gpu_on(...)` are part of the tensor boundary; backend-specific handles
remain internal implementation details. Compute preference stays separate via
`preferred_compute_device()`.

**Why dense only**: structural variants (diagonal, band, triangular, ...)
cause combinatorial explosion in op implementations. Every op must handle
`Dense × Dense`, `Diagonal × Dense`, `Dense × Diagonal`, `Diagonal × Diagonal`,
etc. Adding a new structure type requires touching every op.

StableHLO also assumes dense tensors. Structural variants cannot be
lowered to StableHLO without conversion.

---

## II. Structural information lives in tensor4all-rs

Higher-level structure (diagonal matrices, block-diagonal, etc.) is
managed in **tensor4all-rs**, one layer above tenferro. This matches
the previous tensor4all-rs codebase.

```
tensor4all-rs (structure-aware layer)
  ├── DiagonalTensor { diag: Tensor }        // N-dim vector → diagonal matrix
  ├── BlockDiagonal { blocks: Vec<Tensor> }  // block structure
  ├── ... (other structured types)
  │
  └── uses tenferro einsum with hyper edges to avoid scatter/gather

tenferro (dense layer)
  ├── Tensor — always dense
  ├── TracedTensor — graph-aware wrapper
  ├── einsum — hyper edge support
  └── backends — faer / Custom GPU / StableHLO
```

---

## III. Einsum with hyper edges replaces scatter/gather

### The problem

Reconstructing A = U Σ Vᵀ from SVD naively requires materializing
the diagonal matrix Σ:

```
Naive (scatter needed):
  diag_matrix = scatter(sigma, indices)    // [N] → [N, N]
  temp = matmul(U, diag_matrix)            // [M, N] × [N, N]
  A = matmul(temp, Vt)                     // [M, N] × [N, K]
  → materializes sparse N×N matrix. wasteful.
```

### Hyper edge solution

A **hyper edge** is an index that appears in 3+ tensors simultaneously.
tenferro's einsum supports this natively:

```
einsum("ik,k,kj->ij", U, sigma, Vt)
        ^  ^  ^
        k appears in 3 tensors (hyper edge)

→ sigma stays as 1D vector
→ no diagonal matrix materialized
→ no scatter/gather needed
```

### Einsum capabilities

```rust
// Diagonal embedding: vector → diagonal matrix
einsum("i->ii", &[&v])         // [N] → [N, N]

// Diagonal extraction: matrix → vector
einsum("ii->i", &[&a])         // [N, N] → [N]

// Higher-order diagonal: vector → 3D diagonal tensor
einsum("i->iii", &[&v])        // [N] → [N, N, N]

// Trace: matrix → scalar
einsum("ii->", &[&a])          // [N, N] → scalar

// SVD reconstruction with hyper edge
einsum("ik,k,kj->ij", &[&u, &sigma, &vt])  // no scatter
```

The `Subscripts` data structure represents index equivalence classes:
a label (u32) shared across multiple input tensors defines a hyper edge.
This is a direct encoding of the tensor network structure.

### Multiple equivalence classes

A single einsum can have multiple hyper edges (equivalence classes):

```
einsum("ik,k,kj,jl,l,lm->im", &[&A, &d1, &B, &C, &d2, &D])
        k k k   j j j  l l l
        ^^^^^   ^^^^^   ^^^^^
        class 1 class 2 class 3
```

Each equivalence class contracts independently. The einsum optimizer
chooses the contraction order. No intermediate diagonal matrices needed.

---

## IV. Einsum decomposition for compilation

N-ary einsum and hyper edges are decomposed into binary contractions
using `PrimitiveOp`s (DotGeneral, Reshape, Transpose, etc.) in a
computegraph Fragment:

```
einsum("ik,k,kj->ij", U, sigma, Vt)

Decomposed into Fragment:
  step 1: Mul(U_broadcasted, sigma_broadcasted)   → temp  (element-wise scaling)
  step 2: DotGeneral(temp, Vt, ...)               → A     (matmul)
```

The contraction path optimizer decides the pairwise decomposition order.
This optimization result is cached in `Engine`'s `EinsumCache`.

For StableHLO lowering: the Fragment's binary ops map directly to
`stablehlo.dot_general`, `stablehlo.multiply`, etc.

---

## V. Diagonal structure in tensor4all-rs

### DiagonalTensor

```rust
// tensor4all-rs
struct DiagonalTensor {
    diag: Tensor,   // 1D dense vector [N]
    // Logically represents an [N, N] diagonal matrix
}

impl DiagonalTensor {
    fn to_dense(&self) -> Tensor {
        einsum("i->ii", &[&self.diag])
    }

    fn matmul(&self, rhs: &Tensor) -> Tensor {
        // Efficient: no scatter, no dense diagonal matrix
        einsum("i,ij->ij", &[&self.diag, rhs])
    }
}
```

### BlockDiagonal

```rust
struct BlockDiagonal {
    blocks: Vec<Tensor>,  // each block is dense
    // Logically represents a block-diagonal matrix
}
```

### AD through structured types

Differentiation operates at the `TracedTensor` level (dense).
tensor4all-rs wraps and unwraps:

```
Forward:
  DiagonalTensor.diag (1D Tensor)
    → wrap as TracedTensor
    → einsum with hyper edge
    → result (TracedTensor)

AD:
  differentiate/transpose operate on the dense einsum graph
  → no structural knowledge needed at the AD level
```

tensor4all-rs extracts the dense Tensor leaves from structured types
before entering the AD graph via TracedTensor.

---

## VI. Summary

| Layer | Knows about structure? | Types |
|-------|----------------------|-------|
| computegraph (graph engine) | No | generic `GraphOp` |
| tidu (AD transforms) | No | generic `PrimitiveOp` |
| tenferro | No | `Tensor` (dense), `TracedTensor` |
| tensor4all-rs | **Yes** | `DiagonalTensor`, `BlockDiagonal`, etc. |

| Operation | Without hyper edge | With hyper edge |
|-----------|-------------------|-----------------|
| U Σ Vᵀ | scatter + 2 matmul | einsum 3-input (no scatter) |
| diag(v) × A | scatter + matmul | einsum 2-input "i,ij->ij" |
| Tr(A) | extract_diag + sum | einsum "ii->" |

Structural types live in tensor4all-rs. tenferro provides dense
tensors + einsum with hyper edges. scatter/gather are available but
rarely needed when einsum handles the structure.
