# v2 Tensor Design: Structure and Einsum

**Date:** 2026-04-03
**Status:** Draft
**Related:** `v2-ad-architecture.md`, `v2-backend-architecture.md`

---

## I. Principle: tenferro2::Tensor is always dense

`tenferro2::Tensor` is a plain multi-dimensional array. It carries no
structural metadata (diagonal, symmetric, block-diagonal, sparse, etc.).

```rust
// tenferro2::Tensor — always dense
struct Tensor {
    inner: DynTensor,        // dense data buffer
    device: Device,
    memory_space: MemorySpace,
}
```

**Why**: structural variants (diagonal, band, triangular, ...) cause
combinatorial explosion in op implementations. Every op must handle
`Dense × Dense`, `Diagonal × Dense`, `Dense × Diagonal`, `Diagonal × Diagonal`,
etc. Adding a new structure type requires touching every op.

StableHLO also assumes dense tensors. Structural variants cannot be
lowered to StableHLO without conversion.

---

## II. Structural information lives in tensor4all-rs

Higher-level structure (diagonal matrices, block-diagonal, etc.) is
managed in **tensor4all-rs**, one layer above tenferro2. This matches
the previous tensor4all-rs codebase.

```
tensor4all-rs (structure-aware layer)
  ├── DiagonalTensor { diag: Tensor }        // N-dim vector → diagonal matrix
  ├── BlockDiagonal { blocks: Vec<Tensor> }  // block structure
  ├── ... (other structured types)
  │
  └── uses tenferro2::einsum with hyper edges to avoid scatter/gather

tenferro2 (dense layer)
  ├── Tensor — always dense
  ├── einsum — hyper edge support
  └── backends — faer / Custom GPU / XLA
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
tenferro2's einsum supports this natively:

```
einsum("ik,k,kj->ij", U, sigma, Vt)
        ^  ^  ^
        k appears in 3 tensors (hyper edge)

→ sigma stays as 1D vector
→ no diagonal matrix materialized
→ no scatter/gather needed
```

### tenferro2 einsum capabilities (from tenferro-rs v1)

Already supported:

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

## IV. StableHLO lowering of einsum

StableHLO's `dot_general` only supports **2-input** contractions.
Hyper edges (3+ inputs) must be decomposed into pairwise steps:

```
einsum("ik,k,kj->ij", U, sigma, Vt)

tenferro2 decomposes into:
  step 1: einsum("ik,k->ik", U, sigma)      → temp  (element-wise scaling)
  step 2: einsum("ik,kj->ij", temp, Vt)     → A     (matmul)

StableHLO:
  %temp = stablehlo.mul(%U_broadcasted, %sigma_broadcasted)   // or dot_general
  %A    = stablehlo.dot_general(%temp, %Vt, ...)
```

**tenferro2's einsum optimizer decides the pairwise decomposition order.**
The hyper edge structure is preserved in TenferroIR; only the StableHLO
lowering decomposes into pairwise ops.

For the faer backend: pairwise decomposition is also used (BLAS dgemm
is 2-input), but the optimizer can choose a different order tuned for
CPU cache performance.

---

## V. Diagonal structure in tensor4all-rs

### DiagonalTensor

```rust
// tensor4all-rs
struct DiagonalTensor {
    diag: tenferro2::Tensor,   // 1D dense vector [N]
    // Logically represents an [N, N] diagonal matrix
}

impl DiagonalTensor {
    fn to_dense(&self) -> tenferro2::Tensor {
        einsum("i->ii", &[&self.diag])
    }

    fn matmul(&self, rhs: &tenferro2::Tensor) -> tenferro2::Tensor {
        // Efficient: einsum("i,ij->ij", &[&self.diag, rhs])
        // No scatter, no dense diagonal matrix
        einsum("i,ij->ij", &[&self.diag, rhs])
    }
}
```

### BlockDiagonal

```rust
struct BlockDiagonal {
    blocks: Vec<tenferro2::Tensor>,  // each block is dense
    // Logically represents a block-diagonal matrix
}
```

### AD through structured types

Differentiation operates at the tenferro2::Tensor level (dense).
tensor4all-rs wraps and unwraps:

```
Forward:
  DiagonalTensor.diag (1D Tensor)
    → einsum with hyper edge
    → result (dense Tensor)

AD:
  differentiate/transpose operate on the dense einsum graph
  → no structural knowledge needed at the AD level
```

The `Differentiable` / pytree trait (Appendix B of v2-ad-architecture.md)
extracts the dense tensor leaves from structured types before entering
the AD graph.

---

## VI. Summary

| Layer | Knows about structure? | Tensor type |
|-------|----------------------|-------------|
| tidu2 (AD engine) | No | generic `Operand` |
| tenferro2 | No | `Tensor` (always dense) |
| tensor4all-rs | **Yes** | `DiagonalTensor`, `BlockDiagonal`, etc. |

| Operation | Without hyper edge | With hyper edge |
|-----------|-------------------|-----------------|
| U Σ Vᵀ | scatter + 2 matmul | einsum 3-input (no scatter) |
| diag(v) × A | scatter + matmul | einsum 2-input "i,ij->ij" |
| Tr(A) | extract_diag + sum | einsum "ii->" |

Structural types live in tensor4all-rs. tenferro2 provides dense
tensors + einsum with hyper edges. scatter/gather are available but
rarely needed when einsum handles the structure.
