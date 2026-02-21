# Rustifying TensorKit.jl / MPSKit.jl on top of tenferro-rs

Date: 2026-02-21

## Executive Summary

We analyzed TensorKit.jl and MPSKit.jl to determine whether tenferro-rs
provides a sufficient computation engine for building Rust equivalents of
these libraries on top. The conclusion:

1. **tenferro-rs is sufficient as a minimal computation engine.** Its
   einsum, plan-based execution (`TensorPrims`), linear algebra, and AD
   framework cover the dense-tensor computation needs.

2. **Tensor-network-specific abstractions (vector spaces, sectors,
   codomain/domain) belong in separate upper-layer crates**, not in
   tenferro-rs.

3. **Basic tensor arithmetic** (add, scale, norm, dot) is expressible
   through existing tenferro-rs primitives (`TensorPrims` alpha/beta,
   einsum). Convenience wrappers can live in a future utility crate if
   needed, keeping the core minimal.

4. **Krylov solvers** (Lanczos, Arnoldi, matrix exponential) are required
   by MPSKit algorithms but are out of scope for tenferro-rs. A separate
   crate is needed.

---

## 1. TensorKit.jl Architecture

### 1.1 Core Abstraction: TensorMap as Morphism

TensorKit's central type is **not** a flat multi-dimensional array but a
**linear map** (morphism) between tensor product spaces:

```
TensorMap{T, S, N1, N2, A}
    codomain: ProductSpace{S, N1}   (output spaces)
    domain:   ProductSpace{S, N2}   (input spaces)
    data:     A                      (flat storage for all blocks)
```

Key consequences of this design:
- **SVD/QR** split along the codomain|domain boundary.
- **Transpose** swaps codomain and domain.
- **Adjoint** dualizes spaces and conjugates data.
- **Composition** is matrix multiplication on the codomain|domain structure.

### 1.2 Vector Space Hierarchy

```
VectorSpace (abstract)
  +-- ElementarySpace (single index)
  |     +-- ComplexSpace          (C^d, dual-aware)
  |     +-- CartesianSpace        (R^d, self-dual)
  |     +-- GradedSpace{I, D}    (sector-graded with multiplicities)
  |
  +-- CompositeSpace
        +-- ProductSpace{S, N}   (tensor product V1 x ... x Vn)
        +-- HomSpace             (morphism space, defines TensorMapSpace)
```

Operations on spaces:
- `dual(V)`: dual space (flip contravariance)
- `conj(V)`: complex conjugate
- `fuse(V1, V2)`: fusion to single elementary space
- `dim(V)`: total dimension

### 1.3 Symmetry Sectors and Block-Sparse Storage

For `GradedSpace{I}`, tensors are **block-diagonal** with blocks indexed
by symmetry sectors:

- **Sector types**: `Trivial`, `Z2Irrep`, `U1Irrep`, `SU2Irrep`, ...
- **FusionStyle**: `UniqueFusion` (Abelian), `MultipleFusion` (non-Abelian)
- **BraidingStyle**: `SymmetricBraiding`, `Fermionic`, `Anyonic`
- **FusionTree{I, N}**: tracks how N sectors fuse to a coupled sector

Storage: all blocks stored contiguously in a single flat vector. Block
offsets precomputed in `FusionBlockStructure`.

### 1.4 Tensor Operations

All operations decompose to **block-wise matrix operations**:

| Operation | Signature | Notes |
|-----------|-----------|-------|
| `contract!` | `(C, A, pA, B, pB, pAB, alpha, beta)` | Generalized contraction |
| `add_permute!` | `(C, A, pA, alpha, beta)` | Addition with permutation |
| `trace_permute!` | `(C, A, pA, q, alpha, beta)` | Partial trace |
| `permute!` | `(C, A, p)` | Index reordering |
| `repartition!` | `(C, A)` | Reshape codomain/domain split |

`@tensor` macro (via TensorOperations.jl) provides Einstein notation:
```julia
@tensor C[i,j] := A[i,k] * B[k,l] * C[l,j]
```

### 1.5 Factorizations

Block-wise factorizations in each sector, assembled respecting sector
structure:

- **SVD**: `svd_compact`, `svd_trunc` with truncation strategies
  (`truncrank`, `trunctol`, `truncerror`, `truncspace`)
- **QR/LQ**: `qr_compact`, `lq_compact`, `left_orth`, `right_orth`
- **Eigendecomposition**: `eigh_full`, `eig_full` (Hermitian and general)
- **Polar**: `left_polar`, `right_polar`
- **Null space**: `left_null`, `right_null`
- **Matrix functions**: `exp`, `log`, `sin`, `cos`

### 1.6 Backend / Device

- Storage type parameterized (`Vector{T}` for CPU, `CuVector{T}` for GPU)
- Thread scheduling via `blockscheduler` (serial or dynamic)
- No high-level device abstraction; GPU via extension packages

### 1.7 Key Source Files

| File | Role |
|------|------|
| `spaces/vectorspaces.jl` | VectorSpace trait system |
| `spaces/gradedspace.jl` | Sector-graded space |
| `spaces/productspace.jl` | Tensor product spaces |
| `tensors/tensor.jl` | TensorMap concrete type |
| `tensors/tensoroperations.jl` | TensorOperations.jl backend |
| `tensors/indexmanipulations.jl` | Permute, transpose, repartition |
| `fusiontrees/fusiontrees.jl` | FusionTree structure |
| `fusiontrees/manipulations.jl` | Tree permutations, braiding |
| `factorizations/` | All decompositions + truncation |

---

## 2. MPSKit.jl Architecture

### 2.1 Core Types

**MPS types** (`src/states/`):

| Type | Description |
|------|-------------|
| `FiniteMPS{A,B}` | Finite MPS with AL, AR, AC, C gauges |
| `InfiniteMPS{A,B}` | Infinite MPS with periodic vectors |
| `WindowMPS` | Window into infinite MPS |
| `MultilineMPS` | Multi-row MPS for 2D systems |

Gauge relations: `AL[i] * C[i] = AC[i] = C[i-1] * AR[i]`

**MPO types** (`src/operators/`):

| Type | Description |
|------|-------------|
| `FiniteMPO{O}` | Finite MPO with vector storage |
| `InfiniteMPO{O}` | Infinite MPO with periodic boundaries |
| `MPOHamiltonian` | Upper-triangular block Hamiltonian |

### 2.2 Algorithms

**Ground state** (`src/algorithms/groundstate/`):
- **DMRG** (1-site): single-site variational optimization with sweeps
- **DMRG2** (2-site): two-site variational with SVD bond truncation
- **VUMPS**: variational uniform MPS for infinite systems
- **GradientGrassmann**: Grassmann manifold optimization

**Time evolution** (`src/algorithms/timestep/`):
- **TDVP** (1-site): time-dependent variational principle
- **TDVP2** (2-site): with bond dimension changes

**Bond dimension** (`src/algorithms/changebonds/`):
- `SvdCut`, `OptimalExpand`, `RandExpand`, `VUMPSSvdCut`

**Excitations**: `FiniteExcited`, `QuasiparticleAnsatz`, `ChepigaAnsatz`

### 2.3 Environment Management

Environments (`src/environments/`) cache left/right tensor contractions:

| Type | Description |
|------|-------------|
| `FiniteEnvironments` | GL/GR arrays with staleness tracking |
| `InfiniteEnvironments` | Krylov-based fixed-point solver |
| `MultilineEnvironments` | For 2D systems |

Key operations: `leftenv()`, `rightenv()`, `recalculate!()`, `poison!()`

### 2.4 Transfer Matrix

`TransferMatrix` (`src/transfermatrix/`) represents the single-site
transfer operator. Used for:
- Gauge fixing eigenvalue problems
- Correlation length computation
- Fixed-point equations in infinite MPS

### 2.5 TensorKit Operations Used by MPSKit

**Contractions** (most frequent, 222+ uses of `@plansor`/`@tensor`):
```julia
@plansor AC2[-1 -2; -3 -4] := AC[pos][-1 -2; a] * AR[pos+1][a -3; -4]
```

**Factorizations** (critical for all algorithms):
- `left_orth!(A) -> (Q, R)` — QR for left-canonical gauge
- `right_orth!(A) -> (L, Q)` — LQ for right-canonical gauge
- `svd_trunc!(A; trunc, alg) -> (U, S, V)` — truncated SVD for bond updates

**Space operations**:
- `space()`, `numind()`, `numout()`, `numin()`
- `isometry()`, `left_null()`, `right_null()`
- Tensor product `x`, direct sum `+`, dual `'`

**Basic algebra**:
- `norm()`, `normalize!()`, `dot()`
- Addition, scaling, multiplication

**External algorithms** (via KrylovKit.jl):
- Lanczos/Arnoldi eigensolvers (DMRG, VUMPS)
- `exponentiate()` (TDVP)
- GMRES (linear systems)

---

## 3. Assessment: tenferro-rs as Computation Engine

### 3.1 Design Principle

tenferro-rs is a **minimal computation engine**. Tensor-network-specific
concepts (vector spaces, sectors, codomain/domain, MPS) do not belong in
tenferro-rs. They are built as separate upper-layer crates.

### 3.2 What the Upper Layer Needs from tenferro-rs

The upper layer (TensorKit-like Rust crates) decomposes block-sparse
operations into per-block dense operations on `Tensor<T>`:

```
For each sector c in valid_sectors:
    block_c = einsum("ij,jk->ik", &[&block_a, &block_b])
```

```
For each sector c:
    SvdResult { u, s, vt } = svd(&block_c, options)
```

### 3.3 What tenferro-rs Provides (Sufficient)

| Requirement | tenferro-rs mechanism | Status |
|-------------|----------------------|--------|
| Per-block contraction | `einsum()` on `Tensor<T>` | Sufficient |
| Per-block SVD/QR/LU/eigen | `tenferro-linalg` returns `SvdResult{u,s,vt}`, `QrResult{q,r}` | Sufficient |
| Sector accumulation | `einsum_into(..., alpha, beta)` | Sufficient |
| Arbitrary-stride execution | `TensorPrims::execute()` takes `StridedView<T>` | Sufficient |
| Algebra extensibility | `HasAlgebra` + `TensorPrims<A>` (e.g., tropical) | Sufficient |
| AD framework | `Differentiable` trait applicable to upper-layer types | Sufficient |
| GPU device abstraction | `LogicalMemorySpace` + `ComputeDevice` | Sufficient |
| Tensor construction | `zeros`, `ones`, `eye`, `from_slice`, `from_vec` | Sufficient |
| Contraction tree optimization | `ContractionTree::optimize()` | Sufficient |
| Plan reuse across same-shape blocks | `TensorPrims::plan()` (shape-specific) + `execute()` (data-specific) | Sufficient |

### 3.4 Basic Arithmetic: Covered by Primitives

`Tensor<T>` has no `Add`/`Sub`/`Mul` operator overloads. These are
intentionally handled through existing primitives:

| Operation | How to express |
|-----------|---------------|
| `C = alpha * A` | `Permute` (identity) with `alpha`, `beta=0` |
| `C = A + B` | Two `Permute` calls: `C = 1*A + 0*C`, then `C = 1*B + 1*C` |
| `dot(A, B)` | `einsum("ij,ij->", &[&a, &b])` |
| `norm(A)` | `Reduce{Sum}` on squared elements + `ElementwiseUnary{Sqrt}` |
| `negate(A)` | `ElementwiseUnary{Negate}` |

This is by design: `Tensor<T>` is a data container, computation goes
through `TensorPrims` or `einsum`. If convenience wrappers are needed
later, they belong in a utility crate, not in the core.

### 3.5 What tenferro-rs Does NOT Provide (By Design)

These belong in upper-layer crates, not in tenferro-rs:

| Concept | TensorKit.jl location | Planned upper-layer crate |
|---------|----------------------|--------------------------|
| VectorSpace hierarchy | `spaces/` | `tenferro-spaces` |
| Codomain/domain split | `TensorMap{T,S,N1,N2}` | `tenferro-tensormap` |
| Symmetry sectors | TensorKitSectors.jl | `tenferro-sectors` |
| Fusion trees | `fusiontrees/` | `tenferro-sectors` |
| Block-sparse storage | Block structure in TensorMap | `tenferro-blocksparse` |
| Graded tensor | GradedSpace + blocks | `tenferro-graded` |
| Truncation strategies | `factorizations/truncation.jl` | `tenferro-tensormap` or `tenferro-linalg-ext` |
| Isometry construction | `isometry(V, W)` | `tenferro-spaces` |
| Null space | `left_null()`, `right_null()` | Upper-layer (via SVD) |
| Krylov solvers | KrylovKit.jl | Separate crate |
| Matrix exponential | KrylovKit.jl `exponentiate()` | Separate crate |
| MPS/MPO types | MPSKit `states/`, `operators/` | `tenferro-mps` |
| DMRG/VUMPS/TDVP | MPSKit `algorithms/` | `tenferro-mps` |
| Environments | MPSKit `environments/` | `tenferro-mps` |

### 3.6 Note on TensorView and einsum

Currently `einsum()` accepts only `&[&Tensor<T>]`, not `TensorView`.
This means blocks stored as views into a shared buffer cannot be passed
directly to einsum.

In practice this is not a problem: in Rust, each block is naturally an
independent `Tensor<T>` (clear ownership, no lifetime complexity). The
reason TensorKit.jl packs all blocks into a single flat vector (GC
pressure reduction) does not apply to Rust.

If view-based einsum is ever needed, the lower-level
`TensorPrims::execute()` already accepts arbitrary `StridedView<T>`.

---

## 4. Proposed Upper-Layer Crate Structure

```
                          tenferro-mps
                     (MPS/MPO, DMRG, VUMPS, TDVP)
                              |
                   +----------+----------+
                   |                     |
            tenferro-graded        krylov-solver
         (GradedTensor<T,S>,      (Lanczos, Arnoldi,
          sector-aware ops)        exponentiate)
                   |
          +--------+--------+
          |                 |
   tenferro-blocksparse  tenferro-sectors
   (BlockSparseTensor<T>,  (Sector trait,
    per-block dispatch)     U1, SU2, Z2,
          |                 FusionTree)
          |
   tenferro-tensormap
   (TensorMap<T,S> = Tensor<T> + codomain/domain,
    truncation strategies, isometry, null space)
          |
   tenferro-spaces
   (VectorSpace trait, ComplexSpace, CartesianSpace,
    GradedSpace, ProductSpace, dual, fuse)
          |
   ======================== boundary ========================
          |
   tenferro-rs (computation engine, unchanged)
     tenferro-tensor    (Tensor<T>, data container)
     tenferro-prims     (TensorPrims<A>, plan/execute)
     tenferro-einsum    (einsum, contraction tree)
     tenferro-linalg    (SVD, QR, LU, eigen)
     tenferro-algebra   (Semiring, HasAlgebra)
     tenferro-device    (CPU/GPU abstraction)
     chainrules-core    (AD traits)
     chainrules         (AD engine)
```

### 4.1 tenferro-spaces

Defines the vector space abstraction:

```rust
pub trait VectorSpace: Clone + Eq {
    fn dim(&self) -> usize;
    fn dual(&self) -> Self;
}

pub trait ElementarySpace: VectorSpace {
    fn is_dual(&self) -> bool;
}

pub struct ComplexSpace { d: usize, is_dual: bool }
pub struct CartesianSpace { d: usize }
pub struct GradedSpace<S: Sector> { sectors: Vec<(S, usize)>, is_dual: bool }

pub struct ProductSpace<V: ElementarySpace> { spaces: Vec<V> }
```

### 4.2 tenferro-tensormap

Wraps `Tensor<T>` with codomain/domain metadata:

```rust
pub struct TensorMap<T: Scalar, V: ElementarySpace> {
    data: Tensor<T>,              // dense block (or multiple for block-sparse)
    codomain: ProductSpace<V>,    // first n_codomain dimensions
    domain: ProductSpace<V>,      // remaining dimensions
}

impl<T, V> TensorMap<T, V> {
    pub fn svd(&self, options: &TruncationStrategy) -> SvdDecomposition<T, V>;
    pub fn qr(&self) -> QrDecomposition<T, V>;
    pub fn transpose(&self) -> TensorMap<T, V>;  // swap codomain <-> domain
    pub fn adjoint(&self) -> TensorMap<T, V>;     // transpose + conjugate + dual
}
```

### 4.3 tenferro-sectors

Defines sector algebra (independent of tensors):

```rust
pub trait Sector: Clone + Eq + Hash + Ord {
    fn trivial() -> Self;
    fn dual(&self) -> Self;
    fn fusion_outcomes(&self, other: &Self) -> Vec<(Self, usize)>; // (sector, multiplicity)
    fn dim(&self) -> usize;  // quantum dimension
}

pub struct U1(pub i64);
pub struct Z2(pub bool);
pub struct SU2(pub HalfInt);
pub struct ProductSector<A: Sector, B: Sector>(pub A, pub B);
```

### 4.4 tenferro-blocksparse

Block-sparse tensor where each block is a `Tensor<T>`:

```rust
pub struct BlockSparseTensor<T: Scalar> {
    block_ranges: Vec<Vec<Range<usize>>>,      // axis partitioning
    blocks: HashMap<Vec<usize>, Tensor<T>>,    // block index -> dense block
    shape: Vec<usize>,
}
```

Per-block operations delegate to tenferro-rs:
- `contract()`: iterate valid block pairs, call `einsum()` per pair
- `svd()`: call `tenferro_linalg::svd()` per block

### 4.5 tenferro-graded

Block-sparse tensor with sector labels:

```rust
pub struct GradedTensor<T: Scalar, S: Sector> {
    inner: BlockSparseTensor<T>,
    sector_labels: Vec<Vec<S>>,   // per-axis sector labels
}
```

Sector-aware contraction: only contract blocks where sectors are
compatible under fusion rules.

### 4.6 Krylov Solver (Separate Crate)

Required for DMRG (eigsolve), VUMPS (fixed-point), TDVP (exponentiate).
Interface:

```rust
pub trait LinearOperator<V> {
    fn apply(&self, v: &V) -> V;
}

pub fn eigsolve_lanczos<V, Op: LinearOperator<V>>(
    op: &Op, x0: &V, n_values: usize, which: WhichEigen,
) -> Vec<(f64, V)>;

pub fn exponentiate<V, Op: LinearOperator<V>>(
    op: &Op, v: &V, t: f64,
) -> V;
```

The upper layer implements `LinearOperator` for effective Hamiltonians
and transfer matrices.

---

## 5. How MPSKit Algorithms Map to This Architecture

### 5.1 DMRG Bond Update

```
MPSKit (Julia)                    Rust equivalent
--------------                    ----------------
phi = psi[b] * psi[b+1]          tensormap::contract(&ac, &ar)
phi = eigsolve(H_eff, phi)       krylov::eigsolve_lanczos(&h_eff, &phi, ...)
U, S, V = svd(phi; maxdim=...)   tensormap::svd(&phi, &trunc_strategy)
psi[b] = U                       mps.set_al(b, u)
psi[b+1] = S * V                 tensormap::contract(&s, &v)
```

Each `tensormap::contract` and `tensormap::svd` internally:
1. Iterates over valid sector blocks
2. Calls `tenferro_einsum::einsum()` or `tenferro_linalg::svd()` per block
3. Assembles results with sector metadata

### 5.2 Gauge Fixing

```
left_orth!(A)                     tensormap::qr(&a)  (per-block QR)
right_orth!(A)                    tensormap::lq(&a)  (per-block LQ)
```

### 5.3 Environment Update

```
leftenv = transfer(AL, MPO, AL')  Per-block einsum with sector matching
rightenv = transfer(AR, MPO, AR') Same pattern
```

For infinite MPS, the fixed-point equation requires a Krylov solver:
```
lambda, GL = eigsolve(TransferMatrix, GL0, :LM)
```

---

## 6. Summary

| Aspect | Assessment |
|--------|-----------|
| tenferro-rs as computation engine | **Sufficient.** No modifications needed. |
| einsum for per-block contraction | **Sufficient.** Each block is an independent `Tensor<T>`. |
| SVD/QR for per-block factorization | **Sufficient.** Returns separate tensors (U, S, Vt). |
| alpha/beta for sector accumulation | **Sufficient.** `einsum_into` supports this. |
| AD framework | **Sufficient.** `Differentiable` can be implemented for upper-layer types. |
| Basic arithmetic (add/scale/norm/dot) | **Expressible** via TensorPrims/einsum. Convenience wrappers in utility crate if needed. Keep core minimal. |
| Vector spaces, sectors, codomain/domain | **Not in tenferro-rs** (by design). Upper-layer crates. |
| Krylov solvers | **Not in tenferro-rs** (by design). Separate crate. |
| MPS/MPO algorithms | **Not in tenferro-rs** (by design). Upper-layer crate. |

The layered architecture keeps tenferro-rs focused on dense tensor
computation while allowing the full TensorKit/MPSKit feature set to be
built on top without modifying the engine.
