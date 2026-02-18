# tenferro-rs as Backend for the ITensor Ecosystem

Date: 2026-02-18

## Executive Summary

The ITensor Julia ecosystem comprises ~15 composable packages, many of
which contribute to Julia's precompilation overhead without being
performance-critical. We propose replacing the compute and storage layers
with Rust (via tenferro-rs and new companion crates), while keeping
**Index as a pure-Julia first-class object**.

Key findings:

1. Upper-layer algorithms (DMRG, TDVP, belief propagation) use
   **exclusively bulk tensor operations** — one function call per
   contraction or factorization. No per-element access in the hot path.

2. The **NamedDimsArrays layer already separates** index names (metadata)
   from array data (compute). This separation maps directly to a
   Julia/Rust boundary.

3. **Index must stay in Julia** — it is ITensor's core user-facing
   abstraction, deeply tied to Julia's type system, and not a performance
   bottleneck.

4. The existing tenferro-rs crates require **no modifications**. New
   companion crates (block-sparse, diagonal, graded) are added alongside.

---

## 1. The ITensor Ecosystem: Package Map

### 1.1 Full Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│  ALGORITHM LAYER  (tensor network algorithms)                       │
│                                                                     │
│  ITensorMPS.jl ─────── MPS/MPO, DMRG, TDVP                        │
│       │                 Uses: *, svd, eigsolve, norm, dag, prime    │
│       │                                                             │
│  ITensorNetworks.jl ── Tree TN, PEPS, belief propagation           │
│       │                 Uses: contract, svd, qr, inner, dag, prime  │
│       │                 + graph algorithms (40% of code)            │
│       │                                                             │
│  (Both delegate ALL tensor computation to ITensors.jl)              │
├─────────────────────────────────────────────────────────────────────┤
│  NAMED INDEX LAYER  (index labeling and matching)                   │
│                                                                     │
│  ITensors.jl ──────── User-facing API, Index with tags/primes      │
│       │                Wraps ITensorBase + physics site types        │
│       │                                                             │
│  ITensorBase.jl ───── ITensor struct = NamedDimsArray + IndexName   │
│       │                Adds: id, tags, prime levels to indices       │
│       │                                                             │
│  NamedDimsArrays.jl ─ NamedDimsArray{T,N} = denamed + dimnames     │
│       │                Unwraps names, delegates to TensorAlgebra     │
│       │                                                             │
│       │   contract: TA.contract(denamed(a1), names1, denamed(a2),   │
│       │                         names2)                             │
│       │   svd:      TA.svd(denamed(a), names, codomain, domain)     │
│       │                                                             │
├─────────────────────────────────────────────────────────────────────┤
│  TENSOR ALGEBRA LAYER  (storage-agnostic operations)                │
│                                                                     │
│  TensorAlgebra.jl ─── contract, svd, qr, lq, eigen, polar, exp     │
│       │                Generic over AbstractArray                    │
│       │                Pattern: matricize → matrix op → unmatricize  │
│       │                Dispatches by FusionStyle to handle           │
│       │                  dense (ReshapeFusion) vs                    │
│       │                  block-sparse (BlockReshapeFusion)           │
│       │                                                             │
│  MatrixAlgebraKit ─── Matrix-level SVD, QR, LU (LAPACK wrapper)     │
│       (external)                                                    │
├─────────────────────────────────────────────────────────────────────┤
│  SYMMETRY / GRADING LAYER  (quantum number metadata)                │
│                                                                     │
│  GradedArrays.jl ──── GradedArray = BlockSparseArray + graded axes  │
│       │                Adds: sector labels on blocks, flux           │
│       │                SVD: unfluxify → blockdiagonalize → SVD →     │
│       │                     fluxify                                  │
│       │                Uses TensorKitSectors interface (NOT          │
│       │                SymmetrySectors directly)                     │
│       │                                                             │
│  SymmetrySectors.jl ─ U(1), SU(2), SU(N), Fibonacci, Ising, ...    │
│       │                Fusion rules, quantum dimensions              │
│       │                Pure metadata (~600 LOC, no arrays)           │
│       │                                                             │
├─────────────────────────────────────────────────────────────────────┤
│  STORAGE LAYER  (array data structures)                             │
│                                                                     │
│  BlockSparseArrays.jl  Block-structured sparse arrays               │
│       │                  blocks: SparseArrayDOK of dense matrices    │
│       │                  axes: BlockedUnitRange                      │
│       │                  Per-block factorizations via                │
│       │                    BlockPermutedDiagonalAlgorithm            │
│       │                                                             │
│  SparseArraysBase.jl ─ SparseArrayDOK (dictionary-of-keys)          │
│       │                  stored/unstored separation                  │
│       │                  Style-based dispatch                        │
│       │                                                             │
│  DiagonalArrays.jl ─── DiagonalArray = diag vector + unstored       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE LAYER  (Julia-specific plumbing)                    │
│                                                                     │
│  TypeParameterAccessors.jl ── Julia type parameter reflection       │
│  FunctionImplementations.jl ─ Style-based dispatch patterns         │
│  DerivableInterfaces.jl ───── Interface derivation (Rust derive)    │
│  BlockArrays.jl ─────────── Block axis structure (external)         │
│  FillArrays.jl ──────────── Zeros, Ones, Fill (external)            │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 What Each Layer Does

| Layer | Packages | Responsibility | Compute? |
|-------|----------|----------------|----------|
| Algorithm | ITensorMPS, ITensorNetworks | DMRG loops, graph algorithms, contraction ordering | Orchestration only |
| Named Index | ITensors, ITensorBase, NamedDimsArrays | Index labeling, matching, prime/dag | Metadata only |
| Tensor Algebra | TensorAlgebra, MatrixAlgebraKit | contract, SVD, QR, eigen | **Yes — bulk ops** |
| Symmetry | GradedArrays, SymmetrySectors | QN labels, fusion rules, flux | Metadata + orchestration |
| Storage | BlockSparseArrays, SparseArraysBase, DiagonalArrays | Block structure, sparse indexing | **Yes — per-block ops** |
| Infrastructure | TypeParameterAccessors, FunctionImplementations, ... | Julia type system glue | None |

### 1.3 Data Flow: One DMRG Bond Update

```
ITensorMPS.jl
    phi = psi[b] * psi[b+1]          ──→  1 contract call
    phi = eigsolve(H, phi)            ──→  ~10-50 contract calls (Krylov)
    U, S, V = svd(phi; maxdim=...)    ──→  1 svd call
    psi[b] = U
    psi[b+1] = S * V                 ──→  1 contract call

Each of the above goes through:
    ITensors  →  NamedDimsArrays  →  TensorAlgebra  →  BLAS/LAPACK
    (unwrap names)  (unwrap names)  (matricize+compute) (actual math)
```

**Total: ~15-55 bulk tensor operations per bond update.**
**Per-element access: zero in the hot path.**

---

## 2. tenferro-rs: Current Capabilities

```
tenferro-rs workspace
├── tenferro-tensor ──── Dense Tensor<T>, zero-copy views
├── tenferro-einsum ──── Einstein summation, contraction tree optimization
├── tenferro-linalg ──── Batched SVD, QR, LU, eigen (with AD rules)
├── tenferro-prims ───── Plan-based primitives (batched GEMM, reduce, trace)
├── tenferro-capi ────── C-FFI: DLPack v1.0, opaque handles, status codes
├── tenferro-device ──── Device abstraction (CPU / CUDA / HIP)
├── tenferro-algebra ─── Algebra traits (Standard, Semiring)
├── chainrules-core ──── AD traits (Differentiable, ReverseRule, ForwardRule)
└── chainrules ───────── AD engine (TrackedTensor, DualTensor, tape)

Key properties:
  - Dense tensors only (no sparse / block-sparse / diagonal)
  - C-FFI with DLPack zero-copy exchange
  - AD primitives for einsum and SVD (rrule / frule)
  - GPU-ready architecture (device abstraction)
```

---

## 3. Proposal: Index in Julia, Everything Else in Rust

### 3.1 Design Constraints

**Constraint 1: Index must stay in Julia.**

ITensor's defining abstraction is the `Index`. Users create, inspect, tag,
prime, and reason about Indices constantly:

```julia
i = Index(2, "Site,S=1/2")       # create with tags
j = Index(10, "Link")            # create with tags
A = random_itensor(i, j)         # construct tensor
B = random_itensor(j', i')       # primed indices
C = A * B                        # contract: j matched to j' automatically
U, S, V = svd(C, i)              # specify decomposition by Index
```

Index is:
- **User-facing** — physicists write code in terms of Indices
- **Extensible** — custom tags, QN sectors added via Julia's type system
- **Not a bottleneck** — metadata operations (prime, dag, match) are O(1)
- **Julia ecosystem integrated** — printing, hashing, comparison, dispatch

Moving Index to Rust would break the user experience without performance
benefit.

**Constraint 2: Precompilation time must be minimized.**

The current ITensor ecosystem has ~15 Julia packages between the user and
BLAS. Each package adds precompilation time. By moving everything except
Index and algorithm orchestration to Rust (ahead-of-time compiled, zero
precompile), we eliminate this overhead.

**Constraint 3: The Rust-Julia boundary must be simple.**

A boundary buried deep in the stack (e.g., at BlockSparseArrays or
TensorAlgebra) would create a complex interface with Julia-specific types
(BlockedUnitRange, FusionStyle, Unstored) leaking across FFI. Placing the
boundary high — directly under Index management — keeps the interface to
simple C types: opaque handles + integer arrays (permutations, dimension
indices).

### 3.2 Where the Boundary Sits

```
                                                     Julia │ Rust
                                                           │
  ITensorMPS.jl ── DMRG, TDVP                             │
  ITensorNetworks.jl ── graph TN                           │
  ITensors.jl ── user API, physics sites                   │
  ITensorBase.jl ── ITensor struct                         │
    Index ── id, tags, prime, QN (PURE JULIA)              │
    index matching ── which dims to contract (JULIA)       │
    permutation computation (JULIA, trivial metadata)      │
  ═══════════════════════════════════════ FFI boundary ═════╡
                                                           │
    TensorHandle (opaque) ─ Dense / BlockSparse / Diag /   │
                             Graded                        │
    contract(h1, perm1, h2, perm2)                         │
    svd(h, codomain_dims)                                  │
    qr, norm, add, permutedims, ...                        │
    (Rust has NO concept of Index. Only shapes and perms.) │
                                                           │
```

### 3.3 What Lives Where

```
┌─ Julia ──────────────────────────────────────────────────────────────┐
│                                                                      │
│  ITensorMPS.jl         DMRG sweep loop, Krylov eigsolve             │
│  ITensorNetworks.jl    graph traversal, contraction ordering         │
│  ITensors.jl           user API, physics site types                  │
│       │                                                              │
│  ITensorBase.jl        ITensor = Index[] + Rust TensorHandle         │
│       │                                                              │
│       │  Index (PURE JULIA, non-negotiable):                        │
│       │    struct Index                                               │
│       │      id::UInt64                                               │
│       │      tags::TagSet        ← extensible, user-defined          │
│       │      plev::Int           ← prime level                       │
│       │      space::Any          ← dim, QN sectors, continuous, etc. │
│       │    end                                                        │
│       │                                                              │
│       │  contract(A::ITensor, B::ITensor):                           │
│       │    1. Match indices by id  (Julia, O(n²) on ~4 indices)      │
│       │    2. Compute permutations (Julia, trivial metadata)         │
│       │    3. tfe_contract(A.handle, perms, B.handle, perms) ← 1 FFI│
│       │    4. Attach output Index[] (Julia)                          │
│       │                                                              │
│       │  svd(A::ITensor, left_inds::Index...):                      │
│       │    1. Map left_inds to dimension numbers (Julia)             │
│       │    2. tfe_svd(A.handle, dims) ← 1 FFI                      │
│       │    3. Create new Index for bond dimension (Julia)            │
│       │    4. Wrap U, S, V handles with Index[] (Julia)             │
│       │                                                              │
│       │  prime(A::ITensor):                                         │
│       │    Index[] updated in Julia (no FFI needed)                  │
│       │    handle unchanged                                          │
│       │                                                              │
│       │  dag(A::ITensor):                                           │
│       │    Index[] flipped in Julia                                   │
│       │    handle: tfe_conj(A.handle) or lazy flag                   │
│       │                                                              │
│  ┌─ FFI boundary ──────────────────────────────────────────────────┐ │
│  │  Rust never sees Index objects.                                  │ │
│  │  It receives only: opaque handle + integer permutations.        │ │
│  │                                                                  │ │
│  │  contract(handle_A, perm_A, handle_B, perm_B) → handle_C       │ │
│  │  svd(handle, codomain_dims[]) → (handle_U, handle_S, handle_V) │ │
│  │  norm(handle) → Float64                                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

┌─ Rust (tenferro-rs + new companion crates) ──────────────────────────┐
│                                                                      │
│  TensorHandle (opaque, enum-dispatched)                              │
│    ├── Dense ──────────── tenferro-tensor Tensor<T>                  │
│    ├── BlockSparse ────── new: block structure + dense blocks        │
│    ├── Diagonal ───────── new: diagonal vector                       │
│    └── Graded ─────────── new: block-sparse + sector metadata        │
│                                                                      │
│  Operations (storage-polymorphic, selected by enum tag):             │
│    contract ─── Dense: tenferro-einsum                               │
│              ── BlockSparse: iterate block pairs → batched GEMM      │
│              ── Diagonal: element-wise multiply                      │
│              ── Graded: sector-aware block contraction                │
│                                                                      │
│    svd ──────── Dense: tenferro-linalg                               │
│              ── BlockSparse: per-block SVD                            │
│              ── Graded: blockdiag SVD                                 │
│                                                                      │
│    norm, add, permutedims, copy, scale, ...                          │
│                                                                      │
│  AD primitives:                                                      │
│    einsum_rrule, svd_rrule, einsum_frule, svd_frule                  │
│                                                                      │
│  Device management:                                                  │
│    CPU / GPU transparent to Julia                                    │
│                                                                      │
│  (Rust has NO concept of Index, tags, prime levels, or QN naming.    │
│   It only knows shapes, permutations, block ranges, and sector ids.) │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 Why This Boundary

#### Precompilation: Maximum Reduction

```
Packages ELIMINATED from Julia precompilation:
  TensorAlgebra.jl         ← contract, svd, qr dispatch + matricize
  BlockSparseArrays.jl     ← block structure, 5 type parameters
  SparseArraysBase.jl      ← DOK, style dispatch, zero-preserving traits
  DiagonalArrays.jl        ← diagonal + unstored
  GradedArrays.jl          ← graded structure, flux, unfluxify
  SymmetrySectors.jl       ← fusion rules, quantum dimensions
  TypeParameterAccessors.jl ← @generated functions, type reflection
  FunctionImplementations.jl ← style dispatch metaprogramming
  DerivableInterfaces.jl   ← interface derivation
  MatrixAlgebraKit (dep)   ← LAPACK wrapping
  BlockArrays.jl (dep)     ← BlockedUnitRange, block indexing
  FillArrays.jl (dep)      ← Zeros, Ones

Packages REMAINING in Julia:
  ITensors.jl              ← thin: Index + user API
  ITensorBase.jl           ← thin: ITensor struct + FFI wrappers
  ITensorMPS.jl            ← algorithms (unchanged)
  ITensorNetworks.jl       ← algorithms (unchanged)
```

The eliminated packages are the ones with the most complex type
hierarchies (5-parameter generics, @generated functions, trait-based
dispatch) — exactly what makes Julia precompilation slow.

#### FFI Boundary: Maximum Simplicity

The interface between Julia and Rust consists of:

| From Julia | Type | Example |
|------------|------|---------|
| Tensor handle | Opaque pointer | `Ptr{TfeTensorHandle}` |
| Permutation | `Vector{Int32}` | `[2, 0, 3, 1]` |
| Codomain dims | `Vector{Int32}` | `[0, 2]` |
| Scalars | `Float64` | `1.0`, `0.0` |
| Shape | `Vector{Int64}` | `[2, 3, 4]` |

No Julia-specific types cross the boundary. No `BlockedUnitRange`, no
`FusionStyle`, no `Unstored`, no `SparseArrayDOK`. Just integers and
pointers.

Compare this with a boundary at the BlockSparseArrays level, which would
require serializing block structure metadata, axis partitioning, and
stored/unstored semantics across FFI — a much more complex interface that
would also require keeping TensorAlgebra.jl, GradedArrays.jl, and
SymmetrySectors.jl in Julia (precompilation overhead remains).

#### Index Matching: Trivial Metadata, Stays in Julia

```julia
function contract(A::ITensor, B::ITensor)
    # Step 1: Find shared indices (Julia, ~10 μs for 4 indices)
    shared = intersect(inds(A), inds(B))  # match by id
    free_A = setdiff(inds(A), shared)
    free_B = setdiff(inds(B), shared)

    # Step 2: Compute permutations (Julia, ~1 μs)
    perm_A = [findfirst(==(idx), inds(A)) for idx in (free_A..., shared...)]
    perm_B = [findfirst(==(idx), inds(B)) for idx in (shared..., free_B...)]

    # Step 3: Call Rust (1 FFI call, ~1 ms for typical tensor)
    result_handle = tfe_contract(A.handle, perm_A, B.handle, perm_B)

    # Step 4: Attach output indices (Julia, ~1 μs)
    result_inds = (free_A..., free_B...)
    return ITensor(result_handle, result_inds)
end
```

Steps 1, 2, 4 are O(n²) on the number of indices (typically 2-6).
Step 3 is the actual computation. Keeping steps 1/2/4 in Julia adds
~10 μs overhead to a ~1 ms operation — negligible.

### 3.5 FFI Surface

```c
/* Opaque handle (wraps Dense / BlockSparse / Diagonal / Graded) */
typedef struct TfeTensorHandle TfeTensorHandle;

/* Core operations — ONE call each */
int tfe_contract(TfeTensorHandle *h1, const int *perm1, int n1,
                 TfeTensorHandle *h2, const int *perm2, int n2,
                 TfeTensorHandle **result, tfe_status_t *status);

int tfe_svd(TfeTensorHandle *h,
            const int *codomain_dims, int n_codomain,
            TfeTensorHandle **u, TfeTensorHandle **s, TfeTensorHandle **vt,
            tfe_status_t *status);

int tfe_qr(TfeTensorHandle *h,
           const int *codomain_dims, int n_codomain,
           TfeTensorHandle **q, TfeTensorHandle **r,
           tfe_status_t *status);

double tfe_norm(TfeTensorHandle *h, tfe_status_t *status);

int tfe_add(TfeTensorHandle *dest, TfeTensorHandle *src,
            double alpha, double beta, tfe_status_t *status);

int tfe_permutedims(TfeTensorHandle *h, const int *perm, int n,
                    TfeTensorHandle **result, tfe_status_t *status);

int tfe_conj(TfeTensorHandle *h,
             TfeTensorHandle **result, tfe_status_t *status);

double tfe_scalar(TfeTensorHandle *h, tfe_status_t *status);

/* Construction */
int tfe_from_dense_dlpack(DLManagedTensorVersioned *dl,
                          TfeTensorHandle **result, tfe_status_t *status);

int tfe_from_blocksparse(const int64_t *block_ranges, int n_dims,
                         const int64_t *stored_indices, int n_stored,
                         const double **block_data, const int64_t *block_sizes,
                         TfeTensorHandle **result, tfe_status_t *status);

int tfe_zeros(const int64_t *shape, int ndim,
              TfeTensorHandle **result, tfe_status_t *status);

int tfe_random(const int64_t *shape, int ndim,
               TfeTensorHandle **result, tfe_status_t *status);

/* Metadata query */
int    tfe_ndim(TfeTensorHandle *h);
void   tfe_shape(TfeTensorHandle *h, int64_t *dims_out);
int    tfe_storage_type(TfeTensorHandle *h);  /* Dense=0, BlockSparse=1, ... */

/* Block structure query (for constructing output Index with QN) */
int    tfe_n_stored_blocks(TfeTensorHandle *h);
void   tfe_stored_block_indices(TfeTensorHandle *h, int64_t *out);
void   tfe_block_range(TfeTensorHandle *h, int dim, int block,
                       int64_t *start, int64_t *stop);

/* Lifecycle */
void   tfe_release(TfeTensorHandle *h);
int    tfe_clone(TfeTensorHandle *h, TfeTensorHandle **result,
                 tfe_status_t *status);
```

### 3.6 Julia-Side Changes

```julia
# ITensorBase.jl — the ONLY Julia package that changes significantly

struct ITensor
    handle::Ptr{TfeTensorHandle}   # opaque Rust pointer
    inds::Vector{Index}            # pure Julia Index objects
end

# contract: Index matching in Julia, compute in Rust
function Base.:*(A::ITensor, B::ITensor)
    perm_A, perm_B, out_inds = compute_contraction(inds(A), inds(B))
    result = tfe_contract(A.handle, perm_A, B.handle, perm_B)
    return ITensor(result, out_inds)
end

# svd: Index → dim mapping in Julia, decomposition in Rust
function svd(A::ITensor, left_inds::Index...; maxdim=nothing, cutoff=nothing)
    dims = [findfirst(==(idx), inds(A)) for idx in left_inds]
    hu, hs, hv = tfe_svd(A.handle, dims)
    bond_dim = tfe_shape(hs, ...)[1]
    link = Index(bond_dim, "Link")
    return ITensor(hu, [left_inds..., link]),
           ITensor(hs, [link, link']),
           ITensor(hv, [link', remaining_inds...])
end

# prime: purely Julia, no FFI needed
function prime(A::ITensor, n::Int=1)
    new_inds = [prime(idx, n) for idx in inds(A)]
    return ITensor(A.handle, new_inds)  # same Rust handle!
end

# dag: Index flip in Julia, conjugate in Rust (or lazy)
function dag(A::ITensor)
    new_inds = [dag(idx) for idx in inds(A)]
    new_handle = tfe_conj(A.handle)
    return ITensor(new_handle, new_inds)
end

# norm: pure Rust
norm(A::ITensor) = tfe_norm(A.handle)
```

### 3.7 Scorecard

| Metric | Value |
|--------|-------|
| New Rust code | ~8K LOC (block-sparse + diagonal + graded + sectors + FFI) |
| Julia packages changed | ITensorBase.jl (Index matching + FFI wrappers) |
| Julia packages eliminated | NamedDimsArrays, TensorAlgebra, BlockSparseArrays, SparseArraysBase, DiagonalArrays, GradedArrays, SymmetrySectors, TypeParameterAccessors, FunctionImplementations, DerivableInterfaces |
| Julia packages unchanged | ITensors.jl (user API), ITensorMPS.jl, ITensorNetworks.jl |
| Index | **Pure Julia — non-negotiable** |
| FFI calls per DMRG step | ~3 (contract + svd + contract), constant regardless of storage |
| FFI surface complexity | Simple: opaque handles + integer arrays |
| Precompilation reduction | **Maximum** — 10+ Julia packages eliminated |

---

## 4. Relationship to tenferro-rs: No Changes to Existing Code

### 4.1 Principle

All existing tenferro-rs crates are used **as-is**. No modifications.
New companion crates are added alongside.

### 4.2 How Existing Crates Are Reused

```
tenferro-rs (UNCHANGED)
│
├── tenferro-tensor ── Tensor<T> data type
│     ↑ used as block data type by block-sparse-tensor
│
├── tenferro-prims ─── TensorPrims<A> trait, batched_gemm, reduce, trace
│     ↑ called per block pair during block-sparse contraction
│     ↑ accepts StridedView<T> — each block's data passed directly
│
├── tenferro-einsum ── einsum for dense Tensor<T>
│     ↑ called for Dense path (entire tensor passed)
│     ↑ NOT called for block-sparse (different algorithm needed)
│
├── tenferro-linalg ── SVD, QR, LU, eigen for dense Tensor<T>
│     ↑ called per block during block-sparse SVD
│     ↑ called for entire tensor on Dense path
│
├── tenferro-capi ──── C-FFI for dense operations, DLPack
│     ↑ Dense path reuses existing functions
│
├── tenferro-device ── LogicalMemorySpace, ComputeDevice
│     ↑ shared by all new crates for CPU/GPU abstraction
│
├── chainrules-core ── Differentiable, ReverseRule, ForwardRule
│     ↑ new crates implement Differentiable for their types
│
└── chainrules ─────── TrackedTensor, DualTensor, tape
      ↑ AD engine shared across Dense/BlockSparse/Diagonal/Graded
```

### 4.3 New Companion Crates

```
tenferro-rs workspace
│
├── [existing crates — UNCHANGED]
│
└── NEW companion crates:
    │
    ├── block-sparse-tensor/          (~3K LOC)
    │     [dependencies]
    │     tenferro-tensor, tenferro-prims, tenferro-linalg,
    │     tenferro-device, chainrules-core
    │
    │     pub struct BlockSparseTensor<T> {
    │         block_ranges: Vec<Vec<Range<usize>>>,
    │         stored: HashMap<Vec<usize>, Tensor<T>>,
    │     }
    │     pub fn contract(...)   // iterate block pairs → batched GEMM
    │     pub fn svd(...)        // per-block SVD
    │     impl Differentiable for BlockSparseTensor<T>
    │
    ├── diag-tensor/                  (~500 LOC)
    │     pub struct DiagTensor<T> { diag: Vec<T>, shape: Vec<usize> }
    │
    ├── sectors/                      (~600 LOC)
    │     pub trait Sector: Clone + Eq + Hash { ... }
    │     pub struct U1(i64);
    │     pub struct SU2(HalfInt);
    │     pub fn fusion_rule(s1, s2) → Vec<(S, usize)>
    │
    ├── graded-tensor/                (~2K LOC)
    │     [dependencies]
    │     block-sparse-tensor, sectors
    │
    │     pub struct GradedTensor<T, S: Sector> {
    │         inner: BlockSparseTensor<T>,
    │         sectors: Vec<Vec<S>>,
    │     }
    │     pub fn contract(...)   // sector-aware block contraction
    │     pub fn svd(...)        // blockdiag SVD with flux
    │
    └── tensor-handle-capi/           (~1.5K LOC, unified FFI)
          [dependencies]
          tenferro-capi, block-sparse-tensor, diag-tensor, graded-tensor

          pub enum TensorHandle<T> {
              Dense(Tensor<T>),
              BlockSparse(BlockSparseTensor<T>),
              Diagonal(DiagTensor<T>),
              Graded(GradedTensor<T, DynSector>),
          }

          #[no_mangle] pub extern "C" fn tfe_contract(...)
          #[no_mangle] pub extern "C" fn tfe_svd(...)
          #[no_mangle] pub extern "C" fn tfe_norm(...)
          // enum dispatch: match handle variant → call appropriate impl
```

### 4.4 Workspace Configuration

```toml
# tenferro-rs/Cargo.toml
[workspace]
members = [
    # existing (unchanged)
    "tenferro-device",
    "tenferro-algebra",
    "tenferro-prims",
    "tenferro-tensor",
    "tenferro-einsum",
    "tenferro-linalg",
    "tenferro-capi",
    "extern/chainrules-core",
    "extern/chainrules",
    # ...

    # new companion crates
    "block-sparse-tensor",
    "diag-tensor",
    "sectors",
    "graded-tensor",
    "tensor-handle-capi",
]
```

---

## 5. Implementation Effort

```
Component                      LOC     Notes
─────────────────────────      ─────   ──────────────────────────────
block-sparse storage           1.5K    HashMap<BlockIdx, Tensor<T>>
block-sparse contraction       1.5K    iterate pairs → batched GEMM
block-sparse factorizations    1.0K    per-block SVD/QR via tenferro-linalg
diagonal tensor                0.5K    Vec<T> + shape
sectors (U1, SU2, SUN)         0.6K    trait + 3 concrete types
graded tensor                  2.0K    BlockSparseTensor + sector labels
tensor-handle-capi (FFI)       1.5K    enum dispatch + extern "C" functions
─────────────────────────      ─────
Total new Rust                 ~8.5K

Julia changes:
  ITensorBase.jl               ~500    ITensor struct + FFI wrappers
  ITensors.jl                  ~200    Adjust constructors, re-export
─────────────────────────      ─────
Total Julia changes            ~700
```
