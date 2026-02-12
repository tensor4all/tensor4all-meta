# t4a: Unified Tensor Backend Design Plan

> **Note on naming**: `t4a` is a **working name** (short for tensor4all). The
> final project name is undecided. One candidate is
> **[tensoratu](https://gricad-gitlab.univ-grenoble-alpes.fr/theorypheliqs/tensoratu)**,
> proposed by the Grenoble team (TheoryPheliqs) as a tensor toolkit with
> hybrid indexing for tensor networks. Adopting `tensoratu` would change the
> crate prefix from `t4a-*` to `tensoratu-*` (e.g., `tensoratu-view`,
> `tensoratu-omeinsum`). Throughout this document, `t4a-*` should be read
> as a placeholder for whatever prefix is ultimately chosen.

> **Companion documents**:
> - [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md) — strided-rs vs omeinsum-rs optimization comparison
> - [t4a Einsum Internal Design](./t4a_einsum_internal_design.md) — detailed internal design of t4a-tensorops and t4a-einsum

## Context

Four independent Rust projects exist in tensor4all:
- **strided-rs**: Cache-optimized strided array kernels (view, map/reduce, einsum)
- **omeinsum-rs**: Einsum with tropical algebra, gradient support, GPU dispatch
- **ndtensors-rs**: Tensor types with storage hierarchy, linear algebra, autograd
- **tensor4all-rs**: Tensor network algorithms (TCI, Quantics, MPS) with ad-hoc tensor backend

These have significant overlap (3 einsum implementations, 3 scalar trait definitions, 3 dense storage types) yet critical gaps. The goal is to unify into a coherent, reusable tensor backend library **t4a-\*** that:

1. Integrates strided-rs and omeinsum-rs components directly (not as external dependencies)
2. Provides unified CPU/GPU dispatch via a **cuTENSOR/hipTensor-compatible protocol** (`t4a-tensorops`)
3. Supports both NVIDIA and AMD GPUs via **runtime library loading** (no compile-time vendor lock-in)
4. Supports complex numbers natively
5. Supports custom scalar types (tropical semiring, etc.) with pluggable backends
6. Exposes VJP/JVP through C API for Julia ChainRules.jl
7. Can optionally bridge to Burn for NN workloads

**Key design principles**:
- **strided-rs as foundation**: The general-purpose strided array crates (`strided-traits`, `strided-view`, `strided-kernel`) remain in an independent `strided-rs` workspace. They have no BLAS dependency and can be used standalone. `t4a-rs` depends on them but does not absorb them.
- **cuTENSOR/hipTensor-compatible protocol**: `t4a-tensorops` defines a unified `TensorOps` trait mirroring cuTENSOR's five operation categories (contraction, reduction, permutation, elementwise binary/trinary). CPU, NVIDIA, and AMD backends implement the same trait.
- **Runtime GPU discovery**: GPU vendor libraries (cuTENSOR, hipTensor) are loaded at runtime via `dlopen`. The caller (Julia, Python) provides the `.so` path. No Cargo feature flags for GPU vendor selection.
- **Plan-based execution**: All operations follow the cuTENSOR pattern of descriptor → plan → execute. Plans cache expensive analysis (GPU kernel selection, CPU fusability checks) for reuse.
- **t4a-view is pure type design**: "strided" is about data layout, not operations. The view crate has no GPU dependencies.

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Application                                         │
│   tensor4all-rs (TCI, Quantics, MPS algorithms)              │
│   Julia / Python (C API via t4a-capi)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 3: Einsum Engine (t4a-einsum)                        │
│   N-ary contraction tree optimization (omeco)                │
│   Algebra dispatch (Standard, Tropical, custom)              │
│   Backward pass (VJP/JVP)                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 2: Tensor Operation Protocol (t4a-tensorops)           │
│   cuTENSOR / hipTensor compatible TensorOps trait            │
│   5 operations: Contraction, Reduction, Permutation,         │
│                 ElementwiseBinary, ElementwiseTrinary         │
│   Plan-based execution (create_plan → execute)               │
│   BackendRegistry (runtime GPU discovery via dlopen)         │
│   Caller-injected .so paths (no auto-search)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 1: Backend Implementations                             │
│                                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ CPU              │  │ NVIDIA       │  │ AMD          │    │
│  │                  │  │              │  │              │    │
│  │ Contraction:     │  │ cuTENSOR     │  │ hipTensor    │    │
│  │  strided-einsum2 │  │ (dlopen)     │  │ (dlopen)     │    │
│  │  fusability      │  │              │  │              │    │
│  │  trace reduction │  │ Common vtable│  │              │    │
│  │  EW bypass       │  │ (1:1 API)    │  │              │    │
│  │                  │  │              │  │              │    │
│  │ GEMM (feature):  │  │              │  │              │    │
│  │  faer (default)  │  │              │  │              │    │
│  │  cblas (opt-in)  │  │              │  │              │    │
│  │                  │  │              │  │              │    │
│  │ Elementwise:     │  │              │  │              │    │
│  │  enum-based      │  │              │  │              │    │
│  │  (strided-kernel │  │              │  │              │    │
│  │   cache-opt)     │  │              │  │              │    │
│  └─────────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Crate Structure

```
strided-rs/ (independent workspace) ── Stays as-is ───────
│  General-purpose strided array library. No BLAS dependency.
│  Can be used standalone by projects other than t4a.
│
├── strided-traits       # ScalarBase, ElementOp traits
├── strided-view         # StridedArrayView/Mut (zero-copy strided views)
└── strided-kernel       # Cache-optimized map/reduce/broadcast kernels

t4a-rs/ (workspace) ── Dense array foundation ────────────
│  Depends on strided-rs. Absorbs strided-einsum2 + strided-opteinsum + omeinsum-rs.
│
├── t4a-scalar           # Extends strided-traits: adds Scalar (division, complex), RealScalar
├── t4a-view             # Re-exports strided-view (thin wrapper)
├── t4a-buffer           # DataBuffer<T>: CPU Vec<T> / GPU device memory (Arc-based COW)
├── t4a-algebra          # Semiring/Algebra traits, tropical types (MaxPlus, MinPlus, MaxMul)
├── t4a-tensorops        # Unified tensor operation protocol (TensorOps trait)
│                        #   cuTENSOR/hipTensor compatible API (enum-based, 5 operations)
│                        #   BackendRegistry, runtime GPU discovery
│                        #   Plan-based execution
│                        #   CPU impl uses strided-kernel internally
│                        #   Custom closures: use strided-kernel directly via Tensor::as_strided_view()
├── t4a-einsum           # Einsum engine (absorbs strided-einsum2 + strided-opteinsum + omeinsum-rs)
│                        #   N-ary optimizer, algebra dispatch, backward
│                        #   Delegates binary contraction to t4a-tensorops
├── t4a-tensor           # Tensor<T> = DataBuffer + sizes + strides + offset
│                        #   User-facing API
├── t4a-linalg           # SVD, QR, eigen, polar (CPU: faer, GPU: cuSOLVER)
├── t4a-autograd         # TrackedTensor, DualTensor, VJP/JVP
├── t4a-capi             # C FFI (tensor ops + VJP/JVP + backend loading)
└── burn-t4a             # Burn Backend bridge [OPTIONAL, for NN only]

t4a-structured-rs/ (workspace) ── Structured tensor types ──
│
├── t4a-blocksparse      # BlockSparseTensor (single DataBuffer + block offsets)
├── t4a-diag             # DiagTensor (1D Tensor of diagonal elements)
└── t4a-graded           # GradedTensor (future: quantum number sectors)

tensor4all-rs/ (workspace) ── Tensor network algorithms ────
│
├── TCI, Quantics, MPS, ...
└── depends on t4a-rs + t4a-structured-rs
```

### Dependency Graph

```
strided-rs (independent workspace):
strided-traits → strided-view → strided-kernel

t4a-rs (workspace, depends on strided-rs):

t4a-scalar (← strided-traits)
    │
    ├──────────────────────────────┐
    ↓                              ↓
t4a-view (← strided-view)    t4a-buffer
    │                              │
    ├──── t4a-algebra ←────────────┤
    │                              │
    ↓                              ↓
t4a-tensorops ←────────────────────┘  (TensorOps trait, BackendRegistry,
    │  (← strided-kernel)             libloading, GPU vtable)
    │
    ↓
t4a-einsum (absorbs strided-einsum2 +
    │       strided-opteinsum + omeinsum-rs)
    ↓
t4a-tensor
                   │
         ┌─────────┼───────────┐
         ↓         ↓           ↓
   t4a-linalg  t4a-autograd  t4a-capi
   (← faer)

[separate workspace: t4a-structured-rs]
t4a-blocksparse ← t4a-tensor
t4a-diag        ← t4a-tensor
t4a-graded      ← t4a-blocksparse (future)

[optional]
burn-t4a ← t4a-tensor, burn-backend
```

### Origin of Each Crate

| t4a crate | Origin | What changes |
|-----------|--------|--------------|
| t4a-scalar | Depends on strided-traits | Extends with `Scalar` (division, complex), `RealScalar` |
| t4a-view | Depends on strided-view | Thin re-export wrapper |
| t4a-buffer | New | CPU/GPU buffer abstraction |
| t4a-algebra | omeinsum-rs (Algebra traits) | Standalone crate for Semiring/tropical types |
| t4a-tensorops | **New** (replaces t4a-backend) | cuTENSOR/hipTensor-compatible `TensorOps` trait (enum-based only); CPU impl uses strided-kernel |
| t4a-einsum | **Absorbs** strided-einsum2 + strided-opteinsum + omeinsum-rs | Merge all einsum; delegates binary contraction to `TensorOps` |
| t4a-tensor | New | Tensor<T> API over DataBuffer |
| t4a-linalg | ndtensors-rs (linalg) | Port SVD/QR/eigen |
| t4a-autograd | ndtensors-rs (autodiff) | Port TrackedTensor/DualTensor |
| t4a-capi | ndtensors-rs (capi) + tensor4all-rs (capi) | Port C FFI + backend loading API |
| burn-t4a | New | Burn Backend bridge |
| **t4a-structured-rs (separate workspace):** | | |
| t4a-blocksparse | ndtensors-rs (blocksparse) | Port with single-buffer layout |
| t4a-diag | ndtensors-rs (diag) | Port DiagTensor |
| t4a-graded | New (future) | Quantum number graded tensors |

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** (dlopen) | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** (Cargo feature) | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based only** | cuTENSOR-compatible operator enums; custom closures via strided-kernel directly |
| libloading dependency | **Always ON** | Lightweight, no overhead when GPU absent, no feature gate needed |
| .so path for GPU libs | **Caller-injected** | Rust does not search; Julia/Python provide exact path |

---

## Phase 1: Dense Array Foundation

### t4a-scalar

```rust
/// Minimal scalar trait for map/reduce/einsum.
pub trait ScalarBase: Copy + Send + Sync + Add<Output=Self> + Mul<Output=Self>
    + Zero + One + PartialEq + Debug {}

/// Element operation applied lazily on access (Identity, Conj, Transpose, Adjoint).
pub trait ElementOpApply: ScalarBase {
    fn conjugate(self) -> Self;
    fn transpose(self) -> Self;
    fn adjoint(self) -> Self;
}

/// Rich scalar for linalg and standard numeric types.
pub trait Scalar: ScalarBase + ElementOpApply
    + Div<Output=Self> + Sub<Output=Self> + Neg<Output=Self> + SubAssign + DivAssign
{
    type Real: RealScalar;
    fn conjugate(self) -> Self;
    fn real_part(self) -> Self::Real;
    fn imag_part(self) -> Self::Real;
    fn abs_squared(self) -> Self::Real;
    fn from_real(re: Self::Real) -> Self;
    fn is_complex() -> bool;
}

pub trait RealScalar: Scalar<Real = Self> + PartialOrd {
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn to_f64(self) -> f64;
    fn from_f64(val: f64) -> Self;
}
```

**Implementations**:

| Trait | Types |
|-------|-------|
| `ScalarBase` | f32, f64, Complex\<f32\>, Complex\<f64\>, i32, i64, u32, u64 |
| `ElementOpApply` | f32, f64, Complex\<f32\>, Complex\<f64\> |
| `Scalar` | f32, f64, Complex\<f32\>, Complex\<f64\> |
| `RealScalar` | f32, f64 |

Integer types implement `ScalarBase` only — sufficient for einsum via naive
backend and map/reduce operations.

### t4a-view

Renamed from strided-view. No functional changes.

```rust
pub struct StridedArrayView<'a, T, const N: usize, Op = Identity> { ... }
pub struct StridedArrayViewMut<'a, T, const N: usize, Op = Identity> { ... }
```

- Borrows `&'a [T]` — pure CPU, no GPU dependencies
- Const-generic rank `N`
- Zero-copy: slice, reshape, permute, transpose
- Lazy element operations via `Op` type parameter (Identity, Conj, Transpose, Adjoint)

### t4a-buffer

```rust
pub struct CpuBuffer<T: ScalarBase> {
    data: Vec<T>,
}

pub struct GpuBuffer<T: ScalarBase> {
    ptr: *mut T,
    len: usize,
    device: Device,   // Cuda(id) or Rocm(id)
    // Drop handles deallocation via BackendRegistry
}

pub enum DataBuffer<T: ScalarBase> {
    Cpu(Arc<CpuBuffer<T>>),
    Gpu(Arc<GpuBuffer<T>>),
}
```

Shared ownership via `Arc`. COW via `Arc::make_mut`.

GPU buffers are not gated behind Cargo features. `GpuBuffer` is always
defined; allocation and deallocation go through `BackendRegistry` which
calls the appropriate vendor API via the loaded vtable. If no GPU library
is loaded, GPU allocation returns an error.

### t4a-algebra

Semiring/Algebra abstraction from omeinsum-rs, as a standalone crate:

```rust
pub trait Semiring: ScalarBase {
    fn sem_zero() -> Self;
    fn sem_one() -> Self;
    fn sem_add(self, rhs: Self) -> Self;
    fn sem_mul(self, rhs: Self) -> Self;
}

pub struct Standard<T>(pub T);    // sem_add = +, sem_mul = ×
pub struct MaxPlus<T>(pub T);     // sem_add = max, sem_mul = +
pub struct MinPlus<T>(pub T);     // sem_add = min, sem_mul = +
pub struct MaxMul<T>(pub T);      // sem_add = max, sem_mul = ×
```

Also provides:
- Argmax tracking for tropical backward pass
- Algebra trait extending Semiring with optional backward/gradient support

### t4a-tensorops

**This is the central protocol layer.** It defines a `TensorOps` trait
mirroring cuTENSOR/hipTensor's five operation categories with plan-based
execution. CPU, NVIDIA, and AMD backends all implement this trait.

> **Detailed design**: See [t4a Einsum Internal Design](./t4a_einsum_internal_design.md)
> for the full `TensorOps` trait definition, operator enums, CPU plan
> internals, and GPU vtable structure.

**Five operation categories** (cuTENSOR/hipTensor compatible):

| Operation | Signature pattern | CPU implementation |
|-----------|-------------------|---------------------|
| Contraction | `D = α·A·B + β·C` | strided-einsum2 pipeline |
| Reduction | `D = α·reduce(opA(A)) + β·C` | `reduce_axis` from strided-kernel |
| Permutation | `B = α·opA(A)` | `StridedView::permute` + copy |
| ElementwiseBinary | `D = α·opA(A) ⊕ γ·opC(C)` | `zip_map2_into` |
| ElementwiseTrinary | `D = α·opA(A) ⊕ β·opB(B) ⊕ γ·opC(C)` | `zip_map3_into` |

**Operator enums** (cuTENSOR compatible subset):

```rust
pub enum UnaryOp { Identity, Sqrt, Relu, Conj, Neg, Abs, Exp, Log, ... }
pub enum BinaryOp { Add, Mul, Max, Min }
pub type ReduceOp = BinaryOp;
```

**Plan-based execution** — all operations follow descriptor → plan → execute:

```rust
// Plan-based (reusable for repeated operations)
let plan = backend.create_contraction_plan::<f64>(
    &meta_a, &modes_a, &meta_b, &modes_b, &meta_c, &modes_c,
)?;
backend.contract(&plan, alpha, &a, &b, beta, &mut c)?;
backend.contract(&plan, alpha, &a2, &b2, beta, &mut c2)?;  // reuse plan

// Convenience (one-off, creates and discards plan internally)
contract_once(&backend, alpha, &a, &meta_a, &modes_a, ...)?;
```

**Runtime GPU discovery** via `BackendRegistry`:

```rust
let mut registry = BackendRegistry::new();  // CPU only

// Julia/Python provides .so path
registry.load_cutensor("/path/to/libcutensor.so")?;   // NVIDIA
registry.load_hiptensor("/path/to/libhiptensor.so")?; // AMD
```

C API for library loading:

```c
int t4a_backend_load_cutensor(const char* libcutensor_path);
int t4a_backend_load_hiptensor(const char* libhiptensor_path);
```

**GPU vtable** — cuTENSOR and hipTensor have nearly identical C APIs
(AMD intentionally mirrors NVIDIA). A single function pointer table
(`TensorLibVtable`) abstracts over both, populated at runtime from
whichever library is loaded:

```rust
impl TensorLibVtable {
    fn load_cutensor(lib: &Library) -> Result<Self> {
        Ok(Self {
            create_handle: lib.get(b"cutensorCreate")?,
            contract: lib.get(b"cutensorContract")?,
            // ...
        })
    }
    fn load_hiptensor(lib: &Library) -> Result<Self> {
        Ok(Self {
            create_handle: lib.get(b"hiptensorCreate")?,
            contract: lib.get(b"hiptensorContract")?,
            // ...
        })
    }
}
```

**Elementwise design**: enum-based only (`UnaryOp`, `BinaryOp`), cuTENSOR-compatible,
works on CPU/GPU uniformly. No closure-based API in `TensorOps`.

For **custom element-wise closures** (arbitrary user functions not in the enum),
use strided-kernel directly via `Tensor::as_strided_view()`:

```rust
// TensorOps: enum-based, works on CPU and GPU
backend.elementwise_binary(&plan, alpha, &a, gamma, &c, &mut d)?;

// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.as_strided_view();
let b_view = tensor_b.as_strided_view();
strided_kernel::zip_map2_into(&mut out_view, &a_view, &b_view, |a, b| a * b + 1.0);
```

strided-kernel provides cache-optimized iteration (dimension fusion, L1 blocking,
importance-weighted ordering) and is always available as an independent dependency.

**No Metal (Apple GPU) support**: M-series CPUs are fast enough for our
workloads (tensor network algorithms). Metal lacks a cuTENSOR-equivalent
tensor contraction library, requiring reshape+matmul decomposition that
would be slow for high-rank tensors. Not worth the implementation cost.

**CPU GEMM backend selection** (compile-time Cargo feature):

```toml
[features]
default = ["faer"]
faer = ["dep:faer"]       # Pure Rust, zero external deps (default)
cblas = ["dep:cblas-sys"]  # Requires cblas-src or cblas-inject
```

When `cblas` is selected, the CBLAS provider is supplied downstream:
- `cblas-src`: links OpenBLAS or MKL (standalone Rust apps)
- `cblas-inject`: Julia injects `libblastrampoline` symbols at runtime

### t4a-einsum

Merges strided-einsum2 + strided-opteinsum + omeinsum-rs.

**Delegates binary contraction to `t4a-tensorops`** (`TensorOps::contract`).
The einsum engine handles:
- N-ary contraction tree optimization (omeco: Greedy, TreeSA)
- Algebra-aware dispatch (Standard → GEMM, Tropical → naive/SIMD)
- Backward pass (VJP/JVP for automatic differentiation)
- Single-tensor fast paths (trace, partial trace, permutation-only)
- Buffer pool for intermediate tensor reuse

> **Detailed design**: See [t4a Einsum Internal Design](./t4a_einsum_internal_design.md)
> for the CPU contraction pipeline (6-step strided-einsum2), CPU plan
> internals, algebra dispatch, and backward pass design.

**CPU contraction pipeline** (from strided-einsum2, via `TensorOps`):
1. Trace pre-reduction — sum out axes before GEMM
2. Permutation to canonical order — `A[left, contracted, batch]`
3. Element-wise bypass — Hadamard product skips GEMM
4. Fusability check — `try_fuse_group` avoids unnecessary copies
5. GEMM dispatch — faer or CBLAS (compile-time feature)
6. Copy-back — if output non-contiguous

**Algebra dispatch**:
- `Standard<f64/f32/Complex>` → faer or CBLAS GEMM
- `Standard<i32/i64/u32/u64>` → naive loop
- `MaxPlus<f64>` → tropical-gemm (SIMD, future)
- GPU tensors → cuTENSOR/hipTensor via `TensorOps`

### t4a-tensor

```rust
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    sizes: SmallVec<[usize; 6]>,
    strides: SmallVec<[isize; 6]>,
    storage_offset: usize,
}
```

**Bridge to t4a-view** (CPU path):
```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn as_strided_view(&self) -> StridedArrayView<'_, T, N, Identity>;
    pub fn as_strided_view_mut(&mut self) -> StridedArrayViewMut<'_, T, N>;
}
```

**Zero-copy view operations**: permute, transpose, slice, select, expand, flip, diagonal — all modify metadata only.

**Einsum** (delegates to t4a-einsum):
```rust
pub fn einsum<T: ScalarBase>(
    inputs: &[&Tensor<T>],
    input_labels: &[&[i32]],
    output_labels: &[i32],
) -> Result<Tensor<T>>;
```

---

## Phase 2: t4a-linalg

All decomposition functions:
1. Call `tensor.contiguous_col_major()` (faer is column-major)
2. Create `faer::MatRef` from raw pointer + strides
3. Call faer's decomposition
4. Wrap result back into Tensor

**Operations**: svd, svd_truncated, qr, qr_positive, ql, eigen_hermitian, eigen, polar, matrix_exp.

**N-D tensor decomposition**: specify left_dims for "row" side, reshape to 2D, decompose, reshape back.

**Trait bound**: `T: Scalar + faer::ComplexField` (enforced here, not in t4a-scalar/tensor).

**GPU path**: cuSOLVER/rocSOLVER via runtime-loaded vendor library (same dlopen pattern as t4a-tensorops).

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/backend/faer_interop.rs` — faer bridge pattern
- `ndtensors-rs/crates/ndtensors/src/linalg/` — SVD, QR implementations

---

## Phase 3: t4a-autograd (Primary AD System)

This is the **default** AD system for t4a users. Burn's autodiff is only for NN workloads.

### Reverse-Mode (Backward)

```rust
pub struct TrackedTensor<T: Scalar> {
    tensor: Tensor<T>,
    node: Option<NodeRef>,
    requires_grad: bool,
}

pub fn backward<T: Scalar>(loss: &TrackedTensor<T>) -> Result<Gradients<T>>;
```

### Forward-Mode (JVP)

```rust
pub struct DualTensor<T: Scalar> {
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
}
```

### Contraction VJP/JVP

Both VJP and JVP delegate to `TensorOps::contract`, so they work on
CPU and GPU uniformly.

```rust
pub fn contract_vjp<T: Scalar>(
    a: &Tensor<T>, labels_a: &[i32],
    b: &Tensor<T>, labels_b: &[i32],
    grad_output: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>;

pub fn dual_contract<T: Scalar>(
    a: &DualTensor<T>, labels_a: &[i32],
    b: &DualTensor<T>, labels_b: &[i32],
) -> Result<DualTensor<T>>;
```

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/autodiff/` — backward pass, graph, TrackedTensor
- `ndtensors-rs/crates/ndtensors/src/contract/naive.rs:222` — contract_vjp
- `ndtensors-rs/crates/ndtensors/src/autodiff/ops/dual_contract.rs` — JVP

**Verification**:
- Numerical gradient checks (finite difference vs AD)
- Forward-mode vs reverse-mode consistency
- Complex-valued contraction gradients (Wirtinger calculus)

---

## Phase 4: t4a-capi (C FFI for ChainRules.jl)

### Backend Loading API

```c
// Load GPU vendor library at runtime (Julia/Python provides path)
int t4a_backend_load_cutensor(const char* libcutensor_path);
int t4a_backend_load_hiptensor(const char* libhiptensor_path);

// Query available devices
int t4a_backend_device_count(void);
int t4a_backend_device_type(int device_id);  // 0=CPU, 1=CUDA, 2=ROCm
```

### ChainRules.jl Integration

Julia's ChainRules.jl defines:
- `rrule(f, args...)` → `(result, pullback)` — reverse-mode rule
- `frule((Δself, Δargs...), f, args...)` → `(result, Δresult)` — forward-mode rule

t4a-capi exposes the VJP/JVP primitives that Julia wraps as ChainRules rules:

```c
// Opaque types
typedef struct t4a_tensor_f64 t4a_tensor_f64;
typedef struct t4a_tensor_c64 t4a_tensor_c64;

// Core tensor lifecycle
t4a_tensor_f64* t4a_tensor_f64_from_data(const double* data, const size_t* shape, size_t ndim);
void t4a_tensor_f64_release(t4a_tensor_f64* tensor);

// Contraction
t4a_tensor_f64* t4a_contract_f64(
    const t4a_tensor_f64* a, const int32_t* labels_a,
    const t4a_tensor_f64* b, const int32_t* labels_b,
    int* status);

// VJP (for rrule pullback)
int t4a_contract_vjp_f64(
    const t4a_tensor_f64* a, const int32_t* labels_a, size_t ndim_a,
    const t4a_tensor_f64* b, const int32_t* labels_b, size_t ndim_b,
    const t4a_tensor_f64* grad_c,
    t4a_tensor_f64** grad_a_out,
    t4a_tensor_f64** grad_b_out);

// JVP (for frule)
t4a_tensor_f64* t4a_contract_jvp_f64(
    const t4a_tensor_f64* a, const int32_t* labels_a, size_t ndim_a,
    const t4a_tensor_f64* b, const int32_t* labels_b, size_t ndim_b,
    const t4a_tensor_f64* da,   // nullable (zero tangent)
    const t4a_tensor_f64* db,   // nullable (zero tangent)
    int* status);
```

**Uses integer labels** (i32), not string notation, for C API ergonomics.

Julia example:
```julia
using cuTENSOR_jll

# Load GPU backend via jll-managed path
ccall((:t4a_backend_load_cutensor, libt4a), Cint, (Cstring,),
      cuTENSOR_jll.libcutensor_path)
```

---

## Phase 5: t4a-structured-rs (separate workspace)

Structured tensor types built on top of `Tensor<T>`. These live in a
**separate workspace** so they can be used by projects other than
tensor4all-rs without pulling in application-level dependencies.

### t4a-diag

```rust
pub struct DiagTensor<T: ScalarBase> {
    diag: Tensor<T>,
    full_sizes: Vec<usize>,
}
```

### t4a-blocksparse

Follows the **ITensors.jl/NDTensors pattern**: all blocks in a **single contiguous `DataBuffer`** with block offset mapping.

```rust
pub struct BlockSparseTensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    block_offsets: HashMap<BlockIndex, usize>,
    block_sizes: HashMap<BlockIndex, Vec<usize>>,
    full_sizes: Vec<usize>,
}
```

### t4a-graded (future)

Quantum number graded tensors. BlockSparseTensor with sector-labeled
block indices and fusion-rule-constrained block structure.

---

## Phase 6: burn-t4a (Optional Burn Backend for NN)

**Only needed when**: Users want Burn's NN modules with t4a tensors.

| Burn Method | t4a Implementation |
|---|---|
| `float_add` | `TensorOps::elementwise_binary` (BinaryOp::Add) |
| `float_matmul` | `TensorOps::contract` |
| `float_exp/sin/...` | `TensorOps::permute` with UnaryOp (Exp, etc.) |
| `float_reshape/permute/slice` | Metadata-only (zero-copy) |
| `float_sum/sum_dim` | `TensorOps::reduce` (ReduceOp::Add) |

---

## GPU Strategy

### Runtime Library Loading (Not Compile-Time Features)

GPU support is entirely runtime-based. A single compiled binary supports
CPU, NVIDIA, and AMD:

```
Build once → Ship one binary
                ↓
            Runtime:
            ├── libcutensor.so found? → NVIDIA GPU enabled
            ├── libhiptensor.so found? → AMD GPU enabled
            └── Neither? → CPU only (no error, graceful fallback)
```

No `#[cfg(feature = "cuda")]` or `#[cfg(feature = "rocm")]` in the
codebase (except for GPU-specific buffer allocation internals).

### cuTENSOR / hipTensor API Compatibility

AMD intentionally mirrors NVIDIA's API. The mapping is 1:1:

| cuTENSOR | hipTensor |
|----------|-----------|
| `cutensorCreate` | `hiptensorCreate` |
| `cutensorCreateTensorDescriptor` | `hiptensorCreateTensorDescriptor` |
| `cutensorCreateContraction` | `hiptensorCreateContraction` |
| `cutensorCreatePermutation` | `hiptensorCreatePermutation` |
| `cutensorCreateElementwiseBinary` | `hiptensorCreateElementwiseBinary` |
| `cutensorCreateReduction` | `hiptensorCreateReduction` |
| `cutensorCreatePlan` | `hiptensorCreatePlan` |
| `cutensorContract` | `hiptensorContract` |

This enables a single `TensorLibVtable` struct populated from either library.

### GPU Plan Caching

Plans are expensive to create on GPU. A `PlanCache` (from omeinsum-rs)
avoids re-creation for repeated contraction patterns:

```rust
pub struct PlanCache {
    cache: HashMap<PlanCacheKey, GpuPlan>,
    capacity: usize,
}
```

Cache key: `(shapes, strides, modes, dtype)`.

### No Metal (Apple GPU) Support

M-series CPUs are fast enough for tensor network workloads (TCI, Quantics,
MPS algorithms). Metal is not supported because:

1. **No cuTENSOR equivalent** — Apple has no dedicated N-dimensional tensor
   contraction library. Contraction must be decomposed into reshape + matmul,
   which is slow for high-rank tensors (rank-8+ common in tensor networks).
2. **Incompatible API paradigm** — MPSGraph uses graph-based execution
   (Objective-C), not the operation-level C API used by cuTENSOR/hipTensor.
   Cannot share the dlopen vtable abstraction.
3. **Implementation cost vs benefit** — The primary use case (tensor networks)
   runs well on CPU. GPU acceleration benefits come from NVIDIA/AMD HPC
   clusters, not Apple laptops.

---

## Custom Scalar Type Support

Users extend the system at the appropriate level:

| Level | What to implement | Result |
|-------|-------------------|--------|
| `ScalarBase` only | Basic trait bounds | Einsum via naive loop, map/reduce work |
| `ScalarBase` + custom GEMM | `BgemmBackend<T>` in t4a-einsum | Einsum uses custom GEMM (permute→reshape→bgemm) |
| Full `TensorOps` | Custom backend implementation | Complete control over all operations |

**Algebra-aware dispatch** in t4a-einsum:
- `Standard<f64/f32/Complex>` → faer or CBLAS GEMM
- `Standard<i32/i64>` → naive loop
- `MaxPlus<f64>` → tropical-gemm (SIMD, future)
- GPU tensors → cuTENSOR/hipTensor via `TensorOps`

---

## Migration Strategy (tensor4all-rs)

1. Replace `tensor4all-tensorbackend::Storage` with t4a types
2. Replace `DenseStorage<T>` (mdarray-based) with `Tensor<T>`
3. Adapt `TensorLike` trait to use t4a Tensor
4. Remove mdarray dependency from tensor4all-rs core

This happens **after t4a core is stable**.

---

## Verification Plan

### After Phase 1 (core):
```bash
cd t4a-rs && cargo test -p t4a-scalar -p t4a-view -p t4a-buffer \
    -p t4a-algebra -p t4a-tensorops -p t4a-einsum -p t4a-tensor
```
- Unit tests for all Tensor operations
- Zero-copy verification: assert same Arc pointer after view ops
- Tropical semiring contraction test (t4a-algebra + naive backend)
- Integer type einsum test (i32, i64 via naive backend)
- **TensorOps conformance tests**: verify CPU backend passes the same
  test suite that GPU backends will use
- **Custom type extensibility tests**: `ModInt<P>` test type through
  all three dispatch tiers
- Benchmark: t4a einsum vs current tensor4all-rs mdarray-einsum

### After Phase 2 (linalg):
- Cross-validate SVD/QR results against ndtensors-rs
- Complex SVD test

### After Phase 3 (autograd):
- Numerical gradient checks (finite difference vs reverse-mode AD)
- Forward-mode vs reverse-mode consistency
- Complex-valued gradient test (Wirtinger calculus)

### After Phase 4 (C API):
- Round-trip test: Julia → C API → Rust → C API → Julia
- Backend loading test: `t4a_backend_load_cutensor` / `t4a_backend_load_hiptensor`
- ChainRules.jl integration test with Zygote.jl

---

## Key Files Reference

| Component | Source | Destination |
|---|---|---|
| ScalarBase trait | `strided-rs/strided-traits/src/scalar.rs` | Stays in strided-rs; t4a-scalar depends on it |
| ElementOp | `strided-rs/strided-traits/src/element_op.rs` | Stays in strided-rs; t4a-scalar depends on it |
| StridedArrayView | `strided-rs/strided-view/src/view.rs` | Stays in strided-rs; t4a-view re-exports |
| map/reduce kernels | `strided-rs/strided-kernel/src/` | Stays in strided-rs; t4a-tensorops depends on it |
| einsum2_into | `strided-rs/strided-einsum2/src/lib.rs` | **Absorbed** into t4a-einsum |
| BgemmBackend | `strided-rs/strided-einsum2/src/backend.rs` | **Absorbed** into t4a-einsum |
| opteinsum | `strided-rs/strided-opteinsum/src/lib.rs` | **Absorbed** into t4a-einsum |
| Algebra traits | `omeinsum-rs/src/algebra/` | **Absorbed** into t4a-algebra |
| Backend trait | `omeinsum-rs/src/backend/traits.rs` | **Absorbed** into t4a-tensorops (evolved into TensorOps) |
| cuTENSOR wrapper | `omeinsum-rs/src/backend/cuda/cutensor/` | **Absorbed** into t4a-tensorops (GPU vtable) |
| PlanCache | `omeinsum-rs/src/backend/cuda/cutensor/contract.rs` | **Absorbed** into t4a-tensorops |
| faer bridge | `ndtensors-rs/.../faer_interop.rs` | t4a-linalg |
| contract_vjp | `ndtensors-rs/.../contract/naive.rs` | t4a-autograd |
| TrackedTensor | `ndtensors-rs/.../autodiff/tensor.rs` | t4a-autograd |
| C API patterns | `tensor4all-rs/crates/tensor4all-capi/src/` | t4a-capi |

---

## Future Considerations

### Complex-valued differentiation rules for linear algebra

Complex SVD, QR, eigen decompositions require non-trivial backward rules (Wirtinger calculus). Key references:

- **[BackwardsLinalg.jl](https://github.com/GiggleLiu/BackwardsLinalg.jl)**: Reference implementations for complex backward rules.
- **[MatrixFactorizations.jl](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl)**: Extended factorizations with ChainRules.jl integration.

### JAX / PyTorch integration via C-FFI

t4a-capi supports integration with JAX and PyTorch via ctypes:

```
JAX / PyTorch → Python wrapper → ctypes FFI → t4a-capi → Rust
```

- JAX: `jax.custom_vjp` + `jax.pure_callback()`
- PyTorch: `torch.autograd.Function` with custom forward/backward

### Insights from ITensor Julia ecosystem

| Aspect | ITensor Julia | t4a | Notes |
|---|---|---|---|
| Sparse storage | DOK-of-Arrays | Single DataBuffer + offset map | t4a is GPU-friendly |
| Axis fusion | FusionStyle dispatch | Not yet designed | Critical for quantum number tensors |

### Relationship with mdarray / mdarray-linalg

| | mdarray / mdarray-linalg | t4a-* |
|---|---|---|
| Role | **numpy equivalent** — general-purpose multidimensional array | **PyTorch equivalent** — high-performance tensor library |
| Memory | Owned `Array<T, D>` | `DataBuffer<T>` (Arc-based COW, CPU/GPU) |
| GPU | No | cuTENSOR, hipTensor (no Metal) |
| Autodiff | No | t4a-autograd (VJP/JVP) |
| Dispatch | Direct function calls | `TensorOps` trait (runtime backend selection) |

Both are needed. mdarray is a foundational array library; t4a builds a
richer tensor ecosystem with GPU support and automatic differentiation.

t4a-linalg and mdarray-linalg are **parallel** (both call faer directly),
not serial:

```
faer (SVD, QR, eigen)
    ↑                ↑
t4a-linalg       mdarray-linalg-faer
(Tensor<T>       (Array<T, D>
 → MatRef)        → MatRef)
```
