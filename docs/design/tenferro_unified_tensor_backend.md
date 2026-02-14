# tenferro: Unified Tensor Backend Design Plan

> **POC implementation**: The initial proof-of-concept (4 core crates) is at
> <https://github.com/tensor4all/tenferro-rs/>.

> **Companion documents**:
> - [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md) — strided-rs vs omeinsum-rs optimization comparison
> - [tenferro Einsum Internal Design](./tenferro_einsum_internal_design.md) — detailed internal design of tenferro-prims and tenferro-einsum

## Context

Four independent Rust projects exist in tensor4all:
- **strided-rs**: Cache-optimized strided array kernels (view, map/reduce, einsum)
- **omeinsum-rs**: Einsum with tropical algebra, gradient support, GPU dispatch
- **ndtensors-rs**: Tensor types with storage hierarchy, linear algebra, autograd
- **tensor4all-rs**: Tensor network algorithms (TCI, Quantics, MPS) with ad-hoc tensor backend

These have significant overlap (3 einsum implementations, 3 scalar trait definitions, 3 dense storage types) yet critical gaps. The goal is to unify into a coherent, reusable tensor backend library **tenferro-\*** that:

1. Integrates selected strided-rs add-on crates (`strided-einsum2`, `strided-opteinsum`) and omeinsum-rs components into `tenferro-*` (while keeping `strided-traits/view/kernel` as external foundational dependencies)
2. Provides unified CPU/GPU dispatch via a **cuTENSOR/hipTensor-compatible protocol** (`tenferro-prims`)
3. Supports both NVIDIA and AMD GPUs via **runtime library loading** (no compile-time vendor lock-in)
4. Supports complex numbers natively
5. Supports custom scalar types (tropical semiring, etc.) with pluggable backends
6. Exposes VJP/JVP through C API for Julia ChainRules.jl
7. Can optionally bridge to Burn for NN workloads

**Key design principles**:
- **strided-rs as foundation**: The general-purpose strided array crates (`strided-traits`, `strided-view`, `strided-kernel`) remain in an independent `strided-rs` workspace. They have no BLAS dependency and can be used standalone. `tenferro-rs` depends on them but does not absorb them.
- **cuTENSOR/hipTensor-compatible protocol**: `tenferro-prims` defines a unified `TensorPrims<A>` trait parameterized by algebra `A`, with a cuTENSOR-compatible describe → plan → execute pattern for all operations. CPU, NVIDIA, and AMD backends implement the same trait.
- **Algebra-parameterized dispatch**: `TensorPrims<A>` is parameterized by algebra (e.g., `Standard`, `MaxPlus`). The `HasAlgebra` trait on scalar types enables automatic algebra inference: `Tensor<f64>` → `Standard`, `Tensor<MaxPlus<f64>>` → `MaxPlus`. Users can extend the system by defining new algebras in their own crates (orphan rule compatible).
- **Runtime GPU discovery**: GPU vendor libraries (cuTENSOR, hipTensor) are loaded at runtime via `dlopen`. The caller (Julia, Python) provides the `.so` path. No Cargo feature flags for GPU vendor selection.
- **Plan-based execution**: All operations follow the cuTENSOR pattern of `PrimDescriptor` → plan → execute. Plans cache expensive analysis (GPU kernel selection, CPU fusability checks) for reuse. Extended operations (contract, elementwise_mul) are dynamically queried via `has_extension_for::<T>()`.
- **Enums for all variant types**: All types with variants (`LogicalMemorySpace`, `ComputeDevice`, `OpDescriptor`, etc.) use Rust enums. This preserves exhaustive `match` checking — when a new variant is added, the compiler identifies every call site that needs updating. New variants constitute a semver breaking change (major version bump), managed through Cargo's version resolution. This is preferred over opaque structs because the cost of hidden unhandled variants (runtime bugs) outweighs the cost of explicit version migration (compile-time errors).
- **No implicit cross-space transfer**: Operations across different logical memory spaces fail by default. Data movement is explicit via `to_memory_space_async`.

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Application                                         │
│   tensor4all-rs (TCI, Quantics, MPS algorithms)              │
│   Julia / Python (C API via tenferro-capi) [future]          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 4: Einsum Engine (tenferro-einsum)          [POC]      │
│   High-level API on Tensor<T>: einsum(), string notation     │
│   Subscripts (integer labels + string parse)                 │
│   ContractionTree (optimize, from_pairs)                     │
│   Three API levels: einsum, einsum_with_subscripts,          │
│                     einsum_with_plan                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 3: Tensor Type (tenferro-tensor)             [POC]     │
│   Tensor<T> = DataBuffer + dims + strides + offset + device  │
│   Zero-copy view ops: permute, broadcast, diagonal, reshape  │
│   Bridge to strided-rs via view() / view_mut()               │
│   DataBuffer<T> enum: Cpu(StridedArray<T>)                   │
│   MemoryOrder (ColumnMajor, RowMajor) for allocation only    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 2: Tensor Operation Protocol (tenferro-prims) [POC]│
│   TensorPrims<A> trait — parameterized by algebra A            │
│   cuTENSOR pattern: PrimDescriptor → plan → execute            │
│   Core ops: batched_gemm, reduce, trace, permute,            │
│     anti_trace, anti_diag                                    │
│   Extended ops (dynamic query): contract, elementwise_mul    │
│   HasAlgebra trait: T → A automatic inference                │
│   CpuBackend: impl TensorPrims<Standard>                      │
│   Uses StridedView<T> / StridedViewMut<T> directly           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Shared Infrastructure: Device Layer (tenferro-device) [POC]  │
│   LogicalMemorySpace + ComputeDevice enums                    │
│   preferred_compute_devices(space, op_kind)                   │
│   Error types (thiserror): ShapeMismatch, RankMismatch,      │
│     DeviceError, InvalidArgument, Strided                    │
│   Result<T> type alias                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 1: Backend Implementations                    [future] │
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
│  │  strided-kernel  │  │              │  │              │    │
│  │  cache-opt       │  │              │  │              │    │
│  └─────────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Crate Structure

### POC Crates (implemented in tenferro-rs)

```
strided-rs/ (independent workspace) ── Foundation crates stay as-is ───────
│  General-purpose strided array library. No BLAS dependency.
│  Can be used standalone by projects other than tenferro.
│
├── strided-traits       # ScalarBase, ElementOp traits
├── strided-view         # StridedArray, StridedView, StridedViewMut (zero-copy strided views)
└── strided-kernel       # Cache-optimized map/reduce/broadcast kernels

tenferro-rs/ (workspace) ── 5 POC crates ─────────────────────
│  Depends on strided-rs.
│
├── tenferro-device      # LogicalMemorySpace + ComputeDevice enums
│                        #   preferred_compute_devices(space, op_kind)
│                        #   Error (thiserror): ShapeMismatch, RankMismatch,
│                        #     DeviceError, InvalidArgument, Strided
│                        #   Result<T> type alias
│                        #   Depends on: strided-view (for StridedError), thiserror
│
├── tenferro-algebra     # HasAlgebra trait, Semiring trait, Standard type
│                        #   HasAlgebra: maps T → A (f64 → Standard, etc.)
│                        #   Minimal algebra foundation for TensorPrims<A>
│                        #   Depends on: strided-traits, num-complex
│
├── tenferro-prims   # TensorPrims<A> trait — parameterized by algebra A
│                        #   PrimDescriptor enum (describe → plan → execute)
│                        #   Core ops: batched_gemm, reduce, trace, permute,
│                        #     anti_trace, anti_diag
│                        #   Extended ops (dynamic query): contract, elementwise_mul
│                        #   Associated type Plan<T> (no type erasure)
│                        #   CpuBackend: impl TensorPrims<Standard>
│                        #   Operates on StridedView<T> / StridedViewMut<T> directly
│                        #   Depends on: tenferro-device, tenferro-algebra,
│                        #     strided-view, strided-traits
│
├── tenferro-tensor      # Tensor<T> = DataBuffer + dims + strides + offset + device
│                        #   DataBuffer<T> enum: Cpu(StridedArray<T>)
│                        #   MemoryOrder: ColumnMajor, RowMajor (allocation-time only)
│                        #   Constructors: zeros, ones, from_slice, from_strided_array
│                        #   View ops: view(), view_mut(), permute, broadcast,
│                        #     diagonal, reshape (zero-copy metadata ops)
│                        #   Data ops: contiguous, into_contiguous, is_contiguous, conj
│                        #   Depends on: tenferro-device, strided-view, strided-traits, num-traits
│
└── tenferro-einsum      # High-level einsum on Tensor<T>
                         #   String notation: einsum("ij,jk->ik", &[&a, &b])
                         #   Parenthesized order: einsum("ij,(jk,kl)->il", &[...])
                         #   Subscripts struct (new for integers, parse for strings)
                         #   ContractionTree (optimize, from_pairs)
                         #   Allocating: einsum, einsum_with_subscripts, einsum_with_plan
                         #   Accumulating: einsum_into, einsum_with_subscripts_into,
                         #     einsum_with_plan_into (alpha/beta scaling)
                         #   Depends on: tenferro-device, tenferro-algebra,
                         #     tenferro-prims, tenferro-tensor, strided-traits
```

### Future Crates (not in POC)

```
tenferro-rs/ (future additions) ──────────────────────────────
│
├── tenferro-linalg      # SVD, QR, eigen, polar (CPU: faer, GPU: cuSOLVER via device layer)
├── tenferro-autograd    # TrackedTensor, DualTensor, VJP/JVP
├── tenferro-hdf5        # Tensor<T> HDF5 I/O (via hdf5-rt, dlopen)
├── tenferro-mdarray     # Tensor<T> ←→ mdarray conversion (memory copy)
├── tenferro-ndarray     # Tensor<T> ←→ ndarray conversion (memory copy)
├── tenferro-capi        # C FFI (tensor ops + VJP/JVP + backend loading)
└── burn-tenferro        # Burn Backend bridge [OPTIONAL, for NN only]

tenferro-tropical/ (separate crate — proves extensibility) ──
│  Tropical algebra types and TensorPrims implementations.
│  Being external proves that user-defined algebras can extend
│  the system via the same pattern (orphan rule compatible).
│
├── MaxPlus<T>, MinPlus<T>, MaxMul<T> types
├── impl HasAlgebra for MaxPlus<T> { type Algebra = MaxPlus; }
├── impl TensorPrims<MaxPlus> for CpuBackend   ← orphan OK
├── tropical-gemm SIMD kernels
└── argmax tracking for tropical backward pass

tenferro-structured-rs/ (future separate workspace) ── Structured tensor types ──
│
├── tenferro-blocksparse # BlockSparseTensor (single DataBuffer + block offsets)
├── tenferro-diag        # DiagTensor (1D Tensor of diagonal elements)
└── tenferro-graded      # GradedTensor (future: quantum number sectors)

tensor4all-rs/ (workspace) ── Tensor network algorithms ────
│
├── TCI, Quantics, MPS, ...
└── depends on tenferro-rs + tenferro-structured-rs
```

### Dependency Graph (POC)

```
strided-rs (independent workspace):
strided-traits → strided-view → strided-kernel

tenferro-rs (workspace, depends on strided-rs):

tenferro-device              tenferro-algebra
  (← strided-view,            (← strided-traits,
   ← thiserror)                ← num-complex)
    │                            │
    ├────────────┐       ┌───────┤
    │            ↓       ↓       │
    │       tenferro-prims       │
    │         (← strided-view,   │
    │          ← strided-traits) │
    │            │               │
    ↓            │               │
tenferro-tensor  │               │
  (← strided-view,              │
   ← strided-traits,            │
   ← num-traits)                │
    │            │               │
    └──────┬─────┘       ┌───────┘
           ↓             ↓
      tenferro-einsum
        (← strided-traits)
```

### Future Dependency Graph (full vision)

```
tenferro-device
    │
    ↓
tenferro-algebra (HasAlgebra, Semiring, Standard)
    │
    ├────────────────────────────┐
    ↓                            ↓
tenferro-prims          tenferro-tensor
    │                            │
    └──────────┬─────────────────┘
               ↓
          tenferro-einsum
               │
    ┌──────────┼──────────────┐
    ↓          ↓              ↓
tenferro-  tenferro-     tenferro-
 linalg     autograd       capi

tenferro-hdf5 ← tenferro-tensor, hdf5-rt (dlopen)

[separate crate: tenferro-tropical]
← tenferro-algebra, tenferro-prims
impl TensorPrims<MaxPlus> for CpuBackend (orphan OK)

[separate workspace: tenferro-structured-rs]
tenferro-blocksparse ← tenferro-tensor
tenferro-diag        ← tenferro-tensor
tenferro-graded      ← tenferro-blocksparse (future)

[optional]
burn-tenferro ← tenferro-tensor, burn-backend
```

### Origin of Each Crate

| tenferro crate | Origin | What changes |
|----------------|--------|--------------|
| tenferro-device | **New** (POC) | LogicalMemorySpace/ComputeDevice enums, preferred device selection, Error/Result types |
| tenferro-prims | **New** (POC), will absorb strided-einsum2 | TensorPrims\<A\> trait (algebra-parameterized), PrimDescriptor, CpuBackend |
| tenferro-tensor | **New** (POC) | Tensor\<T\>, DataBuffer\<T\>, MemoryOrder, zero-copy view ops |
| tenferro-einsum | **New** (POC), will absorb strided-opteinsum + omeinsum-rs | Subscripts, ContractionTree, einsum/einsum_with_subscripts/einsum_with_plan |
| tenferro-algebra | omeinsum-rs (Algebra traits) | Standalone crate for Semiring/tropical types [future] |
| tenferro-linalg | ndtensors-rs (linalg) | Port SVD/QR/eigen [future] |
| tenferro-autograd | ndtensors-rs (autodiff) | Port TrackedTensor/DualTensor [future] |
| tenferro-capi | ndtensors-rs (capi) + tensor4all-rs (capi) | Port C FFI + backend loading API [future] |
| tenferro-hdf5 | New, uses hdf5-rt | Tensor\<T\> HDF5 I/O via runtime library loading [future] |
| burn-tenferro | New | Burn Backend bridge [future] |
| **tenferro-structured-rs (separate workspace):** | | |
| tenferro-blocksparse | ndtensors-rs (blocksparse) | Port with single-buffer layout [future] |
| tenferro-diag | ndtensors-rs (diag) | Port DiagTensor [future] |
| tenferro-graded | New (future) | Quantum number graded tensors [future] |

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** (dlopen) [future] | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** (Cargo feature) [future] | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based** in TensorPrims; closures via strided-kernel | cuTENSOR-compatible operator enums for GPU; custom closures via strided-kernel directly (CPU only) |
| libloading dependency | **Always ON** (in tenferro-device) [future] | Lightweight, no overhead when GPU absent, no feature gate needed |
| .so path for GPU libs | **Caller-injected** [future] | Rust does not search; Julia/Python provide exact path |

---

## Phase 1: Dense Array Foundation (POC)

### tenferro-device

The device crate provides shared infrastructure used across all tenferro crates.

```rust
/// Logical memory space where tensor buffers reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalMemorySpace {
    /// Always-available host memory.
    MainMemory,
    /// GPU-accessible memory space. A space may be attached to one or many GPUs.
    GpuMemory { space_id: usize },
}

/// Compute device where kernels execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    Cpu { device_id: usize },
    Cuda { device_id: usize },
    Hip { device_id: usize },
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeDevice::Cpu { device_id } => write!(f, "cpu:{device_id}"),
            ComputeDevice::Cuda { device_id } => write!(f, "cuda:{device_id}"),
            ComputeDevice::Hip { device_id } => write!(f, "hip:{device_id}"),
        }
    }
}

/// Operation kind used for capability filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind { Contract, BatchedGemm, Reduce, Trace, Permute, ElementwiseMul }

/// Returns executable compute devices in descending preference order.
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    op_kind: OpKind,
) -> Result<Vec<ComputeDevice>>;
```

**Error types** using `thiserror`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("device error: {0}")]
    DeviceError(String),

    #[error("no compatible compute device for {op:?} in memory space {space:?}")]
    NoCompatibleComputeDevice { space: LogicalMemorySpace, op: OpKind },

    #[error("operation across distinct logical memory spaces is not allowed by default")]
    CrossMemorySpaceOperation,

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Dependencies**: `strided-view` (for `StridedError`), `thiserror`.

**Note**: `BackendRegistry`, `GpuBackend`, and `TensorLibVtable` are **not** in the POC. They are planned for future GPU support (see [GPU Strategy](#gpu-strategy) section below).

### tenferro-algebra

Minimal algebra foundation for `TensorPrims<A>`. Provides the `HasAlgebra`
trait for automatic algebra inference and the `Standard` type for standard
arithmetic.

```rust
/// Maps a scalar type T to its default algebra A.
/// Enables automatic inference: Tensor<f64> → Standard, Tensor<MaxPlus<f64>> → MaxPlus.
pub trait HasAlgebra {
    type Algebra;
}

/// Standard arithmetic algebra (add = +, mul = *).
pub struct Standard;

impl HasAlgebra for f64 { type Algebra = Standard; }
impl HasAlgebra for f32 { type Algebra = Standard; }
impl HasAlgebra for Complex64 { type Algebra = Standard; }
// etc.

/// Semiring trait for algebra-generic operations.
pub trait Semiring {
    type Scalar: ScalarBase;
    fn zero() -> Self::Scalar;
    fn one() -> Self::Scalar;
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
```

**Note**: Tropical types (`MaxPlus`, `MinPlus`, `MaxMul`) are in the separate
`tenferro-tropical` crate, not here. This separation proves that the algebra
extension mechanism works for external crates.

### tenferro-prims

The central protocol layer. Defines `TensorPrims<A>` parameterized by algebra `A`,
with a cuTENSOR-compatible describe → plan → execute pattern.

> **Detailed design**: See [tenferro Einsum Internal Design](./tenferro_einsum_internal_design.md)
> for the full internal design including CPU contraction pipeline details.

#### Design Overview

GiggleLiu proposed a **universal set** of primitive operations plus an
**extended set** of optimized composites. The trait is parameterized by
algebra `A` so different scalar types can plug in their own implementations.

```
tenferro-einsum (engine)
    │
    │  T: HasAlgebra → infers A automatically
    │
    ├── [has_extension_for::<T>(Contract)?]
    │   YES → execute Contract plan (fused permute+GEMM)
    │
    └── [otherwise]
        decompose into core ops:
        diag → trace/reduce → permute → batched_gemm → permute
```

**Dispatch is dynamic**: `has_extension_for::<T>(ext)` queries at runtime
whether a specific extended operation is available for scalar type `T`.
This is important because:
- GPU backends are loaded at runtime (dlopen)
- cuTENSOR supports `f32`/`f64`/Complex but not tropical types
- CPU backends may support `contract` for `f64` (faer) but not for custom types

Note: `diag` (diagonal extraction) and `repeat` (broadcast) are **zero-copy
stride tricks** handled at the `Tensor<T>` level (see below), not in `TensorPrims`.

#### Adjoint Pairs for AD

The core operations form adjoint pairs, enabling clean VJP/JVP rules:

| Forward | Backward (adjoint) |
|---------|-------------------|
| trace | anti_trace |
| diag (on Tensor) | anti_diag |
| reduce | repeat (on Tensor) |
| permute | inverse permute |
| batched_gemm | Leibniz rule |

#### Key Types

```rust
/// Describes any TensorPrims operation (cuTENSOR pattern: describe → plan → execute).
pub enum PrimDescriptor {
    BatchedGemm { batch_dims: Vec<usize>, m: usize, n: usize, k: usize },
    Reduce { modes_a: Vec<u32>, modes_c: Vec<u32>, op: ReduceOp },
    Trace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    Permute { modes_a: Vec<u32>, modes_b: Vec<u32> },
    AntiTrace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    AntiDiag { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    // Extended
    Contract { modes_a: Vec<u32>, modes_b: Vec<u32>, modes_c: Vec<u32> },
    ElementwiseMul,
}

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Max, Min }

/// Extended operation identifiers for dynamic capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension { Contract, ElementwiseMul }
```

#### TensorPrims\<A\> Trait

```rust
/// Backend trait parameterized by algebra A.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute, anti_trace,
/// anti_diag) must be implemented. Extended ops (contract, elementwise_mul)
/// have default implementations that decompose into core ops.
///
/// The algebra parameter A enables extensibility: external crates can
/// implement TensorPrims<MyAlgebra> for CpuBackend (orphan rule compatible).
pub trait TensorPrims<A> {
    /// Backend-specific plan type (no type erasure).
    type Plan<T: ScalarBase>;

    /// Create an execution plan (cuTENSOR: describe → plan).
    fn plan<T: ScalarBase>(
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    /// Execute a plan (cuTENSOR: plan → execute).
    fn execute<T: ScalarBase>(
        plan: &Self::Plan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for scalar type T.
    /// Enables dynamic dispatch: GPU may support Contract for f64 but not
    /// for tropical types.
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}
```

#### CpuBackend

```rust
pub struct CpuBackend;

/// Standard arithmetic on CPU (faer GEMM for f64/f32, naive for others).
impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;

    fn plan<T: ScalarBase>(desc: &PrimDescriptor, shapes: &[&[usize]])
        -> Result<CpuPlan<T>> { ... }

    fn execute<T: ScalarBase>(plan: &CpuPlan<T>, ...) -> Result<()> { ... }

    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
        // CPU supports Contract and ElementwiseMul for all standard types
        true
    }
}

/// CPU plan — concrete enum, no type erasure.
enum CpuPlan<T: ScalarBase> {
    BatchedGemm { m: usize, n: usize, k: usize, ... },
    Reduce { axis: usize, op: ReduceOp },
    Trace { paired: Vec<(u32, u32)> },
    Permute { perm: Vec<usize> },
    Contract { /* strided-einsum2 cached analysis */ },
    ElementwiseMul,
    ...
}
```

**Tropical backend** (in separate `tenferro-tropical` crate):

```rust
// tenferro-tropical crate — external, proves extensibility
pub struct MaxPlus;

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus; }

/// Tropical GEMM on CPU (SIMD-optimized tropical-gemm kernel).
impl TensorPrims<MaxPlus> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;

    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
        false  // tropical uses core ops decomposition, no fused contract
    }
    ...
}
```

**User-defined algebra** (in user crate):

```rust
// User crate — same pattern as tenferro-tropical
struct MyScalar(f64);
struct MyAlgebra;

impl ScalarBase for MyScalar { ... }
impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }

impl TensorPrims<MyAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = MyPlan<T>;
    ...
}

// Just works:
let a = Tensor::<MyScalar>::zeros(&[3, 4], ...);
einsum("ij,jk->ik", &[&a, &b])?;  // MyAlgebra auto-inferred
```

**Backend implementation matrix**:

| Backend | Algebra | Extended ops | Notes |
|---------|---------|-------------|-------|
| CpuBackend | Standard | Contract, ElementwiseMul | faer/cblas GEMM |
| CpuBackend | MaxPlus | None (decompose to core) | tropical-gemm SIMD |
| CpuBackend | MyAlgebra | User choice | User-provided kernels |
| GpuBackend [future] | Standard | Contract, ElementwiseMul | cuTENSOR/hipTensor |
| GpuBackend [future] | MaxPlus | None | No cuTENSOR tropical support |

**Usage examples**:

```rust
use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, ReduceOp, Standard};
use strided_view::StridedArray;

// Plan + execute: GEMM
let desc = PrimDescriptor::BatchedGemm { batch_dims: vec![], m: 3, n: 5, k: 4 };
let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();

// Plan + execute: Reduction
let desc = PrimDescriptor::Reduce { modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum };
let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[3]]).unwrap();
CpuBackend::execute(&plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

// Dynamic extension check
if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
    let desc = PrimDescriptor::Contract { modes_a: vec![0,1], modes_b: vec![1,2], modes_c: vec![0,2] };
    let plan = CpuBackend::plan::<f64>(&desc, &shapes).unwrap();
    CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
}
```

**No Metal (Apple GPU) support**: M-series CPUs are fast enough for our
workloads (tensor network algorithms). Metal lacks a cuTENSOR-equivalent
tensor contraction library, requiring reshape+matmul decomposition that
would be slow for high-rank tensors. Not worth the implementation cost.

### tenferro-tensor

`Tensor<T>` is the core data type. It wraps a `DataBuffer<T>` with
shape/stride metadata and provides zero-copy view operations.

```rust
/// Memory ordering for new allocations only.
/// Not stored on the tensor — strides fully describe the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    ColumnMajor,  // First dimension has stride 1 (Fortran/Julia)
    RowMajor,     // Last dimension has stride 1 (C/NumPy)
}

/// Owned data buffer, device-aware.
pub enum DataBuffer<T> {
    Cpu(StridedArray<T>),
    // Future: Cuda(CudaBuffer<T>), Hip(HipBuffer<T>)
}

/// Multi-dimensional dense tensor.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,  // None = ready, Some = pending accelerator work
}
```

**Constructors**:

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn ones(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self>;
    pub fn from_strided_array(array: StridedArray<T>) -> Self;
}
```

**Metadata**:

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn dims(&self) -> &[usize];
    pub fn strides(&self) -> &[isize];
    pub fn ndim(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn logical_memory_space(&self) -> LogicalMemorySpace;
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice>;
    pub fn set_preferred_compute_device(&mut self, d: Option<ComputeDevice>);
    pub fn effective_compute_devices(&self, op: OpKind) -> Result<Vec<ComputeDevice>>;
}
```

**Explicit memory movement**:

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Asynchronous explicit move between logical memory spaces.
    ///
    /// - Same source/destination space: always Ok, zero-copy no-op.
    /// - Different spaces: explicit transfer (never implicit in ops).
    pub fn to_memory_space_async(&self, dst: LogicalMemorySpace) -> Result<Tensor<T>>;
}
```

**View operations** (zero-copy, modify only metadata):

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Immutable strided view for use with TensorPrims or strided-kernel.
    pub fn view(&self) -> StridedView<'_, T>;

    /// Mutable strided view.
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T>;

    /// Permute (reorder) dimensions. Zero-copy.
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>>;

    /// Broadcast to a larger shape. Zero-copy via stride 0.
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>>;

    /// Extract diagonal view by merging pairs of axes. Zero-copy stride trick.
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>>;

    /// Reshape. Requires contiguous data.
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>>;
}
```

**Data operations**:

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Return a contiguous copy in the given memory order.
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;

    /// Consume this tensor and return a contiguous version.
    /// If already contiguous in the requested order, returns self
    /// without copying or allocating.
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;

    /// Check if tensor data is contiguous in memory.
    pub fn is_contiguous(&self) -> bool;

    /// Return a tensor with complex-conjugated elements.
    ///
    /// For real types (f32, f64), returns a copy unchanged.
    /// For complex types (Complex32, Complex64), negates the imaginary part.
    pub fn conj(&self) -> Tensor<T>;

    /// Wait for any pending accelerator computation to complete.
    /// No-op for CPU tensors or already-completed operations.
    pub fn wait(&self);

    /// Check if data is ready without blocking.
    /// Always returns true for CPU tensors.
    pub fn is_ready(&self) -> bool;
}
```

**Key differences from original design**:
- `DataBuffer<T>` is an enum in `tenferro-tensor` (not a separate crate, no `Arc` wrapping)
- Fields: `dims` (`Vec<usize>`), `strides` (`Vec<isize>`), `offset` (`isize`) -- no `SmallVec`
- `MemoryOrder` is only used at allocation time, **not stored** on the tensor
- Bridge to strided-rs via `view()` / `view_mut()` (not `as_strided_view()`)
- No `TensorMeta` struct -- not needed since `TensorPrims` uses `StridedView` directly
- No type casting methods yet (future work)

**Creating and using tensors**:

```rust
use tenferro_tensor::{Tensor, MemoryOrder};
use tenferro_device::LogicalMemorySpace;

// Create tensors
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let m = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();

// Zero-copy transpose
let mt = m.permute(&[1, 0]).unwrap();
assert_eq!(mt.dims(), &[3, 2]);

// Broadcasting (zero-copy via stride 0)
let col = Tensor::<f64>::ones(&[3, 1], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
let expanded = col.broadcast(&[3, 4]).unwrap();
assert_eq!(expanded.dims(), &[3, 4]);

// Get strided views for low-level operations
let view = a.view();
let mut b = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
let view_mut = b.view_mut();
```

For **custom element-wise closures** (arbitrary user functions not in the
`TensorPrims` enum), use strided-kernel directly via `view()`:

```rust
// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.view();
let b_view = tensor_b.view();
strided_kernel::zip_map2_into(&mut out.view_mut(), &a_view, &b_view, |a, b| a * b + 1.0);
```

### tenferro-einsum

High-level einsum API on `Tensor<T>`. Supports string notation with
parenthesized contraction order, integer label notation, and pre-optimized
contraction trees.

**Subscripts**:

```rust
/// Einsum subscripts using integer labels (omeinsum-rs compatible).
#[derive(Debug, Clone)]
pub struct Subscripts {
    pub inputs: Vec<Vec<u32>>,
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create from integer label arrays.
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self;

    /// Parse from string notation: "ij,jk->ik"
    /// Supports parenthesized order: "ij,(jk,kl)->il"
    pub fn parse(notation: &str) -> Result<Self>;
}
```

**ContractionTree**:

```rust
pub struct ContractionTree { /* internal */ }

impl ContractionTree {
    /// Automatically optimize contraction order (cost-based heuristic).
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self>;

    /// Manually specify pairwise contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self>;
}
```

**Three API levels**:

```rust
/// Level 1: String notation — parse + optimize + execute.
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Level 2: Pre-built subscripts — optimize + execute.
pub fn einsum_with_subscripts<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Level 3: Pre-optimized tree — execute only.
pub fn einsum_with_plan<T: ScalarBase + HasAlgebra>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;
```

| Level | Parsing | Optimization | Execution | Use case |
|-------|---------|-------------|-----------|----------|
| `einsum` | Yes | Yes | Yes | One-off, convenience |
| `einsum_with_subscripts` | Cached | Yes | Yes | Same pattern, varying shapes |
| `einsum_with_plan` | Cached | Cached | Yes | Hot loops, same shapes |

**User examples**:

```rust
use tenferro_einsum::einsum;
use tenferro_tensor::{Tensor, MemoryOrder};
use tenferro_device::LogicalMemorySpace;

let col = MemoryOrder::ColumnMajor;
let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();

// Matrix multiplication
let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();

// Trace
let tr = einsum("ii->", &[&a]).unwrap();

// Batch matrix multiplication
let ba = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
let bb = Tensor::<f64>::zeros(&[10, 4, 5], LogicalMemorySpace::MainMemory, col);
let bc = einsum("bij,bjk->bik", &[&ba, &bb]).unwrap();

// Explicit contraction order via parentheses
let d = einsum("ij,(jk,kl)->il", &[&a, &b, &c]).unwrap();

// Integer label notation (for programmatic use)
use tenferro_einsum::{einsum_with_subscripts, Subscripts};
let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let c = einsum_with_subscripts(&subs, &[&a, &b]).unwrap();

// Pre-optimized tree (hot loops)
use tenferro_einsum::ContractionTree;
let tree = ContractionTree::optimize(&subs, &[&[2, 2], &[2, 2]]).unwrap();
let c = einsum_with_plan(&tree, &[&a, &b]).unwrap();
```

**Key differences from original design**:
- String-first API: `einsum("ij,jk->ik", &[&a, &b])` instead of integer labels as primary
- Parenthesized contraction order in string notation
- `Subscripts::parse()` handles string-to-integer conversion
- Six API functions: three allocating (`einsum`, `einsum_with_subscripts`, `einsum_with_plan`) and three accumulating (`einsum_into`, `einsum_with_subscripts_into`, `einsum_with_plan_into`) with BLAS-style `alpha`/`beta` scaling
- `Tensor::into_contiguous(self)` consuming variant avoids allocation when already contiguous
- No mixed-type inputs: all inputs and output must be the same type `T`

---

## Future Phase: tenferro-tropical (Separate Crate)

Tropical algebra types and `TensorPrims` implementations, as a separate crate
that proves the extensibility of the algebra-parameterized design:

```rust
// tenferro-tropical crate
pub struct MaxPlus<T>(pub T);    // sem_add = max, sem_mul = +
pub struct MinPlus<T>(pub T);    // sem_add = min, sem_mul = +
pub struct MaxMul<T>(pub T);     // sem_add = max, sem_mul = *

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus; }

impl TensorPrims<MaxPlus> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;
    // SIMD-optimized tropical-gemm kernels
    ...
}
```

Also provides:
- Argmax tracking for tropical backward pass
- SIMD-optimized tropical-gemm via `TypeId`-based runtime dispatch (from omeinsum-rs)

Being in a separate crate proves that external crates can extend the system
by implementing `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule compatible).

---

## Future Phase: tenferro-linalg

All decomposition functions:
1. Call `tensor.contiguous(MemoryOrder::ColumnMajor)` (faer is column-major)
2. Create `faer::MatRef` from raw pointer + strides
3. Call faer's decomposition
4. Wrap result back into Tensor

**Operations**: svd, svd_truncated, qr, qr_positive, ql, eigen_hermitian, eigen, polar, matrix_exp.

**N-D tensor decomposition**: specify left_dims for "row" side, reshape to 2D, decompose, reshape back.

**Trait bound**: `T: Scalar + faer::ComplexField` (enforced here, not in strided-traits/tenferro-tensor).

**GPU path**: cuSOLVER/rocSOLVER via runtime-loaded vendor library (same dlopen pattern).

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/backend/faer_interop.rs` -- faer bridge pattern
- `ndtensors-rs/crates/ndtensors/src/linalg/` -- SVD, QR implementations

---

## Future Phase: tenferro-autograd (Primary AD System)

This is the **default** AD system for tenferro users. Burn's autodiff is only for NN workloads.

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

Both VJP and JVP delegate to `TensorPrims`, so they work on CPU and GPU uniformly.
The two-tier TensorPrims design provides clean **adjoint pairs** for each
primitive operation:

| Forward | Backward (adjoint) |
|---------|-------------------|
| `trace(A)` | `anti_trace(∂y)` — scatter-add to diagonal |
| `diag(A)` (on Tensor) | `anti_diag(∂y)` — write to diagonal positions |
| `reduce(A, dim)` | `repeat(∂y, dim)` — broadcast gradient |
| `permute(A, p)` | `permute(∂y, p⁻¹)` — inverse permutation |
| `batched_gemm(A, B)` | `∂A = batched_gemm(∂C, B^T)`, `∂B = batched_gemm(A^T, ∂C)` |

```rust
pub fn contract_vjp<T: Scalar>(
    a: &Tensor<T>, labels_a: &[u32],
    b: &Tensor<T>, labels_b: &[u32],
    grad_output: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>;

pub fn dual_contract<T: Scalar>(
    a: &DualTensor<T>, labels_a: &[u32],
    b: &DualTensor<T>, labels_b: &[u32],
) -> Result<DualTensor<T>>;
```

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/autodiff/` -- backward pass, graph, TrackedTensor
- `ndtensors-rs/crates/ndtensors/src/contract/naive.rs:222` -- contract_vjp
- `ndtensors-rs/crates/ndtensors/src/autodiff/ops/dual_contract.rs` -- JVP

**Verification**:
- Numerical gradient checks (finite difference vs AD)
- Forward-mode vs reverse-mode consistency
- Complex-valued contraction gradients (Wirtinger calculus)

---

## Future Phase: tenferro-capi (C FFI for ChainRules.jl)

### Backend Loading API

Wraps the device layer's `BackendRegistry` as C functions:

```c
// Load GPU vendor library at runtime
int tenferro_backend_load_cutensor(const char* libcutensor_path);
int tenferro_backend_load_hiptensor(const char* libhiptensor_path);

// Query available devices
int tenferro_backend_device_count(void);
int tenferro_backend_device_type(int device_id);  // 0=CPU, 1=CUDA, 2=HIP
```

### ChainRules.jl Integration

Julia's ChainRules.jl defines:
- `rrule(f, args...)` -> `(result, pullback)` -- reverse-mode rule
- `frule((Dself, Dargs...), f, args...)` -> `(result, Dresult)` -- forward-mode rule

tenferro-capi exposes the VJP/JVP primitives that Julia wraps as ChainRules rules:

```c
// Opaque types
typedef struct tenferro_tensor_f64 tenferro_tensor_f64;
typedef struct tenferro_tensor_c64 tenferro_tensor_c64;

// Core tensor lifecycle
tenferro_tensor_f64* tenferro_tensor_f64_from_data(
    const double* data, const size_t* shape, size_t ndim);
void tenferro_tensor_f64_release(tenferro_tensor_f64* tensor);

// Contraction
tenferro_tensor_f64* tenferro_contract_f64(
    const tenferro_tensor_f64* a, const uint32_t* labels_a,
    const tenferro_tensor_f64* b, const uint32_t* labels_b,
    int* status);

// VJP (for rrule pullback)
int tenferro_contract_vjp_f64(
    const tenferro_tensor_f64* a, const uint32_t* labels_a, size_t ndim_a,
    const tenferro_tensor_f64* b, const uint32_t* labels_b, size_t ndim_b,
    const tenferro_tensor_f64* grad_c,
    tenferro_tensor_f64** grad_a_out,
    tenferro_tensor_f64** grad_b_out);

// JVP (for frule)
tenferro_tensor_f64* tenferro_contract_jvp_f64(
    const tenferro_tensor_f64* a, const uint32_t* labels_a, size_t ndim_a,
    const tenferro_tensor_f64* b, const uint32_t* labels_b, size_t ndim_b,
    const tenferro_tensor_f64* da,   // nullable (zero tangent)
    const tenferro_tensor_f64* db,   // nullable (zero tangent)
    int* status);
```

**Uses integer labels** (`u32`), not string notation, for C API ergonomics.

Julia example:
```julia
using cuTENSOR_jll

# Load GPU backend via jll-managed path
ccall((:tenferro_backend_load_cutensor, libtenferro), Cint, (Cstring,),
      cuTENSOR_jll.libcutensor_path)
```

---

## Future Phase: tenferro-hdf5

HDF5 I/O for `Tensor<T>`, using [`hdf5-rt`](https://github.com/tensor4all/hdf5-rt)
(runtime library loading via dlopen — same pattern as GPU backends).

The primary goal is to define a **common HDF5 file format** that Rust,
Julia, and Python implementations all follow, enabling cross-language
interoperability.

### File Format Specification

A tensor is stored as an HDF5 **N-dimensional dataset** with metadata
attributes. The dataset shape directly represents the tensor's logical
shape — no flattening to 1D.

```
/group/tensor_name           (HDF5 Dataset: N-D array, shape = tensor shape)
├── @format_version          (Attribute: string, e.g. "1.0")
├── @dtype                   (Attribute: string, e.g. "float64", "complex128")
└── @memory_order            (Attribute: string, "column_major" or "row_major")
```

**N-D dataset** (not 1D flat array): The HDF5 dataset's shape matches
the tensor's logical shape (e.g., a `[3, 4, 5]` tensor is stored as a
3×4×5 dataset). This enables direct reading by h5py (`f['tensor'][:]`
→ NumPy array) and HDF5.jl (`read(f, "tensor")` → Julia array) without
custom deserialization logic.

Note: ITensors.jl stores data as a 1D flat array with separate Index
metadata. We deliberately choose N-D datasets for broader interoperability
at the cost of ITensors.jl format compatibility.

**Data type encoding**:

| dtype string | Rust type | Julia type | NumPy dtype |
|-------------|-----------|------------|-------------|
| `float32` | `f32` | `Float32` | `np.float32` |
| `float64` | `f64` | `Float64` | `np.float64` |
| `complex64` | `Complex<f32>` | `ComplexF32` | `np.complex64` |
| `complex128` | `Complex<f64>` | `ComplexF64` | `np.complex128` |
| `int32` | `i32` | `Int32` | `np.int32` |
| `int64` | `i64` | `Int64` | `np.int64` |

**Complex number storage**: HDF5 has no native complex type (until
HDF5 2.0.0, which is adding one). Complex numbers are stored as an
HDF5 compound type. The field naming varies across ecosystems:

| Library | Field names |
|---------|------------|
| h5py (Python) | `"r"`, `"i"` |
| HDF5.jl (Julia) | reads both; writes `"r"`, `"i"` |
| Octave | `"real"`, `"imag"` |

**tenferro convention**: Use `"r"` and `"i"` (h5py/HDF5.jl compatible).
Readers should also accept `"real"`/`"imag"` for interoperability.
When HDF5 2.0.0 native complex types are available, prefer those.

**Memory order**: HDF5 internally stores N-D data in row-major (C) order.
The `memory_order` attribute records the writer's logical convention
(`"column_major"` or `"row_major"`). Details are an open issue (see below).

### Rust API

```rust
use tenferro_hdf5::{read_tensor, write_tensor};
use tenferro_device::LogicalMemorySpace;

// Write tensor to HDF5 file
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
write_tensor(&file, "group/tensor_name", &a)?;

// Read tensor from HDF5 file
let b: Tensor<f64> = read_tensor(&file, "group/tensor_name")?;
```

### Design Notes

- **Format first**: The file format spec is the primary deliverable.
  Rust, Julia, and Python I/O libraries are implementations of that spec.
- Separate crate to avoid mandatory C library dependency
- Uses `hdf5-rt` (dlopen): no compile-time HDF5 linking required
- Julia integration: Julia provides `libhdf5` path at runtime (same
  as cuTENSOR/hipTensor injection pattern)

### Open Issues

1. **Memory order and dimension convention**: HDF5 is internally
   row-major. Column-major writers (Julia, Fortran, tenferro) must
   choose between:
   - **Option A**: Follow HDF5.jl — reverse dimensions in HDF5
     (`[3,4]` Julia → `[4,3]` HDF5). Julia↔Julia round-trip is
     transparent, but h5py sees reversed shape.
   - **Option B**: Write dimensions as-is (`[3,4]` → `[3,4]` HDF5).
     h5py sees correct shape, but raw byte order is column-major
     inside a row-major container. Requires `memory_order` attribute
     for correct interpretation.
   - **Option C**: Always materialize as row-major before writing.
     All readers see consistent data, but Julia/Fortran writers pay
     a transpose cost.

2. **Complex number field names**: `"r"`/`"i"` is proposed, but
   HDF5 2.0.0 will introduce native complex types. Migration strategy
   (when to switch, backward compat) is TBD.

3. **Structured tensor formats**: How to store `BlockSparseTensor`,
   `DiagTensor` in HDF5. ITensors.jl uses a `"type"` attribute
   (`"Dense{Float64}"`, `"BlockSparse{Float64}"`) with type-specific
   sub-structure. Whether to adopt a similar convention or define a
   new one is TBD.

4. **ITensors.jl format compatibility**: ITensors.jl stores data as
   1D flat arrays with separate Index metadata. Our N-D dataset format
   is intentionally different for broader interoperability. Whether to
   provide an ITensors.jl compatibility reader/writer is TBD.

---

## Future Phase: tenferro-mdarray / tenferro-ndarray (Bridge Crates)

Separate bridge crates for converting between `Tensor<T>` and other
Rust array libraries. Conversions involve memory copies (not zero-copy),
which is acceptable since these are used at ecosystem boundaries, not
in hot loops.

```rust
// tenferro-mdarray crate
use tenferro_tensor::Tensor;
use mdarray::Array;

impl<T: ScalarBase> From<&Tensor<T>> for Array<T, Dyn> {
    fn from(tensor: &Tensor<T>) -> Self {
        let t = tensor.contiguous(MemoryOrder::RowMajor);  // mdarray is row-major
        Array::from_slice(t.as_slice(), t.dims())
    }
}

impl<T: ScalarBase> From<&Array<T, Dyn>> for Tensor<T> {
    fn from(array: &Array<T, Dyn>) -> Self {
        Tensor::from_slice(array.as_slice(), array.shape(), MemoryOrder::RowMajor).unwrap()
    }
}
```

```rust
// tenferro-ndarray crate
use tenferro_tensor::Tensor;
use ndarray::ArrayD;

impl<T: ScalarBase> From<&Tensor<T>> for ArrayD<T> {
    fn from(tensor: &Tensor<T>) -> Self {
        let t = tensor.contiguous(MemoryOrder::RowMajor);  // ndarray is row-major
        ArrayD::from_shape_vec(t.dims().to_vec(), t.as_slice().to_vec()).unwrap()
    }
}

impl<T: ScalarBase> From<&ArrayD<T>> for Tensor<T> {
    fn from(array: &ArrayD<T>) -> Self {
        // ndarray may not be contiguous; make contiguous first
        let standard = array.as_standard_layout();
        Tensor::from_slice(standard.as_slice().unwrap(), array.shape(), MemoryOrder::RowMajor).unwrap()
    }
}
```

**Design notes**:
- Each bridge crate depends only on `tenferro-tensor` + the target library
- Memory order is handled automatically: both mdarray and ndarray use
  row-major; tenferro defaults to column-major, so conversions transpose
- `contiguous()` call ensures data is laid out correctly before copy
- No feature flags on `tenferro-tensor` — dependencies are fully isolated

---

## Future Phase: tenferro-structured-rs (separate workspace)

Structured tensor types built on top of `Tensor<T>`. These live in a
**separate workspace** so they can be used by projects other than
tensor4all-rs without pulling in application-level dependencies.

### tenferro-diag

```rust
pub struct DiagTensor<T: ScalarBase> {
    diag: Tensor<T>,
    full_sizes: Vec<usize>,
}
```

### tenferro-blocksparse

Follows the **ITensors.jl/NDTensors pattern**: all blocks in a **single contiguous `DataBuffer`** with block offset mapping.

```rust
pub struct BlockSparseTensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    block_offsets: HashMap<BlockIndex, usize>,
    block_sizes: HashMap<BlockIndex, Vec<usize>>,
    full_sizes: Vec<usize>,
}
```

### tenferro-graded (future)

Quantum number graded tensors. BlockSparseTensor with sector-labeled
block indices and fusion-rule-constrained block structure.

---

## Future Phase: burn-tenferro (Optional Burn Backend for NN)

**Only needed when**: Users want Burn's NN modules with tenferro tensors.

| Burn Method | tenferro Implementation |
|---|---|
| `float_add` | `TensorPrims::reduce` or strided-kernel `axpy` |
| `float_matmul` | `TensorPrims contract (extended op)` (or decomposed to `batched_gemm`) |
| `float_mul` | `TensorPrims elementwise_mul (extended op)` |
| `float_exp/sin/...` | strided-kernel map (CPU only) |
| `float_reshape/permute/slice` | Metadata-only (zero-copy) |
| `float_sum/sum_dim` | `TensorPrims::reduce` (ReduceOp::Sum) |

---

## GPU Strategy

### Runtime Library Loading (Not Compile-Time Features)

GPU support is entirely runtime-based. A single compiled binary supports
CPU, NVIDIA, and AMD:

```
Build once -> Ship one binary
                |
            Runtime:
            |-- libcutensor.so found? -> NVIDIA GPU enabled
            |-- libhiptensor.so found? -> AMD GPU enabled
            +-- Neither? -> CPU only (no error, graceful fallback)
```

No `#[cfg(feature = "cuda")]` or `#[cfg(feature = "hip")]` in the
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

### GPU Backend Design (Future)

When GPU support is added, the `tenferro-device` crate will be extended with:

```rust
/// BackendRegistry manages all devices (CPU + GPU).
pub struct BackendRegistry {
    cpu: CpuBackend,
    gpu: Option<GpuBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self;  // CPU only
    pub fn load_cutensor(&mut self, path: &str) -> Result<()>;
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()>;
    pub fn available_compute_devices(&self) -> Vec<ComputeDevice>;
}

/// GPU vtable abstracting over cuTENSOR / hipTensor.
struct TensorLibVtable {
    create_handle: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32>,
    contract: Symbol<unsafe extern "C" fn(/* ... */) -> i32>,
    // ...
}

pub struct GpuBackend {
    vtable: TensorLibVtable,
    handle: *mut c_void,
    _lib: Library,  // prevent unloading
}
```

`libloading` will be an unconditional dependency (lightweight, no overhead when GPU absent).

### GPU Plan Caching

Plans are expensive to create on GPU. A `PlanCache` avoids re-creation
for repeated contraction patterns:

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

1. **No cuTENSOR equivalent** -- Apple has no dedicated N-dimensional tensor
   contraction library. Contraction must be decomposed into reshape + matmul,
   which is slow for high-rank tensors (rank-8+ common in tensor networks).
2. **Incompatible API paradigm** -- MPSGraph uses graph-based execution
   (Objective-C), not the operation-level C API used by cuTENSOR/hipTensor.
   Cannot share the dlopen vtable abstraction.
3. **Implementation cost vs benefit** -- The primary use case (tensor networks)
   runs well on CPU. GPU acceleration benefits come from NVIDIA/AMD HPC
   clusters, not Apple laptops.

---

## Custom Scalar Type Support

Users extend the system at the appropriate level:

| Level | What to implement | Result |
|-------|-------------------|--------|
| `ScalarBase` only | Basic trait bounds | Einsum via naive loop, map/reduce work |
| `ScalarBase` + custom GEMM | Custom `batched_gemm` in tenferro-prims | Einsum decomposes to diag → trace → permute → custom batched_gemm |
| Full `TensorPrims` | Custom backend implementation | Complete control over all core operations |
| Full `TensorPrims<A>` with extensions | Custom backend with extended ops (contract, elementwise_mul) | Maximum performance (has_extension_for returns true) |

**Algebra-parameterized dispatch** (via `TensorPrims<A>`):
- `impl TensorPrims<Standard> for CpuBackend` → faer/cblas GEMM for f64/f32/Complex, naive for i32/i64
- `impl TensorPrims<MaxPlus> for CpuBackend` (tenferro-tropical) → tropical-gemm SIMD
- `impl TensorPrims<Standard> for GpuBackend` [future] → cuTENSOR/hipTensor
- `impl TensorPrims<MyAlgebra> for CpuBackend` (user crate) → user-provided kernels

---

## Migration Strategy (tensor4all-rs)

1. Replace `tensor4all-tensorbackend::Storage` with tenferro types
2. Replace `DenseStorage<T>` (mdarray-based) with `Tensor<T>`
3. Adapt `TensorLike` trait to use tenferro Tensor
4. Remove mdarray dependency from tensor4all-rs core

This happens **after tenferro core is stable**.

---

## Verification Plan

### After POC (Phase 1 core):
```bash
cd tenferro-rs && cargo test -p tenferro-device -p tenferro-prims \
    -p tenferro-tensor -p tenferro-einsum
```
- Unit tests for all Tensor operations
- Zero-copy verification: assert same buffer pointer after view ops
- **TensorPrims conformance tests**: verify CpuBackend passes the same
  test suite that GPU backends will use
- Integer type einsum test (i32, i64 via naive backend)
- Benchmark: tenferro einsum vs current tensor4all-rs mdarray-einsum

### After future phases:

**tenferro-algebra**:
- Tropical semiring contraction test (tenferro-algebra + naive backend)
- Custom type extensibility tests: `ModInt<P>` test type through
  all three dispatch tiers

**tenferro-linalg**:
- Cross-validate SVD/QR results against ndtensors-rs
- Complex SVD test

**tenferro-autograd**:
- Numerical gradient checks (finite difference vs reverse-mode AD)
- Forward-mode vs reverse-mode consistency
- Complex-valued gradient test (Wirtinger calculus)

**tenferro-capi**:
- Round-trip test: Julia -> C API -> Rust -> C API -> Julia
- Backend loading test: `tenferro_backend_load_cutensor` / `tenferro_backend_load_hiptensor`
- ChainRules.jl integration test with Zygote.jl

---

## Key Files Reference

| Component | Source | Destination |
|---|---|---|
| ScalarBase trait | `strided-rs/strided-traits/src/scalar.rs` | Stays in strided-rs; used by tenferro via dependency |
| ElementOp | `strided-rs/strided-traits/src/element_op.rs` | Stays in strided-rs; used by tenferro via dependency |
| StridedArray/View | `strided-rs/strided-view/src/` | Stays in strided-rs; used directly in tenferro-tensor (DataBuffer), tenferro-prims (TensorPrims), tenferro-device (StridedError) |
| map/reduce kernels | `strided-rs/strided-kernel/src/` | Stays in strided-rs; tenferro-prims will depend on it |
| einsum2_into | `strided-rs/strided-einsum2/src/lib.rs` | **Absorbed** into tenferro-prims `PrimDescriptor::Contract` (CPU contraction) [future] |
| BgemmBackend | `strided-rs/strided-einsum2/src/backend.rs` | **Absorbed** into tenferro-prims `TensorPrims::batched_gemm` [future] |
| reduce_trace_axes | `strided-rs/strided-einsum2/src/trace.rs` | **Absorbed** into tenferro-prims `TensorPrims::trace` [future] |
| diagonal_view | `strided-rs/strided-view/src/view.rs` | Stays in strided-rs; used by `Tensor::diagonal()` [future] |
| opteinsum | `strided-rs/strided-opteinsum/src/lib.rs` | **Absorbed** into tenferro-einsum [future] |
| Algebra traits | `omeinsum-rs/src/algebra/` | **Absorbed** into tenferro-algebra [future] |
| Backend trait | `omeinsum-rs/src/backend/traits.rs` | **Absorbed** into tenferro-prims (evolved into TensorPrims) |
| cuTENSOR wrapper | `omeinsum-rs/src/backend/cuda/cutensor/` | **Absorbed** into tenferro-device (GPU vtable) [future] |
| PlanCache | `omeinsum-rs/src/backend/cuda/cutensor/contract.rs` | **Absorbed** into tenferro-device [future] |
| faer bridge | `ndtensors-rs/.../faer_interop.rs` | tenferro-linalg [future] |
| contract_vjp | `ndtensors-rs/.../contract/naive.rs` | tenferro-autograd [future] |
| TrackedTensor | `ndtensors-rs/.../autodiff/tensor.rs` | tenferro-autograd [future] |
| C API patterns | `tensor4all-rs/crates/tensor4all-capi/src/` | tenferro-capi [future] |

---

## Future Considerations

### Complex-valued differentiation rules for linear algebra

Complex SVD, QR, eigen decompositions require non-trivial backward rules (Wirtinger calculus). Key references:

- **[BackwardsLinalg.jl](https://github.com/GiggleLiu/BackwardsLinalg.jl)**: Reference implementations for complex backward rules.
- **[MatrixFactorizations.jl](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl)**: Extended factorizations with ChainRules.jl integration.

### JAX / PyTorch integration via C-FFI

tenferro-capi supports integration with JAX and PyTorch via ctypes:

```
JAX / PyTorch -> Python wrapper -> ctypes FFI -> tenferro-capi -> Rust
```

- JAX: `jax.custom_vjp` + `jax.pure_callback()`
- PyTorch: `torch.autograd.Function` with custom forward/backward

### Multi-GPU distributed contraction

The current design targets single-GPU execution. For large-scale tensor
network computations, distributing contractions across multiple GPUs
is desirable.

**Chosen memory/compute separation**:

The design now separates memory placement from execution target:

1. **Logical memory space**: where the tensor buffer lives
   (`MainMemory`, GPU memory spaces, etc.)
2. **Compute device**: where the kernel executes (`Cpu`, `Cuda`, `Hip`)

Rules:
- `MainMemory` always exists.
- A logical GPU memory space may be attached to one GPU or multiple GPUs.
- A single GPU compute device is attached to exactly one logical memory space.
- Cross-space operations are **not implicit**; they fail by default.
- Movement across spaces is explicit via `to_memory_space_async`.

```rust
pub enum LogicalMemorySpace {
    MainMemory,
    GpuMemory { space_id: usize },
}

pub enum ComputeDevice {
    Cpu { device_id: usize },
    Cuda { device_id: usize },
    Hip { device_id: usize },
}

pub enum OpKind { Contract, BatchedGemm, Reduce, Trace, Permute, ElementwiseMul }

pub struct Tensor<T> {
    // ...
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>, // None => infer from memory space + op
}

// tenferro-device API:
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    op_kind: OpKind,
) -> Result<Vec<ComputeDevice>>; // Err(NoCompatibleComputeDevice) when empty

impl<T> Tensor<T> {
    pub fn to_memory_space_async(&self, dst: LogicalMemorySpace) -> Result<Tensor<T>>;
    // same src/dst space: always Ok, zero-copy no-op
}
```

**Open questions**:
- **Batch-level parallelism**: Distribute independent batch slices of
  `batched_gemm` across GPUs (simplest, no inter-GPU communication
  during GEMM).
- **Tensor splitting**: Split a large contraction into sub-problems
  across GPUs (e.g., split the contracted dimension `k` across devices,
  then reduce). Requires inter-GPU communication (NCCL/RCCL).
- **Contraction tree parallelism**: In N-ary einsum, independent
  sub-trees can execute on different GPUs concurrently.
- **P2P capability detection**: Not all GPU pairs support P2P (depends
  on NVLink/PCIe topology). Need runtime query for P2P availability.
- **Interaction with MPI**: For multi-node multi-GPU (e.g., 4 nodes ×
  4 GPUs), how does GPU-level distribution interact with
  [`rsmpi-rt`](https://github.com/tensor4all/rsmpi-rt)-based
  node-level distribution in tensor4all-rs?

### GPU/CPU overlap and asynchronous execution

The current `TensorPrims::execute` API is synchronous — it blocks until
the operation completes. For GPU backends, this leaves the CPU idle during
GPU computation:

```
GPU: [== einsum A ==][== einsum B ==]
CPU: [    idle       ][    idle       ]
```

With asynchronous execution, the CPU could prepare the next operands or
perform independent computation while the GPU is busy:

```
GPU: [== einsum A ==][== einsum B ==][== einsum C ==]
CPU:                  [prepare B     ][prepare C     ]
```

#### Chosen approach: Tensor embeds async state

Rather than introducing separate `einsum` / `einsum_async` functions, the
`Tensor<T>` struct itself carries an optional `CompletionEvent` that tracks
pending accelerator computation. This follows the PyTorch model where
accelerator tensors are always potentially async:

```rust
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,  // None = ready, Some = pending accelerator work
}
```

This enables transparent accelerator-to-accelerator operation chaining
without CPU synchronization:

```rust
// Accelerator operations return immediately with event attached
let c = einsum("ij,jk->ik", &[&a_gpu, &b_gpu])?;
//  → GPU submit, c.event = Some(event_1), returns immediately

let d = einsum("ij,jk->ik", &[&c, &e_gpu])?;
//  → detects c.event → sets up stream dependency → no CPU wait
//  → GPU submit, d.event = Some(event_2), returns immediately

// CPU data access triggers implicit synchronization
println!("{:?}", d.view());  // view() calls wait() internally
```

The `einsum` API does not change — the same function handles both sync
(CPU) and async (accelerator) execution transparently. For CPU tensors,
`event` is always `None` with zero overhead.

Key methods:

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Wait for any pending accelerator computation to complete.
    /// No-op for CPU tensors or already-completed operations.
    pub fn wait(&self) { ... }

    /// Check if data is ready without blocking.
    pub fn is_ready(&self) -> bool { ... }

    // Existing methods that access data auto-sync:
    // view(), view_mut(), conj(), contiguous() — all call wait() internally
}
```

#### Alternatives considered

1. **Separate `einsum_async` function**: Rejected — splits the API
   unnecessarily. A single `einsum` that works for both sync and async
   is simpler.

2. **Trait-based (`TensorArg` trait with `wait()`)**: More extensible
   (could also support lazy expressions), but adds API complexity
   (`&[&dyn TensorArg<T>]` or generics explosion). Can be introduced
   later if needed — implementing the trait for `Tensor<T>` is backward
   compatible.

3. **Contraction tree-level pipelining only**: Useful as an internal
   optimization within N-ary einsum, but does not help when the user
   chains independent einsum calls manually. The Tensor-embedded event
   approach handles both cases.

#### Applicability beyond GPU: multi-threaded CPU parallelism

The `CompletionEvent` mechanism is not limited to GPU/accelerator backends.
It applies equally to **multi-threaded CPU execution**:

- **Contraction tree parallelism**: Independent subtrees of an N-ary einsum
  can be dispatched to different CPU threads. Each subtree result is a
  `Tensor` with a `CompletionEvent` that completes when the thread finishes.
  The parent contraction waits on both child events before proceeding.

- **User-level parallelism**: Independent `einsum` calls can run on separate
  threads. Passing the resulting `Tensor` (with pending event) to a
  subsequent `einsum` automatically chains via event dependencies — no
  explicit synchronization needed.

- **Compute-device model extension**: `ComputeDevice::Cpu { device_id }`
  matches `Cuda/Hip` and supports multiple CPU execution contexts.
  `device_id: 0` represents the default global thread pool (all cores).
  Higher IDs can map to Rayon `ThreadPool` instances bound to specific
  core sets via `core_affinity`, enabling NUMA-aware and cache-local
  execution. Memory placement remains independent via
  `LogicalMemorySpace`. The `CompletionEvent` mechanism remains unchanged.

This means the same `einsum` API and `wait()` / `is_ready()` interface
covers GPU async, CPU multi-thread, and potentially other execution models
(FPGA, distributed) without modification.

**Current status**: The `event` field is present in the POC `Tensor<T>`
struct as `Option<CompletionEvent>` (placeholder type) to signal the design
intent. Actual async execution will be implemented with accelerator and
multi-threaded backends.

### Tensor / TensorView ownership split

#### Motivation

The POC defines `permute(&self) -> Tensor<T>` as "zero-copy," but `Tensor<T>`
owns its `DataBuffer<T>`. For true zero-copy view operations, the new tensor
must share the original's data buffer. Two main approaches exist:

- **Arc-based sharing** (PyTorch model): Single `Tensor` type, buffer wrapped
  in `Arc`. Simple API, but buffer uniqueness is a runtime check.
- **Tensor / TensorView split** (ndarray model, `String` / `&str` pattern):
  Owned `Tensor` and borrowed `TensorView<'a>`. Buffer uniqueness is a
  compile-time guarantee.

#### Chosen direction: Tensor + TensorView

The Tensor/TensorView split is more idiomatic Rust and provides stronger
compile-time guarantees for buffer reuse:

```rust
/// Owned tensor. Holds exclusive ownership of the data buffer.
/// Can be moved (consumed) for buffer reuse.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,
}

/// Borrowed tensor view. References a Tensor's data buffer.
/// Zero-copy, lifetime-tied to the source Tensor.
///
/// Public API always returns TensorView with event = None
/// (wait is performed before construction). The event field
/// is used only by crate-internal as_operand_view() for
/// accelerator pipeline chaining.
pub struct TensorView<'a, T: ScalarBase> {
    data: &'a DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<&'a CompletionEvent>,  // Public API: None; internal operand path: propagated
}
```

#### API design

**Tensor (owned) methods:**

```rust
impl<T: ScalarBase> Tensor<T> {
    // --- Public: Borrow → TensorView (zero-copy, waits if pending) ---
    fn view(&self) -> TensorView<'_, T>;
    fn permute(&self, perm: &[usize]) -> Result<TensorView<'_, T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<TensorView<'_, T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<TensorView<'_, T>>;

    // --- Internal: Non-blocking operand view (event propagated) ---
    pub(crate) fn as_operand_view(&self) -> TensorView<'_, T>;

    // --- Consume self → Tensor (buffer reuse, guaranteed) ---
    fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;
    fn into_conj(self) -> Tensor<T>;

    // --- Borrow → new Tensor (new allocation, waits if pending) ---
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;
}
```

**TensorView methods:**

```rust
impl<'a, T: ScalarBase> TensorView<'a, T> {
    // Further view ops (data already ready — event is None in public API)
    fn permute(&self, perm: &[usize]) -> Result<TensorView<'a, T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<TensorView<'a, T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<TensorView<'a, T>>;

    // Materialize: copy data into a new owned Tensor
    fn to_tensor(&self) -> Tensor<T>;
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;
}
```

**einsum takes &Tensor references (not TensorView):**

```rust
// einsum takes Tensor references. Internally uses as_operand_view()
// to propagate pending events without blocking (pipeline-safe).
// Einstein notation handles permute/broadcast — no need to create
// TensorView before calling einsum.
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

// Consuming variant: moves Tensors, enabling buffer reuse for intermediates.
// Buffer reuse is guaranteed (compile-time) — no runtime refcount check.
pub fn einsum_owned<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: Vec<Tensor<T>>,
) -> Result<Tensor<T>>;
```

#### Ownership safety examples

```rust
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, ColumnMajor);
let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, ColumnMajor);

// einsum takes &Tensor — notation handles permutation/broadcast
let c = einsum("ij,jk->ik", &[&a, &b])?;       // → new Tensor
let c_t = einsum("ji,jk->ik", &[&a, &b])?;     // transposed a via notation

// View operations for CPU data inspection (waits if pending)
let at = a.permute(&[1, 0])?;           // TensorView borrowing a
assert_eq!(at.dims(), &[4, 3]);         // data is ready to read

// Compile-time safety: can't consume while borrowed
let at = a.permute(&[1, 0])?;
let d = einsum_owned("...", vec![a]);    // ERROR: at borrows a
drop(at);
let d = einsum_owned("...", vec![a])?;   // OK: borrow released

// Consuming einsum: buffer reuse is deterministic
let d = einsum_owned("ij,jk->ik", vec![a, b])?;
// a, b are moved — compiler guarantees no other references exist
// Their buffers can be reused for intermediates — zero allocation
```

#### Comparison with Arc-based approach

| Aspect | TensorView (chosen) | Arc-based |
|--------|---------------------|-----------|
| Buffer uniqueness | Compile-time guarantee | Runtime `Arc::strong_count` check |
| `into_` buffer reuse | Always succeeds | May fail if views exist (fallback to alloc) |
| API types | `Tensor` + `TensorView` | `Tensor` only |
| Lifetime complexity | Yes (`'a` propagates) | None |
| Runtime overhead | Zero | Atomic refcount on clone/drop |
| Rust idiom | `String`/`&str`, `Vec`/`&[T]` | `Arc<T>` |
| Misuse detection | Compile error | Silent fallback to allocation |

#### Coherence with async execution (CompletionEvent propagation)

Public view APIs (`view()`, `permute()`, etc.) **wait** before returning,
so users always receive ready-to-read data. Accelerator pipeline chaining
is handled internally by `as_operand_view()`, which propagates pending
events without blocking.

**Two-tier API contract:**

| Tier | Methods | Event handling | User visibility |
|------|---------|---------------|-----------------|
| **Public (CPU-read)** | `view()`, `permute()`, `broadcast()`, `diagonal()`, `view_mut()`, `to_tensor()`, `contiguous()`, `conj()` | **Wait** if pending, return ready data | Yes |
| **Internal (pipeline)** | `pub(crate) as_operand_view()` | **Propagate** event as `Option<&'a CompletionEvent>` | No (crate-internal) |
| **Accelerator ops** | `einsum` (takes `&[&Tensor]`) | Internally calls `as_operand_view()`, **detects** events → sets up stream dependency | Yes (but event handling is transparent) |

**GPU pipeline preserved via internal path:**

```rust
let a = einsum("ij,jk->ik", &[&x_gpu, &y_gpu])?;
//  → einsum internally: as_operand_view(&x_gpu), as_operand_view(&y_gpu)
//  → GPU submit, a.event = Some(event_1), returns immediately

let b = einsum("ij,jk->ik", &[&a, &z_gpu])?;
//  → einsum internally: as_operand_view(&a) → event_1 propagated
//  → detects event_1 → sets up stream dependency → no CPU wait
//  → GPU submit, b.event = Some(event_2), returns immediately

// Public API access triggers synchronization
let bv = b.view();  // wait(event_2), then return TensorView (event=None)
```

Note: Einstein notation subsumes `permute`/`broadcast` for einsum operands
(`"ji,jk->ik"` transposes the first operand), so users rarely need to
create TensorView explicitly when chaining accelerator operations.

**Implementation note — `wait(&self)` requires interior mutability:**

`Tensor::wait(&self)` must clear the `event` field through a shared
reference. This requires interior mutability (e.g., `Cell<Option<CompletionEvent>>`
or similar). This is an implementation detail that does not affect the
public API contract.

**Potential issues to monitor:**

1. **Non-einsum accelerator operations**: Future element-wise operations
   (add, mul, etc.) on accelerators will also need the `as_operand_view()`
   internal path to maintain pipelines. This is architecturally consistent
   but must be applied to every new accelerator-capable operation.

2. **TensorView cannot be passed to einsum**: Since `einsum` takes
   `&[&Tensor]`, a user cannot pass a `TensorView` directly. This is
   intentional — Einstein notation already handles permute/broadcast,
   and `einsum` uses `as_operand_view()` internally for pipeline safety.
   If a use case arises that genuinely requires passing views to einsum,
   a trait-based approach can be introduced later.

The Arc approach remains viable if lifetime ergonomics prove too burdensome
in practice. Switching from TensorView to Arc is backward-compatible for
callers (TensorView disappears, all methods return Tensor).

**Current status**: POC uses `Tensor<T>` only (no TensorView yet).
View operations (`permute`, `broadcast`, `diagonal`) return `Tensor<T>`
with `todo!()` bodies. The TensorView split will be implemented when
view operations are filled in.

### einsum_into (implemented in POC)

The POC includes accumulating `einsum_*_into` variants alongside the allocating versions:

- **`einsum_into`**, **`einsum_with_subscripts_into`**, **`einsum_with_plan_into`** --
  write into a caller-provided output buffer with BLAS-style accumulation
  (`output = alpha * einsum(...) + beta * output`). Avoids output allocation in hot loops.

**Not yet implemented** (future optimization):
- **`einsum_owned_into`** -- consumes input tensors, reuses their buffers as intermediate
  workspace when reference count is 1 (buffer pool pattern from strided-opteinsum).
  Maximum performance for N-ary contraction trees.

### Insights from ITensor Julia ecosystem

| Aspect | ITensor Julia | tenferro | Notes |
|---|---|---|---|
| Sparse storage | DOK-of-Arrays | Single DataBuffer + offset map | tenferro is GPU-friendly |
| Axis fusion | FusionStyle dispatch | Not yet designed | Critical for quantum number tensors |

### Relationship with mdarray / mdarray-linalg

| | mdarray / mdarray-linalg | tenferro-* |
|---|---|---|
| Role | **numpy equivalent** -- general-purpose multidimensional array | **PyTorch equivalent** -- high-performance tensor library |
| Memory | Owned `Array<T, D>` | `DataBuffer<T>` (CPU/GPU) |
| GPU | No | cuTENSOR, hipTensor (no Metal) |
| Autodiff | No | tenferro-autograd (VJP/JVP) [future] |
| Dispatch | Direct function calls | `TensorPrims` trait (backend selection) |

Both are needed. mdarray is a foundational array library; tenferro builds a
richer tensor ecosystem with GPU support and automatic differentiation.

tenferro-linalg and mdarray-linalg are **parallel** (both call faer directly),
not serial:

```
faer (SVD, QR, eigen)
    ^                ^
tenferro-linalg  mdarray-linalg-faer
(Tensor<T>       (Array<T, D>
 -> MatRef)       -> MatRef)
```
