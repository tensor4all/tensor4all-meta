# tenferro-einsum Internal Design

> This document details the internal architecture of the tenferro einsum
> subsystem, covering the tensor operation protocol layer (`tenferro-tensorops`),
> the einsum engine (`tenferro-einsum`), and backend implementations for CPU
> and GPU.
>
> This is a companion to the higher-level design in
> `tenferro_unified_tensor_backend.md` and the algorithm comparison in
> `einsum_algorithm_comparison.md`.
>
> **POC repository**: <https://github.com/tensor4all/tenferro-rs/>

## Layered Architecture

```
Layer 4: Einsum Engine (tenferro-einsum)
         High-level API on Tensor<T>
         Absorbs strided-opteinsum + omeinsum-rs
         N-ary contraction tree, algebra dispatch, backward
         ↓
Layer 3: Tensor Type (tenferro-tensor)
         Tensor<T> = DataBuffer + shape + strides
         Zero-copy view ops, view()/view_mut() bridge
         ↓
Layer 2: Tensor Operation Protocol (tenferro-tensorops)
         "Tensor BLAS": cuTENSOR / hipTensor compatible unified trait
         Absorbs strided-einsum2 (binary contraction pipeline)
         TensorOps on StridedView<T> / StridedViewMut<T>
         4 operations: Contraction, Reduction, Permutation,
                       ElementwiseBinary
         Plan-based execution (contraction)
         CPU impl depends on strided-kernel (stays in strided-rs)
         GPU dispatch via tenferro-device
         ↓
Shared:  Device Layer (tenferro-device)
         Device enum (Cpu, Cuda, Hip), Error, Result
         Used by: tenferro-tensorops, tenferro-tensor, tenferro-einsum

Foundation: strided-rs (independent workspace, not absorbed)
         strided-traits -> strided-view -> strided-kernel
         General-purpose strided array library, no BLAS dependency
```

### Design Rationale

GiggleLiu proposed that strided-rs should serve as a "tensor-level BLAS" --
the CPU counterpart to cuTENSOR -- with standardized interfaces applicable to
CPU, GPU, and tropical tensors. We introduce `tenferro-tensorops` as a low-level
"Tensor BLAS" protocol layer that abstracts over all backends using
a cuTENSOR-compatible interface on `StridedView<T>` / `StridedViewMut<T>`.
`tenferro-tensor` defines the `Tensor<T>` type above it, and `tenferro-einsum`
provides a high-level einsum API on `Tensor<T>`, internally delegating
binary contractions to `tenferro-tensorops`. Custom closure-based operations
(which cannot run on GPU) are not part of `TensorOps`; users access
strided-kernel directly via `Tensor::view()`.

This is motivated by the observation that **cuTENSOR and hipTensor have
nearly identical APIs** (AMD intentionally mirrors NVIDIA's API). A single
trait can abstract over both, plus the CPU backend.

**POC status**: The [tenferro-rs POC](https://github.com/tensor4all/tenferro-rs/)
implements the four-crate structure (`tenferro-device`, `tenferro-tensorops`,
`tenferro-tensor`, `tenferro-einsum`) with stub implementations. The `TensorOps`
trait and public einsum API are defined. `CpuBackend` is the only backend; GPU
backends, `BackendRegistry`, and `TensorLibVtable` are future work.

---

## Layer 2: tenferro-tensorops

### Operation Categories

Mirroring cuTENSOR/hipTensor, `tenferro-tensorops` defines four operation
categories in the POC:

| Operation | cuTENSOR | hipTensor | CPU (strided-rs) |
|-----------|----------|-----------|-------------------|
| **Contraction** | `cutensorContract` | `hiptensorContract` | strided-einsum2 pipeline |
| **Reduction** | `cutensorReduce` | `hiptensorReduce` | `reduce_axis` |
| **Permutation** | `cutensorPermute` | `hiptensorPermute` | `StridedView::permute` + copy |
| **ElementwiseBinary** | `cutensorElementwiseBinaryExecute` | `hiptensorElementwiseBinaryExecute` | `zip_map2_into` |

ElementwiseTrinary is not included in the POC. It may be added in the
future when needed.

### ReduceOp Enum

The POC defines a standalone `ReduceOp` enum (not a type alias for
`BinaryOp`):

```rust
/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}
```

There are no `UnaryOp` or `BinaryOp` enums in the POC. These may be added
in the future for cuTENSOR-compatible elementwise operations.

### Plan-Based Execution

Contraction follows the cuTENSOR pattern of descriptor -> plan -> execute:

```rust
/// Describes a tensor contraction in terms of index modes.
///
/// Each mode is a u32 label. Modes shared between two tensors are
/// contracted (summed over). Follows the cuTENSOR
/// cutensorOperationDescriptor pattern.
#[derive(Debug, Clone)]
pub struct ContractionDescriptor {
    /// Index mode labels for tensor A.
    pub modes_a: Vec<u32>,
    /// Index mode labels for tensor B.
    pub modes_b: Vec<u32>,
    /// Index mode labels for the output tensor C.
    pub modes_c: Vec<u32>,
}

/// Pre-computed execution plan for a tensor contraction.
///
/// Created by TensorOps::plan_contraction. Encapsulates kernel
/// selection and workspace allocation so that TensorOps::contract
/// can execute with zero additional allocation.
pub struct ContractionPlan<T: ScalarBase> {
    _marker: PhantomData<T>,
}
```

The other three operations (elementwise_binary, reduce, permute) do not
use a plan in the POC -- they execute directly. Plan-based variants for
these may be added in the future for GPU kernel reuse.

### TensorOps Trait

The POC `TensorOps` trait has four operations. All functions are
**associated functions** (no `&self` receiver), and the trait has **no
associated types**. Inputs and outputs are `StridedView<T>` /
`StridedViewMut<T>` directly (not `Storage<T>` + `TensorMeta`).

```rust
/// Backend trait for tensor operations (cuTENSOR-compatible protocol).
///
/// Provides plan-based contraction, element-wise binary operations,
/// reductions, and permutations. All operations follow the BLAS/cuTENSOR
/// C = alpha * op(inputs) + beta * C pattern.
pub trait TensorOps {
    /// Create an execution plan for a tensor contraction.
    fn plan_contraction<T: ScalarBase>(
        desc: &ContractionDescriptor,
        dims_a: &[usize],
        dims_b: &[usize],
        dims_c: &[usize],
    ) -> Result<ContractionPlan<T>>;

    /// Execute a tensor contraction: C = alpha * contract(A, B) + beta * C.
    fn contract<T: ScalarBase>(
        plan: &ContractionPlan<T>,
        alpha: T,
        a: &StridedView<T>,
        b: &StridedView<T>,
        beta: T,
        c: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Element-wise binary operation with mode alignment:
    /// C_{modes_c} = alpha * A_{modes_a} + beta * C_{modes_c}.
    fn elementwise_binary<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        beta: T,
        c: &mut StridedViewMut<T>,
        modes_c: &[u32],
    ) -> Result<()>;

    /// Reduction over modes not present in the output:
    /// C_{modes_c} = alpha * reduce_op(A_{modes_a}) + beta * C_{modes_c}.
    fn reduce<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        beta: T,
        c: &mut StridedViewMut<T>,
        modes_c: &[u32],
        op: ReduceOp,
    ) -> Result<()>;

    /// Permute tensor modes: B_{modes_b} = alpha * A_{modes_a}.
    fn permute<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        b: &mut StridedViewMut<T>,
        modes_b: &[u32],
    ) -> Result<()>;
}
```

Key differences from the original design:

1. **No associated types** -- The trait is not generic over storage or plan
   types. `ContractionPlan<T>` is a concrete struct.
2. **Associated functions, not methods** -- No `&self` receiver. Call as
   `CpuBackend::contract(...)` instead of `backend.contract(...)`.
3. **StridedView/StridedViewMut directly** -- Not `Storage<T>` + `TensorMeta`.
   This simplifies the interface for the CPU backend.
4. **Four operations** -- No `ElementwiseTrinary`.
5. **Modes are `u32`** -- Not `i32` (matching cuTENSOR's unsigned mode labels).

### Custom Closures: Use strided-kernel Directly

The `TensorOps` trait does not provide a closure-based API, because GPU
backends cannot execute arbitrary Rust closures (cuTENSOR only supports
predefined operator enums, and GPU shaders require special compilation).

For custom element-wise operations, users access strided-kernel directly
via `Tensor::view()`:

```rust
// TensorOps: enum-based, works on CPU and GPU
CpuBackend::elementwise_binary(alpha, &a, &modes_a, beta, &mut c, &modes_c)?;

// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.view();
let b_view = tensor_b.view();
strided_kernel::zip_map2_into(&mut out_view, &a_view, &b_view, |a, b| a * b + 1.0);
```

This keeps `tenferro-tensorops` purely cuTENSOR/hipTensor-compatible.
strided-kernel (in the independent strided-rs workspace) provides
cache-optimized iteration for arbitrary closures: dimension fusion,
L1 blocking, importance-weighted ordering, SIMD auto-vectorization.

---

## Layer 1: Backend Implementations

### Device Layer (tenferro-device)

The POC `tenferro-device` crate provides a minimal device abstraction:

```rust
/// Compute device on which tensor data resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device.
    Cpu,
    /// NVIDIA CUDA device with a specific device ID.
    Cuda { device_id: usize },
    /// AMD HIP device with a specific device ID.
    Hip { device_id: usize },
}

/// Error type used across the tenferro workspace.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("device error: {0}")]
    DeviceError(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

/// Result type alias using Error.
pub type Result<T> = std::result::Result<T, Error>;
```

The POC does **not** include `BackendRegistry`, `GpuBackend`, or
`TensorLibVtable`. These are future work for GPU support. The current
`tenferro-device` crate is intentionally minimal: `Device` enum, `Error`,
and `Result`.

### CPU Backend

The CPU backend (`CpuBackend`) implements `TensorOps` using strided-rs
kernels. In the POC, all operations are stubs (`todo!()`). The planned
implementation follows the strided-einsum2 pipeline.

#### GEMM Backend Selection (Compile-Time, Future)

```toml
# tenferro-tensorops/Cargo.toml (planned)
[features]
default = ["faer"]
faer = ["dep:faer"]
cblas = ["dep:cblas-sys"]

[dependencies]
faer = { version = "...", optional = true }
cblas-sys = { version = "...", optional = true }
```

| Feature | GEMM source | Use case |
|---------|------------|----------|
| `faer` (default) | Pure Rust, zero external deps | Standalone apps, guaranteed build |
| `cblas` | Requires `cblas-src` or `cblas-inject` | HPC, Julia integration |

When `cblas` is selected, the actual CBLAS implementation is provided by the
downstream user:

- **`cblas-src`**: Links OpenBLAS or MKL (standalone Rust apps)
- **`cblas-inject`**: Julia injects `libblastrampoline` symbols at runtime

`tenferro-tensorops` depends only on `cblas-sys` function signatures and is
agnostic to the CBLAS provider.

#### CPU Contraction Plan (Future)

The CPU plan will cache the analysis results from strided-einsum2:

```rust
pub struct CpuContractionPlan {
    /// Canonical mode ordering (left, contracted, batch for A; etc.)
    a_perm: Vec<usize>,
    b_perm: Vec<usize>,
    c_perm: Vec<usize>,

    /// Fusability results from try_fuse_group
    a_fusable: bool,
    b_fusable: bool,

    /// Dimension sizes after classification
    batch_size: usize,
    left_size: usize,
    right_size: usize,
    contract_size: usize,

    /// GEMM dispatch
    gemm: GemmDispatch,  // Faer | Cblas | Naive

    /// Pre-computed workspace size
    workspace_size: usize,

    /// Element-wise bypass flag (no contraction axes -> Hadamard product)
    elementwise_bypass: bool,

    /// Blocking plan from strided-kernel (for cache-optimized iteration)
    blocking: Option<BlockingPlan>,
}
```

#### CPU Contraction Pipeline (strided-einsum2, Future)

The six-step pipeline from strided-einsum2, adapted to work with
the `TensorOps` interface:

```
1. Trace pre-reduction
   Sum out axes appearing only in one operand before GEMM.
   Conjugation materialized during reduce (conj flag -> false for GEMM).

2. Permutation to canonical order
   A[left, contracted, batch], B[contracted, right, batch], C[left, right, batch]
   Batch-last for column-major contiguity.

3. Element-wise bypass
   If contracted, left, and right are all empty (pure Hadamard product),
   call zip_map2_into instead of GEMM.

4. Fusability check (try_fuse_group)
   Test whether dimension groups can be fused into a single contiguous
   dimension without copying. Sorts (dim, stride) pairs by |stride|
   ascending and verifies stride[i] * dim[i] == stride[i+1].
   If fusable -> zero-copy metadata extraction.
   If not -> allocate col-major buffer and copy.

5. GEMM dispatch
   Call selected backend: faer::bgemm or cblas::dgemm/zgemm.
   Naive loop fallback for non-Scalar types (integers, tropical).

6. Copy-back
   If output was non-contiguous, copy from internal buffer back to
   the original strided destination.
```

Steps 1-4 are analyzed during `plan_contraction`. Steps 5-6
are executed during `contract`.

### GPU Backend (Future)

GPU support via cuTENSOR/hipTensor is planned but not in the POC. The
design involves:

#### GPU Backend Discovery (Future, via tenferro-device)

The caller (Julia, Python, or standalone Rust) will provide the path to
the shared library. This avoids:
- Requiring separate builds for NVIDIA vs AMD
- Automatic search finding wrong library versions
- Forcing users to set environment variables

```rust
// Future tenferro-device additions
pub struct BackendRegistry {
    cpu: CpuBackend,
    gpu: Option<GpuBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self { ... }  // CPU only
    pub fn load_cutensor(&mut self, path: &str) -> Result<()> { ... }
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()> { ... }
    pub fn available_devices(&self) -> Vec<Device> { ... }
}
```

#### GPU Vtable (Future, in tenferro-device)

cuTENSOR and hipTensor have nearly identical C APIs. A single function
pointer table will abstract over both:

```rust
struct TensorLibVtable {
    // Handle lifecycle
    create_handle: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32>,
    destroy_handle: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,

    // Tensor descriptor
    create_tensor_descriptor: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, desc: *mut *mut c_void,
        num_modes: u32, extent: *const i64, stride: *const i64,
        data_type: u32, alignment: u32,
    ) -> i32>,
    destroy_tensor_descriptor: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,

    // Contraction
    create_contraction: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, desc: *mut *mut c_void,
        desc_a: *mut c_void, modes_a: *const i32, op_a: u32,
        desc_b: *mut c_void, modes_b: *const i32, op_b: u32,
        desc_c: *mut c_void, modes_c: *const i32, op_c: u32,
        desc_d: *mut c_void, modes_d: *const i32,
        compute: *mut c_void,
    ) -> i32>,

    // Plan
    create_plan: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, plan: *mut *mut c_void,
        desc: *mut c_void, pref: *mut c_void, ws_size: u64,
    ) -> i32>,
    destroy_plan: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,

    // Execution
    contract: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, plan: *mut c_void,
        alpha: *const c_void, a: *const c_void, b: *const c_void,
        beta: *const c_void, c: *const c_void, d: *mut c_void,
        workspace: *mut c_void, ws_size: u64, stream: *mut c_void,
    ) -> i32>,

    // Permutation, reduction, elementwise (same pattern)
    create_permutation: Symbol<...>,
    permute: Symbol<...>,
    create_reduction: Symbol<...>,
    reduce_exec: Symbol<...>,
    create_elementwise_binary: Symbol<...>,
    elementwise_binary_exec: Symbol<...>,

    // Workspace
    estimate_workspace_size: Symbol<...>,

    // Cleanup
    destroy_operation_descriptor: Symbol<...>,
}

impl TensorLibVtable {
    fn load_cutensor(lib: &Library) -> Result<Self> {
        Ok(Self {
            create_handle: lib.get(b"cutensorCreate")?,
            destroy_handle: lib.get(b"cutensorDestroy")?,
            create_contraction: lib.get(b"cutensorCreateContraction")?,
            contract: lib.get(b"cutensorContract")?,
            // ...
        })
    }

    fn load_hiptensor(lib: &Library) -> Result<Self> {
        Ok(Self {
            create_handle: lib.get(b"hiptensorCreate")?,
            destroy_handle: lib.get(b"hiptensorDestroy")?,
            create_contraction: lib.get(b"hiptensorCreateContraction")?,
            contract: lib.get(b"hiptensorContract")?,
            // ...
        })
    }
}
```

The `GpuBackend` struct will wrap the vtable:

```rust
pub struct GpuBackend {
    vtable: TensorLibVtable,
    handle: *mut c_void,
    _lib: Library,  // prevent unloading
}
```

#### GPU Plan Caching (Future, in tenferro-device)

Plans are expensive to create on GPU. A cache will avoid re-creation for
repeated contraction patterns (common in tensor network algorithms):

```rust
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct PlanCacheKey {
    pub shapes: Vec<Vec<usize>>,
    pub strides: Vec<Vec<usize>>,
    pub modes: Vec<Vec<u32>>,
    pub dtype: u32,
}

pub struct PlanCache {
    cache: HashMap<PlanCacheKey, GpuPlan>,
    capacity: usize,
}
```

---

## Layer 4: Einsum Engine (tenferro-einsum)

High-level einsum API on `Tensor<T>`. Internally extracts `view()` from
each tensor and delegates binary contractions to `tenferro-tensorops`.

### Public API

The POC defines three API levels, all same-type `T` (no mixed-type inputs):

```rust
/// Execute einsum using string notation.
///
/// Parses the subscript string, optimizes the contraction order, and
/// executes the contraction. The backend is selected automatically from
/// the tensors' device.
///
/// Parentheses in the subscript string specify contraction order
/// explicitly (e.g., "ij,(jk,kl)->il" contracts B and C first).
pub fn einsum<T: ScalarBase>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Execute einsum with pre-built Subscripts.
///
/// Avoids re-parsing the subscript string on each call. Useful when
/// the same contraction pattern is applied to tensors of varying shapes.
pub fn einsum_with_subscripts<T: ScalarBase>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Execute einsum with a pre-optimized ContractionTree.
///
/// Avoids both subscript parsing and contraction order optimization.
/// Ideal for hot loops where the same contraction is executed repeatedly
/// on tensors of the same shape.
pub fn einsum_with_plan<T: ScalarBase>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;
```

The three levels form a progressive optimization ladder:

| Level | Avoids | Use case |
|-------|--------|----------|
| `einsum` | Nothing | One-off contractions, prototyping |
| `einsum_with_subscripts` | String parsing | Same pattern, varying shapes |
| `einsum_with_plan` | Parsing + tree optimization | Hot loops, fixed shapes |

**Future API extensions**: `einsum_into` and `einsum_owned_into` are not in
the POC but are planned for accumulation (`C = alpha * einsum + beta * C`)
and buffer reuse optimizations.

### Subscripts

```rust
/// Einsum subscripts using integer labels (omeinsum-rs compatible).
#[derive(Debug, Clone)]
pub struct Subscripts {
    /// Index labels for each input tensor.
    pub inputs: Vec<Vec<u32>>,
    /// Index labels for the output tensor.
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create subscripts from integer label arrays.
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self { ... }

    /// Parse subscripts from NumPy/PyTorch-style string notation.
    ///
    /// Each character (a-z, A-Z) represents a dimension label.
    /// Parentheses specify contraction order explicitly.
    ///
    /// Examples:
    ///   "ij,jk->ik"           -- matrix multiplication
    ///   "ii->i"               -- diagonal extraction
    ///   "ij,(jk,kl)->il"      -- contract B*C first, then A
    pub fn parse(notation: &str) -> Result<Self> { ... }
}
```

### ContractionTree

```rust
/// Contraction tree determining pairwise contraction order for N-ary einsum.
pub struct ContractionTree { ... }

impl ContractionTree {
    /// Automatically compute an optimized contraction order.
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self>;

    /// Manually build a contraction tree from a pairwise contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self>;
}
```

### String Notation Examples

```rust
// Matrix multiplication
let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();

// Trace
let tr = einsum("ii->", &[&a]).unwrap();

// Outer product
let outer = einsum("i,j->ij", &[&v, &v]).unwrap();

// Batch GEMM
let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();

// Explicit contraction order with parentheses
let d = einsum("ij,(jk,kl)->il", &[&a, &b, &c]).unwrap();

// Integer label notation
let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let c = einsum_with_subscripts(&subs, &[&a, &b]).unwrap();

// Pre-optimized plan for hot loops
let tree = ContractionTree::optimize(&subs, &[&[3, 4], &[4, 5]]).unwrap();
for _ in 0..n_steps {
    let c = einsum_with_plan(&tree, &[&a, &b]).unwrap();
}
```

### N-ary Contraction (Internal)

For N > 2 inputs, the einsum engine uses contraction tree optimization
to find the optimal pairwise contraction order, then dispatches each
pairwise contraction through `TensorOps::contract`.

```rust
/// Internal: dispatches to TensorOps backend.
fn einsum_impl<T: ScalarBase>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>> {
    match operands.len() {
        0 => Err(Error::InvalidArgument("no inputs".into())),
        1 => single_tensor_op(operands[0], &subscripts),
        2 => {
            // Direct binary contraction via TensorOps
            let plan = CpuBackend::plan_contraction::<T>(...)?;
            CpuBackend::contract(&plan, ...)?;
            Ok(result)
        }
        _ => {
            // N-ary: optimize contraction order, then execute pairwise
            let tree = ContractionTree::optimize(&subscripts, ...)?;
            execute_tree(&tree, operands)
        }
    }
}
```

#### Optimizations from strided-opteinsum (Future)

- **Borrowed-view passthrough**: Leaf nodes return borrows, not clones.
- **Permutation-only detection**: Metadata-only transformation for
  nodes that only permute axes (no contraction).
- **Buffer pool** (opt-in): Reuse intermediate buffers across pairwise
  contractions. Trades memory for speed.
- **Direct root write**: Final contraction writes into user's output
  buffer (no extra allocation).

#### Single-Tensor Fast Paths

From strided-opteinsum, five-step pipeline with zero-allocation fast paths:

1. **Full trace** (`"ii->"` or all-same indices): Single loop over diagonal.
2. **Partial trace** (`"iij->j"`): Detect repeated pair, direct loop.
3. **General reduce**: `reduce_axis()` per summed axis.
4. **Broadcast/repeat**: Stride-0 view + materialize.
5. **Duplicate** (`"i->ii"`): Write to diagonal positions.

### Algebra Dispatch (Future)

Algebra-aware backend selection, integrated with `TensorOps`:

```rust
/// Dispatch GEMM/contraction backend based on algebra and scalar type.
fn dispatch_contraction<T: ScalarBase>(...) -> Result<()> {
    // For GPU tensors: always use TensorOps (cuTENSOR/hipTensor)
    //
    // For CPU:
    //   Standard<f64/f32/Complex> -> faer or cblas GEMM
    //   Standard<i32/i64/u32/u64> -> naive loop
    //   MaxPlus<f64>              -> tropical-gemm (future SIMD)
    //   Custom algebra            -> user-provided BgemmBackend or naive
}
```

Tropical semiring (MaxPlus, MinPlus, MaxMul) types from a future
`tenferro-algebra` crate use the naive loop fallback on CPU. Future
optimization: SIMD-optimized tropical-gemm via `TypeId`-based runtime
dispatch (from omeinsum-rs).

### Backward Pass (VJP/JVP, Future)

Contraction VJP for automatic differentiation:

```rust
/// VJP: grad_A = contract(grad_C, B), grad_B = contract(A, grad_C)
pub fn contract_vjp<T: ScalarBase>(
    a: &Tensor<T>, modes_a: &[u32],
    b: &Tensor<T>, modes_b: &[u32],
    grad_c: &Tensor<T>, modes_c: &[u32],
) -> Result<(Tensor<T>, Tensor<T>)>;

/// JVP: dC = contract(dA, B) + contract(A, dB)  (Leibniz rule)
pub fn contract_jvp<T: ScalarBase>(...) -> Result<Tensor<T>>;
```

Both VJP and JVP go through `TensorOps::contract`, so they work on
CPU and GPU uniformly.

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** dlopen (future) | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** feature (future) | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based only** | cuTENSOR-compatible operator enums; custom closures via strided-kernel directly |
| libloading | **Always ON** (future, in tenferro-device) | Lightweight, no overhead when GPU absent |
| .so path | **Caller-injected** (future, via tenferro-device) | No auto-search; Julia/Python manage library versions |

---

## Crate Dependency Graph

The POC has four crates with the following dependency structure:

```
strided-rs (independent workspace):
strided-traits -> strided-view -> strided-kernel

tenferro-rs (workspace, depends on strided-rs):

tenferro-device
    (Device enum, Error, Result)
    (depends on: strided-view for StridedError)
        │
        ├──────────────────────────────┐
        ↓                              ↓
tenferro-tensorops              tenferro-tensor
    (TensorOps trait,               (Tensor<T> = DataBuffer
     CpuBackend,                     + dims + strides + offset,
     ContractionDescriptor,          zero-copy view ops)
     ContractionPlan,               (depends on: tenferro-device,
     ReduceOp)                       strided-view, strided-traits,
    (depends on: tenferro-device,    num-traits)
     strided-view, strided-traits)      │
        │                              │
        └──────────┬───────────────────┘
                   ↓
            tenferro-einsum
                (einsum, einsum_with_subscripts,
                 einsum_with_plan,
                 Subscripts, ContractionTree)
                (depends on: tenferro-device,
                 tenferro-tensorops,
                 tenferro-tensor,
                 strided-traits)
```

Future crates (not in POC):
- `tenferro-algebra` -- Semiring/Algebra traits, tropical types
- `tenferro-linalg` -- SVD, QR, eigen (CPU: faer, GPU: cuSOLVER)
- `tenferro-autograd` -- TrackedTensor, DualTensor, VJP/JVP
- `tenferro-capi` -- C FFI for Julia/Python integration

---

## References

- [cuTENSOR API Reference](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html)
- [hipTensor API Reference](https://rocm.docs.amd.com/projects/hipTensor/en/latest/api-reference/api-reference.html)
- [cudarc cuTENSOR bindings](https://github.com/coreylowman/cudarc/tree/main/src/cutensor)
- [omeinsum-rs cuTENSOR wrapper](https://github.com/tensor4all/omeinsum-rs) (internal)
- [strided-rs einsum2 pipeline](https://github.com/tensor4all/strided-rs) (internal)
- [tenferro-rs POC](https://github.com/tensor4all/tenferro-rs/)
- [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md)
- [tenferro Unified Tensor Backend Design](./tenferro_unified_tensor_backend.md)
