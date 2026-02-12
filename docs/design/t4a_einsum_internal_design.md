# t4a-einsum Internal Design

> This document details the internal architecture of the t4a einsum subsystem,
> covering the tensor operation protocol layer (`t4a-tensorops`), the einsum
> engine (`t4a-einsum`), and backend implementations for CPU and GPU.
>
> This is a companion to the higher-level design in
> `t4a_unified_tensor_backend.md` and the algorithm comparison in
> `einsum_algorithm_comparison.md`.

## Layered Architecture

```
Layer 4: Einsum Engine (t4a-einsum)
         High-level API on Tensor<T>
         Absorbs strided-opteinsum + omeinsum-rs
         N-ary contraction tree, algebra dispatch, backward
         ↓
Layer 3: Tensor Type (t4a-tensor)
         Tensor<T> = DataBuffer + shape + strides
         Zero-copy view ops, as_strided_view() bridge
         ↓
Layer 2: Tensor Operation Protocol (t4a-tensorops)
         "Tensor BLAS": cuTENSOR / hipTensor compatible unified trait
         Absorbs strided-einsum2 (binary contraction pipeline)
         TensorOps on raw Storage<T> + TensorMeta
         5 operations: Contraction, Reduction, Permutation,
                       ElementwiseBinary, ElementwiseTrinary
         Plan-based execution
         CPU impl depends on strided-kernel (stays in strided-rs)
         GPU dispatch via t4a-device
         ↓
Shared:  GPU Device Layer (t4a-device)
         Device enum, BackendRegistry, TensorLibVtable
         Runtime GPU discovery (dlopen, caller-injected .so paths)
         Used by: t4a-buffer, t4a-tensorops, t4a-linalg
         ↓
Layer 1: Backend Implementations
         CPU: strided-einsum2 pipeline (faer or cblas)
         NVIDIA: cuTENSOR via t4a-device (dlopen)
         AMD: hipTensor via t4a-device (dlopen)

Foundation: strided-rs (independent workspace, not absorbed)
         strided-traits → strided-view → strided-kernel
         General-purpose strided array library, no BLAS dependency
```

### Design Rationale

GiggleLiu proposed that strided-rs should serve as a "tensor-level BLAS" —
the CPU counterpart to cuTENSOR — with standardized interfaces applicable to
CPU, GPU, and tropical tensors. We introduce `t4a-tensorops` as a low-level
"Tensor BLAS" protocol layer that abstracts over all backends using
cuTENSOR-compatible operator enums on raw `Storage<T>` + `TensorMeta`.
`t4a-tensor` defines the `Tensor<T>` type above it, and `t4a-einsum`
provides a high-level einsum API on `Tensor<T>`, internally delegating
binary contractions to `t4a-tensorops`. Custom closure-based operations
(which cannot run on GPU) are not part of `TensorOps`; users access
strided-kernel directly via `Tensor::as_strided_view()`.

This is motivated by the observation that **cuTENSOR and hipTensor have
nearly identical APIs** (AMD intentionally mirrors NVIDIA's API). A single
trait can abstract over both, plus the CPU backend.

---

## Layer 2: t4a-tensorops

### Operation Categories

Mirroring cuTENSOR/hipTensor, `t4a-tensorops` defines five operation categories:

| Operation | cuTENSOR | hipTensor | CPU (strided-rs) |
|-----------|----------|-----------|-------------------|
| **Contraction** | `cutensorContract` | `hiptensorContract` | strided-einsum2 pipeline |
| **Reduction** | `cutensorReduce` | `hiptensorReduce` | `reduce_axis` |
| **Permutation** | `cutensorPermute` | `hiptensorPermute` | `StridedView::permute` + copy |
| **ElementwiseBinary** | `cutensorElementwiseBinaryExecute` | `hiptensorElementwiseBinaryExecute` | `zip_map2_into` |
| **ElementwiseTrinary** | `cutensorElementwiseTrinaryExecute` | `hiptensorElementwiseTrinaryExecute` | `zip_map3_into` |

### Plan-Based Execution

All operations follow the cuTENSOR pattern of descriptor → plan → execute:

```rust
/// Metadata describing a tensor's layout (no data pointer).
pub struct TensorMeta {
    pub shape: SmallVec<[usize; 6]>,
    pub strides: SmallVec<[usize; 6]>,
}

/// Unary operators (cuTENSOR compatible subset).
#[repr(u32)]
pub enum UnaryOp {
    Identity = 1,
    Sqrt = 2,
    Relu = 8,
    Conj = 9,
    Neg = 25,
    Abs = 24,
    Exp = 22,
    Log = 23,
    // ... extensible
}

/// Binary operators for elementwise and reduction.
#[repr(u32)]
pub enum BinaryOp {
    Add = 3,
    Mul = 5,
    Max = 6,
    Min = 7,
}

/// Reduce operators (same as BinaryOp, used for clarity).
pub type ReduceOp = BinaryOp;
```

### TensorOps Trait

```rust
pub trait TensorOps {
    type Storage<T>;
    type ContractionPlan;
    type PermutationPlan;
    type ReductionPlan;
    type ElementwiseBinaryPlan;
    type ElementwiseTrinaryPlan;

    // === Contraction ===
    fn create_contraction_plan<T: OpScalar>(
        &self,
        a: &TensorMeta, modes_a: &[i32],
        b: &TensorMeta, modes_b: &[i32],
        c: &TensorMeta, modes_c: &[i32],
    ) -> Result<Self::ContractionPlan>;

    fn contract<T: OpScalar>(
        &self,
        plan: &Self::ContractionPlan,
        alpha: T, a: &Self::Storage<T>,
                  b: &Self::Storage<T>,
        beta: T,  c: &mut Self::Storage<T>,
    ) -> Result<()>;

    // === Reduction ===
    fn create_reduction_plan<T: OpScalar>(
        &self,
        a: &TensorMeta, modes_a: &[i32], op_a: UnaryOp,
        c: &TensorMeta, modes_c: &[i32],
        op_reduce: ReduceOp,
    ) -> Result<Self::ReductionPlan>;

    fn reduce<T: OpScalar>(
        &self,
        plan: &Self::ReductionPlan,
        alpha: T, a: &Self::Storage<T>,
        beta: T,  c: &mut Self::Storage<T>,
    ) -> Result<()>;

    // === Permutation ===
    fn create_permutation_plan<T: OpScalar>(
        &self,
        a: &TensorMeta, modes_a: &[i32], op_a: UnaryOp,
        b: &TensorMeta, modes_b: &[i32],
    ) -> Result<Self::PermutationPlan>;

    fn permute<T: OpScalar>(
        &self,
        plan: &Self::PermutationPlan,
        alpha: T, a: &Self::Storage<T>,
                  b: &mut Self::Storage<T>,
    ) -> Result<()>;

    // === Elementwise Binary ===
    fn create_elementwise_binary_plan<T: OpScalar>(
        &self,
        a: &TensorMeta, modes_a: &[i32], op_a: UnaryOp,
        c: &TensorMeta, modes_c: &[i32], op_c: UnaryOp,
        d: &TensorMeta, modes_d: &[i32],
        op_ac: BinaryOp,
    ) -> Result<Self::ElementwiseBinaryPlan>;

    fn elementwise_binary<T: OpScalar>(
        &self,
        plan: &Self::ElementwiseBinaryPlan,
        alpha: T, a: &Self::Storage<T>,
        gamma: T, c: &Self::Storage<T>,
                  d: &mut Self::Storage<T>,
    ) -> Result<()>;

    // === Elementwise Trinary ===
    fn create_elementwise_trinary_plan<T: OpScalar>(
        &self,
        a: &TensorMeta, modes_a: &[i32], op_a: UnaryOp,
        b: &TensorMeta, modes_b: &[i32], op_b: UnaryOp,
        c: &TensorMeta, modes_c: &[i32], op_c: UnaryOp,
        d: &TensorMeta, modes_d: &[i32],
        op_ab: BinaryOp,
        op_abc: BinaryOp,
    ) -> Result<Self::ElementwiseTrinaryPlan>;

    fn elementwise_trinary<T: OpScalar>(
        &self,
        plan: &Self::ElementwiseTrinaryPlan,
        alpha: T, a: &Self::Storage<T>,
        beta: T,  b: &Self::Storage<T>,
        gamma: T, c: &Self::Storage<T>,
                  d: &mut Self::Storage<T>,
    ) -> Result<()>;
}
```

### Custom Closures: Use strided-kernel Directly

The `TensorOps` trait uses cuTENSOR-compatible operator enums exclusively.
It does **not** provide a closure-based API, because GPU backends cannot
execute arbitrary Rust closures (cuTENSOR only supports predefined operator
enums, and GPU shaders require special compilation).

For custom element-wise operations not covered by the enum set, users
access strided-kernel directly via `Tensor::as_strided_view()`:

```rust
// TensorOps: enum-based, works on CPU and GPU
backend.elementwise_binary(&plan, alpha, &a, gamma, &c, &mut d)?;

// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.as_strided_view();
let b_view = tensor_b.as_strided_view();
strided_kernel::zip_map2_into(&mut out_view, &a_view, &b_view, |a, b| a * b + 1.0);
```

This keeps `t4a-tensorops` purely cuTENSOR/hipTensor-compatible. strided-kernel
(in the independent strided-rs workspace) provides cache-optimized iteration
for arbitrary closures: dimension fusion, L1 blocking, importance-weighted
ordering, SIMD auto-vectorization.

### Convenience Functions (Plan-Free API)

For one-off operations, convenience functions hide plan creation:

```rust
/// Convenience: creates plan internally, executes, discards plan.
pub fn contract_once<B: TensorOps, T: OpScalar>(
    backend: &B,
    alpha: T, a: &B::Storage<T>, meta_a: &TensorMeta, modes_a: &[i32],
              b: &B::Storage<T>, meta_b: &TensorMeta, modes_b: &[i32],
    beta: T,  c: &mut B::Storage<T>, meta_c: &TensorMeta, modes_c: &[i32],
) -> Result<()> {
    let plan = backend.create_contraction_plan::<T>(
        meta_a, modes_a, meta_b, modes_b, meta_c, modes_c,
    )?;
    backend.contract(&plan, alpha, a, b, beta, c)
}
```

For repeated operations with the same shape, users create a plan once and
reuse it.

---

## Layer 1: Backend Implementations

### Runtime Backend Discovery (via t4a-device)

GPU device discovery, handle management, and vendor library loading
are provided by the **t4a-device** crate — a shared infrastructure
crate used by `t4a-buffer`, `t4a-tensorops`, and `t4a-linalg`.

The caller (Julia, Python, or standalone Rust) provides the path to
the shared library. This avoids:
- Requiring separate builds for NVIDIA vs AMD
- Automatic search finding wrong library versions
- Forcing users to set environment variables

```rust
// t4a-device crate
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

C API for Julia/Python integration (exposed via `t4a-capi`):

```c
int t4a_backend_load_cutensor(const char* libcutensor_path);
int t4a_backend_load_hiptensor(const char* libhiptensor_path);
```

Julia example:

```julia
using cuTENSOR_jll
ccall((:t4a_backend_load_cutensor, libt4a), Cint, (Cstring,),
      cuTENSOR_jll.libcutensor_path)
```

`libloading` is an unconditional dependency of `t4a-device` (always
linked, lightweight, no overhead when GPU libraries are absent).

### GPU Backend: Common Vtable (in t4a-device)

cuTENSOR and hipTensor have nearly identical C APIs. A single function
pointer table in `t4a-device` abstracts over both:

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
    create_elementwise_trinary: Symbol<...>,
    elementwise_trinary_exec: Symbol<...>,

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

The `GpuBackend` struct (in `t4a-device`) wraps the vtable. `t4a-tensorops`
implements `TensorOps` for `GpuBackend`:

```rust
// t4a-device crate
pub struct GpuBackend {
    vtable: TensorLibVtable,
    handle: *mut c_void,
    _lib: Library,  // prevent unloading
}

// t4a-tensorops crate
impl TensorOps for GpuBackend {
    type Storage<T> = GpuBuffer<T>;  // device memory
    type ContractionPlan = GpuPlan;

    fn create_contraction_plan<T: OpScalar>(...) -> Result<GpuPlan> {
        // 1. Create tensor descriptors via vtable
        // 2. Create operation descriptor via vtable
        // 3. Estimate workspace via vtable
        // 4. Create plan via vtable
        // 5. Cleanup temporaries
    }

    fn contract<T: OpScalar>(...) -> Result<()> {
        // Allocate workspace, call vtable.contract
    }
}
```

### GPU Plan Caching (in t4a-device)

Plans are expensive to create on GPU. A cache (in `t4a-device`) avoids
re-creation for repeated contraction patterns (common in tensor network
algorithms):

```rust
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct PlanCacheKey {
    pub shapes: Vec<Vec<usize>>,
    pub strides: Vec<Vec<usize>>,
    pub modes: Vec<Vec<i32>>,
    pub dtype: u32,
}

pub struct PlanCache {
    cache: HashMap<PlanCacheKey, GpuPlan>,
    capacity: usize,
}

impl PlanCache {
    pub fn get_or_create<T: OpScalar>(
        &mut self,
        backend: &GpuBackend,
        key: PlanCacheKey,
        meta_a: &TensorMeta, modes_a: &[i32],
        meta_b: &TensorMeta, modes_b: &[i32],
        meta_c: &TensorMeta, modes_c: &[i32],
    ) -> Result<&GpuPlan> {
        if !self.cache.contains_key(&key) {
            if self.cache.len() >= self.capacity {
                // Evict oldest entry
                let k = self.cache.keys().next().cloned().unwrap();
                self.cache.remove(&k);
            }
            let plan = backend.create_contraction_plan::<T>(
                meta_a, modes_a, meta_b, modes_b, meta_c, modes_c,
            )?;
            self.cache.insert(key.clone(), plan);
        }
        Ok(self.cache.get(&key).unwrap())
    }
}
```

### CPU Backend

The CPU backend implements `TensorOps` using strided-rs kernels.

#### GEMM Backend Selection (Compile-Time)

```toml
# t4a-einsum/Cargo.toml
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

`t4a-einsum` depends only on `cblas-sys` function signatures and is
agnostic to the CBLAS provider.

#### CPU Contraction Plan

The CPU plan caches the analysis results from strided-einsum2:

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

    /// Element-wise bypass flag (no contraction axes → Hadamard product)
    elementwise_bypass: bool,

    /// Blocking plan from strided-kernel (for cache-optimized iteration)
    blocking: Option<BlockingPlan>,
}
```

Plan creation:

```rust
impl TensorOps for CpuBackend {
    type Storage<T> = Vec<T>;
    type ContractionPlan = CpuContractionPlan;

    fn create_contraction_plan<T: OpScalar>(
        &self,
        meta_a: &TensorMeta, modes_a: &[i32],
        meta_b: &TensorMeta, modes_b: &[i32],
        meta_c: &TensorMeta, modes_c: &[i32],
    ) -> Result<CpuContractionPlan> {
        // 1. Classify modes: batch, left, right, contracted
        let (batch, left, right, contracted) =
            classify_modes(modes_a, modes_b, modes_c);

        // 2. Compute permutations to canonical order
        //    A[left, contracted, batch], B[contracted, right, batch]
        let a_perm = compute_permutation(modes_a, &left, &contracted, &batch);
        let b_perm = compute_permutation(modes_b, &contracted, &right, &batch);

        // 3. Fusability check (from strided-einsum2)
        //    Can dimension groups be fused without copying?
        let a_fusable = try_fuse_group(&meta_a.strides, &a_perm, ...);
        let b_fusable = try_fuse_group(&meta_b.strides, &b_perm, ...);

        // 4. Element-wise bypass detection
        let elementwise_bypass = contracted.is_empty()
            && left.is_empty() && right.is_empty();

        // 5. GEMM dispatch selection
        let gemm = select_gemm::<T>();  // faer or cblas based on feature

        Ok(CpuContractionPlan { a_perm, b_perm, a_fusable, b_fusable, ... })
    }

    fn contract<T: OpScalar>(
        &self,
        plan: &CpuContractionPlan,
        alpha: T, a: &Vec<T>, b: &Vec<T>,
        beta: T,  c: &mut Vec<T>,
    ) -> Result<()> {
        // Execute using cached plan:
        // - Skip fusability re-check (cached)
        // - Skip mode classification (cached)
        // - Use pre-computed permutations
        // - Element-wise bypass if flagged
        // - GEMM via selected backend
    }
}
```

#### CPU Contraction Pipeline (strided-einsum2)

The six-step pipeline from strided-einsum2, adapted to work with
`TensorOps` interface:

```
1. Trace pre-reduction
   Sum out axes appearing only in one operand before GEMM.
   Conjugation materialized during reduce (conj flag → false for GEMM).

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
   If fusable → zero-copy metadata extraction.
   If not → allocate col-major buffer and copy.

5. GEMM dispatch
   Call selected backend: faer::bgemm or cblas::dgemm/zgemm.
   Naive loop fallback for non-Scalar types (integers, tropical).

6. Copy-back
   If output was non-contiguous, copy from internal buffer back to
   the original strided destination.
```

Steps 1-4 are analyzed during `create_contraction_plan`. Steps 5-6
are executed during `contract`.

#### CPU Elementwise Implementation

The CPU backend translates operator enums into closures and delegates to
strided-kernel internally:

```rust
impl TensorOps for CpuBackend {
    fn elementwise_binary<T: OpScalar>(
        &self,
        plan: &Self::ElementwiseBinaryPlan,
        alpha: T, a: &Vec<T>,
        gamma: T, c: &Vec<T>,
                  d: &mut Vec<T>,
    ) -> Result<()> {
        // Dispatch based on plan.op_a, plan.op_c, plan.op_ac
        // Internally calls strided-kernel's zip_map2_into with
        // the appropriate closure derived from the operator enums
        let f = match (plan.op_a, plan.op_c, plan.op_ac) {
            (UnaryOp::Identity, UnaryOp::Identity, BinaryOp::Add) =>
                |a, c| alpha * a + gamma * c,
            (UnaryOp::Conj, UnaryOp::Identity, BinaryOp::Add) =>
                |a, c| alpha * a.conj() + gamma * c,
            // ... other combinations
        };
        zip_map2_into(&mut dest_view, &a_view, &c_view, f)
    }
}
```

For operations not covered by the enum set, users bypass `TensorOps` and
call strided-kernel directly via `Tensor::as_strided_view()`. See the
"Custom Closures" section above.

---

## Layer 4: Einsum Engine (t4a-einsum)

High-level einsum API on `Tensor<T>`. Internally extracts `storage()`
and `meta()` from each tensor and delegates binary contractions to
`t4a-tensorops`.

### Public API

Three variants, all same-type `T` (no mixed-type inputs):

```rust
/// Allocating — returns new tensor.
pub fn einsum<T: ScalarBase>(
    inputs: &[&Tensor<T>],
    input_labels: &[&[i32]],
    output_labels: &[i32],
) -> Result<Tensor<T>>;

/// Into — writes into caller's output buffer.
/// Supports accumulation: output = α·einsum(...) + β·output.
pub fn einsum_into<T: ScalarBase>(
    inputs: &[&Tensor<T>],
    input_labels: &[&[i32]],
    output: &mut Tensor<T>,
    output_labels: &[i32],
    alpha: T, beta: T,
) -> Result<()>;

/// Owned — consumes input tensors, reuses their buffers as
/// intermediate workspace (avoids allocation when Arc refcount == 1).
pub fn einsum_owned_into<T: ScalarBase>(
    inputs: Vec<Tensor<T>>,
    input_labels: &[&[i32]],
    output: &mut Tensor<T>,
    output_labels: &[i32],
    alpha: T, beta: T,
) -> Result<()>;
```

`einsum` is a convenience wrapper over `einsum_into` with `alpha=1, beta=0`.
`einsum_owned_into` enables the buffer pool optimization: when
`Arc::strong_count() == 1`, input buffers are recycled as workspace
for intermediate contractions in the N-ary tree.

### N-ary Contraction

For N > 2 inputs, the einsum engine uses contraction tree optimization
(omeco) to find the optimal pairwise contraction order, then dispatches
each pairwise contraction through `TensorOps::contract`.

```rust
/// Internal: dispatches to TensorOps backend.
fn einsum_impl<B: TensorOps, T: OpScalar>(
    backend: &B,
    inputs: &[(&B::Storage<T>, &TensorMeta, &[i32])],
    output_modes: &[i32],
) -> Result<B::Storage<T>> {
    match inputs.len() {
        0 => Err(Error::NoInputs),
        1 => single_tensor_op(backend, inputs[0], output_modes),
        2 => {
            // Direct binary contraction via TensorOps
            let plan = backend.create_contraction_plan::<T>(...)?;
            backend.contract(&plan, ...)?;
            Ok(result)
        }
        _ => {
            // N-ary: optimize contraction order, then execute pairwise
            let tree = omeco::optimize(inputs, output_modes, Strategy::Greedy)?;
            execute_tree(backend, &tree, inputs)
        }
    }
}
```

#### Optimizations from strided-opteinsum

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

### Algebra Dispatch

Algebra-aware backend selection, integrated with `TensorOps`:

```rust
/// Dispatch GEMM/contraction backend based on algebra and scalar type.
fn dispatch_contraction<A: Algebra, B: TensorOps>(
    backend: &B,
    a: ..., b: ..., c: ...,
) -> Result<()> {
    // For GPU tensors: always use TensorOps::contract
    //   (cuTENSOR/hipTensor handles everything)
    //
    // For CPU:
    //   Standard<f64/f32/Complex> → faer or cblas GEMM
    //   Standard<i32/i64/u32/u64> → naive loop
    //   MaxPlus<f64>              → tropical-gemm (future SIMD)
    //   Custom algebra            → user-provided BgemmBackend or naive
}
```

Tropical semiring (MaxPlus, MinPlus, MaxMul) types from `t4a-algebra`
use the naive loop fallback on CPU. Future optimization: SIMD-optimized
tropical-gemm via `TypeId`-based runtime dispatch (from omeinsum-rs).

### Backward Pass (VJP/JVP)

Contraction VJP for automatic differentiation:

```rust
/// VJP: grad_A = contract(grad_C, B), grad_B = contract(A, grad_C)
pub fn contract_vjp<B: TensorOps, T: OpScalar>(
    backend: &B,
    a: &B::Storage<T>, meta_a: &TensorMeta, modes_a: &[i32],
    b: &B::Storage<T>, meta_b: &TensorMeta, modes_b: &[i32],
    grad_c: &B::Storage<T>, meta_c: &TensorMeta, modes_c: &[i32],
) -> Result<(B::Storage<T>, B::Storage<T>)> {
    // grad_A = contract(grad_C, B) with appropriate mode relabeling
    // grad_B = contract(A, grad_C) with appropriate mode relabeling
}

/// JVP: dC = contract(dA, B) + contract(A, dB)  (Leibniz rule)
pub fn contract_jvp<B: TensorOps, T: OpScalar>(...) -> Result<B::Storage<T>> {
    // Two contractions + elementwise addition
}
```

Both VJP and JVP go through `TensorOps::contract`, so they work on
CPU and GPU uniformly.

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** dlopen | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** feature | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based only** | cuTENSOR-compatible operator enums; custom closures via strided-kernel directly |
| libloading | **Always ON** (in t4a-device) | Lightweight, no overhead when GPU absent |
| .so path | **Caller-injected** (via t4a-device) | No auto-search; Julia/Python manage library versions |

---

## Crate Dependency Graph

```
t4a-scalar
    │
    ├──────────────────────────────┐
    ↓                              ↓
t4a-view                      t4a-buffer ←── t4a-device
    │                              │          (Device enum, BackendRegistry,
    ├──── t4a-algebra ←────────────┤           TensorLibVtable, libloading)
    │                              │               │
    ↓                              ↓               │
t4a-tensorops ←────────────────────┘               │
    │  (← strided-kernel, ← t4a-device)           │
    │  ("Tensor BLAS": TensorOps on Storage<T>     │
    │   + TensorMeta, plan-based execution)        │
    ↓                                              │
t4a-tensor (Tensor<T> = DataBuffer + shape         │
    │       + strides, zero-copy view ops)         │
    │                                              │
    ├──────────────────────────────┐               │
    ↓                              ↓               │
t4a-einsum                   t4a-linalg            │
    │  (high-level einsum      (← faer,            │
    │   on Tensor<T>)           ← t4a-device)      │
    │                              │               │
    ├──────────────────────────────┤               │
    ↓                              ↓               │
t4a-autograd                 t4a-capi              │
                              (wraps t4a-device     │
                               for library loading) │
                                   │               │
                                   └───────────────┘
```

---

## References

- [cuTENSOR API Reference](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html)
- [hipTensor API Reference](https://rocm.docs.amd.com/projects/hipTensor/en/latest/api-reference/api-reference.html)
- [cudarc cuTENSOR bindings](https://github.com/coreylowman/cudarc/tree/main/src/cutensor)
- [omeinsum-rs cuTENSOR wrapper](https://github.com/tensor4all/omeinsum-rs) (internal)
- [strided-rs einsum2 pipeline](https://github.com/tensor4all/strided-rs) (internal)
- [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md)
- [t4a Unified Tensor Backend Design](./t4a_unified_tensor_backend.md)
