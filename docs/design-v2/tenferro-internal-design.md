# v2 tenferro-rs Internal Design

**Date:** 2026-04-03
**Status:** Draft
**Repo:** tenferro-rs
**Parent:** `README.md`
**Related:** `computegraph-design.md`, `chainrules-design.md`, `tidu-design.md`, `backend-architecture.md`, `primitive-catalog.md`

---

## I. Purpose

This document defines the internal crate structure and type design of
tenferro-rs v2. The key design driver is that **all computation is
graph-based**: every operation (einsum, linalg, elementwise) produces nodes in
a `Fragment<Op>`, and execution is always lazy through
`materialize_merge -> compile -> eval`.

---

## II. v1 to v2 Transformation

### What disappears

v1 organizes around eager execution families and tape-based AD. In v2 these
are replaced by the graph + fragment model:

| v1 crate | v2 | Reason |
|---|---|---|
| `internal/ad-core` | deleted | Fragment replaces tape |
| `internal/ad-ops` | → `tenferro-ops` PrimitiveOp impl | AD rules live on TensorOp |
| `internal/ad-linalg` | → `tenferro-ops` PrimitiveOp impl | AD rules in ops/ad/linalg.rs |
| `internal/ad-surface` | → tidu-rs `differentiate`/`transpose` | External crate |
| `internal/frontend-core` | → `tenferro` TracedTensor | Lazy, not eager |
| `internal/runtime` | → `tenferro` Engine | |
| `tenferro-dynamic-compute` | deleted | Always graph |
| `tenferro-tensor-compute` | → `tenferro-ops` | |
| `tenferro-linalg-prims` | → `tenferro-ops` | No need to separate |
| `tenferro-capi` | deferred | Phase 4+ |
| `extension/*` | deferred | |

### What remains

| v1 crate | v2 crate | Notes |
|---|---|---|
| `tenferro-device` | `tenferro-device` | Mostly unchanged |
| `tenferro-algebra` | `tenferro-algebra` | Mostly unchanged |
| `tenferro-tensor` | `tenferro-tensor` | Simplified |
| `tenferro-prims` | `tenferro-ops` | Rewritten: single TensorOp enum |
| `tenferro-einsum` | `tenferro-einsum` | Rewritten: graph builder |
| `tenferro-linalg` | → `tenferro-ops` + `tenferro` | AD rules → tenferro-ops, LAPACK kernels → tenferro backend |
| `tenferro` (facade) | `tenferro` | TracedTensor, Engine, backends |

**29 crates → 6 crates** (plus 3 external: computegraph-rs, chainrules-rs,
tidu-rs).

---

## III. Crate Dependency Graph

```text
tenferro-device
    |
tenferro-algebra
    |
tenferro-tensor ──── computegraph-rs (Operand)
    |
tenferro-ops ─────── computegraph-rs (GraphOp, Fragment)
    |                 chainrules-rs   (PrimitiveOp)
    |
    ├── tenferro-einsum (SemiringOps → Fragment construction)
    |
tenferro ──────────── tidu-rs (differentiate, transpose)
    (TracedTensor, Engine, backends)
```

---

## IV. Two Op Types

The fundamental design constraint is that `GraphOp::Operand` is an associated
type, so a single Op type can only serve one `Operand` type. Since standard
algebra (`Tensor`) and custom algebras (`TropicalTensor`, etc.) have
different `Operand` types, tenferro provides two Op types:

### StdTensorOp — standard algebra, full vocabulary, AD-capable

```rust
enum StdTensorOp {
    // Tier 1: semiring-compatible core
    Semiring(SemiringOpKind),
    Neg, Conj,

    // Tier 2: standard arithmetic only
    Div, Abs, Sign, Maximum, Minimum,
    Compare(CompareDir), Select, Clamp,

    // Analytic
    Exp, Log, Sin, Cos, Tanh, Sqrt, Rsqrt, Pow, Expm1, Log1p,

    // Indexing & structure
    Gather(GatherConfig), Scatter(ScatterConfig),
    Slice(SliceConfig), DynamicSlice,
    Pad(PadConfig), Concatenate { axis: usize }, Reverse { axes: Vec<usize> },

    // Additional reductions
    ReduceProd { axes: Vec<usize> },
    ReduceMax { axes: Vec<usize> },
    ReduceMin { axes: Vec<usize> },

    // Linalg (AD rules in ops/ad/linalg.rs, execution via custom_call)
    Svd, Qr, Cholesky, Eigh, Solve,
}

impl GraphOp for StdTensorOp {
    type Operand = Tensor;       // f32, f64, Complex<f32>, Complex<f64>
    type Context = CpuContext;      // default backend (faer/BLAS)
    type InputKey = TensorInputKey;
    // ...
}

impl PrimitiveOp for StdTensorOp {
    // linearize + transpose_rule for every variant
}

impl SemiringOps for StdTensorOp {
    // delegates to Semiring(...) variant
}
```

### SemiringOp\<T\> — custom algebra, semiring subset, no AD

```rust
struct SemiringOp<T: Operand> {
    kind: SemiringOpKind,
    _phantom: PhantomData<T>,
}

impl<T: Operand> GraphOp for SemiringOp<T> {
    type Operand = T;
    type Context = SemiringContext<T>;
    type InputKey = String;

    fn eval(&self, ctx: &mut SemiringContext<T>, inputs: &[&T]) -> Vec<T> {
        // delegates entirely to T: Operand methods
        match &self.kind {
            SemiringOpKind::Add => vec![inputs[0].add(inputs[1])],
            SemiringOpKind::Mul => vec![inputs[0].multiply(inputs[1])],
            SemiringOpKind::DotGeneral(cfg) => vec![inputs[0].dot_general(inputs[1], cfg)],
            SemiringOpKind::ReduceSum { axes } => vec![inputs[0].reduce_sum(axes)],
            SemiringOpKind::Transpose { perm } => vec![inputs[0].transpose(perm)],
            SemiringOpKind::Reshape { shape } => vec![inputs[0].reshape(shape)],
            SemiringOpKind::BroadcastInDim { shape, dims } =>
                vec![inputs[0].broadcast_in_dim(shape, dims)],
        }
    }
}

impl<T: Operand> SemiringOps for SemiringOp<T> { ... }

// PrimitiveOp is NOT implemented — no AD for custom algebras
```

Users extend tenferro by implementing `Operand` for their tensor type:

```rust
// User's crate
struct TropicalTensor { ... }

impl Operand for TropicalTensor {
    fn zero(shape: &[usize]) -> Self { ... }
    fn one(shape: &[usize]) -> Self { ... }    // multiplicative identity
    fn add(&self, other: &Self) -> Self { ... } // tropical: max
    fn multiply(&self, other: &Self) -> Self { ... } // tropical: +
    fn dot_general(&self, other: &Self, config: &DotGeneralConfig) -> Self { ... }
    fn reduce_sum(&self, axes: &[usize]) -> Self { ... }
    // ...
}

type TropicalOp = SemiringOp<TropicalTensor>;
// einsum, compile, eval all work
```

---

## V. SemiringOpKind — Shared Vocabulary

`SemiringOpKind` is the set of operations that all algebras must support.
It is shared between `StdTensorOp` and `SemiringOp<T>`:

```rust
#[derive(Clone, Hash, Eq, PartialEq)]
enum SemiringOpKind {
    Add,
    Mul,
    DotGeneral(DotGeneralConfig),
    ReduceSum { axes: Vec<usize> },
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
}
```

`StdTensorOp::Semiring(SemiringOpKind)` wraps it as a variant.
`SemiringOp<T>` wraps it as a newtype.

This avoids duplicating op definitions across algebra types.

---

## VI. SemiringOps Trait — Generic Einsum

`SemiringOps` is the trait that einsum Fragment construction is generic over:

```rust
trait SemiringOps: GraphOp + Sized {
    fn add() -> Self;
    fn mul() -> Self;
    fn dot_general(config: DotGeneralConfig) -> Self;
    fn reduce_sum(axes: Vec<usize>) -> Self;
    fn transpose(perm: Vec<usize>) -> Self;
    fn reshape(shape: Vec<usize>) -> Self;
    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self;
}
```

Both `StdTensorOp` and `SemiringOp<T>` implement `SemiringOps`.

Einsum is algebra-agnostic:

```rust
fn build_einsum_fragment<Op: SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    path: &ContractionPath,
    inputs: &[ValRef<Op>],
) -> LocalValId {
    // Constructs DotGeneral, Transpose, Reshape, etc. nodes
    // Does not know which algebra is in use
}
```

The contraction path optimization is also algebra-agnostic (it only depends
on shapes and subscripts):

```rust
fn optimize_contraction_path(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
) -> ContractionPath;
```

---

## VII. Einsum: N-ary to Graph

N-ary einsum is decomposed into a graph of binary operations:

```text
einsum("ij,jk,kl->il", A, B, C)
    |
    | optimize_contraction_path (shape-based, algebra-agnostic)
    v
ContractionPath: [(A,B) -> T, (T,C) -> result]
    |
    | build_einsum_fragment<Op: SemiringOps>
    v
Fragment<Op>:
    t0 = DotGeneral(A, B, {contract=[j]})    // "ij,jk->ik"
    t1 = DotGeneral(t0, C, {contract=[k]})   // "ik,kl->il"
```

Each binary contraction step may insert `Transpose`, `Reshape`, or
`BroadcastInDim` nodes as needed to align axes for `DotGeneral`.

For standard algebra, the resulting `Fragment<StdTensorOp>` can be
differentiated and transposed by tidu-rs. For custom algebras,
`Fragment<SemiringOp<T>>` goes directly to `materialize_merge -> compile ->
eval`.

---

## VIII. Backend Architecture

### Design principle: Backend trait separate from GraphOp::eval

`GraphOp::eval` is a built-in reference implementation (CPU, single-threaded).
It is useful for testing and simple execution, but it is **not** the primary
execution path for production backends.

`CompiledProgram<Op>` is pure data — an instruction sequence with slot
assignments. Any backend can interpret it independently of `GraphOp::eval`,
using its own execution strategy.

### Backend trait

```rust
trait Backend<Op> {
    fn eval_program(
        &mut self,
        prog: &CompiledProgram<Op>,
        inputs: &[Tensor],
    ) -> Vec<Tensor>;
}
```

For standard algebra backends, `Tensor` is the shared runtime value type across
CPU and GPU. A GPU backend may internally materialize `XlaBuffer`,
`CudaBuffer`, or other backend-native handles, but those remain backend
implementation details and are wrapped back into device-aware `Tensor` values
at the API boundary.

### StableHLO IR representation

tenferro defines its own Rust data structures that mirror StableHLO semantics.
This is neither binary nor text — it is an in-process Rust struct passed
directly to backends. No serialization for faer/GPU backends.

```rust
/// A single StableHLO instruction
struct StableHloInstruction {
    op: StableHloOp,
    inputs: Vec<usize>,             // slot indices
    outputs: Vec<usize>,            // slot indices
    input_types: Vec<TensorType>,   // shape + dtype per input
    output_types: Vec<TensorType>,  // shape + dtype per output
}

/// The complete StableHLO program (lowered from CompiledProgram)
struct StableHloProgram {
    instructions: Vec<StableHloInstruction>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
}

/// StableHLO ops — mirrors the StableHLO spec
enum StableHloOp {
    // Elementwise
    Add, Multiply, Negate, Divide, Abs, Sign,
    Exponential, Log, Sine, Cosine, Tanh, Sqrt, Rsqrt, Power,
    ExponentialMinusOne, LogPlusOne,
    Compare(CompareDir), Select, Clamp,
    Maximum, Minimum,

    // Shape
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    Reshape { shape: Vec<usize> },
    Transpose { perm: Vec<usize> },
    Reverse { axes: Vec<usize> },
    Concatenate { axis: usize },
    Pad(PadConfig),

    // Contraction
    DotGeneral(DotGeneralConfig),

    // Reduction
    Reduce { axes: Vec<usize>, combiner: Combiner },

    // Indexing
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice,

    // Custom call — linalg, user kernels, etc.
    CustomCall {
        target: String,
        config: Vec<u8>,  // opaque serialized config
    },

    // Control flow (future)
    If, While,
}
```

### StableHLO lowering: StdTensorOp → StableHloOp

Most `StdTensorOp` variants map 1:1 to a `StableHloOp`. Some require 1:N
expansion:

| StdTensorOp | StableHLO | Mapping |
|---|---|---|
| `Add`, `Mul`, `Neg`, `Div`, `Exp`, `Log`, ... | `add`, `multiply`, `negate`, `divide`, `exponential`, `log`, ... | 1:1 |
| `DotGeneral(cfg)` | `dot_general(cfg)` | 1:1 |
| `Transpose`, `Reshape`, `BroadcastInDim` | same | 1:1 |
| `Compare`, `Select`, `Gather`, `Scatter`, ... | same | 1:1 |
| `ReduceSum { axes }` | `reduce { axes, combiner: add_region }` | 1:1 but combiner is a sub-computation in StableHLO |
| `Conj` | `real` + `imag` + `negate` + `complex` | 1:4 composite |
| `Svd` | `custom_call("gesvd")` + `get_tuple_element` × 3 | 1:4 |
| `Qr` | `custom_call("geqrf")` + `get_tuple_element` × 2 | 1:3 |
| `Solve` | `custom_call("getrf")` + `custom_call("getrs")` | 1:2+ |

```rust
fn lower_instruction(inst: &Instruction<StdTensorOp>) -> Vec<StableHloInstruction> {
    match &inst.op {
        // 1:1 cases
        StdTensorOp::Semiring(SemiringOpKind::Add) =>
            vec![hlo_inst(StableHloOp::Add, &inst)],
        StdTensorOp::Exp =>
            vec![hlo_inst(StableHloOp::Exponential, &inst)],
        StdTensorOp::Semiring(SemiringOpKind::DotGeneral(c)) =>
            vec![hlo_inst(StableHloOp::DotGeneral(c.clone()), &inst)],
        StdTensorOp::Semiring(SemiringOpKind::ReduceSum { axes }) =>
            vec![hlo_inst(StableHloOp::Reduce {
                axes: axes.clone(), combiner: Combiner::Add,
            }, &inst)],

        // 1:N expansion
        StdTensorOp::Conj => lower_conj(&inst),  // real + imag + negate + complex
        StdTensorOp::Svd => lower_svd(&inst),    // custom_call + get_tuple_element × 3
        StdTensorOp::Solve => lower_solve(&inst), // getrf + getrs

        // ...
    }
}
```

### custom_call — linalg and user-defined kernels

Operations that have no direct StableHLO op lower to `CustomCall`:

| StdTensorOp | StableHLO | custom_call target |
|---|---|---|
| `Svd` | `CustomCall` | `"lapack_gesvd"` / `"cusolver_gesvd"` |
| `Qr` | `CustomCall` | `"lapack_geqrf"` / `"cusolver_geqrf"` |
| `Cholesky` | `stablehlo.cholesky` | (direct op, not custom_call) |
| `Eigh` | `CustomCall` | `"lapack_syevd"` / `"cusolver_syevd"` |
| `Solve` | `CustomCall` | `"lapack_getrf"` + `"lapack_getrs"` |
| `LuFullPivot` | `CustomCall` | `"lapack_getc2"` |

Users can also register custom kernels:

```rust
engine.register_custom_call("my_decomposition", MyDecompositionKernel);

// Then use it in the graph:
StdTensorOp::CustomCall {
    target: "my_decomposition".into(),
    n_inputs: 1, n_outputs: 2,
}
```

The `config: Vec<u8>` field in `StableHloOp::CustomCall` carries opaque
per-call configuration (e.g., "full pivot" vs "partial pivot" for LU).
Backends deserialize this when dispatching.

### Standard algebra backends: all go through StableHLO

All standard algebra backends consume `StableHloProgram`. The IR
representation is Rust data structures passed in-process — no serialization
for interpreter backends.

```text
CompiledProgram<StdTensorOp>
    │
    │ lower_to_stablehlo() — mostly 1:1, some 1:N expansion
    ↓
StableHloProgram (Rust struct, in-process)
    │
    ├── FaerBackend:      iterate instructions → faer/BLAS/LAPACK dispatch
    │                     custom_call → registered kernel lookup
    │                     (no serialization)
    │
    ├── CustomGpuBackend: iterate instructions → CUDA kernel dispatch
    │                     custom_call → registered kernel lookup
    │                     (no serialization)
    │
    ├── XlaBackend:       build HLO via xla-rs builder API (programmatic)
    │                     custom_call → XLA custom_call with target name
    │                     (no text/binary serialization needed)
    │
    └── IREE (future):    emit MLIR text → IREE compile
                          (only this path needs serialization)
```

Why all through StableHLO:

- **Same work either way**: interpreting CompiledProgram directly and
  interpreting StableHLO both dispatch each op to a kernel. Skipping the
  StableHLO lowering step saves no implementation effort.
- **v1 reuse**: the existing faer/LAPACK backend already implements every
  needed kernel. Each kernel maps 1:1 to a StableHLO op.
- **Backend portability**: new backends only need to interpret StableHLO,
  not understand CompiledProgram internals.
- **Shared correctness tests**: all standard backends consume the same IR,
  so tests are shared.
- **custom_call extensibility**: linalg ops and user kernels flow through
  the same pipeline as standard ops.

```rust
struct FaerBackend {
    ctx: CpuContext,
    custom_calls: HashMap<String, Box<dyn CustomCallKernel<Tensor>>>,
}

impl Backend<StdTensorOp> for FaerBackend {
    fn eval_program(&mut self, prog, inputs) {
        let hlo = lower_to_stablehlo(prog);
        for inst in &hlo.instructions {
            match &inst.op {
                StableHloOp::Add => faer_add(...),
                StableHloOp::DotGeneral(cfg) => faer_matmul(cfg, ...),
                StableHloOp::Reduce { combiner: Combiner::Add, .. } => faer_sum(...),
                StableHloOp::CustomCall { target, config } =>
                    self.custom_calls[target].execute(config, ...),
                // ...
            }
        }
    }
}

struct XlaBackend { client: XlaClient }
impl Backend<StdTensorOp> for XlaBackend {
    fn eval_program(&mut self, prog, inputs) {
        let hlo = lower_to_stablehlo(prog);
        let device_buffers = self.upload_inputs(inputs)?;
        // Build HLO computation via xla-rs builder API
        let builder = XlaBuilder::new("program");
        for inst in &hlo.instructions {
            match &inst.op {
                StableHloOp::Add => builder.add(...),
                StableHloOp::DotGeneral(cfg) => builder.dot_general(cfg, ...),
                StableHloOp::CustomCall { target, config } =>
                    builder.custom_call(target, config, ...),
                // ...
            }
        }
        let executable = self.client.compile(&builder.build())?;
        let outputs = executable.execute(&device_buffers)?;
        self.wrap_outputs(outputs)
    }
}
```

### Custom algebra backends

Users define their own backends for custom algebras by implementing
`Backend<SemiringOp<T>>`:

```text
CompiledProgram<SemiringOp<TropicalTensor>>
    |
    ├── CpuBackend                   Tensor = TropicalTensor
    ├── TropicalGpuBackend           Tensor = GpuTropicalTensor
    └── (user-defined backends...)   Tensor = ...
```

```rust
// Generic CPU backend — works for any Operand type
struct CpuSemiringBackend;
impl<T: Operand> Backend<SemiringOp<T>> for CpuSemiringBackend {
    type Tensor = T;
    fn eval_program(&mut self, prog, inputs) {
        // Iterate instructions, delegate to T::add, T::dot_general, ...
        // (equivalent to GraphOp::eval, but through Backend trait)
    }
}

// User-defined GPU backend for Tropical
struct TropicalGpuBackend { cuda_ctx: CudaContext }
impl Backend<SemiringOp<TropicalTensor>> for TropicalGpuBackend {
    type Tensor = GpuTropicalTensor;
    fn eval_program(&mut self, prog, inputs) {
        for inst in &prog.instructions {
            match &inst.op.kind {
                SemiringOpKind::DotGeneral(cfg) => self.cuda_gemm(cfg, ...),
                SemiringOpKind::Add => self.cuda_add(...),
                SemiringOpKind::ReduceSum { axes } => self.cuda_reduce(axes, ...),
                SemiringOpKind::Mul => self.cuda_mul(...),
                SemiringOpKind::Transpose { .. } => ...,
                SemiringOpKind::Reshape { .. } => ...,
                SemiringOpKind::BroadcastInDim { .. } => ...,
            }
        }
    }
}
```

### GraphOp::eval vs Backend trait

| | `GraphOp::eval` | `Backend<Op>` |
|---|---|---|
| Defined in | computegraph-rs | tenferro-rs |
| Tensor type | fixed (`Op::Operand`) | fixed (`Tensor`) for standard algebra; backend-internal buffers are opaque |
| Multiple backends | no (one `Context`) | yes |
| Standard algebra | reference impl only | StableHLO → faer/XLA/GPU |
| Custom algebra | reference impl only | user-defined (CPU/GPU/...) |

`GraphOp::eval` remains useful for:
- computegraph-rs unit tests (no backend dependency)
- Quick prototyping and debugging

`Backend<Op>` is the primary execution path for all production use.
For standard algebra, all backends go through StableHLO lowering.
For custom algebras, backends interpret `CompiledProgram` directly
(no StableHLO — custom algebras cannot be lowered to StableHLO).

### Backend dispatch in Engine

```rust
struct Engine<B: Backend<StdTensorOp>> {
    backend: B,
    compile_cache: CompileCache,
    einsum_cache: EinsumCache,
}
```

For custom algebras, users construct their own evaluation pipeline:

```rust
let path = optimize_contraction_path(&subscripts, &shapes);
let fragment = build_einsum_fragment::<TropicalOp>(&mut builder, &path, &inputs);
let view = resolve(vec![fragment]);
let graph = materialize_merge(&view, &outputs);
let prog = compile(&graph);

// Choose backend
let mut backend = TropicalGpuBackend::new(cuda_ctx);
let result = backend.eval_program(&prog, &input_tensors);
```

---

## IX. TracedTensor and Engine

`TracedTensor` is the user-facing lazy type for standard algebra:

```rust
struct TracedTensor {
    shape: Vec<usize>,
    dtype: DType,
    fragment: Arc<Fragment<StdTensorOp>>,
    val: LocalValId,
    data: Option<Tensor>,
}
```

Key operations:

```rust
impl TracedTensor {
    /// Create from concrete data
    fn from(tensor: Tensor) -> Self;

    /// Lazy evaluation (single output, no intermediate sharing)
    fn eval(&mut self, engine: &mut Engine) -> &Tensor;

    /// VJP: differentiate → transpose (via tidu-rs), still lazy
    fn grad(&self, wrt: &TracedTensor) -> TracedTensor;

    /// JVP: differentiate only (via tidu-rs), still lazy
    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor;
}

impl Engine {
    /// Evaluate multiple outputs together.
    /// All fragments are resolved into one MaterializedGraph, so shared
    /// intermediate nodes (primal values needed by both output and gradient)
    /// are computed only once via GlobalValKey deduplication.
    fn eval_all(&mut self, outputs: &mut [&mut TracedTensor]) -> Vec<&Tensor>;
}
```

`eval_all` is the recommended API when primal outputs and their derivatives
are needed together. Single-output `eval` is a convenience wrapper.

For custom algebras, users work with `Fragment<SemiringOp<T>>` and
`CompiledProgram<SemiringOp<T>>` directly through the computegraph-rs API,
without `TracedTensor`.

---

## X. User Extension Points

| Goal | What to implement |
|---|---|
| New scalar algebra for einsum (CPU) | `impl Operand for MyTensor` |
| Custom GPU backend for custom algebra | `impl Backend<SemiringOp<MyTensor>> for MyGpuBackend` |
| Custom CPU backend with optimized kernels | `impl Backend<SemiringOp<MyTensor>> for MyOptCpuBackend` |
| Custom linalg kernel (standard algebra) | `engine.register_custom_call("name", kernel)` |
| AD for custom algebra | Define own Op enum, impl `PrimitiveOp` (advanced) |

The minimal extension path (CPU only):

1. `impl Operand for MyTensor` — define semiring operations
2. Use `SemiringOp<MyTensor>` as the Op type
3. Use `CpuSemiringBackend` — einsum + compile + eval work immediately

Adding a GPU backend:

1. Define `GpuMyTensor` — GPU-resident tensor type
2. `impl Backend<SemiringOp<MyTensor>> for MyGpuBackend` — map each
   `SemiringOpKind` to GPU kernels
3. Use the same `CompiledProgram<SemiringOp<MyTensor>>` — graph construction
   and compilation are backend-agnostic

---

## XI. Operand Trait

`Operand` (defined in computegraph-rs) provides the minimum operations needed
for graph evaluation. For einsum execution via `SemiringOp<T>`, the tensor
type must support all of these:

```rust
trait Operand: Clone + Send + Sync + 'static {
    fn zero(shape: &[usize]) -> Self;
    fn one(shape: &[usize]) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn dot_general(&self, other: &Self, config: &DotGeneralConfig) -> Self;
    fn reduce_sum(&self, axes: &[usize]) -> Self;
    fn transpose(&self, perm: &[usize]) -> Self;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self;
    fn conj(&self) -> Self;
}
```

`zero` is needed for zero propagation in AD. `one` is needed for reverse-mode
seeding (`ct_y = one`). For custom algebras without AD, `zero` and `one`
still serve as identity elements for the semiring.

---

## XII. Per-Crate Contents

### tenferro-device

Unchanged from v1. Device enum, memory spaces, error types.

### tenferro-algebra

Unchanged from v1. `SemiringAlgebra` trait, `StandardAlgebra`, scalar type
constraints.

### tenferro-tensor

Simplified from v1. No AD-related code.

- `TensorData<T: Scalar>` — generic typed tensor (buffer, shape, strides)
- `Tensor` — type-erased enum over `TensorData<f32/f64/c32/c64>`
- `DType` — scalar type discriminator
- `impl Operand for Tensor`

### tenferro-ops

The core crate:

- `SemiringOpKind` enum (shared vocabulary)
- `SemiringOps` trait
- `SemiringOp<T>` generic wrapper + `GraphOp` impl
- `StdTensorOp` enum (full vocabulary)
- `impl GraphOp for StdTensorOp`
- `impl PrimitiveOp for StdTensorOp` (linearize + transpose_rule)
- `impl SemiringOps for StdTensorOp`
- `TensorInputKey` + `impl ADKey`

Depends on: computegraph-rs, chainrules-rs, tenferro-tensor.

### tenferro-einsum

Graph builder for N-ary einsum:

- `Subscripts` parsing and validation
- `ContractionPath` optimization
- `build_einsum_fragment<Op: SemiringOps>` (algebra-agnostic)

Depends on: computegraph-rs, tenferro-ops.

### tenferro

Top-level facade:

- `TracedTensor` (lazy graph-aware wrapper)
- `Engine` (compilation cache, backend dispatch, einsum cache, custom_call
  registry)
- Public API: `einsum()`, `grad()`, `jvp()`, `eval()`, `eval_all()`
- `Backend<Op>` trait
- `StableHloProgram`, `StableHloOp`, `StableHloInstruction` (Rust IR)
- `lower_to_stablehlo()` (`CompiledProgram<StdTensorOp>` → `StableHloProgram`,
  mostly 1:1, some 1:N expansion for `Conj`, multi-output linalg, `Solve`)
- Standard backends:
  - `FaerBackend` — StableHLO interpreter, CPU (faer/BLAS/LAPACK)
  - `XlaBackend` — StableHLO → xla-rs builder API → JIT
- Custom algebra backends:
  - `CpuSemiringBackend<T>` — generic, interprets `CompiledProgram` directly
    via `Operand` trait methods

Depends on: all of the above + tidu-rs.

---

## XIII. Roadmap

### Phase 1: Scalar fragment AD

- computegraph-rs: Fragment, resolve, materialize_merge, compile, eval
- chainrules-rs: PrimitiveOp trait
- tidu-rs: differentiate, transpose
- tenferro-ops: scalar subset of StdTensorOp (Add, Mul, Exp, Neg, Conj)
- tenferro: minimal Engine with CPU eval
- Tests: forward, backward, second order on `exp(a*x)`

### Phase 2: Tensor primitives + einsum

- tenferro-ops: full StdTensorOp (DotGeneral, ReduceSum, BroadcastInDim, ...)
- tenferro-ops: SemiringOp\<T\>, SemiringOps trait
- tenferro-tensor: Tensor, DType, impl Operand
- tenferro-einsum: contraction path + Fragment construction
- Tests: vector AD examples, einsum correctness

### Phase 3: Linalg + backends

- tenferro-ops: SVD, QR, Cholesky PrimitiveOp impls (in ad/linalg.rs)
- tenferro: StableHLO lowering, XLA backend
- tenferro: CPU backend with faer/BLAS
- Tests: linalg AD, StableHLO round-trip

### Phase 4: Custom algebra + optimization

- SemiringOp\<T\> end-to-end with Tropical
- Custom GPU backend (reuse v1 CUDA kernels)
- tenferro-capi (C FFI for Julia/Python)
- Logical-DAG-aware checkpoint scheduling
- Operator fusion in compiled IR
