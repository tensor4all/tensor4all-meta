# v2 tenferro-rs Internal Design

**Date:** 2026-04-04
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

StdTensorOp is **flat** — most variants map 1:1 to a StableHLO op (documented
exceptions include composite lowerings like `Conj` -> 4 ops and multi-output
linalg ops like `Svd`). There is no `Semiring(SemiringOpKind)` wrapper; the
semiring-compatible operations (`Add`, `Mul`, `DotGeneral`, etc.) are top-level
variants, making the StdTensorOp -> StableHLO lowering a trivial match.

```rust
enum StdTensorOp {
    // Tier 1: semiring-compatible core (flat, mostly mirrors StableHLO 1:1)
    Add, Mul, Neg, Conj,
    DotGeneral(DotGeneralConfig),
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    ReduceSum { axes: Vec<usize> },

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
    Cholesky, Svd, Qr, Eigh, Solve,

    CustomCall { target: String, n_inputs: usize, n_outputs: usize },
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
    fn add() -> Self { StdTensorOp::Add }
    fn mul() -> Self { StdTensorOp::Mul }
    fn dot_general(c: DotGeneralConfig) -> Self { StdTensorOp::DotGeneral(c) }
    fn reduce_sum(axes: Vec<usize>) -> Self { StdTensorOp::ReduceSum { axes } }
    fn transpose(perm: Vec<usize>) -> Self { StdTensorOp::Transpose { perm } }
    fn reshape(shape: Vec<usize>) -> Self { StdTensorOp::Reshape { shape } }
    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self {
        StdTensorOp::BroadcastInDim { shape, dims }
    }
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
        // Algebraic ops → Operand trait methods
        // Structural ops → generic functions over TensorData
        match &self.kind {
            SemiringOpKind::Add => vec![inputs[0].add(inputs[1])],
            SemiringOpKind::Mul => vec![inputs[0].multiply(inputs[1])],
            SemiringOpKind::DotGeneral(cfg) => vec![inputs[0].dot_general(inputs[1], cfg)],
            SemiringOpKind::ReduceSum { axes } => vec![inputs[0].reduce_sum(axes)],
            SemiringOpKind::Transpose { perm } => vec![transpose(inputs[0], perm)],
            SemiringOpKind::Reshape { shape } => vec![reshape(inputs[0], shape)],
            SemiringOpKind::BroadcastInDim { shape, dims } =>
                vec![broadcast_in_dim(inputs[0], shape, dims)],
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
}

impl TensorData for TropicalTensor {
    type Scalar = f64;
    fn shape(&self) -> &[usize] { ... }
    fn strides(&self) -> &[usize] { ... }
    fn data(&self) -> &[f64] { ... }
    fn from_data(shape: Vec<usize>, data: Vec<f64>) -> Self { ... }
}
// transpose, reshape, broadcast_in_dim are provided automatically

type TropicalOp = SemiringOp<TropicalTensor>;
// einsum, compile, eval all work
```

---

## V. SemiringOpKind — Shared Vocabulary

`SemiringOpKind` is the set of operations that all algebras must support.
It is used **only** inside `SemiringOp<T>` — the generic custom-algebra op
type. `StdTensorOp` does **not** wrap `SemiringOpKind`; it has its own flat
variants that mostly mirror StableHLO 1:1 (with documented exceptions for
composite lowerings and multi-output linalg ops).

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

`SemiringOp<T>` wraps it as a newtype. The `SemiringOps` trait bridges both
worlds: `StdTensorOp` implements it by mapping to flat variants,
`SemiringOp<T>` implements it by mapping to `SemiringOpKind` variants.

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
`StdTensorOp` maps directly to its flat variants (e.g. `fn add() -> Self { StdTensorOp::Add }`).
`SemiringOp<T>` maps to the corresponding `SemiringOpKind` variants.

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

## VIII. Backend Architecture — 2-Level IR

### Design principle

All execution flows through a 2-level IR with StableHLO as the cut point:

```text
CompiledProgram<StdTensorOp>
    │
    │ lower_to_stablehlo() — flat 1:1 mapping (+ some 1:N for Conj, linalg)
    ↓
StableHloProgram (Rust struct, in-process)    ← CUT POINT
    │
    ├── XlaBackend:  StableHLO → XLA directly (unchanged)
    │
    └── FaerBackend: StableHLO → optimizing compiler → LowLevelProgram
                         → generic execution engine → SemiringCore trait
```

XLA consumes StableHLO directly (it already does its own optimization).
All other backends go through the optimizing compiler to produce a
`LowLevelProgram`, which a generic engine interprets by dispatching to
backend traits.

For custom algebras (`SemiringOp<T>`), the same 2-level structure applies:
`SemiringOp<T>` lowers to StableHLO, then to `LowLevelProgram`.

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

Because `StdTensorOp` is flat (no `Semiring(...)` wrapper), most variants map
1:1 to a `StableHloOp` with a trivial match. Some require 1:N expansion:

| StdTensorOp | StableHLO | Mapping |
|---|---|---|
| `Add`, `Mul`, `Neg`, `Div`, `Exp`, `Log`, ... | `add`, `multiply`, `negate`, `divide`, `exponential`, `log`, ... | 1:1 |
| `DotGeneral(cfg)` | `dot_general(cfg)` | 1:1 |
| `Transpose`, `Reshape`, `BroadcastInDim` | same | 1:1 |
| `Compare`, `Select`, `Gather`, `Scatter`, ... | same | 1:1 |
| `ReduceSum { axes }` | `reduce { axes, combiner: add_region }` | 1:1 but combiner is a sub-computation in StableHLO |
| `Conj` | `real` + `imag` + `negate` + `complex` | 1:4 composite |
| `Svd` | `custom_call("gesvd")` + `get_tuple_element` x 3 | 1:4 |
| `Qr` | `custom_call("geqrf_orgqr")` + `get_tuple_element` x 2 | 1:3 |
| `Solve` | `custom_call("getrf")` + `custom_call("getrs")` | 1:2+ |

```rust
fn lower_instruction(inst: &Instruction<StdTensorOp>) -> Vec<StableHloInstruction> {
    match &inst.op {
        // 1:1 cases — flat variants map directly
        StdTensorOp::Add =>
            vec![hlo_inst(StableHloOp::Add, &inst)],
        StdTensorOp::Mul =>
            vec![hlo_inst(StableHloOp::Multiply, &inst)],
        StdTensorOp::Exp =>
            vec![hlo_inst(StableHloOp::Exponential, &inst)],
        StdTensorOp::DotGeneral(c) =>
            vec![hlo_inst(StableHloOp::DotGeneral(c.clone()), &inst)],
        StdTensorOp::ReduceSum { axes } =>
            vec![hlo_inst(StableHloOp::Reduce {
                axes: axes.clone(), combiner: Combiner::Add,
            }, &inst)],
        StdTensorOp::Transpose { perm } =>
            vec![hlo_inst(StableHloOp::Transpose { perm: perm.clone() }, &inst)],

        // 1:N expansion
        StdTensorOp::Conj => lower_conj(&inst),  // real + imag + negate + complex
        StdTensorOp::Svd => lower_svd(&inst),    // custom_call + get_tuple_element x 3
        StdTensorOp::Solve => lower_solve(&inst), // getrf + getrs

        // ...
    }
}
```

#### SemiringOp\<T\> lowering

`SemiringOp<T>` also lowers to StableHLO. The mapping is through
`SemiringOpKind`:

```rust
fn lower_semiring_instruction<T: Operand>(
    inst: &Instruction<SemiringOp<T>>,
) -> Vec<StableHloInstruction> {
    match &inst.op.kind {
        SemiringOpKind::Add => vec![hlo_inst(StableHloOp::Add, &inst)],
        SemiringOpKind::Mul => vec![hlo_inst(StableHloOp::Multiply, &inst)],
        SemiringOpKind::DotGeneral(c) =>
            vec![hlo_inst(StableHloOp::DotGeneral(c.clone()), &inst)],
        SemiringOpKind::ReduceSum { axes } =>
            vec![hlo_inst(StableHloOp::Reduce {
                axes: axes.clone(), combiner: Combiner::Add,
            }, &inst)],
        SemiringOpKind::Transpose { perm } =>
            vec![hlo_inst(StableHloOp::Transpose { perm: perm.clone() }, &inst)],
        SemiringOpKind::Reshape { shape } =>
            vec![hlo_inst(StableHloOp::Reshape { shape: shape.clone() }, &inst)],
        SemiringOpKind::BroadcastInDim { shape, dims } =>
            vec![hlo_inst(StableHloOp::BroadcastInDim {
                shape: shape.clone(), dims: dims.clone(),
            }, &inst)],
    }
}
```

### custom_call — linalg and user-defined kernels

Operations that have no direct StableHLO op lower to `CustomCall`:

| StdTensorOp | StableHLO | custom_call target |
|---|---|---|
| `Svd` | `CustomCall` | `"lapack_gesvd"` / `"cusolver_gesvd"` |
| `Qr` | `CustomCall` | `"lapack_geqrf_orgqr"` / `"cusolver_geqrf_orgqr"` |
| `Cholesky` | `stablehlo.cholesky` | (direct op, not custom_call) |
| `Eigh` | `CustomCall` | `"lapack_syevd"` / `"cusolver_syevd"` |
| `Solve` | `CustomCall` | `"lapack_getrf"` + `"lapack_getrs"` |

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

### Optimizing compiler (StableHLO → Low-level IR)

For non-XLA backends, an **optimizing compiler** transforms `StableHloProgram`
into `LowLevelProgram`. This compiler is algebra-agnostic — it works
identically for standard and custom algebras.

Optimization passes:

- **TransposeFolding**: eliminate or fuse adjacent `Transpose` instructions
- **DotDecomposer**: decompose complex `DotGeneral` configurations into
  sequences of simpler operations (permute + batched GEMM)
- **LinalgCustomCallPassthrough**: pass linalg `CustomCall` ops through to the
  low-level IR as-is

Note: contiguous materialization is **not** a compiler pass. It happens at
eval() time as a pre-processing step: permuted-contiguous views pass the raw
buffer with a `stablehlo.transpose` inserted at the program head; truly
non-contiguous views are physically copied outside the IR. All tensors
entering the compiler are already contiguous and column-major.

```rust
fn compile_to_lowlevel(hlo: &StableHloProgram) -> LowLevelProgram {
    let hlo = transpose_folding(hlo);
    let hlo = dot_decomposer(&hlo);
    let ll = lower_to_lowlevel(&hlo);      // StableHLO → LowLevelOp (including CustomCall passthrough)
    ll
}
```

### Low-level IR

`LowLevelProgram` is a flat instruction sequence where all tensors are
contiguous and column-major. This IR is what the generic execution engine
interprets.

```rust
enum LowLevelOp {
    BatchedGemm { batch_dims: Vec<usize>, m: usize, n: usize, k: usize },
    ReduceSum { axes: Vec<usize> },
    Permute { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    CustomCall { target: String, config: Vec<u8> },
}

struct LowLevelInstruction {
    op: LowLevelOp,
    inputs: Vec<usize>,             // slot indices
    outputs: Vec<usize>,            // slot indices
    input_types: Vec<TensorType>,
    output_types: Vec<TensorType>,
}

struct LowLevelProgram {
    instructions: Vec<LowLevelInstruction>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
}
```

All tensors at this level are contiguous column-major. The optimizing compiler
guarantees this invariant.

### Generic execution engine

The generic engine interprets `LowLevelProgram` by dispatching each
instruction to the appropriate method on backend traits:

```rust
fn execute_lowlevel<Alg: Semiring, B: SemiringCore<Alg>>(
    backend: &B,
    prog: &LowLevelProgram,
    inputs: &[B::Buffer],
) -> Vec<B::Buffer> {
    let mut slots: Vec<Option<B::Buffer>> = vec![None; prog.n_slots];
    // ... load inputs into slots ...
    for inst in &prog.instructions {
        match &inst.op {
            LowLevelOp::BatchedGemm { batch_dims, m, n, k } =>
                backend.batched_gemm(batch_dims, *m, *n, *k, ...),
            LowLevelOp::ReduceSum { axes } =>
                backend.reduce_sum(axes, ...),
            LowLevelOp::Permute { perm } =>
                backend.permute(perm, ...),
            LowLevelOp::Reshape { shape } =>
                backend.reshape(shape, ...),
            LowLevelOp::CustomCall { target, config } =>
                backend.dispatch_custom_call(target, config, ...),
        }
    }
    // ... collect outputs from slots ...
}
```

### Backend traits

Two traits define what a backend must implement:

```rust
/// Minimum required operations for any semiring backend.
trait SemiringCore<Alg: Semiring> {
    type Buffer;
    fn batched_gemm(
        &self, batch_dims: &[usize], m: usize, n: usize, k: usize,
        a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer,
    );
    fn reduce_sum(&self, axes: &[usize], input: &Self::Buffer, out: &mut Self::Buffer);
}

/// Optional fast-path operations. Returning `false` falls back to the
/// generic engine's decomposition.
trait SemiringFastPath<Alg: Semiring>: SemiringCore<Alg> {
    /// Full contraction (einsum-level). Returns true if handled.
    fn contract(
        &self, subscripts: &Subscripts, inputs: &[&Self::Buffer],
        out: &mut Self::Buffer,
    ) -> bool { false }

    /// Fast path for Hadamard products. Returns true if handled.
    fn elementwise_mul(
        &self, a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer,
    ) -> bool { false }

    /// Fast path for semiring accumulation. Returns true if handled.
    fn elementwise_add(
        &self, a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer,
    ) -> bool { false }
}
```

The minimum a custom backend must implement is `SemiringCore` — specifically
`batched_gemm` and `reduce_sum`. Structural ops (`Permute`, `Reshape`) are
handled by common infrastructure shared across all backends. Fast-path methods
(`elementwise_mul`, `elementwise_add`, `contract`) live on `SemiringFastPath`
and are optional.

### Standard backends

```text
CompiledProgram<StdTensorOp>
    │
    │ lower_to_stablehlo() — flat 1:1 mapping
    ↓
StableHloProgram
    │
    ├── XlaBackend:  StableHLO → XLA directly
    │                (XLA does its own optimization)
    │
    └── FaerBackend: StableHLO → optimizing compiler → LowLevelProgram
                         → generic execution engine
                         → SemiringCore<StandardAlgebra> impl (faer/BLAS/LAPACK)
```

```rust
struct FaerBackend {
    custom_calls: HashMap<String, Box<dyn CustomCallKernel<Tensor>>>,
}

impl SemiringCore<StandardAlgebra> for FaerBackend {
    type Buffer = Tensor;
    fn batched_gemm(&self, batch_dims, m, n, k, a, b, out) {
        faer_matmul(batch_dims, m, n, k, a, b, out);
    }
    fn reduce_sum(&self, axes, input, out) {
        faer_sum(axes, input, out);
    }
    // ... other SemiringCore methods ...
}

impl Backend<StdTensorOp> for FaerBackend {
    fn eval_program(&mut self, prog, inputs) {
        let hlo = lower_to_stablehlo(prog);
        let ll = compile_to_lowlevel(&hlo);
        execute_lowlevel::<StandardAlgebra, _>(self, &ll, &inputs)
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
`SemiringCore<Alg>`:

```rust
// Generic CPU backend — works for any Operand type
struct CpuSemiringBackend;
impl<T: Operand> SemiringCore<T::Algebra> for CpuSemiringBackend {
    type Buffer = T;
    fn batched_gemm(&self, batch_dims, m, n, k, a, b, out) {
        // Delegate to T::dot_general
    }
    fn reduce_sum(&self, axes, input, out) {
        // Delegate to T::reduce_sum
    }
    // ...
}

// User-defined GPU backend for Tropical
struct TropicalGpuBackend { cuda_ctx: CudaContext }
impl SemiringCore<TropicalAlgebra> for TropicalGpuBackend {
    type Buffer = GpuTropicalTensor;
    fn batched_gemm(&self, batch_dims, m, n, k, a, b, out) {
        self.cuda_tropical_gemm(batch_dims, m, n, k, a, b, out);
    }
    fn reduce_sum(&self, axes, input, out) {
        self.cuda_tropical_reduce(axes, input, out);
    }
    // ...
}
```

The full pipeline for custom algebras:

```text
CompiledProgram<SemiringOp<TropicalTensor>>
    │
    │ lower_semiring_to_stablehlo()
    ↓
StableHloProgram
    │
    │ compile_to_lowlevel()  (same optimizing compiler — algebra-agnostic)
    ↓
LowLevelProgram
    │
    │ execute_lowlevel::<TropicalAlgebra, TropicalGpuBackend>(...)
    ↓
Results
```

### GraphOp::eval vs Backend trait vs SemiringCore trait

| | `GraphOp::eval` | `Backend<Op>` | `SemiringCore<Alg>` |
|---|---|---|---|
| Defined in | computegraph-rs | tenferro | tenferro |
| Operates on | `CompiledProgram` | `CompiledProgram` | `LowLevelProgram` |
| Tensor type | fixed (`Op::Operand`) | `Tensor` / backend-internal | `Buffer` (backend-defined) |
| Standard algebra | reference impl | StableHLO → XLA or lowlevel | faer/BLAS kernels |
| Custom algebra | reference impl | StableHLO → lowlevel | user-defined kernels |
| Use case | unit tests, prototyping | top-level entry point | kernel implementation |

`GraphOp::eval` remains useful for:
- computegraph-rs unit tests (no backend dependency)
- Quick prototyping and debugging

`Backend<Op>` is the top-level entry point that orchestrates
lowering + compilation + execution. `SemiringCore<Alg>` is the low-level
trait that backend authors implement to provide kernels.

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

## XI. Operand and TensorData Traits

The previous single `Operand` trait is split into two concerns:

### Operand — pure algebra

`Operand` (defined in computegraph-rs) provides the **algebraic** operations
needed for semiring evaluation. These are the operations that change meaning
across different algebras:

```rust
trait Operand: Clone + Send + Sync + 'static {
    fn zero(shape: &[usize]) -> Self;
    fn one(shape: &[usize]) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn dot_general(&self, other: &Self, config: &DotGeneralConfig) -> Self;
    fn reduce_sum(&self, axes: &[usize]) -> Self;
}
```

`zero` is needed for zero propagation in AD. `one` is needed for reverse-mode
seeding (`ct_y = one`). For custom algebras without AD, `zero` and `one`
still serve as identity elements for the semiring.

### TensorData — buffer access

`TensorData` provides structural access to the underlying buffer. This is
needed by backends and the generic execution engine, but is **not** part of
the algebra:

```rust
trait TensorData: Operand {
    type Scalar;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn data(&self) -> &[Self::Scalar];
    fn from_data(shape: Vec<usize>, data: Vec<Self::Scalar>) -> Self;
}
```

### Structural ops — generic functions, not trait methods

Structural operations (`transpose`, `reshape`, `broadcast_in_dim`) are the
same for all algebras — they only rearrange data, they do not depend on the
semiring. They are implemented as generic functions over `TensorData`, not as
methods on `Operand`:

```rust
fn transpose<T: TensorData>(tensor: &T, perm: &[usize]) -> T { ... }
fn reshape<T: TensorData>(tensor: &T, shape: &[usize]) -> T { ... }
fn broadcast_in_dim<T: TensorData>(tensor: &T, shape: &[usize], dims: &[usize]) -> T { ... }
```

This separation means:
- Implementors of custom algebras only need to define the algebraic operations
  (`add`, `multiply`, `dot_general`, `reduce_sum`) plus buffer access.
- Structural ops are provided for free by the framework.
- The `SemiringOp<T>::eval` implementation calls `Operand` methods for
  algebraic ops and the generic functions for structural ops.

---

## XII. Per-Crate Contents

### tenferro-device

Defines the v2 placement vocabulary and shared runtime errors. `Placement`
contains `memory_kind` plus `resident_device`, while `ComputeDevice` remains a
separate notion for execution. Public memory kinds follow JAX/XLA-style names:
`Device`, `PinnedHost`, `UnpinnedHost`, and `Other(String)`.

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

- `SemiringOpKind` enum (shared vocabulary, used only in `SemiringOp<T>`)
- `SemiringOps` trait
- `SemiringOp<T>` generic wrapper + `GraphOp` impl
- `StdTensorOp` enum — **flat**, most variants mirror a StableHLO op 1:1 (documented exceptions: `Conj`, multi-output linalg)
- `impl GraphOp for StdTensorOp`
- `impl PrimitiveOp for StdTensorOp` (linearize + transpose_rule)
- `impl SemiringOps for StdTensorOp` — maps to flat variants directly
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
  flat 1:1 mapping, some 1:N expansion for `Conj`, multi-output linalg, `Solve`)
- `lower_semiring_to_stablehlo()` (`CompiledProgram<SemiringOp<T>>` →
  `StableHloProgram`)
- Optimizing compiler: `compile_to_lowlevel()` (StableHLO → `LowLevelProgram`)
  - TransposeFolding, DotDecomposer, LinalgCustomCallPassthrough passes
  - Algebra-agnostic — same passes for standard and custom algebras
- `LowLevelProgram`, `LowLevelOp`, `LowLevelInstruction`
- Generic execution engine: `execute_lowlevel()` — interprets `LowLevelProgram`,
  dispatches to `SemiringCore`/`SemiringFastPath` trait methods
- `SemiringCore<Alg>` trait — minimum kernel interface (batched_gemm, reduce_sum, ...)
- `SemiringFastPath<Alg>` trait — optional fast-path operations (contract, fused ops)
- Standard backends:
  - `FaerBackend` — StableHLO → optimizing compiler → LowLevelProgram →
    generic engine → `SemiringCore<StandardAlgebra>` (faer/BLAS/LAPACK)
  - `XlaBackend` — StableHLO → XLA directly (unchanged)
- Custom algebra backends:
  - `CpuSemiringBackend<T>` — generic, implements `SemiringCore` via
    `Operand` trait methods

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
