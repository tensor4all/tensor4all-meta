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
| `internal/ad-linalg` | → `tenferro-linalg` PrimitiveOp impl | Same |
| `internal/ad-surface` | → tidu-rs `differentiate`/`transpose` | External crate |
| `internal/frontend-core` | → `tenferro` TracedTensor | Lazy, not eager |
| `internal/runtime` | → `tenferro` Engine | |
| `tenferro-dynamic-compute` | deleted | Always graph |
| `tenferro-tensor-compute` | → `tenferro-ops` | |
| `tenferro-linalg-prims` | → `tenferro-linalg` | No need to separate |
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
| `tenferro-linalg` | `tenferro-linalg` | AD rules → PrimitiveOp impl |
| `tenferro` (facade) | `tenferro` | TracedTensor, Engine, backends |

**29 crates → 7 crates** (plus 3 external: computegraph-rs, chainrules-rs,
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
    ├── tenferro-linalg (linalg TensorOp + PrimitiveOp impls)
    |
tenferro ──────────── tidu-rs (differentiate, transpose)
    (TracedTensor, Engine, backends)
```

---

## IV. Two Op Types

The fundamental design constraint is that `GraphOp::Operand` is an associated
type, so a single Op type can only serve one `Operand` type. Since standard
algebra (`DynTensor`) and custom algebras (`TropicalTensor`, etc.) have
different `Operand` types, tenferro provides two Op types:

### StdTensorOp — standard algebra, full vocabulary, AD-capable

```rust
enum StdTensorOp {
    // Tier 1: semiring-compatible core
    Semiring(SemiringOpKind),
    Neg, Conj, Dup,

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

    // Linalg (re-exported from tenferro-linalg)
    Svd, Qr, Cholesky, Eigh, Solve,
}

impl GraphOp for StdTensorOp {
    type Operand = DynTensor;       // f32, f64, Complex<f32>, Complex<f64>
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

### Standard algebra backends

```text
CompiledProgram<StdTensorOp>
    |
    |── CPU (default): GraphOp::eval with CpuContext (faer/BLAS)
    |
    |── XLA (optional): StableHLO lowering → XLA JIT compile → execute
    |                    (bypasses GraphOp::eval)
    |
    └── Custom GPU: StableHLO lowering → op-by-op CUDA dispatch
                    (bypasses GraphOp::eval)
```

The CPU backend uses `GraphOp::eval` directly. XLA and GPU backends lower
`CompiledProgram<StdTensorOp>` to StableHLO first, then use their own
execution paths. `GraphOp::Context = CpuContext` is the default; StableHLO
backends have their own execution pipeline.

### Custom algebra backends

```text
CompiledProgram<SemiringOp<T>>
    |
    |── CPU: GraphOp::eval with SemiringContext<T>
    |        (delegates to T: Operand methods)
    |
    └── GPU: custom kernel dispatch (e.g., v1 CUDA kernels for Tropical)
```

Custom algebras do not go through StableHLO. The CPU backend evaluates
`CompiledProgram` instruction-by-instruction using `GraphOp::eval`, which
delegates to `Operand` trait methods.

### Backend dispatch in Engine

```rust
struct Engine {
    backend: BackendKind,
    compile_cache: CompileCache,
    einsum_cache: EinsumCache,
}

enum BackendKind {
    Cpu(CpuContext),
    Xla(XlaContext),
    // Custom algebra uses SemiringContext<T> separately
}
```

For `StdTensorOp`, the `Engine` manages backend selection. For
`SemiringOp<T>`, the user provides a `SemiringContext<T>` directly.

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

    /// Lazy evaluation: resolve → materialize_merge → compile → eval
    fn eval(&mut self, engine: &mut Engine) -> &Tensor;

    /// VJP: differentiate → transpose (via tidu-rs)
    fn grad(&self, wrt: &TracedTensor) -> TracedTensor;

    /// JVP: differentiate only (via tidu-rs)
    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor;
}
```

For custom algebras, users work with `Fragment<SemiringOp<T>>` and
`CompiledProgram<SemiringOp<T>>` directly through the computegraph-rs API,
without `TracedTensor`.

---

## X. User Extension Points

| Goal | What to implement |
|---|---|
| New scalar algebra for einsum | `impl Operand for MyTensor` |
| Custom GPU backend for custom algebra | Custom `SemiringContext<T>` |
| AD for custom algebra | Define own Op enum, impl `PrimitiveOp` (advanced) |

The minimal extension path: implement `Operand`, use `SemiringOp<MyTensor>`,
and einsum + compile + eval work immediately.

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

Simplified from v1. `Tensor`, `DType`, `Buffer`. No AD-related code.
`impl Operand for DynTensor`.

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

### tenferro-linalg

Linalg primitives as `StdTensorOp` variants:

- SVD, QR, Cholesky, Eigh, Solve
- `PrimitiveOp` implementations (linearize + transpose_rule)
- Backend kernels (LAPACK, cuSOLVER)
- `custom_call` lowering support

Depends on: computegraph-rs, chainrules-rs, tenferro-ops, tenferro-tensor.

### tenferro

Top-level facade:

- `TracedTensor` (lazy graph-aware wrapper)
- `Engine` (compilation cache, backend dispatch, einsum cache)
- Public API: `einsum()`, `grad()`, `jvp()`, `eval()`
- StableHLO lowering (`StdTensorOp` → StableHLO)
- Backend implementations (CPU/faer, XLA)

Depends on: all of the above + tidu-rs.

---

## XIII. Roadmap

### Phase 1: Scalar fragment AD

- computegraph-rs: Fragment, resolve, materialize_merge, compile, eval
- chainrules-rs: PrimitiveOp trait
- tidu-rs: differentiate, transpose
- tenferro-ops: scalar subset of StdTensorOp (Add, Mul, Exp, Neg, Dup, Conj)
- tenferro: minimal Engine with CPU eval
- Tests: forward, backward, second order on `exp(a*x)`

### Phase 2: Tensor primitives + einsum

- tenferro-ops: full StdTensorOp (DotGeneral, ReduceSum, BroadcastInDim, ...)
- tenferro-ops: SemiringOp\<T\>, SemiringOps trait
- tenferro-tensor: Tensor, DynTensor, impl Operand
- tenferro-einsum: contraction path + Fragment construction
- Tests: vector AD examples, einsum correctness

### Phase 3: Linalg + backends

- tenferro-linalg: SVD, QR, Cholesky with PrimitiveOp impls
- tenferro: StableHLO lowering, XLA backend
- tenferro: CPU backend with faer/BLAS
- Tests: linalg AD, StableHLO round-trip

### Phase 4: Custom algebra + optimization

- SemiringOp\<T\> end-to-end with Tropical
- Custom GPU backend (reuse v1 CUDA kernels)
- tenferro-capi (C FFI for Julia/Python)
- Logical-DAG-aware checkpoint scheduling
- Operator fusion in compiled IR
