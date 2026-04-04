# v2 tenferro-rs Internal Design

**Date:** 2026-04-04
**Status:** Draft
**Repo:** tenferro-rs
**Parent:** `../README.md`
**Related:** `computegraph.md`, `chainrules.md`, `tidu.md`, `../spec/backend-contract.md`, `../spec/primitive-catalog.md`

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

`StdTensorOp` is a **flat** enum whose variants mostly mirror StableHLO ops
1:1 (documented exceptions: composite lowerings like `Conj`, multi-output
linalg ops like `Svd`). It implements `GraphOp`, `PrimitiveOp`, and
`SemiringOps`.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV -- Tenferro IR Vocabulary).
AD trait (`PrimitiveOp`): [`spec/ad-contract.md`](../spec/ad-contract.md).

### SemiringOp\<T\> — custom algebra, semiring subset, no AD

`SemiringOp<T>` is a generic wrapper around `SemiringOpKind` that implements
`GraphOp` for any `T: Operand`. It delegates algebraic ops to `Operand` trait
methods and structural ops to generic `TensorData` functions. `PrimitiveOp` is
**not** implemented -- no AD for custom algebras.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV).

Users extend tenferro by implementing `Operand` (algebraic ops) and
`TensorData` (buffer access) for their tensor type, then use
`SemiringOp<MyTensor>` as the op type. Structural ops (`transpose`,
`reshape`, `broadcast_in_dim`) are provided automatically.

Canonical `Operand` definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md).
Canonical `TensorData` definition: [`spec/tensor-semantics.md`](../spec/tensor-semantics.md).

---

## V. SemiringOpKind — Shared Vocabulary

`SemiringOpKind` is the set of operations that all algebras must support.
It is used **only** inside `SemiringOp<T>` — the generic custom-algebra op
type. `StdTensorOp` does **not** wrap `SemiringOpKind`; it has its own flat
variants that mostly mirror StableHLO 1:1 (with documented exceptions for
composite lowerings and multi-output linalg ops).

`SemiringOpKind` is the minimal set of ops all algebras must support:
`Add`, `Mul`, `DotGeneral`, `ReduceSum`, `Transpose`, `Reshape`,
`BroadcastInDim`.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV -- AD-closed graph core + structural ops).

`SemiringOp<T>` wraps it as a newtype. The `SemiringOps` trait bridges both
worlds: `StdTensorOp` implements it by mapping to flat variants,
`SemiringOp<T>` implements it by mapping to `SemiringOpKind` variants.

---

## VI. SemiringOps Trait — Generic Einsum

`SemiringOps` bridges both `StdTensorOp` and `SemiringOp<T>` so that einsum
Fragment construction is algebra-agnostic. Both op types implement it.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md).

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
    └── FaerBackend: StableHLO → optimizing compiler → ExecProgram
                         → generic execution engine → SemiringCore trait
```

XLA consumes StableHLO directly (it already does its own optimization).
All other backends go through the optimizing compiler to produce a
`ExecProgram`, which a generic engine interprets by dispatching to
backend traits.

For custom algebras (`SemiringOp<T>`), the same 2-level structure applies:
`SemiringOp<T>` lowers to the same `StableHloOp` types, then to `ExecProgram`.
**Note:** for custom algebra, the ops have semiring-specific semantics (Add=⊕,
Mul=⊗). This IR is **not** serializable to StableHLO MLIR — the XLA path is
not available. Custom algebra always goes through the optimizing compiler →
Execution IR → stride-aware engine path.

### StableHLO IR representation

tenferro defines its own Rust data structures that mirror StableHLO semantics.
This is neither binary nor text — it is an in-process Rust struct passed
directly to backends. No serialization for faer/GPU backends.

The `StableHloProgram`, `StableHloOp`, and `StableHloInstruction` types
mirror StableHLO semantics as in-process Rust structs.

Canonical definition: [`spec/backend-contract.md`](../spec/backend-contract.md).

### StableHLO lowering

`StdTensorOp` lowering is mostly 1:1 (flat variants map directly to
`StableHloOp`), with documented 1:N exceptions (`Conj`, linalg).
`SemiringOp<T>` lowers to the same `StableHloOp` types but with semiring
semantics (not MLIR-serializable).

Canonical lowering rules and custom_call targets:
[`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section VI -- StableHLO Alignment)
and [`spec/backend-contract.md`](../spec/backend-contract.md).

### Optimizing compiler and Execution IR

The optimizing compiler transforms `StableHloProgram` into `ExecProgram`
(algebra-agnostic). `ExecOp` uses the same op vocabulary as StableHLO with
one substitution: `DotGeneral` is replaced by `BatchedGemm`.

Canonical `ExecOp` definition and pass list:
[`spec/backend-contract.md`](../spec/backend-contract.md).
Pass algorithms: [`spec/optimizer-passes.md`](../spec/optimizer-passes.md).

### Generic execution engine

The generic engine interprets `ExecProgram` by dispatching each instruction
to `SemiringCore`/`SemiringFastPath` methods, standard kernels, or common
infrastructure, depending on the dispatch category.

*(illustrative, non-normative -- see [`spec/backend-contract.md`](../spec/backend-contract.md) for canonical definition)*

### Backend traits

`SemiringCore` (required: `batched_gemm` + `reduce_sum`) and
`SemiringFastPath` (optional: `contract`, `elementwise_mul`,
`elementwise_add`) define what a backend must implement.

Canonical trait signatures: [`spec/backend-contract.md`](../spec/backend-contract.md).

### Standard and custom algebra backends

Two standard backends are provided: `FaerBackend` (StableHLO -> optimizing
compiler -> ExecProgram -> generic engine -> faer/BLAS/LAPACK) and
`XlaBackend` (StableHLO -> XLA directly). Custom algebra backends implement
`SemiringCore<Alg>` with a minimum of `batched_gemm` + `reduce_sum`.

The relationship between `GraphOp::eval` (computegraph-rs, for unit tests),
`Backend<Op>` (tenferro, top-level entry point), and `SemiringCore<Alg>`
(tenferro, kernel implementation) is documented in
[`spec/backend-contract.md`](../spec/backend-contract.md).

`GraphOp::eval` remains useful for:
- computegraph-rs unit tests (no backend dependency)
- Quick prototyping and debugging

`Backend<Op>` is the top-level entry point that orchestrates
lowering + compilation + execution. `SemiringCore<Alg>` is the kernel-level
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

### Operand -- pure algebra

`Operand` (defined in computegraph-rs) provides the algebraic operations
needed for semiring evaluation (`zero`, `one`, `add`, `multiply`,
`dot_general`, `reduce_sum`). These change meaning across different algebras.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV).

### TensorData -- buffer access

`TensorData` extends `Operand` with structural buffer access (`shape`,
`strides`, `data`, `from_data`). Needed by backends and the execution engine
but not part of the algebra.

Canonical definition: [`spec/tensor-semantics.md`](../spec/tensor-semantics.md).

### Structural ops -- generic functions, not trait methods

`transpose`, `reshape`, `broadcast_in_dim` are generic functions over
`TensorData`. They are the same for all algebras and are provided by the
framework. Custom algebra implementors only define algebraic operations plus
buffer access.

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
- Optimizing compiler: `compile_to_exec()` (StableHLO → `ExecProgram`)
  - TransposeFolding, DotDecomposer, LinalgCustomCallPassthrough passes
  - Algebra-agnostic — same passes for standard and custom algebras
- `ExecProgram`, `ExecOp`, `ExecInstruction`
- Generic execution engine: `execute_exec()` — interprets `ExecProgram`,
  dispatches to `SemiringCore`/`SemiringFastPath` trait methods
- `SemiringCore<Alg>` trait — minimum kernel interface (batched_gemm, reduce_sum, ...)
- `SemiringFastPath<Alg>` trait — optional fast-path operations (contract, fused ops)
- Standard backends:
  - `FaerBackend` — StableHLO → optimizing compiler → ExecProgram →
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
