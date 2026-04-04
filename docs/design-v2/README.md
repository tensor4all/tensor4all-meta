# Design v2 Overview

**Date:** 2026-04-04
**Status:** Draft

This `README.md` is the umbrella/index document for the `docs/design-v2/`
directory.

---

## Vision

Build a differentiable programming stack in Rust with clean separation between
graph infrastructure, AD traits, AD transforms, and concrete primitives.

---

## 3-Layer IR Architecture

The v2 design is organized around three intermediate representations:

```
┌─────────────────────────────────────────┐
│ Tenferro IR                             │
│ (StdTensorOp / SemiringOp<T>)           │
│ Fragment construction, AD, einsum        │
└──────────────┬──────────────────────────┘
               │ lower_to_stablehlo()
┌──────────────▼──────────────────────────┐
│ StableHLO IR                            │
│ (StableHloOp)                           │
│ Serializable cut point                  │
│ XLA backend takes this directly         │
└──────────────┬──────────────────────────┘
               │ optimizing compiler
┌──────────────▼──────────────────────────┐
│ Execution IR                            │
│ (ExecOp)                                │
│ faer / custom backends execute this     │
│ stride-aware engine dispatch            │
└─────────────────────────────────────────┘
```

**Tenferro IR** (`StdTensorOp` / `SemiringOp<T>`) — the graph-level vocabulary
for fragment construction, AD, and einsum decomposition.

**StableHLO IR** (`StableHloOp`) — the **single cut point** between graph/AD
and execution. Serializable to StableHLO MLIR. XLA backend takes this directly.

**Execution IR** (`ExecOp`) — output of the optimizing compiler. Same ops as
StableHLO + `BatchedGemm` − `DotGeneral`. faer/custom backends execute this
via stride-aware engine dispatch.

See [`primitive-catalog.md`](primitive-catalog.md) for the full specification
of each layer.

---

## Crate Hierarchy

```text
computegraph-rs    General-purpose computation graph engine
    ↓              (GraphOp, Operand, Fragment, resolve,
    ↓               materialize_merge, compile, eval, cache)
    ↓
chainrules-rs      AD trait definitions
    ↓              (PrimitiveOp: GraphOp + add + linearize + transpose_rule)
    ↓
tidu-rs            AD graph transforms
    ↓              (differentiate, transpose — generic over PrimitiveOp)
    ↓
tenferro-rs        Concrete tensor primitives + backends
                   (Tensor, StableHLO lowering, optimizing compiler,
                    Execution IR, CPU/GPU dispatch)
```

Each layer depends only on the layers above it. No layer references specific
primitives except tenferro-rs.

---

## Per-Crate Design Documents

| Crate | Document | Key Contents |
|-------|----------|--------------|
| computegraph-rs | [`computegraph-design.md`](computegraph-design.md) | AD-agnostic graph IR: `GraphOp`, `Operand`, `Fragment`, `resolve`, `materialize_merge`, `compile`, `eval`, cache |
| chainrules-rs | [`chainrules-design.md`](chainrules-design.md) | `PrimitiveOp` trait, cotangent accumulation via `add()`, and primitive-local `linearize` / `transpose_rule` contract |
| tidu-rs | [`tidu-design.md`](tidu-design.md) | `differentiate`, `transpose`, `LinearFragment`, higher-order AD pipelines |
| tenferro-rs | [`backend-architecture.md`](backend-architecture.md) | StableHLO-targeted backend architecture, `custom_call`, CPU/GPU/custom-algebra split |
| tenferro-rs | [`tensor-design.md`](tensor-design.md) | Dense tensor principle, structural tensor information, einsum decomposition |
| tenferro-rs | [`tensor-api-pseudocode.md`](tensor-api-pseudocode.md) | `TracedTensor` API, lazy evaluation, AD usage examples |
| tenferro-rs | [`primitive-catalog.md`](primitive-catalog.md) | v2 core primitive vocabulary, backend-facing execution subsets, per-op definitions, canonical lowering |
| tenferro-rs | [`tenferro-internal-design.md`](tenferro-internal-design.md) | Internal crate structure, `StdTensorOp` / `SemiringOp<T>` dual Op types, `SemiringOps` trait, einsum graph builder, backend dispatch, user extension points |
---

## Related Documents

| Document | Contents |
|----------|----------|
| [`ad-architecture.md`](ad-architecture.md) | Detailed AD theory, examples (scalar + vector), golden tests |
| [`primitive-catalog.md`](primitive-catalog.md) | Conceptual v2 primitive catalog: core primitives, standard-arithmetic-only ops, execution subsets, StableHLO alignment |
| [`backend-architecture.md`](backend-architecture.md) | Backend-specific details |
| [`stablehlo-primitives.md`](stablehlo-primitives.md) | StableHLO op inventory with concrete per-op descriptions, plus deprecated and legacy names |
| [`jax-primitives.md`](jax-primitives.md) | JAX primitive inventory with concrete per-primitive descriptions for tensor ops, AD, control flow, and linalg |
| [`jax-stablehlo-primitives-needed-for-tenferro.md`](jax-stablehlo-primitives-needed-for-tenferro.md) | Phase-1 StableHLO target set, current tenferro-to-StableHLO correspondence tables, and the JAX-like primitive layer to add above StableHLO |
| [`ad-graph-experiments.md`](ad-graph-experiments.md) | Earlier experiments and notes that informed the fragment-based AD design |
| [`vector_ad_examples_check.py`](vector_ad_examples_check.py) | Sanity-check script for the vector AD examples in `ad-architecture.md` |

---

## How To Read The Primitive Documents

The primitive-related documents have different roles:

- [`primitive-catalog.md`](primitive-catalog.md)
  - the conceptual v2 primitive catalog
  - defines the internal tensor-op vocabulary and its canonical lowering
- [`jax-stablehlo-primitives-needed-for-tenferro.md`](jax-stablehlo-primitives-needed-for-tenferro.md)
  - the implementation-planning document
  - maps current tenferro operations to the StableHLO ops that actually need to exist below the new v2 primitive layer
- [`jax-primitives.md`](jax-primitives.md)
  - reference document for what JAX primitives concretely do
- [`stablehlo-primitives.md`](stablehlo-primitives.md)
  - reference document for what StableHLO ops concretely do

## Key Design Principles

### computegraph-rs is AD-agnostic

The graph engine knows nothing about differentiation. It is equally usable for
multi-tensor einsum (graph of binary contractions) or any DAG-structured
computation.

**Note on `Operand`:** While `Operand` is defined in computegraph-rs, its
methods are tensor-oriented algebraic operations (`dot_general`, `reduce_sum`,
`conj`, etc.). Structural operations (`transpose`, `reshape`,
`broadcast_in_dim`) live in a separate `TensorData` trait. This is a deliberate
design choice -- computegraph-rs is a **tensor computation graph engine**, not a
fully generic DAG engine. The abstraction boundary is AD-agnostic, not
tensor-agnostic.

### tidu-rs is primitive-agnostic

AD transforms (`differentiate`, `transpose`) are fully generic over
`Op: PrimitiveOp`. tidu-rs never references specific primitives such as `Add`
or `Mul`.

The concrete tensor vocabulary that downstream tenferro supplies is documented
in [`primitive-catalog.md`](primitive-catalog.md).

### Closure is downstream responsibility

The only rule for primitives is:

> `linearize` and `transpose_rule` must emit only ops that themselves
> implement `PrimitiveOp`.

tenferro-rs is responsible for satisfying this.

### Physical merge is late

```text
differentiate -> resolve -> differentiate -> resolve -> ...  (logical, cheap)
materialize_merge  (physical, once, before compile)
```

Higher-order AD requires `resolve` (logical), not `materialize_merge`
(physical).

### Execution context is an associated type

```rust
trait GraphOp {
    type Context;
    fn eval(&self, ctx: &mut Self::Context, ...) -> ...;
}
```

This allows backends to inject execution state (CPU context, GPU context)
without constraining the graph engine.

### All operations are lazy, two tensor types

tenferro-rs exposes two types:

```rust
// Typed tensor (internal)
struct TensorData<T: Scalar> { buffer: Vec<T>, shape, strides }

// Type-erased tensor (user-facing)
enum Tensor { F32(TensorData<f32>), F64(...), C32(...), C64(...) }

// Graph-aware wrapper — tracks computation for AD and compilation
struct TracedTensor {
    shape: Vec<usize>,
    dtype: DType,
    fragment: Arc<Fragment<StdTensorOp>>,  // graph info (always present)
    val: LocalValId,
    data: Option<Tensor>,                  // Some for inputs / eval'd results
}
```

- `TracedTensor::from(Tensor)` creates a Fragment input node with `data = Some(...)`.
- Operations (einsum, exp, ...) build graph, return `TracedTensor` with `data = None`.
- `eval()` triggers compile (cached) + execute, fills in `data`, returns `&Tensor`.
- Same function works for plain computation and AD — no mode switching.

`TracedTensor::from()` consumes the `Tensor` (move semantics, no implicit copy).
Clone explicitly if the original is still needed.

See `tensor-api-pseudocode.md` for full usage examples.

---

## Typical Pipeline

```text
build → resolve → differentiate → transpose → resolve → materialize_merge → compile
        ╰─ computegraph ─╯   ╰── tidu ──╯              ╰──── computegraph ────╯

    → lower_to_stablehlo → [XLA: direct] or [optimizing compiler → Execution IR → eval]
      ╰──────────────────────── tenferro ──────────────────────────────────────────────╯
```

---

## Relationship to Existing tenferro-rs

### Semiring Core stays separate

`TensorSemiringCore<Alg>` is an execution protocol for semiring-compatible
tensor operations (BatchedGemm, ReduceSum, etc.), parameterized by scalar
algebra. It serves einsum execution and supports exotic algebras (tropical).

It is orthogonal to `PrimitiveOp` which operates at the tensor-op / AD level.
The two do not conflict and do not need unification.

The current-tenferro-to-StableHLO correspondence for these execution families
is summarized in
[`jax-stablehlo-primitives-needed-for-tenferro.md`](jax-stablehlo-primitives-needed-for-tenferro.md).

### AD-specific ops leave Semiring Core

`AntiTrace` and `AntiDiag` are adjoint operations that exist for AD. In v2
they belong as `transpose_rule` outputs of `Trace`/`Diag` primitives in
tenferro-rs, not in the Semiring Core execution protocol.

### Tape-based AD is superseded

The current `chainrules-core` + `tidu` (tape-based AD) is replaced by the
graph-based fragment AD described here.

---

## Implementation Strategy

This is a clean-slate rewrite. Each repo uses a `feat/v2` branch with all
existing v1 code removed. v2 code is built from scratch based on these design
documents.

### Branch Setup

| Repo | Branch | Action |
|------|--------|--------|
| `computegraph-rs` | (new repo) | Create repo, implement from scratch |
| `chainrules-rs` | `feat/v2` | (exists) Delete all v1 src, rewrite from scratch |
| `tidu-rs` | `feat/v2` | (exists) Delete all v1 src, rewrite from scratch |
| `tenferro-rs` | `feat/v2` | (create) Delete all v1 src, rewrite from scratch |

v1 code is preserved on `main`. The `feat/v2` branches are not expected to
merge back into `main` — when v2 is ready, `main` will be replaced.

### Implementation Order

```text
1. computegraph-rs   ← no dependencies, start here
2. chainrules-rs     ← depends on computegraph-rs
3. tidu-rs           ← depends on computegraph-rs + chainrules-rs
4. tenferro-rs       ← depends on all three
```

Each step should be independently testable before proceeding to the next.

---

## Roadmap

### Phase 1: Scalar fragment AD

- `computegraph-rs`: Fragment, resolve, materialize_merge, compile, eval
- `chainrules-rs`: PrimitiveOp trait
- `tidu-rs`: differentiate, transpose
- Scalar primitives: `Add`, `Mul`, `Exp`, `Neg`, `Conj`
- Tests: forward, backward, and second order on `exp(a*x)`

### Phase 2: Tensor primitives

- `StdTensorOp` with full primitive set
- `Tensor` (type-erased enum) implementing `Operand`
- Vector and reduction transpose rules
- Batched JVP via tensor-valued tangent inputs

### Phase 3: Backend compilation

- `MaterializedGraph -> CompiledProgram`
- StableHLO lowering
- CPU / GPU backends consume only compiled or lowered materialized programs

### Phase 4: Optimization

- Logical-DAG-aware checkpoint scheduling
- Partial transpose / cross-country mode
- Late materialization heuristics
- Operator fusion in compiled IR
