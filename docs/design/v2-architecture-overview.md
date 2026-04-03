# v2 Architecture Overview

**Date:** 2026-04-03
**Status:** Draft

---

## Vision

Build a differentiable programming stack in Rust with clean separation between
graph infrastructure, AD traits, AD transforms, and concrete primitives.

---

## Crate Hierarchy

```text
computegraph-rs    General-purpose computation graph engine
    ↓              (GraphOp, Operand, Fragment, resolve,
    ↓               materialize_merge, compile, eval, cache)
    ↓
chainrules-rs      AD trait definitions
    ↓              (PrimitiveOp: GraphOp + linearize + transpose_rule)
    ↓
tidu-rs            AD graph transforms
    ↓              (differentiate, transpose — generic over PrimitiveOp)
    ↓
tenferro-rs        Concrete tensor primitives + backends
                   (DynTensor, StableHLO lowering, CPU/GPU dispatch)
```

Each layer depends only on the layers above it. No layer references specific
primitives except tenferro-rs.

---

## Per-Crate Design Documents

| Crate | Document | Key Contents |
|-------|----------|--------------|
| computegraph-rs | [`v2-computegraph-design.md`](v2-computegraph-design.md) | GraphOp, Operand, Fragment, resolve, materialize_merge, compile (SSA), eval, compilation cache |
| chainrules-rs | [`v2-chainrules-design.md`](v2-chainrules-design.md) | PrimitiveOp trait (linearize + transpose_rule), closure contract |
| tidu-rs | [`v2-tidu-design.md`](v2-tidu-design.md) | differentiate, transpose, LinearFragment, pipelines (JVP/VJP/HVP/higher-order) |
| tenferro-rs | [`v2-backend-architecture.md`](v2-backend-architecture.md) | StableHLO lowering, CPU/GPU backends |

---

## Related Documents

| Document | Contents |
|----------|----------|
| [`v2-ad-architecture.md`](v2-ad-architecture.md) | Detailed AD theory, examples (scalar + vector), golden tests |
| [`v2-transpose-rules.md`](v2-transpose-rules.md) | Per-primitive transpose rule table |
| [`v2-backend-architecture.md`](v2-backend-architecture.md) | Backend-specific details |

---

## Key Design Principles

### computegraph-rs is AD-agnostic

The graph engine knows nothing about differentiation. It is equally usable for
multi-tensor einsum (graph of binary contractions) or any DAG-structured
computation.

### tidu-rs is primitive-agnostic

AD transforms (`differentiate`, `transpose`) are fully generic over
`Op: PrimitiveOp`. tidu-rs never references specific primitives such as `Add`
or `Mul`.

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

### User-facing Tensor is an enum

tenferro-rs exposes a single `Tensor` type that supports both eager and
traced execution:

```rust
enum Tensor {
    Eager(DynTensor),          // immediate data
    Traced(TracedDynTensor),   // graph node reference
}
```

- **Eager**: operations execute immediately. For debugging, tropical algebra,
  or any case where AD/compilation is not needed.
- **Traced**: operations build a computation graph. Required for AD and
  compilation caching. `eval()` triggers compile + execute and returns
  `DynTensor`.

When eager and traced tensors are mixed in an operation, the eager side is
consumed as a constant node in the graph (result is traced).

`trace.input()` consumes a `DynTensor` (move semantics, no implicit copy).
Clone explicitly if the original is still needed.

---

## Typical Pipeline

```text
build → resolve → differentiate → transpose → resolve → materialize_merge → compile → eval
        ╰─ computegraph ─╯   ╰── tidu ──╯              ╰──── computegraph ────╯
```

---

## Relationship to Existing tenferro-rs

### Semiring Core stays separate

`TensorSemiringCore<Alg>` is an execution protocol for semiring-compatible
tensor operations (BatchedGemm, ReduceAdd, etc.), parameterized by scalar
algebra. It serves einsum execution and supports exotic algebras (tropical).

It is orthogonal to `PrimitiveOp` which operates at the tensor-op / AD level.
The two do not conflict and do not need unification.

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
- Scalar primitives: `Add`, `Mul`, `Exp`, `Dup`, `Neg`, `Conj`
- Tests: forward, backward, and second order on `exp(a*x)`

### Phase 2: Tensor primitives

- `TensorOp` with full primitive set
- `DynTensor` implementing `Operand`
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
