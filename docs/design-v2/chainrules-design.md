# v2 chainrules-rs Design

**Date:** 2026-04-03
**Status:** Draft
**Repo:** chainrules-rs
**Parent:** `README.md`
**Depends on:** `computegraph-rs`

---

## I. Purpose

`chainrules-rs` defines the AD trait (`PrimitiveOp`) that extends
`computegraph::GraphOp` with linearization and transpose rules. It contains
no graph infrastructure and no concrete primitives.

This is the v2 counterpart of the AD behavior that JAX stores in
`primitive_jvps` and `primitive_transposes`. The information is the same kind
of information, but the representation is different:

- JAX: global registries keyed by primitive object
- v2: methods on the concrete primitive type itself

---

## II. PrimitiveOp Trait

```rust
use computegraph::{GraphOp, FragmentBuilder, GlobalValKey, LocalValId, ValRef, OpMode};

pub trait PrimitiveOp: GraphOp {
    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;
}
```

---

## III. Linearization Rules

A primitive's `linearize` must be linear in tangent inputs. It may:

- reference primal inputs or outputs through `External(GlobalValKey)`
- emit primitives in `OpMode::Linear`
- emit `Dup` or `Conj` when required by transpose semantics

It must not introduce nonlinear dependence on tangent inputs.

The intended mental model is close to JAX `linearize`:

- JAX `linearize` applies a primitive's JVP rule and emits a new composition of
  JAX primitives in a jaxpr
- v2 `PrimitiveOp::linearize` emits a new composition of downstream concrete
  primitives into a fragment

---

## IV. Transpose Rules

A primitive's `transpose_rule` receives cotangent outputs and produces
cotangent inputs. It must only emit primitives that themselves implement
`PrimitiveOp`.

---

## V. ADKey Trait

`chainrules-rs` defines the `ADKey` trait that constrains `InputKey` for AD
use. `tidu-rs` uses this trait to generate tangent input keys during
`differentiate`.

```rust
pub type DiffPassId = u64;

pub trait ADKey: Clone + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `differentiate` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

`PrimitiveOp` requires `Self::InputKey: ADKey`:

```rust
pub trait PrimitiveOp: GraphOp where Self::InputKey: ADKey {
    fn linearize(...) -> ...;
    fn transpose_rule(...) -> ...;
}
```

The concrete implementation of `ADKey` is the downstream implementor's
choice. A typical pattern is a recursive enum:

```rust
// tenferro-rs
enum TensorInputKey {
    User(String),
    Tangent { of: Box<TensorInputKey>, pass: DiffPassId },
}
```

This gives debuggable keys like
`Tangent { of: Tangent { of: User("x"), pass: 1 }, pass: 3 }` for
higher-order AD.

---

## VI. Closure Responsibility

`chainrules-rs` defines the contract but does not enforce closure. The
downstream implementor (e.g. tenferro-rs) is responsible for ensuring that
the set of primitives reachable through `linearize` and `transpose_rule`
is closed — i.e., every emitted op also implements `PrimitiveOp`.

---

## VI. Design Boundaries

```text
chainrules-rs owns:
  - PrimitiveOp trait (linearize + transpose_rule)

chainrules-rs does NOT own:
  - graph infrastructure → computegraph-rs
  - AD transforms (differentiate, transpose) → tidu-rs
  - concrete primitives → downstream (tenferro-rs)
```
