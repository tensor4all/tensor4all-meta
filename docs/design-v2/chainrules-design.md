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

---

## IV. Transpose Rules

A primitive's `transpose_rule` receives cotangent outputs and produces
cotangent inputs. It must only emit primitives that themselves implement
`PrimitiveOp`.

---

## V. Closure Responsibility

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
