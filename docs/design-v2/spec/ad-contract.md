# AD Contract

**Date:** 2026-04-04
**Status:** Draft
**Parent:** [`../README.md`](../README.md)
**Related:** [`primitive-catalog.md`](primitive-catalog.md), [`../architecture/chainrules.md`](../architecture/chainrules.md), [`../architecture/tidu.md`](../architecture/tidu.md)

---

## Purpose

This document is the normative specification for the AD trait contract that
concrete primitives must satisfy. It owns the `PrimitiveOp` trait signature
and the rules that `linearize` and `transpose_rule` must follow.

For the AD pipeline architecture (differentiate, transpose, higher-order AD),
see [`../architecture/ad-pipeline.md`](../architecture/ad-pipeline.md).

For the AD trait design rationale, see
[`../architecture/chainrules.md`](../architecture/chainrules.md).

---

## PrimitiveOp trait (canonical signature)

```rust
trait PrimitiveOp: GraphOp {
    /// Produce a linear approximation of this op at the given primal point.
    ///
    /// Returns a `LinearFragment` whose outputs are linear functions of the
    /// tangent inputs. The fragment may reference primal values via external
    /// `GlobalValKey` references.
    fn linearize(
        &self,
        primals: &[GlobalValKey<Self>],
        builder: &mut FragmentBuilder<Self>,
    ) -> LinearFragment<Self>;

    /// Transpose a linear use of this op.
    ///
    /// Given cotangent(s) for the output(s), produce cotangent(s) for the
    /// input(s). Only active inputs (indicated by `active_mask`) receive
    /// cotangents; inactive inputs are ignored.
    fn transpose_rule(
        &self,
        cotangents: &[ValRef<Self>],
        primals: &[GlobalValKey<Self>],
        active_mask: &[bool],
        builder: &mut FragmentBuilder<Self>,
    ) -> Vec<Option<LocalValId>>;
}
```

## Rules

1. **Closure**: `linearize` and `transpose_rule` must emit only ops that
   themselves implement `PrimitiveOp`. This is the sole closure requirement.
   tenferro-rs is responsible for satisfying it.

2. **Cotangent accumulation**: when a value fans out to multiple consumers,
   tidu's `transpose` accumulates cotangents via `Op::add()`. This means
   `Add` must implement `PrimitiveOp` and its transpose rule must be the
   identity (cotangent passes through to both inputs).

3. **Linear ops**: an op whose `linearize` returns itself (identity tangent
   map) only needs a `transpose_rule`. Examples: `Transpose`, `Reshape`,
   `BroadcastInDim`.

4. **Primal reuse**: `linearize` may reference primal values in its
   `LinearFragment` via `GlobalValKey`. These are resolved during
   `materialize_merge` so that shared primal computations are not duplicated.

5. **No AD for custom algebra**: `SemiringOp<T>` does NOT implement
   `PrimitiveOp`. AD is only available for `StdTensorOp` (standard algebra).

## Owned by this document

- `PrimitiveOp` trait signature
- Closure rule
- Cotangent accumulation rule
- Linear op rule
- Primal reuse rule

Other documents link here for the AD contract; they do not re-state
these definitions.
