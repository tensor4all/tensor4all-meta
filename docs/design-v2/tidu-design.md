# v2 tidu-rs Design

**Date:** 2026-04-03
**Status:** Draft
**Repo:** tidu-rs
**Parent:** `README.md`
**Depends on:** `computegraph-rs`, `chainrules-rs`

---

## I. Purpose

`tidu-rs` provides AD-specific graph transforms (`differentiate`, `transpose`)
that are fully generic over `Op: PrimitiveOp`. It owns no graph infrastructure
(that belongs to `computegraph-rs`) and references no specific primitives.

---

## II. Transforms

### `differentiate`

Consumes a resolved view and returns a new linear fragment (JVP).

```rust
use computegraph::{ResolvedView, GlobalValKey, LocalValId, Fragment};
use chainrules::PrimitiveOp;

struct LinearFragment<Op> {
    fragment: Fragment<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    tangent_outputs: Vec<Option<LocalValId>>,
}

fn differentiate<Op: PrimitiveOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
) -> LinearFragment<Op>;
```

Algorithm:

1. Traverse the reachable logical DAG in topological order.
2. Seed tangent inputs for the requested primal `InputKey`s.
3. For each reachable primitive, call `Op::linearize`.
4. Emit new local linear nodes into the new fragment.
5. Reference primal values through `External(GlobalValKey)`.
6. Skip unreachable tangent flow with zero propagation.

### `transpose`

Consumes a linear fragment and produces another with active inputs and outputs
reversed.

```rust
fn transpose<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
) -> LinearFragment<Op>;
```

Traverses the linear fragment in reverse topological order and, for each op
node, calls `Op::transpose_rule` to obtain the local transposed contribution.

Transpose accumulation must use global identity: when multiple reverse
contributions flow back to the same original tangent node, bucket by the
**global key of that tangent value**, not by a fragment-local id.

---

## III. Typical Pipelines

```text
JVP:
  build -> resolve -> differentiate -> materialize_merge -> compile -> eval

VJP (grad):
  build -> resolve -> differentiate -> transpose -> materialize_merge -> compile -> eval

2nd directional derivative (FoF):
  build -> resolve -> differentiate -> resolve -> differentiate
       -> materialize_merge -> compile -> eval

HVP (FoR = jvp(vjp(f))):
  build -> resolve -> differentiate -> transpose -> resolve -> differentiate
       -> materialize_merge -> compile -> eval

n-th derivative:
  build -> (resolve -> differentiate) x n -> [transpose] -> materialize_merge -> compile -> eval
```

`resolve`, `materialize_merge`, `compile`, `eval` are provided by
`computegraph-rs`. `tidu-rs` only adds `differentiate` and `transpose`.

---

## IV. Linear Nodes

The linear graph uses the same primitive set as the primal graph. There is no
dedicated `Scale` primitive.

```text
Mul(a, dx)   mode=Linear { active_mask=[fixed, active] }
Add(dx, dy)  mode=Linear { active_mask=[active, active] }
Exp(x)       mode=Primal
```

The linearization of `Exp(x)` emits:

```text
Mul(External(exp(x)), dx) mode=Linear { active_mask=[fixed, active] }
```

Active mask is part of identity (`OpMode::Linear` vs `Primal`). Nodes that
evaluate the same way but transpose differently must not alias.

---

## V. Higher-Order AD

The rule for higher order is:

```text
resolve before the next differentiate
materialize_merge before compile
```

`resolve` creates a logical view over fragments. No physical merge is
required between successive `differentiate` calls.

---

## VI. Design Boundaries

```text
tidu-rs owns:
  - differentiate (JVP transform)
  - transpose (reverse linear flow)
  - LinearFragment data structure

tidu-rs does NOT own:
  - graph infrastructure → computegraph-rs
  - PrimitiveOp trait → chainrules-rs
  - concrete primitives → downstream (tenferro-rs)
```
