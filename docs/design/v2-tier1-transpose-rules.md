# v2 Tier 1 Transpose Rules

**Date:** 2026-04-03
**Status:** Draft
**Related:** `v2-ad-architecture.md`, `v2-backend-architecture.md`

---

## I. Scope

This document defines the detailed transpose rules for the **Tier 1** primitive
set used by `tidu2::transpose`.

The intent is:

- `tidu2` directly knows how to transpose only a **closed Tier 1 set**
- `tenferro2` may define richer primitives downstream
- any downstream primitive that participates in reverse-mode AD must have a
  `linearize` rule whose output uses only this transpose-closed Tier 1 set

So the contract is:

```text
downstream primitive
  -> linearize
  -> Tier 1 linear fragment only
  -> transpose in tidu2
```

This is the intended answer for primitives such as `SVD`, `Solve`, `QR`, and
other Tier 2 or downstream-only ops:

- they are defined downstream
- their forward evaluation is downstream
- their `linearize` rule is downstream
- but after `linearize`, `tidu2::transpose` should only see Tier 1 primitives

---

## II. Invariants

### Linear fragments are linear, not merely affine

Only nodes that are linear in the active inputs may appear in a linear
fragment.

Tier 1 also assumes that broadcasting is explicit:

- `Add` and `Mul` do not hide shape expansion
- shape expansion must appear as `BroadcastInDim`

That means:

- `Add(active, active)` is allowed
- `Mul(fixed, active)` is allowed
- `Mul(active, fixed)` is allowed
- `DotGeneral(fixed, active)` is allowed
- `DotGeneral(active, fixed)` is allowed
- unary maps such as `Neg`, `Conj`, `Transpose`, `Reshape`,
  `BroadcastInDim`, `ReduceAdd` are allowed on active values

These are not allowed as linear nodes:

- `Mul(active, active)`
- `DotGeneral(active, active)`
- `Add(fixed, active)` as a linear node

If a downstream `linearize` rule needs those forms, it has not finished
linearizing.

### Active mask conventions

For this document, the active mask is read as:

- `active`: this input carries tangent or cotangent flow
- `fixed`: this input is a coefficient or residual

Typical masks:

```text
Add(dx, dy)              active_mask=[active, active]
Mul(a, dx)               active_mask=[fixed, active]
Mul(dx, a)               active_mask=[active, fixed]
DotGeneral(W, dx)        active_mask=[fixed, active]
DotGeneral(dx, W)        active_mask=[active, fixed]
ReduceAdd(dx)            active_mask=[active]
BroadcastInDim(dx)       active_mask=[active]
Transpose(dx)            active_mask=[active]
Reshape(dx)              active_mask=[active]
Neg(dx)                  active_mask=[active]
Conj(dx)                 active_mask=[active]
```

### Complex convention

For complex tensors, transpose means the adjoint of the linear map.

So whenever a rule reuses a fixed operand in reverse mode:

- use it unchanged for real dtypes
- wrap it in `Conj(...)` for complex dtypes

This matters especially for `Mul` and `DotGeneral`.

---

## III. Rule Table

### Summary

| Primitive | Allowed linear form | Reverse rule |
|-----------|---------------------|--------------|
| `Add` | `[active, active]` | `Dup(ct)` |
| `Mul` | `[fixed, active]` or `[active, fixed]` | reuse `Mul` with fixed side preserved |
| `Neg` | `[active]` | `Neg(ct)` |
| `Conj` | `[active]` | `Conj(ct)` for complex, identity for real |
| `Dup` | active input, 2 active outputs | `Add(ct_0, ct_1)` |
| `DotGeneral` | one side fixed, one side active | new `DotGeneral` plus axis-restoring `Transpose` |
| `Transpose` | `[active]` | `Transpose(ct, inverse_perm)` |
| `Reshape` | `[active]` | inverse `Reshape` |
| `BroadcastInDim` | `[active]` | `ReduceAdd` over broadcasted axes, then `Reshape` to input shape |
| `ReduceAdd` | `[active]` | `BroadcastInDim` back to unreduced shape |

The rest of this document spells these out in detail.

---

## IV. Elementwise Rules

### `Add`

Forward linear form:

```text
y = Add(x0, x1)          active_mask=[active, active]
```

Reverse:

```text
(ct_x0, ct_x1) = Dup(ct_y)
```

Meaning:

- the output cotangent is copied to both inputs
- if both inputs refer to the same original tangent value, accumulation later
  combines them with `Add`

### `Mul`

Forward linear forms:

```text
y = Mul(a, x)            active_mask=[fixed, active]
y = Mul(x, a)            active_mask=[active, fixed]
```

Reverse:

```text
Mul(a, x)  -> ct_x = Mul(a, ct_y)
Mul(x, a)  -> ct_x = Mul(ct_y, a)
```

For complex tensors:

```text
Mul(a, x)  -> ct_x = Mul(Conj(a), ct_y)
Mul(x, a)  -> ct_x = Mul(ct_y, Conj(a))
```

`Mul(active, active)` must not appear in a linear fragment.

### `Neg`

Forward:

```text
y = Neg(x)
```

Reverse:

```text
ct_x = Neg(ct_y)
```

### `Conj`

Forward:

```text
y = Conj(x)
```

Reverse:

```text
complex input: ct_x = Conj(ct_y)
real input:    ct_x = ct_y
```

`Conj` is self-adjoint on complex values.

---

## V. Fan-out and Reduction Rules

### `Dup`

Forward:

```text
(y0, y1) = Dup(x)
```

Reverse:

```text
ct_x = Add(ct_y0, ct_y1)
```

This is the explicit reverse of fan-out.

### `ReduceAdd`

Forward:

```text
y = ReduceAdd(x, axes=axes)
```

Reverse:

```text
ct_x = BroadcastInDim(
    ct_y,
    shape=input_shape(x),
    dims=nonreduced_axes(x, axes),
)
```

Here `dims` is the list of output axes that survived the reduction.

Example:

```text
ReduceAdd : [b, m, n] -> [b, n]   axes=[1]
transpose -> BroadcastInDim : [b, n] -> [b, m, n]   dims=[0, 2]
```

### `BroadcastInDim`

Forward:

```text
y = BroadcastInDim(x, shape=out_shape, dims=dims)
```

Reverse:

```text
ct_tmp = ReduceAdd(ct_y, axes=broadcasted_axes)
ct_x   = Reshape(ct_tmp, input_shape(x))
```

Where `broadcasted_axes` means:

- output axes not listed in `dims`
- plus axes corresponding to input dimensions of size `1` that were expanded

This is the reverse of duplication by broadcasting.

Example:

```text
BroadcastInDim : [1, n] -> [m, n]   dims=[0, 1]
transpose      : ReduceAdd over axis 0, then Reshape to [1, n]
```

If no dimension was expanded and only rank was increased, the `Reshape` may be
trivial.

---

## VI. Structural Tensor Rules

### `Transpose`

Forward:

```text
y = Transpose(x, permutation=perm)
```

Reverse:

```text
ct_x = Transpose(ct_y, permutation=inverse(perm))
```

This is self-inverse up to the inverse permutation.

### `Reshape`

Forward:

```text
y = Reshape(x, shape=out_shape)
```

Reverse:

```text
ct_x = Reshape(ct_y, shape=input_shape(x))
```

This rule assumes `Reshape` only changes grouping of axes, not their order.

If a transformation needs both reordering and reshaping, represent it as:

```text
Transpose -> Reshape
```

not as one overloaded `Reshape`.

That keeps the transpose rule simple and exact.

---

## VII. `DotGeneral`

`DotGeneral` is the most structured Tier 1 rule.

### Forward form

We write:

```text
y = DotGeneral(lhs, rhs; ((lhs_contract, rhs_contract),
                          (lhs_batch,    rhs_batch)))
```

Allowed linear forms are:

```text
DotGeneral(fixed, active)
DotGeneral(active, fixed)
```

The case where both inputs are active is bilinear and must not appear in a
linear fragment.

### Output axis order

The logical output order is:

```text
[batch axes, lhs free axes, rhs free axes]
```

Where:

- `lhs free axes` are the lhs axes not in `lhs_contract` or `lhs_batch`
- `rhs free axes` are the rhs axes not in `rhs_contract` or `rhs_batch`

### Reverse rule: lhs active, rhs fixed

Forward:

```text
y = DotGeneral(x, w; ((x_contract, w_contract), (x_batch, w_batch)))
```

Reverse:

1. Contract the cotangent against the **rhs free axes** of `w`
2. Keep the batch axes paired with `w_batch`
3. The resulting tensor has axes ordered as:

```text
[x_batch, x_free, x_contract(sorted like w_contract)]
```

4. Apply a final `Transpose` so the result matches the original axis order of
   `x`

In shorthand:

```text
tmp  = DotGeneral(
         ct_y,
         maybe_conj(w),
         contract=(ct_rhs_free, w_free),
         batch=(ct_batch, w_batch),
       )
ct_x = Transpose(tmp, restore_x_axis_order)
```

For complex tensors, use `maybe_conj(w) = Conj(w)`.

### Reverse rule: rhs active, lhs fixed

This is symmetric.

Forward:

```text
y = DotGeneral(w, x; ((w_contract, x_contract), (w_batch, x_batch)))
```

Reverse:

```text
tmp  = DotGeneral(
         ct_y,
         maybe_conj(w),
         contract=(ct_lhs_free, w_free),
         batch=(ct_batch, w_batch),
       )
ct_x = Transpose(tmp, restore_x_axis_order)
```

Again, use `Conj(w)` in the complex case.

### Why `Transpose` appears in the reverse rule

`DotGeneral` chooses a canonical output axis order.

The reverse result must match the original axis order of the active input, so
the reverse rule is:

```text
DotGeneral + Transpose
```

not just `DotGeneral` alone.

### Matrix multiplication example

Forward:

```text
Y = DotGeneral(A, X)   // A:[m,k], X:[k,n], Y:[m,n]
```

If `X` is active and `A` is fixed:

```text
ct_X = DotGeneral(Conj(Transpose(A)), ct_Y)
```

If `A` is active and `X` is fixed:

```text
ct_A = DotGeneral(ct_Y, Conj(Transpose(X)))
```

This is the familiar matrix rule, expressed through the general `DotGeneral`
machinery.

---

## VIII. What `tidu2::transpose` Must Know

`tidu2::transpose` must dispatch on:

- primitive kind
- arity
- linear active mask
- primitive metadata for structured ops such as `DotGeneral`,
  `Transpose`, `Reshape`, `BroadcastInDim`, and `ReduceAdd`

That metadata is part of the primitive value itself, so `tidu2` does not need
to know any downstream `SVD` or `Solve` details.

It only needs the Tier 1 rule table above.

---

## IX. Downstream Contract

Downstream primitive sets, including Tier 2 tensor ops, must satisfy:

```text
primitive.eval           lives downstream
primitive.linearize      lives downstream
primitive.transpose      not required downstream
```

provided that:

```text
primitive.linearize(...)
```

emits only a Tier 1 linear fragment whose transpose behavior is defined here.

This is the intended design boundary between:

- `tidu2`: graph transforms over a transpose-closed Tier 1 set
- `tenferro2`: concrete tensor primitives and their linearization rules
