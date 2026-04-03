# JAX/StableHLO Primitives Needed For tenferro

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`
**Related:** `jax-primitives.md`, `stablehlo-primitives.md`, `primitive-catalog.md`

---

## I. Goal

This note answers a narrower question than `primitive-catalog.md`:

> If tenferro v2 wants to reuse JAX's `linearize` design with minimal changes,
> which JAX primitives must tenferro expose, and which StableHLO ops are
> needed below that layer?

The goal here is **not** to design a backend-minimal semiring substrate.
`Semiring Core` is intentionally ignored in this document.

The target feature set for this first pass is:

- `einsum`
- `lu`
- `solve`
- `qr`
- `svd`

Both **primal execution** and **forward AD / `linearize` support** are in
scope.

---

## II. Sources

This note is based on:

- current `tenferro-rs` code in `~/tensor4all/tenferro-rs`
- current JAX code in `~/tensor4all/jax`
- the StableHLO spec summarized in `stablehlo-primitives.md`

The most relevant source files were:

- `tenferro-rs/docs/design/supported-ops.md`
- `tenferro-rs/docs/design/tensor-prims.md`
- `tenferro-rs/tenferro-prims/src/families/*.rs`
- `tenferro-rs/tenferro-linalg/src/lib.rs`
- `tenferro-rs/tenferro-linalg/src/prims_bridge.rs`
- `jax/jax/_src/interpreters/ad.py`
- `jax/jax/_src/numpy/einsum.py`
- `jax/jax/_src/lax/control_flow/solves.py`
- `jax/jax/_src/lax/linalg.py`
- `jax/jax/_src/numpy/linalg.py`

---

## III. Phase-1 JAX Primitive List To Implement

This is the current recommended **first-wave primitive list** if the goal is:

- keep tenferro close enough to JAX that `jax.linearize` can be ported with
  minimal semantic drift
- support `einsum`, `lu`, `solve`, `qr`, and `svd`

### Primitive layer to expose in tenferro

These are the JAX-like primitive IDs that should exist at tenferro's
primitive layer.

AD helpers:

- `add_jaxvals_p` / `add_any`
- `zeros_like_p`
- `stop_gradient_p`

Tensor primitives:

- arithmetic:
  `add_p`, `sub_p`, `mul_p`, `div_p`, `neg_p`
- complex/real helpers:
  `conj_p`, `real_p`
- contraction/reduction/shape:
  `dot_general_p`, `reduce_p`, `broadcast_in_dim_p`, `reshape_p`,
  `transpose_p`, `squeeze_p`, `pad_p`
- predicate/select/index helpers:
  `select_n_p`, `eq_p`, `iota_p`, `sort_p`, `convert_element_type_p`

Structured linalg primitives:

- `lu_p`
- `triangular_solve_p`
- `qr_p`
- `svd_p`

Control-flow / implicit-diff primitive:

- `linear_solve_p` (`custom_linear_solve`)

Important consequence:

- `einsum` stays a composite over tensor primitives
- `solve` stays a composite over `linear_solve_p`, `lu_p`,
  `triangular_solve_p`, and tensor primitives

### Infrastructure that must exist around those primitives

These are not primitives, but JAX `linearize` expects this surrounding
machinery:

- `primitive_jvps`
- `primitive_transposes`
- partial-evaluation support for zero tangents / known primals

### Helper kernels that may exist below the primitive layer

These are useful lowering helpers, but they do **not** need to be part of the
first public tenferro primitive catalog:

- `geqrf_p`
- `geqp3_p`
- `householder_product_p`
- `lu_pivots_to_permutation_p`

### Optional only if tenferro wants pure JAX fallback algorithms

These are not required in the first wave if `lu_p`, `qr_p`, and `svd_p` lower
directly to backend kernels / custom calls:

- `argmax_p`
- `gather_p`
- `scatter_p`
- `while_p`

---

## IV. Current tenferro-rs Inventory

### Tensor/view layer

Current `tenferro-tensor` already has the view/metadata operations that JAX
usually treats as tensor primitives:

- `reshape`
- `permute`
- `broadcast`
- `diagonal`
- `select`
- `narrow`
- `contiguous`
- `tril`
- `triu`
- `eye`

These exist today, but they are **tensor/view APIs**, not first-class JAX-like
primitive IDs with per-primitive JVP registrations.

### Execution families in `tenferro-prims`

Current `tenferro-prims` is organized by backend execution families:

- `TensorSemiringCore`
  - `BatchedGemm`
  - `ReduceAdd`
  - `Trace`
  - `AntiTrace`
  - `AntiDiag`
  - `MakeContiguous`
- `TensorSemiringFastPath`
  - `Contract`
  - `ElementwiseBinary::{Add, Mul}`
- `TensorScalarPrims`
  - unary: `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`
  - binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`,
    `Greater`, `GreaterEqual`, `ClampMin`, `ClampMax`
  - ternary: `Where`
  - reductions: `Sum`, `Prod`, `Mean`, `Max`, `Min`
- `TensorAnalyticPrims`
  - unary: `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Ceil`, `Log`, `Log1p`, `Sin`,
    `Cos`, `Tan`, `Tanh`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`,
    `Acosh`, `Atanh`
  - binary: `Pow`, `Atan2`, `Hypot`, `Xlogy`
  - reductions: `Var`, `Std`

This is already a substantial operation inventory, but it is **not aligned to
the JAX primitive layer**. It groups by backend family rather than by JAX-style
primitive name such as `dot_general_p`, `qr_p`, or `svd_p`.

### Structured linalg in `tenferro-linalg`

Current `tenferro-linalg` already has backend kernels and public APIs for the
target structured operations:

- backend-facing kernels:
  - `lu`
  - `solve`
  - `solve_triangular`
  - `qr`
  - `svd`
- public primal APIs:
  - `lu`
  - `solve`
  - `solve_triangular`
  - `qr`
  - `svd`
- stateless AD rules already exist for:
  - `lu`
  - `solve`
  - `solve_triangular`
  - `qr`
  - `svd`

So tenferro can already execute and differentiate the target math. The mismatch
is not mathematical coverage. The mismatch is the **primitive boundary** used
to express that coverage.

---

## V. JAX Primitives Needed To Port `linearize`

### Common AD substrate

Before looking at individual tensor ops, tenferro needs the common AD-facing
primitive layer that JAX `linearize` assumes exists:

- `add_jaxvals_p` / `add_any`
  - tangent accumulation primitive used by `ad.py`
- `zeros_like_p`
  - create zero tangents during partial evaluation / instantiation
- `stop_gradient_p`
  - identity on primals, zero on tangents
- per-primitive JVP registry
  - JAX `linearize` calls into `ad.primitive_jvps[...]`

Strictly speaking, `linearize` itself is forward-mode and does not require the
full transpose registry. But if tenferro wants the same primitive layer shape
as JAX, the natural pair is:

- `primitive_jvps`
- `primitive_transposes`

### `einsum`

JAX does **not** treat `einsum` itself as a primitive. The implementation in
`jax/_src/numpy/einsum.py` is a composite built from tensor primitives.

Primitives directly visible in the primal path:

- `reduce_p`
- `dot_general_p`
- `transpose_p`
- `squeeze_p`
- `select_n_p`

Additional helper primitives pulled in by internal helpers:

- `iota_p`
- `broadcast_in_dim_p`
- `eq_p`
- `convert_element_type_p`

Meaning:

- if tenferro wants to run a JAX-style `einsum` implementation unchanged, then
  `dot_general_p` alone is not enough
- it also needs explicit reduction, masking/select, and shape primitives

### `lu`

For JAX, LU is a first-class primitive:

- `lu_p`

If tenferro wants JAX-like forward AD at the primitive layer, the JVP rule for
`lu_p` additionally uses:

- `triangular_solve_p`
- `dot_general_p`
- `transpose_p`
- `conj_p`
- `pad_p`
- `add_p`

And through triangular/diagonal mask helpers:

- `iota_p`
- `compare`-style predicates
- `select_n_p`

If tenferro wants to reuse JAX's pure fallback algorithm for primal LU instead
of lowering `lu_p` directly to a backend kernel, the required JAX primitive
surface becomes larger again:

- `argmax_p`
- scatter/update family
- `while_p` / loop lowering
- `div_p`
- `sub_p`

So the clean JAX-aligned choice is:

- keep `lu_p` as a first-class primitive
- lower it directly to a backend kernel or custom call
- do **not** force LU down into only `dot_general + transpose + reshape`

### `solve`

JAX does **not** have a dedicated `solve_p`.

Instead, `solve` is built from:

- `linear_solve_p` (`custom_linear_solve`)
- `stop_gradient_p`
- `lu_p`
- `broadcast_in_dim_p`
- `dot_general_p`
- `triangular_solve_p`

The composite `lu_solve` path also needs:

- `reshape_p`
- `sort_p`
- `iota_p`
- permutation/gather-style indexing helpers

This point matters a lot:

- if tenferro wants to preserve JAX's `solve` differentiation strategy, it
  needs `linear_solve_p`
- replacing `solve` with a plain traced decomposition of `lu + triangular_solve`
  changes the AD boundary and loses JAX's implicit-diff structure

### `qr`

For JAX, QR is also a first-class primitive:

- `qr_p`

Its JVP rule uses:

- `triangular_solve_p`
- `dot_general_p`
- `transpose_p`
- `conj_p`
- `add_p`
- `sub_p`
- `mul_p`
- `real_p`

And via triangular / identity helpers:

- `broadcast_in_dim_p`
- `iota_p`
- `select_n_p`

The current JAX primal lowering of `qr_p` internally uses additional linalg
primitives:

- `geqrf_p`
- `geqp3_p`
- `householder_product_p`

So for QR, the clean JAX-aligned surface is again:

- expose `qr_p` as a first-class primitive
- allow backend lowering to use lower-level helper kernels if desired
- do not make those helper kernels the user-visible primitive boundary

### `svd`

For JAX, SVD is a first-class primitive:

- `svd_p`

Its JVP rule uses a relatively rich tensor primitive set:

- `dot_general_p`
- `transpose_p`
- `conj_p`
- `real_p`
- `add_p`
- `sub_p`
- `mul_p`
- `div_p`

And through diagonal / eye / zero-safe inverse helpers:

- `broadcast_in_dim_p`
- `iota_p`
- `select_n_p`

As with LU and QR, the important design signal is:

- `svd_p` should remain a first-class primitive
- its JVP formula depends on normal tensor primitives around it
- JAX does not expect users to differentiate through a hand-written SVD
  algorithm expressed only in low-level shape ops

---

## VI. StableHLO Ops Needed Below That Layer

The StableHLO working set splits into two groups:

- ops that exist directly in StableHLO
- JAX primitives that have **no** direct StableHLO op and therefore need
  `custom_call` or another backend-only lowering path

### Direct StableHLO ops that matter for this target set

These are the main StableHLO ops that recur across the target JAX primitive
set:

- `add`
- `subtract`
- `multiply`
- `divide`
- `compare`
- `select`
- `constant`
- `convert`
- `broadcast_in_dim`
- `reshape`
- `transpose`
- `pad`
- `reduce`
- `dot_general`
- `iota`
- `sort`
- `gather`
- `scatter`
- `triangular_solve`
- `while`
- `custom_call`

This is the concrete StableHLO vocabulary that shows up once we go below the
JAX primitive layer for `einsum`, `solve`, `lu`, `qr`, and `svd`.

### `einsum` -> StableHLO

Because JAX `einsum` is already a tensor-primitive composite, the StableHLO
story is straightforward:

- `reduce_p` -> `reduce`
- `dot_general_p` -> `dot_general`
- `transpose_p` -> `transpose`
- `squeeze_p` -> `reshape`
- `select_n_p` -> `select`
- helpers -> `iota`, `compare`, `broadcast_in_dim`, `convert`, `constant`

### `lu_p` -> StableHLO

There is **no** native StableHLO `lu` op.

Current JAX practice is therefore:

- lower `lu_p` to backend-specific `custom_call` where possible
- or use a pure fallback algorithm that needs
  `while`, `iota`, `compare`, `select`, scatter/update ops, `divide`,
  `subtract`, `dot_general`, `triangular_solve`, and `pad`

So StableHLO does not give tenferro a clean 1:1 target for LU. The realistic
lowering boundary is:

- `lu_p` at the JAX-like primitive layer
- `custom_call` or backend kernel below it

### `linear_solve_p` / `solve` -> StableHLO

There is no StableHLO `solve` op and no StableHLO `custom_linear_solve` op.

Current JAX `solve` therefore lowers by inlining the solve body. In the
current implementation that means the StableHLO-side ingredients are:

- `custom_call` through `lu_p`
- `broadcast_in_dim`
- `dot_general`
- `triangular_solve`
- `reshape`
- `sort`
- `iota`
- gather/permutation helpers

This is the clearest case where the JAX primitive layer is **above** the
StableHLO layer and should stay there.

### `qr_p` -> StableHLO

There is no native StableHLO `qr` op.

Current JAX practice is:

- keep `qr_p` as the JAX primitive
- lower it via helper linalg kernels such as `geqrf_p`, `geqp3_p`,
  `householder_product_p`
- those helper kernels themselves are custom-call-backed on CPU/GPU

So the important StableHLO fact is simple:

- `qr_p` ultimately needs `custom_call`
- tensor helper ops such as `pad`, `broadcast_in_dim`, `transpose`,
  `select`, and `dot_general` are still needed around the edges

### `svd_p` -> StableHLO

There is no native StableHLO `svd` op.

Current JAX practice is:

- keep `svd_p` as the JAX primitive
- lower it to backend-specific `custom_call`
- use ordinary tensor ops only in the JVP rule and result post-processing

So for tenferro the StableHLO implication is:

- `svd_p` should stay above StableHLO
- StableHLO needs `custom_call` plus the normal tensor algebra ops used by the
  JVP formula

---

## VII. Immediate Gap Summary

### Already present in current tenferro

- structured tensor/view operations:
  `reshape`, `permute`, `broadcast`, `diagonal`, `select`, `narrow`,
  `contiguous`, `tril`, `triu`
- backend linalg kernels:
  `lu`, `solve`, `solve_triangular`, `qr`, `svd`
- stateless linalg AD rules for the target structured ops
- scalar and analytic pointwise families

### Missing if the goal is "port JAX `linearize` mostly as-is"

- AD helper primitives:
  `add_any`, `zeros_like`, `stop_gradient`
- a JAX-like primitive registry keyed by primitive ID with per-primitive JVP
  rules
- JAX-style tensor primitives such as:
  `dot_general`, `reduce`, `broadcast_in_dim`, `transpose`, `squeeze`,
  `select`, `compare`, `iota`, `convert_element_type`
- control-flow/implicit-diff primitive:
  `custom_linear_solve`
- structured linalg primitives as first-class primitive IDs:
  `lu_p`, `qr_p`, `svd_p`, `triangular_solve_p`

### Design consequence

If the target is really "copy JAX `linearize` into tenferro with minimal
semantic change", then tenferro v2 should not use the current execution-family
surface as its primary primitive catalog.

It needs a JAX-like primitive layer above the current backend families:

- tensor primitives such as `dot_general`, `reduce`, `broadcast_in_dim`,
  `transpose`
- AD helpers such as `add_any`, `zeros_like`, `stop_gradient`
- structured linalg primitives such as `lu`, `qr`, `svd`, `triangular_solve`
- `custom_linear_solve` for `solve`

The current semiring/scalar/analytic/linalg backend families can still exist,
but they belong **below** that layer.
