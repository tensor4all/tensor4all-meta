# JAX Primitives

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`
**Related:** `primitive-catalog.md`, `stablehlo-primitives.md`, `backend-architecture.md`

---

## I. Purpose

This document records the JAX primitives that are most relevant to tensor
operations and AD design in v2.

The source of truth here is the local checkout at:

- `~/tensor4all/jax`

This is intentionally **not** a catalog of every frontend helper in JAX.
Instead, it is a catalog of the actual primitive layer that JAX uses beneath
the public `jax.numpy` / `jax.lax` surface.

---

## II. What Counts As A Primitive In JAX

At the source level, the key markers are:

- `core.Primitive(...)`
- `standard_primitive(...)`
- `standard_unop(...)`
- `standard_naryop(...)`

And the AD-relevant registrations are attached separately via:

- `ad.deflinear2(...)`
- `ad.defbilinear(...)`
- `ad.primitive_jvps[...] = ...`
- `ad.primitive_transposes[...] = ...`

So in JAX, a primitive is not just a name. It is a bundle of:

- abstract evaluation / shape rules
- batching rules
- MLIR / StableHLO lowering
- JVP / transpose / linearization behavior

---

## III. Source Files Surveyed

The main source files used for this note are:

- `jax/_src/ad_util.py`
- `jax/_src/lax/lax.py`
- `jax/_src/lax/convolution.py`
- `jax/_src/lax/fft.py`
- `jax/_src/lax/slicing.py`
- `jax/_src/lax/windowed_reductions.py`
- `jax/_src/lax/control_flow/conditionals.py`
- `jax/_src/lax/control_flow/loops.py`
- `jax/_src/lax/control_flow/solves.py`
- `jax/_src/lax/linalg.py`

This note does **not** try to catalog hardware-specific auxiliary primitives in
areas such as `pallas`, `cudnn`, or TPU/Mosaic lowering internals.

---

## IV. AD Helper Primitives

From `jax/_src/ad_util.py`:

- `add_any` (`add_jaxvals_p`)
- `stop_gradient` (`stop_gradient_p`)
- `zeros_like` (`zeros_like_p`)

These are important because they show that JAX keeps some AD-facing helpers
outside the tensor-kernel vocabulary itself.

In other words, JAX does **not** force every AD concern to be expressed only in
terms of pure tensor contraction / reshape primitives.

---

## V. Core Tensor Primitives In `jax/_src/lax/lax.py`

### Arithmetic and comparison

- `neg_p`
- `sign_p`
- `abs_p`
- `add_p`
- `sub_p`
- `mul_p`
- `div_p`
- `rem_p`
- `max_p`
- `min_p`
- `clamp_p`
- `eq_p`
- `ne_p`
- `ge_p`
- `gt_p`
- `le_p`
- `lt_p`
- `and_p`
- `or_p`
- `xor_p`
- `not_p`
- `shift_left_p`
- `shift_right_arithmetic_p`
- `shift_right_logical_p`
- `population_count_p`
- `clz_p`

### Analytic and complex-valued math

- `exp_p`
- `exp2_p`
- `log_p`
- `expm1_p`
- `log1p_p`
- `tanh_p`
- `logistic_p`
- `sin_p`
- `cos_p`
- `tan_p`
- `asin_p`
- `acos_p`
- `atan_p`
- `atan2_p`
- `sinh_p`
- `cosh_p`
- `asinh_p`
- `acosh_p`
- `atanh_p`
- `sqrt_p`
- `rsqrt_p`
- `cbrt_p`
- `square_p`
- `pow_p`
- `integer_pow_p`
- `real_p`
- `imag_p`
- `complex_p`
- `conj_p`
- `is_finite_p`

### Tensor structure, contraction, and conversion

- `dot_general_p`
- `ragged_dot_general_p`
- `broadcast_in_dim_p`
- `reshape_p`
- `transpose_p`
- `rev_p`
- `concatenate_p`
- `pad_p`
- `squeeze_p`
- `tile_p`
- `iota_p`
- `select_n_p`
- `convert_element_type_p`
- `bitcast_convert_type_p`
- `to_edtype_p`
- `from_edtype_p`

### Reductions and ordering

- `reduce_p`
- `reduce_sum_p`
- `reduce_prod_p`
- `reduce_max_p`
- `reduce_min_p`
- `reduce_or_p`
- `reduce_and_p`
- `reduce_xor_p`
- `reduce_precision_p`
- `sort_p`
- `top_k_p`

### Miscellaneous tensor/runtime helpers in the same file

- `composite_p`
- `create_token_p`
- `after_all_p`
- `rng_uniform_p`
- `rng_bit_generator_p`
- `copy_p`
- `tie_p`
- `optimization_barrier_p`

---

## VI. Slicing, Gather, Scatter, And Windowed Reduction Primitives

From `jax/_src/lax/slicing.py`:

- `slice_p`
- `dynamic_slice_p`
- `dynamic_update_slice_p`
- `gather_p`
- `scatter_p`
- `scatter_add_p`
- `scatter_sub_p`
- `scatter_mul_p`
- `scatter_min_p`
- `scatter_max_p`

From `jax/_src/lax/windowed_reductions.py`:

- `reduce_window_p`
- `reduce_window_sum_p`
- `reduce_window_max_p`
- `reduce_window_min_p`
- `select_and_scatter_p`
- `select_and_scatter_add_p`
- `select_and_gather_add_p`

These matter because they show that JAX does not collapse everything into only
`slice` / `gather` / `scatter` / generic `reduce`; it keeps some specialized
window and AD-friendly primitives as first-class surface at the primitive
layer.

---

## VII. Control Flow Primitives

From `jax/_src/lax/control_flow/conditionals.py`,
`jax/_src/lax/control_flow/loops.py`, and
`jax/_src/lax/control_flow/solves.py`:

- `cond_p`
- `scan_p`
- `while_p`
- `custom_linear_solve` (`linear_solve_p`)
- `cumsum_p`
- `cumlogsumexp_p`
- `cumprod_p`
- `cummax_p`
- `cummin_p`

Also present as internal control-flow support:

- `eval_jaxpr_p`
- `platform_index_p`

The important point for v2 is that JAX keeps control flow as dedicated
primitives instead of forcing everything through a tensor-only core.

---

## VIII. Additional Tensor Primitives Outside `lax.py`

From `jax/_src/lax/convolution.py`:

- `conv_general_dilated_p`

From `jax/_src/lax/fft.py`:

- `fft_p`

These are relevant because they show another JAX design choice: some structured
ops remain explicit primitives rather than being lowered away into only
`dot_general` plus reshapes.

---

## IX. Structured Linalg Primitives

From `jax/_src/lax/linalg.py`:

- `cholesky_p`
- `cholesky_update_p`
- `eig_p`
- `eigh_p`
- `hessenberg_p`
- `householder_product_p`
- `ormqr_p`
- `lu_p`
- `lu_pivots_to_permutation_p`
- `geqrf_p`
- `geqp3_p`
- `qr_p`
- `schur_p`
- `svd_p`
- `symmetric_product_p`
- `triangular_solve_p`
- `tridiagonal_p`
- `tridiagonal_solve_p`

This is useful for v2 because it shows a clear split:

- some tensor ops live in a relatively small contraction/shape core
- structured matrix algorithms stay explicit and have their own AD rules

---

## X. What JAX Suggests For v2

### The JAX primitive layer is not identical to StableHLO

Examples:

- JAX uses `dot_general_p`, which is close to StableHLO
- JAX uses `broadcast_in_dim_p`, `reshape_p`, and `transpose_p`, also close to
  StableHLO
- JAX uses `reduce_sum_p` as a named primitive, even though StableHLO also has
  a more general `reduce`
- JAX uses `select_n_p`, not just raw StableHLO-style `select`
- JAX has AD helper primitives like `add_any`, `stop_gradient`, and
  `zeros_like` which have no direct role as StableHLO tensor ops

### The most v2-relevant JAX primitives are these

If the goal is to study the JAX-style "core tensor vocabulary" that supports
AD and tensor programming, the most relevant primitives are:

- `add_p`
- `mul_p`
- `neg_p`
- `dot_general_p`
- `broadcast_in_dim_p`
- `reshape_p`
- `transpose_p`
- `reduce_sum_p`
- `select_n_p`
- `convert_element_type_p`
- `slice_p`
- `dynamic_slice_p`
- `gather_p`
- `scatter_p`
- `concatenate_p`
- `pad_p`
- `cond_p`
- `scan_p`
- `while_p`
- `stop_gradient_p`

That is a better starting point for v2 discussion than the entire JAX surface.

### Design takeaway

The JAX source suggests a useful separation:

- a relatively small tensor/contraction/shape primitive layer
- explicit control-flow primitives
- explicit structured linalg primitives
- a small set of AD helper primitives

This is a strong argument against trying to force every concern into one flat
"tensor primitive" list.
