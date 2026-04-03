# StableHLO Primitives

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`
**Related:** `primitive-catalog.md`, `backend-architecture.md`, `jax-primitives.md`

---

## I. Purpose

This document records the current StableHLO operation inventory as an external
reference for the v2 design.

StableHLO itself uses the word "op", not "primitive". This document uses
"primitive" only to match the naming used elsewhere in `docs/design-v2/`.

There are two goals:

1. record the full current StableHLO op inventory in one place
2. make it easy to compare tenferro v2 candidates against the official
   StableHLO naming and op set

---

## II. Primary Sources

The primary source is the official StableHLO specification:

- `https://openxla.org/stablehlo/spec`

This document was compiled from the current op sections in that specification
on 2026-04-03. The spec also has an explicit `Deprecated Operations` section,
which is summarized below.

---

## III. Current StableHLO Op Inventory

The current spec has **106** op sections after excluding non-op section
headings.

### Elementwise arithmetic, comparison, and math

- `abs`
- `add`
- `and`
- `atan2`
- `bitcast_convert`
- `cbrt`
- `ceil`
- `clamp`
- `compare`
- `complex`
- `convert`
- `cosine`
- `count_leading_zeros`
- `divide`
- `exponential`
- `exponential_minus_one`
- `floor`
- `imag`
- `is_finite`
- `log`
- `log_plus_one`
- `logistic`
- `maximum`
- `minimum`
- `multiply`
- `negate`
- `not`
- `or`
- `popcnt`
- `power`
- `real`
- `remainder`
- `round_nearest_afz`
- `round_nearest_even`
- `rsqrt`
- `shift_left`
- `shift_right_arithmetic`
- `shift_right_logical`
- `sign`
- `sine`
- `sqrt`
- `subtract`
- `tan`
- `tanh`
- `xor`

### Shape, layout, and tensor construction

- `broadcast_in_dim`
- `concatenate`
- `constant`
- `dynamic_broadcast_in_dim`
- `dynamic_iota`
- `dynamic_pad`
- `dynamic_reshape`
- `get_dimension_size`
- `iota`
- `optimization_barrier`
- `pad`
- `reshape`
- `reverse`
- `transpose`

### Slicing, gather/scatter, and indexed updates

- `dynamic_gather`
- `dynamic_slice`
- `dynamic_update_slice`
- `gather`
- `scatter`
- `select`
- `slice`

### Reductions, windows, and ordering

- `reduce`
- `reduce_precision`
- `reduce_scatter`
- `reduce_window`
- `select_and_scatter`
- `sort`

### Contraction, linalg, and NN-oriented ops

- `batch_norm_grad`
- `batch_norm_inference`
- `batch_norm_training`
- `cholesky`
- `convolution`
- `custom_call`
- `dot_general`
- `dynamic_conv`
- `fft`
- `triangular_solve`

### Control flow and higher-order ops

- `case`
- `if`
- `map`
- `while`

### Collectives, communication, tokens, and side-effect ordering

- `after_all`
- `all_gather`
- `all_reduce`
- `all_to_all`
- `async_done`
- `async_start`
- `collective_broadcast`
- `collective_permute`
- `infeed`
- `outfeed`
- `partition_id`
- `recv`
- `replica_id`
- `send`

### Tuple, RNG, and compatibility-oriented ops

- `get_tuple_element`
- `rng`
- `rng_bit_generator`
- `tuple`

### Quantization

- `uniform_dequantize`
- `uniform_quantize`

### Alphabetical Appendix

```text
abs
add
after_all
all_gather
all_reduce
all_to_all
and
async_done
async_start
atan2
batch_norm_grad
batch_norm_inference
batch_norm_training
bitcast_convert
broadcast_in_dim
case
cbrt
ceil
cholesky
clamp
collective_broadcast
collective_permute
compare
complex
concatenate
constant
convert
convolution
cosine
count_leading_zeros
custom_call
divide
dot_general
dynamic_broadcast_in_dim
dynamic_conv
dynamic_gather
dynamic_iota
dynamic_pad
dynamic_reshape
dynamic_slice
dynamic_update_slice
exponential
exponential_minus_one
fft
floor
gather
get_dimension_size
get_tuple_element
if
imag
infeed
iota
is_finite
log
log_plus_one
logistic
map
maximum
minimum
multiply
negate
not
optimization_barrier
or
outfeed
pad
partition_id
popcnt
power
real
recv
reduce
reduce_precision
reduce_scatter
reduce_window
remainder
replica_id
reshape
reverse
rng
rng_bit_generator
round_nearest_afz
round_nearest_even
rsqrt
scatter
select
select_and_scatter
send
shift_left
shift_right_arithmetic
shift_right_logical
sign
sine
slice
sort
sqrt
subtract
tan
tanh
transpose
triangular_solve
tuple
uniform_dequantize
uniform_quantize
while
xor
```

---

## IV. Deprecated And Legacy Names In The Spec

The spec explicitly calls out these deprecated operations:

- `broadcast`
- `create_token`
- `cross-replica-sum`
- `dot`
- `einsum`
- `torch_index_select`
- `unary_einsum`

The spec also says that removal is still being explored for some compatibility
or legacy ops, including:

- `complex`
- `get_tuple_element`
- `map`
- `rng`
- `torch_index_select`
- `tuple`

This matters for v2 because a primitive catalog that wants to stay close to
StableHLO should avoid building itself around already-deprecated names such as
`dot`, `einsum`, or old `broadcast`.

---

## V. Immediate Relevance To v2

For the current v2 discussion, the most important StableHLO ops are:

- `dot_general`
- `broadcast_in_dim`
- `reshape`
- `transpose`
- `reduce`
- `add`
- `multiply`
- `negate`
- `compare`
- `select`
- `gather`
- `scatter`
- `slice`
- `dynamic_slice`
- `pad`
- `concatenate`
- `while`
- `if`
- `case`
- `custom_call`

These are the main comparison points for deciding:

- which tenferro v2 ops should align 1:1 with StableHLO
- which names should remain tenferro-side convenience or AD-facing wrappers
- which backend-only execution contracts should stay below the v2 primitive
  layer
