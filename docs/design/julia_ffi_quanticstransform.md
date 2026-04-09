# Julia Frontend QuanticsTransform Support

**Last updated**: 2026-04-10

## Purpose

This document covers the `Tensor4all.QuanticsTransform` submodule.

`Tensor4all.QuanticsTransform` owns transform operators in the Julia frontend: affine coordinate transforms, shifts, flips, phase rotations, cumulative operators, Fourier-like transforms. It is a **provider** of `LinearOperator` instances, not the owner of the `LinearOperator` type itself.

This document does not cover quantics grid semantics or quantics interpolation convenience. Those belong to [julia_ffi_quanticsgrids.md](./julia_ffi_quanticsgrids.md) and [julia_ffi_quanticstci.md](./julia_ffi_quanticstci.md).

## Design Direction

Port the transform/operator-facing parts of `Quantics.jl` into `Tensor4all.QuanticsTransform`, with operator construction backed by `tensor4all-rs`.

The goal is:

- Rust owns operator construction and heavy TT-side kernels
- Julia owns surface API, index mapping logic, and module structure

## Layer Position

```text
Tensor4all.TreeTN
    |
    +---- LinearOperator        (generic MPO + index mapping, lives here)
    |     set_input_space!
    |     set_output_space!
    |     apply
    |
    +---- TensorTrain (MPS/MPO)

Tensor4all.QuanticsTransform
    |
    +---- shift_operator(r, ...) -> LinearOperator
    |     flip_operator(r, ...) -> LinearOperator
    |     fourier_operator(r, ...) -> LinearOperator
    |     affine_operator(...) -> LinearOperator
    |     ...                                          (construction only)
    |
    +---- depends on TreeTN.LinearOperator
```

`QuanticsTransform` depends on:

- `Tensor4all.TreeTN` for `TensorTrain`, `LinearOperator`, `apply`
- `Tensor4all` core for `Index`, `Tensor`

## LinearOperator (pure Julia, chain-only, owned by TreeTN)

### Motivation

The Rust `LinearOperator<T, V>` wraps an MPO with input/output index mappings.
Rather than keeping this as an opaque C API type, the Julia side implements its
own `LinearOperator` for flexibility:

- Site index subset selection (operate on part of a multi-variable MPS)
- Tag-based index selection (select by `tag="x"`)
- No prime-level convention dependency

This type belongs in `TreeTN` because it wraps `TensorTrain` (MPO) and is
useful beyond quantics transforms (any MPO-based operator can use it).

### Structure

```julia
struct LinearOperator
    mpo::TensorTrain                             # internal MPO
    input_indices::Vector{Index}                 # per-site internal input index
    output_indices::Vector{Index}                # per-site internal output index
    true_input::Vector{Union{Index, Nothing}}    # bound true input index (nothing = unbound)
    true_output::Vector{Union{Index, Nothing}}   # bound true output index (nothing = unbound)
end
```

- `mpo`: The MPO as a `TensorTrain` (= `TreeTensorNetwork{Int}`), chain topology only.
- `input_indices` / `output_indices`: The MPO's internal site indices identifying
  which index is "input" and which is "output" at each site. Fixed at construction time.
- `true_input` / `true_output`: The "true" site indices of the state the operator
  will act on. Initially `nothing` (unbound). Set via `set_input_space!` / `set_output_space!`.

Input/output distinction is **not** based on prime levels. The mapping is stored
explicitly. This is more general and avoids coupling to ITensors conventions.

### set_input_space!

Three overloads for binding true input indices:

```julia
# 1. From a TensorTrain -- bind all sites
function set_input_space!(op::LinearOperator, state::TensorTrain)

# 2. From a TensorTrain with tag selection -- bind matching subset
function set_input_space!(op::LinearOperator, state::TensorTrain; tag::AbstractString)

# 3. From explicit index list
function set_input_space!(op::LinearOperator, indices::Vector{Index})
```

`set_output_space!` is analogous. `set_iospaces!` sets both (output defaults to
same as input when omitted).

Validation: the number of indices must match the operator's site count, and
dimensions must match at each site.

### apply

```julia
function apply(op::LinearOperator, state::TensorTrain;
               method=:naive, rtol=0.0, maxdim=0)
    # 1. Validate: all true_input entries are bound
    # 2. Replace state's site indices with MPO's internal input indices
    # 3. Contract MPO with aligned state (Rust C API)
    # 4. Replace result's output indices with true output indices
    return result
end
```

When the operator covers only a **subset** of the state's sites, the Rust
contraction engine will be extended to handle partial-site application directly.
Until then, the operator must cover all sites of the state.

### Tag-based site index lookup

Ported from `Quantics.jl`'s `findallsites_by_tag`:

```julia
function findallsiteinds_by_tag(state::TensorTrain; tag::AbstractString)
    # Find site indices matching tag=1, tag=2, ... pattern
end
```

This utility also belongs in `TreeTN` since it operates on `TensorTrain` site indices.

## QuanticsTransform: Construction Functions

`QuanticsTransform` is responsible only for constructing `LinearOperator` instances
from quantics-specific parameters. It uses Rust C API to build the MPO, then
wraps it in a Julia `LinearOperator`.

### Construction flow

```julia
function shift_operator(r::Integer, offset::Integer; bc=Periodic)
    # 1. Build opaque t4a_linop via Rust C API
    raw_op = _build_raw_linop(r, offset, bc)

    # 2. Extract MPO as TensorTrain
    mpo = _extract_mpo(raw_op)

    # 3. Extract per-site input/output index IDs, match to MPO site indices
    input_indices, output_indices = _extract_index_mappings(raw_op, mpo)

    # 4. Release raw_op, return Julia LinearOperator
    _release_raw_linop(raw_op)

    return LinearOperator(
        mpo, input_indices, output_indices,
        fill(nothing, length(input_indices)),
        fill(nothing, length(output_indices)),
    )
end
```

### Available constructors

| Julia function | C-API | Description |
|----------------|-------|-------------|
| `shift_operator(r, offset; bc)` | `t4a_qtransform_shift` | coordinate shift |
| `shift_operator_multivar(r, offset, nvars, target; bc)` | `t4a_qtransform_shift_multivar` | shift on one variable |
| `flip_operator(r; bc)` | `t4a_qtransform_flip` | flip / reversal |
| `flip_operator_multivar(r, nvars, target; bc)` | `t4a_qtransform_flip_multivar` | flip on one variable |
| `phase_rotation_operator(r, theta)` | `t4a_qtransform_phase_rotation` | phase multiplication |
| `phase_rotation_operator_multivar(r, theta, nvars, target)` | `t4a_qtransform_phase_rotation_multivar` | phase on one variable |
| `cumsum_operator(r)` | `t4a_qtransform_cumsum` | cumulative sum |
| `fourier_operator(r; forward, maxbonddim, tolerance)` | `t4a_qtransform_fourier` | quantics Fourier transform |
| `affine_operator(r, a_num, a_den, b_num, b_den; bc)` | `t4a_qtransform_affine` | affine `y = A x + b` |
| `affine_pullback_operator(r, params; bc)` | `t4a_qtransform_affine_pullback` | affine pullback |
| `binaryop_operator(r, a1, b1, a2, b2; bc1, bc2)` | `t4a_qtransform_binaryop` | binary operation |

## C-API Extensions

New C API functions to extract MPO and index mappings from `t4a_linop`:

| C-API | Description |
|-------|-------------|
| `t4a_linop_get_mpo` | Clone the internal MPO as a `t4a_treetn` |
| `t4a_linop_num_sites` | Get the number of operator sites |
| `t4a_linop_get_input_index_ids` | Get internal input index IDs per site |
| `t4a_linop_get_output_index_ids` | Get internal output index IDs per site |

## Pipeline Integration

### Data flow

```
QuanticsGrids (grid) --> QuanticsTCI (interpolation) --> SimpleTensorTrain
                                                              |
                                                         MPS(tt)
                                                              |
                                                         TensorTrain (MPS)
                                                              |
                    QuanticsTransform (operator) -------> apply(op, mps) --> TensorTrain
```

### Module responsibilities

| Module | Input | Output | Responsibility |
|--------|-------|--------|----------------|
| QuanticsGrids | domain params | Grid object | Coordinate system definition |
| QuanticsTCI | Grid + function | SimpleTensorTrain | Function interpolation via TCI |
| SimpleTT | -- | SimpleTensorTrain | Lightweight TT without Index objects |
| TreeTN | Tensors with Index | TensorTrain, LinearOperator | Index-based TT operations + generic operator |
| QuanticsTransform | operator params | LinearOperator | Quantics operator construction |

### Type visibility

Both `SimpleTensorTrain` and `TensorTrain` (MPS) are user-visible types.
Users choose the conversion point explicitly via `MPS(tt::SimpleTensorTrain)`.

## Typical User Code

### Basic: interpolate and Fourier transform

```julia
using Tensor4all
using Tensor4all.QuanticsGrids
using Tensor4all.QuanticsTCI
using Tensor4all.QuanticsTransform
using Tensor4all.TreeTN: apply, set_input_space!

grid = DiscretizedGrid(1, 8, [0.0], [1.0])
qtci = quanticscrossinterpolate(grid, x -> exp(-x))
tt = to_tensor_train(qtci)
mps = MPS(tt)

op = fourier_operator(8)
set_input_space!(op, mps)
result = apply(op, mps)
```

### Subset: operate on one variable of a multi-variable MPS

```julia
op = shift_operator(r, 1)
set_input_space!(op, mps; tag="x")   # bind to x=1, x=2, ... only
result = apply(op, mps)
```

### Explicit index list

```julia
op = flip_operator(r)
target_indices = siteinds(mps)[[1, 3, 5]]
set_input_space!(op, target_indices)
result = apply(op, mps)
```

## Migration Path

1. **Phase 1**: Add C API extensions (`t4a_linop_get_mpo`, `t4a_linop_get_index_ids`)
   to `tensor4all-rs`
2. **Phase 2**: Implement `LinearOperator` in `TreeTN` module (pure Julia),
   add `findallsiteinds_by_tag`
3. **Phase 3**: Migrate `QuanticsTransform` construction functions to return
   Julia `LinearOperator`
4. **Phase 4**: Deprecate and remove opaque `t4a_linop` wrapper from Julia side

The existing C API `t4a_linop_apply` remains available during migration.

## Relationship to `Quantics.jl`

`Quantics.jl` contains the legacy high-level transform-oriented Julia surface.
Key patterns ported:

- `findallsites_by_tag` / `findallsiteinds_by_tag`: tag-based site index selection
- `matchsiteinds`: embedding operator into full site chain (replaced by
  `set_input_space!` + Rust-side partial contraction in the new design)
- `_find_target_sites`: unified tag/explicit-index lookup (-> `set_input_space!` overloads)

## Scope Exclusions

- **BubbleTeaCI integration**: out of scope
- **Grid-aware operators** (`shift_operator_on_grid` etc.): future work
- **Rust-side partial-site contraction**: separate Rust implementation task
- **General tree topology**: chain-only for now
