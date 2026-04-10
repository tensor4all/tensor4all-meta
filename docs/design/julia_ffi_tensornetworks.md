# Julia Frontend TensorNetworks Support

## Purpose

This document covers the `Tensor4all.TensorNetworks` submodule.

`Tensor4all.TensorNetworks` is the indexed tensor-network layer in the Julia frontend. It sits above the low-level `Core` primitives and provides both chain (TensorTrain) and tree (TreeTensorNetwork) representations for indexed tensor-network workflows.

This document does not define high-level `TTFunction` / `GriddedFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## In Scope

- `AbstractTensorNetwork` abstract type and common API via multiple dispatch
- `TensorTrain` as a chain-topology tensor network
- `TreeTensorNetwork` as a tree-topology tensor network (future)
- Contraction, truncation, and arithmetic at the indexed layer
- `ITensorMPS.MPS` / `MPO` compatibility (for `TensorTrain`)
- HDF5-level compatibility for indexed TT workflows

## Type Hierarchy

```julia
abstract type AbstractTensorNetwork end

mutable struct TensorTrain <: AbstractTensorNetwork
    data::Vector{Tensor}   # Rust-backed, indexed tensors
    llim::Int              # left orthogonality limit
    rlim::Int              # right orthogonality limit
end

mutable struct TreeTensorNetwork <: AbstractTensorNetwork
    # tree-topology tensor network (design TBD, implement after TensorTrain)
end
```

- `TensorTrain` is the chain specialization.
- `TreeTensorNetwork` will support general tree topologies. Its internal structure is to be designed later, but the abstract type and common API are defined now to avoid future breakage.
- Both types are separate concrete types (not aliases), but share a common interface through `AbstractTensorNetwork`.

### MPS/MPO distinction

`TensorTrain` represents both MPS-like and MPO-like objects, distinguished by index structure at runtime:

- `is_mps_like(tt)` — 1 site index per vertex
- `is_mpo_like(tt)` — 2 site indices per vertex

There is no type-level MPS vs. MPO distinction.

## Common API (AbstractTensorNetwork)

These functions are defined with `AbstractTensorNetwork` dispatch and work for both `TensorTrain` and `TreeTensorNetwork`:

```julia
# Addition: direct sum
Base.:+(a::T, b::T) where {T<:AbstractTensorNetwork} -> T

# Truncation
truncate(tn::AbstractTensorNetwork; rtol=..., maxdim=...)   -> AbstractTensorNetwork
truncate(tn::AbstractTensorNetwork; cutoff=..., maxdim=...) -> AbstractTensorNetwork

# Contraction
contract(a::AbstractTensorNetwork, b::AbstractTensorNetwork;
         method=:zipup, rtol=..., maxdim=...) -> AbstractTensorNetwork

# Inner product
dot(a::AbstractTensorNetwork, b::AbstractTensorNetwork) -> Number

# Norms
norm(tn::AbstractTensorNetwork) -> Number
```

Method choices for contraction remain backend-shaped: `:zipup`, `:fit`, `:naive`.

## TensorTrain-Specific API

```julia
# Chain-specific queries
is_mps_like(tt::TensorTrain) -> Bool
is_mpo_like(tt::TensorTrain) -> Bool
```

## Re-export at Top Level

The primary types and common functions are re-exported from `Tensor4all`:

```julia
# Users access via:
using Tensor4all

tt = TensorTrain(...)
ttn = TreeTensorNetwork(...)
result = contract(tt1, tt2; method=:zipup)
```

Users should not need `using Tensor4all.TensorNetworks` for standard workflows.

## Conversion

- `TensorTrain` ↔ `ITensorMPS.MPS`
- `TensorTrain` ↔ `ITensorMPS.MPO`
- `TensorTrain` ↔ `SimpleTT.TensorTrain{V,N}` with explicit site-index information

## C-API Surface

| C-API function | Parameters | Description |
|----------------|------------|-------------|
| `t4a_treetn_add(a, b) -> out` | — | Direct sum. Bond dim = dim_a + dim_b. No truncation. |
| `t4a_treetn_truncate(ttn, rtol, cutoff, maxdim)` | `rtol`, `cutoff`, `maxdim` | In-place truncation. |
| `t4a_treetn_contract(a, b, method, rtol, cutoff, maxdim) -> out` | method + truncation parameters | Contraction with integrated truncation. |
| `t4a_treetn_contract_partial(op, state, target_sites, n, method, rtol, cutoff, maxdim) -> out` | target site positions + truncation parameters | Contract an operator on a subset of sites. |

Notes:

- `rtol` is the native tolerance in `tensor4all-rs`.
- `cutoff` is an ITensors.jl-convention parameter, converted to `rtol = sqrt(cutoff)` in the C-API.
- `maxdim = 0` means no limit. `rtol = 0.0` / `cutoff = 0.0` means not set.
- The Rust C-API already uses `treetn` naming. Both `TensorTrain` and `TreeTensorNetwork` map to the same Rust backend; the Julia layer dispatches appropriately.

## Additional Rust-Exposed Primitives

- TT scale
- TT dot / inner product
- TT reverse
- full tensor export
- construction from site tensors

## ITensorMPS / HDF5 Compatibility

- `TensorNetworks` is the place where MPS/MPO compatibility belongs (via `TensorTrain`).
- HDF5 save/load should preserve compatibility with the existing ITensors/ITensorMPS ecosystem at the MPS/MPO representation level.
- Preserve round-trips with existing Julia MPS/MPO workflows, even when the in-memory representation becomes Rust-backed.
- This compatibility requirement applies to `TensorTrain` only. `TreeTensorNetwork` will define its own serialization format.

## Implementation Priority

1. **Now**: `TensorTrain` with full API
2. **Future**: `TreeTensorNetwork` — internal structure and tree-specific operations TBD

The abstract type `AbstractTensorNetwork` and common API signatures are defined now so that downstream code (QuanticsTransform, BubbleTeaCI) can program against the abstract interface where appropriate.

## Relationship to Other Docs

- [julia_ffi_core.md](./julia_ffi_core.md) defines the `Index` / `Tensor` primitives used here.
- [julia_ffi_simplett.md](./julia_ffi_simplett.md) defines the raw-array TT representation used for TCI-oriented workflows.
- [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) builds transform application on top of `TensorNetworks`.

## Open Questions

- What is the internal structure of `TreeTensorNetwork`? (deferred until implementation)
- Should the public API remain thin and backend-shaped, or expose more Julia-side convenience methods?
- Which `ITensorMPS` naming and behavioral conventions should be mirrored exactly, and which should be intentionally simplified?
