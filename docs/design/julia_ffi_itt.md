# Julia Frontend ITT Support

## Purpose

This document covers the `Tensor4all.ITT` submodule.

`Tensor4all.ITT` is the indexed tensor-train layer in the Julia frontend. It sits above the low-level `Core` primitives and provides the MPS/MPO-facing TT representation used for indexed tensor-network workflows.

This document does not define high-level `TTFunction` / `GriddedFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## In Scope

- `ITT.ITensorTrain` as a Julia-owned chain of `Tensor`
- TT-level contraction and truncation at the indexed layer
- indexed TT arithmetic and structural operations
- `ITensorMPS.MPS` / `MPO` compatibility
- HDF5-level compatibility for indexed TT workflows

## `Tensor4all.ITT.ITensorTrain`

```julia
mutable struct ITensorTrain
    data::Vector{Tensor}   # Rust-backed, indexed tensors
    llim::Int              # left orthogonality limit
    rlim::Int              # right orthogonality limit
end
```

- Julia owns the chain structure; individual site tensors are Rust-backed `Tensor` objects.
- `llim`/`rlim` track orthogonality center, matching `ITensorMPS.MPS`/`MPO` conventions.
- The same structure represents both MPS-like and MPO-like objects, distinguished by index structure.
- This replaces `SimpleTensorTrains.SimpleTensorTrain`, which stores `Vector{ITensor}` and relies on `ITensorMPS`.

## Conversion

- `ITT.ITensorTrain` ↔ `ITensorMPS.MPS`
- `ITT.ITensorTrain` ↔ `ITensorMPS.MPO`
- `ITT.ITensorTrain` ↔ `SimpleTensorTrains.SimpleTensorTrain`
- `ITT.ITensorTrain` ↔ `SimpleTT.TensorTrain{V,N}` with explicit site-index information

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

## Julia API Direction

```julia
# Addition: direct sum only
Base.:+(a::ITensorTrain, b::ITensorTrain) -> ITensorTrain

# Truncation
truncate(tt::ITensorTrain; rtol=..., maxdim=...)   -> ITensorTrain
truncate(tt::ITensorTrain; cutoff=..., maxdim=...) -> ITensorTrain

# Contraction
contract(a::ITensorTrain, b::ITensorTrain;
         method=:zipup, rtol=..., maxdim=...) -> ITensorTrain
```

Method choices for contraction remain backend-shaped:

- `:zipup`
- `:fit`
- `:naive`

## Additional Rust-Exposed Primitives

- TT scale
- TT dot / inner product
- TT reverse
- full tensor export
- construction from site tensors

## ITensorMPS / HDF5 Compatibility

- `Tensor4all.ITT` is the place where MPS/MPO compatibility belongs.
- HDF5 save/load should preserve compatibility with the existing ITensors/ITensorMPS ecosystem at the MPS/MPO representation level.
- Preserve round-trips with existing Julia MPS/MPO workflows, even when the in-memory representation becomes Rust-backed.
- This compatibility requirement is part of the indexed TT layer, not a separate frontend extension boundary.

## Relationship to Other Docs

- [julia_ffi_core.md](./julia_ffi_core.md) defines the `Index` / `Tensor` primitives used here.
- [julia_ffi_simplett.md](./julia_ffi_simplett.md) defines the raw-array TT representation used for TCI-oriented workflows.
- [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) builds transform application on top of `ITT`.

## Open Questions

- Should the public `ITensorTrain` API remain thin and backend-shaped, or expose more Julia-side convenience methods?
- Which `ITensorMPS` naming and behavioral conventions should be mirrored exactly, and which should be intentionally simplified?
