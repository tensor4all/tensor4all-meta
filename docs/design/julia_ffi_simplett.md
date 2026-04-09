# Julia Frontend SimpleTT Support

## Purpose

This document covers the `Tensor4all.SimpleTT` submodule.

`Tensor4all.SimpleTT` is the raw-array tensor-train layer in the Julia frontend. It is the low-overhead TT representation used most naturally as the output of TCI/QTCI construction.

This document does not define indexed MPS/MPO compatibility or high-level `TTFunction` semantics. Indexed TT workflows belong to [julia_ffi_itt.md](./julia_ffi_itt.md), while high-level workflows remain in [bubbleteaCI.md](./bubbleteaCI.md).

## In Scope

- `SimpleTT.TensorTrain{V,N}`
- raw-array TT arithmetic
- raw-array TT evaluation and reduction
- compression entry points at the `SimpleTT` layer
- conversion to and from `TensorCrossInterpolation.TensorTrain{V,N}`
- conversion to and from `ITT.ITensorTrain`

## `Tensor4all.SimpleTT.TensorTrain{V,N}`

```julia
module SimpleTT

mutable struct TensorTrain{V,N}
    sitetensors::Vector{Array{V,N}}
end

end # module SimpleTT
```

- Pure Julia arrays without index metadata.
- Direct replacement for `TensorCrossInterpolation.TensorTrain{V,N}`.
- Primary output type from `Tensor4all.TensorCI`.
- `N=3` for MPS-like chains and `N=4` for MPO-like chains.

## Conversion

- `SimpleTT.TensorTrain{V,N}` ↔ `TensorCrossInterpolation.TensorTrain{V,N}`
- `SimpleTT.TensorTrain{V,N}` → `ITT.ITensorTrain`: requires explicit site indices
- `ITT.ITensorTrain` → `SimpleTT.TensorTrain{V,N}`: extract raw data arrays from Rust-backed tensors

Conversions should live in the target type's module, following Julia conventions.

## C-API Surface

| C-API function | Parameters | Description |
|----------------|------------|-------------|
| `t4a_simplett_f64_add(a, b) -> out` | — | Direct sum. Bond dim = dim_a + dim_b. No truncation. |
| `t4a_simplett_c64_add(a, b) -> out` | — | Same for complex. |

## Julia API Direction

```julia
# Accessors
sitetensors(tt)       -> Vector{Array{V,N}}
sitetensor(tt, i)     -> Array{V,N}
linkdims(tt)          -> Vector{Int}
linkdim(tt, i)        -> Int
sitedims(tt)          -> Vector{Vector{Int}}
sitedim(tt, i)        -> Vector{Int}
rank(tt)              -> Int
length(tt)            -> Int
tt[i]                 -> Array{V,N}

# Evaluation
evaluate(tt, indexset) -> V
tt(indexset)           -> V

# Arithmetic
Base.:+(a, b) -> TensorTrain{V,N}
Base.:-(a, b) -> TensorTrain{V,N}
add(a, b; factorlhs=1, factorrhs=1,
    tolerance=0.0, maxbonddim=typemax(Int)) -> TensorTrain{V,N}
subtract(a, b; tolerance=0.0, maxbonddim=typemax(Int)) -> TensorTrain{V,N}

# Compression
compress!(tt, method=:LU; tolerance=1e-12, maxbonddim=typemax(Int))

# Norms
LinearAlgebra.norm(tt)  -> Float64
LinearAlgebra.norm2(tt) -> Float64

# Reduction
sum(tt; dims=nothing)   -> TensorTrain or scalar
```

Notes:

- `+`/`-` delegate to Rust for the direct-sum step.
- `add`/`subtract` include optional compression after direct sum.
- `compress!` stays Julia-side for LU/CI/SVD-style workflows unless the backend later grows equivalent APIs.

## Relationship to `TensorCrossInterpolation.jl`

- `SimpleTT.TensorTrain{V,N}` is the intended frontend-native replacement for `TensorCrossInterpolation.TensorTrain{V,N}`.
- Provide explicit conversion in both directions between `SimpleTT.TensorTrain{V,N}` and `TensorCrossInterpolation.TensorTrain{V,N}`.
- `Tensor4all.TensorCI` should target this type directly.
- This keeps the generic TCI layer aligned with one TT representation inside the frontend while still allowing interop with existing `TensorCrossInterpolation.jl` code.

## Relationship to Other Docs

- [julia_ffi_tci.md](./julia_ffi_tci.md) treats `SimpleTT.TensorTrain{V,N}` as the output boundary of core TCI.
- [julia_ffi_itt.md](./julia_ffi_itt.md) defines the indexed TT layer that `SimpleTT` can convert into.

## Open Questions

- Which `TensorCrossInterpolation.jl` conventions should be preserved exactly at the API level?
- Should more `SimpleTT` operations move into Rust over time, or should some remain intentionally Julia-native?
