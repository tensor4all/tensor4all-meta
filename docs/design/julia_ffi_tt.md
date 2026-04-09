# Julia Frontend TT Support

## Purpose

This document covers backend tensor-train support in the Julia frontend. It is the layer between the low-level core primitives and the higher-level `BubbleTeaCI` function semantics.

## In Scope

- `ITensorTrain` as a Julia-owned chain of `Tensor` (indexed, MPS/MPO-compatible)
- `SimpleTT.TensorTrain{V,N}` as a pure Julia raw-array tensor train (TCI-compatible)
- TT-level contraction and compression
- TT arithmetic and structural operations
- TT-level transform operators

This document does not define `TTFunction` / `GriddedFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## Two TensorTrain Types

The Julia frontend provides two TensorTrain types in different modules, mirroring the existing ecosystem split:

| Type | Module | Data | Replaces | Use case |
|------|--------|------|----------|----------|
| `ITensorTrain` | `Tensor4all.ITT` | `Vector{Tensor}` + `llim/rlim` | `SimpleTensorTrains.SimpleTensorTrain` (`Vector{ITensor}`) | MPS/MPO workflows, index-aware contraction, ITensors interop |
| `TensorTrain{V,N}` | `Tensor4all.SimpleTT` | `Vector{Array{V,N}}` | `TensorCrossInterpolation.TensorTrain{V,N}` | TCI results, raw numerical computation without index overhead |

### `Tensor4all.ITT.ITensorTrain` (indexed, MPS/MPO-compatible)

```julia
mutable struct ITensorTrain
    data::Vector{Tensor}   # Rust-backed, indexed tensors
    llim::Int              # left orthogonality limit
    rlim::Int              # right orthogonality limit
end
```

- Julia owns the chain structure; individual site tensors are Rust-backed `Tensor` objects.
- `llim`/`rlim` track orthogonality center, matching `ITensorMPS.MPS`/`MPO` conventions.
- The same structure represents both MPS-like and MPO-like objects (distinguished at runtime by index structure).
- Replaces `SimpleTensorTrains.SimpleTensorTrain`, which wraps `ITensor` and delegates arithmetic/truncation to `ITensorMPS`.
- In the new design, arithmetic and truncation are delegated to Rust via C-API instead of ITensorMPS.

**Conversion (via package extension):**
- `ITensorTrain` ↔ `ITensorMPS.MPS` / `ITensorMPS.MPO`
- `ITensorTrain` ↔ `SimpleTensorTrains.SimpleTensorTrain`

### `Tensor4all.SimpleTT.TensorTrain{V,N}` (raw arrays, TCI-compatible)

```julia
module SimpleTT

mutable struct TensorTrain{V,N}
    sitetensors::Vector{Array{V,N}}
end

end # module SimpleTT
```

- Pure Julia arrays without index metadata.
- Direct replacement for `TensorCrossInterpolation.TensorTrain{V,N}`.
- Primary output type from TCI/QTCI construction.
- `N=3` for standard MPS-like chains (bond, site, bond); `N=4` for MPO-like.
- Operations delegate to Rust via `t4a_simplett_*` C-API (no pure Julia reimplementation).

**Conversion:**
- `SimpleTT.TensorTrain{V,N}` → `ITensorTrain`: requires supplying `Vector{Index}` for site indices.
- `ITensorTrain` → `SimpleTT.TensorTrain{V,N}`: extract raw data arrays from Rust-backed tensors.

## TT-Level Operations on `SimpleTT.TensorTrain{V,N}` (in `Tensor4all.SimpleTT`)

### C-API Surface

| C-API function | Parameters | Description |
|----------------|------------|-------------|
| `t4a_simplett_f64_add(a, b) → out` | — | Direct sum. Bond dim = dim_a + dim_b. No truncation. |
| `t4a_simplett_c64_add(a, b) → out` | — | Same for complex. |

### Julia API (following `TensorCrossInterpolation.jl` conventions)

```julia
# --- Accessors ---
sitetensors(tt)        → Vector{Array{V,N}}
sitetensor(tt, i)      → Array{V,N}
linkdims(tt)           → Vector{Int}      # bond dimensions
linkdim(tt, i)         → Int
sitedims(tt)           → Vector{Vector{Int}}
sitedim(tt, i)         → Vector{Int}
rank(tt)               → Int              # max bond dimension
length(tt)             → Int              # number of sites
tt[i]                  → Array{V,N}       # getindex

# --- Evaluation ---
evaluate(tt, indexset)  → V
tt(indexset)            → V               # callable syntax

# --- Arithmetic (direct sum, delegates to t4a_simplett_*_add) ---
Base.:+(a, b)          → TensorTrain{V,N}  # direct sum, no truncation
Base.:-(a, b)          → TensorTrain{V,N}
add(a, b; factorlhs=1, factorrhs=1,
    tolerance=0.0, maxbonddim=typemax(Int)) → TensorTrain{V,N}
    # direct sum + optional compression
subtract(a, b; tolerance=0.0, maxbonddim=typemax(Int)) → TensorTrain{V,N}

# --- Compression (pure Julia, LU/CI/SVD) ---
compress!(tt, method=:LU; tolerance=1e-12, maxbonddim=typemax(Int))
# method ∈ {:LU, :CI, :SVD}
# LU = RRLU (default), CI = MatrixLUCI, SVD = standard SVD

# --- Norms ---
LinearAlgebra.norm(tt)  → Float64
LinearAlgebra.norm2(tt) → Float64   # squared Frobenius norm

# --- Reduction ---
sum(tt; dims=nothing)   → TensorTrain or scalar
```

Notes:
- `+`/`-` delegate to Rust (`t4a_simplett_*_add`) for the direct sum step.
- `add`/`subtract` include optional compression after direct sum.
- `compress!` stays pure Julia (Rust simplett does not expose LU/CI-based compression).

### Conversion (constructor pattern)

```julia
# SimpleTT.TensorTrain → ITT.ITensorTrain (needs site indices)
ITT.ITensorTrain(tt::TensorTrain{V,N}, sites::Vector{Vector{Index}})

# ITT.ITensorTrain → SimpleTT.TensorTrain (extract raw arrays)
SimpleTT.TensorTrain(itt::ITT.ITensorTrain)
```

Conversions live in the **target type's module**, following Julia conventions.

## TT-Level Operations on `ITensorTrain` (in `Tensor4all.ITT`)

### C-API Surface

| C-API function | Parameters | Description |
|----------------|------------|-------------|
| `t4a_treetn_add(a, b) → out` | — | Direct sum. Bond dim = dim_a + dim_b. No truncation. |
| `t4a_treetn_truncate(ttn, rtol, cutoff, maxdim)` | `rtol`, `cutoff` (→ `rtol=sqrt(cutoff)`), `maxdim` | In-place truncation. |
| `t4a_treetn_contract(a, b, method, rtol, cutoff, maxdim) → out` | method (Zipup/Fit/Naive), `rtol`, `cutoff`, `maxdim` | Contraction with integrated truncation. |

Notes:
- `rtol` is the native tolerance in `tensor4all-rs`.
- `cutoff` is an ITensors.jl-convention parameter, converted to `rtol = sqrt(cutoff)` in the C-API.
- `maxdim = 0` means no limit. `rtol = 0.0` / `cutoff = 0.0` means not set.

### Julia API

```julia
# Addition: direct sum only (no truncation)
Base.:+(a::ITensorTrain, b::ITensorTrain) → ITensorTrain
# Delegates to t4a_treetn_add. Caller truncates separately if needed.

# Truncation: rtol-based (cutoff as ITensors-compatible convenience)
truncate(tt::ITensorTrain; rtol=..., maxdim=...)  → ITensorTrain
truncate(tt::ITensorTrain; cutoff=..., maxdim=...) → ITensorTrain
# cutoff is converted to rtol = sqrt(cutoff).

# Contraction: method + truncation
contract(a::ITensorTrain, b::ITensorTrain;
         method=:zipup, rtol=..., maxdim=...) → ITensorTrain
# method ∈ {:zipup, :fit, :naive}
```

### Additional Rust-Exposed Primitives

- TT scale
- TT dot / inner product
- TT reverse
- full tensor export
- construction from site tensors

## Relationship to BubbleTeaCI

- This is infrastructure for `BubbleTeaCI`, not a replacement for its high-level function semantics.
- `BubbleTeaCI` should build on this layer rather than duplicate TT backend functionality.

## Open Questions

- Should the public `ITensorTrain` API remain thin and backend-shaped, or expose more Julia-side convenience methods?
