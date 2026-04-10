# Julia Frontend TCI Support

## Purpose

This document covers the core TCI (Tensor Cross Interpolation) module in the Julia frontend. `Tensor4all.TensorCI` is responsible for generic tensor-train construction from function evaluations.

This is the `Tensor4all.TensorCI` submodule.

This document does not cover the quantics-specific convenience layer. That belongs to [julia_ffi_quanticstci.md](./julia_ffi_quanticstci.md).

## Relationship to `TensorCrossInterpolation.jl`

`TensorCrossInterpolation.jl` is the existing pure Julia TCI implementation. It provides:

- `crossinterpolate1` — TCI1 algorithm
- `crossinterpolate2` — TCI2 algorithm (primary)
- `optfirstpivot` — pivot initialization
- `TensorTrain{V,N}` — output type
- `contract` — TT contraction via TCI
- Matrix CI internals: `MatrixCI`, `MatrixACA`, `MatrixLU`, `MatrixLUCI`
- Sweep strategies, batch evaluation, cached function evaluation, integration

### Source structure of `TensorCrossInterpolation.jl`

| File | Contents |
|------|----------|
| `abstracttensortrain.jl` | `AbstractTensorTrain{V}`, accessors, evaluate, arithmetic, norm, sum |
| `tensortrain.jl` | `TensorTrain{V,N}`, `compress!` (LU/CI/SVD) |
| `tensorci1.jl` | TCI1 algorithm |
| `tensorci2.jl` | TCI2 algorithm |
| `matrixci.jl` | Matrix CI base |
| `matrixlu.jl` | Rank-revealing LU |
| `matrixluci.jl` | Matrix LUCI |
| `matrixaca.jl` | Adaptive Cross Approximation |
| `conversion.jl` | TCI ↔ TT conversions |
| `integration.jl` | Integration via TCI |
| `contraction.jl` | TT contraction |
| `globalpivotfinder.jl` | Global pivot search |
| `globalsearch.jl` | Global search utilities |
| `cachedtensortrain.jl` | Lazy/cached TT |
| `cachedfunction.jl` | Cached function evaluation |
| `batcheval.jl` | Batch evaluation |

## Design Decision: Re-export vs. Port

### Option A: Re-export `TensorCrossInterpolation.jl`

```julia
module TensorCI
    using TensorCrossInterpolation
    # re-export public API
end
```

- **Pro**: No code duplication. Immediate availability.
- **Con**: TCI outputs `TensorCrossInterpolation.TensorTrain{V,N}`, not `SimpleTT.TensorTrain{V,N}`. Conversion always required between the two types.

### Option B: Port into `Tensor4all.TensorCI`

```julia
module TensorCI
    using ..SimpleTT: TensorTrain
    # TCI algorithms ported, outputting SimpleTT.TensorTrain{V,N} directly
end
```

- **Pro**: Output type is `SimpleTT.TensorTrain{V,N}` natively. No conversion overhead. Single type system.
- **Con**: Code duplication / maintenance burden. Must keep in sync or fully fork.

### Analysis: Does `SimpleTT` require porting?

If we re-export, the workflow is:

```julia
tci_result = crossinterpolate2(f, ...)       # → TCI.TensorTrain{V,N}
tt = SimpleTT.TensorTrain(tci_result)         # conversion needed every time
```

If we port:

```julia
tt = crossinterpolate2(f, ...)               # → SimpleTT.TensorTrain{V,N} directly
```

The conversion is trivial (both hold `Vector{Array{V,N}}`), but it is an extra step for every TCI call. Porting eliminates this friction and unifies the type system.

A middle ground is also possible:

### Option C: Re-export + type alias

Make `SimpleTT.TensorTrain{V,N}` a type alias for `TensorCrossInterpolation.TensorTrain{V,N}`, or vice versa. This requires one package to depend on the other.

## Decision: Option B (Port)

Port `TensorCrossInterpolation.jl` into `Tensor4all.TensorCI`, outputting `SimpleTT.TensorTrain{V,N}` directly. This unifies the type system and avoids any conversion step.

## Open Questions

- Should the matrix CI internals (MatrixLU, MatrixLUCI, MatrixACA) also move into `Tensor4all.TensorCI`, or stay as a separate dependency?
- Should `TensorCI` expose TCI1, or only TCI2 (the primary algorithm)?
- How should TCI diagnostics (convergence info, error estimates) be exposed?

## Julia API (regardless of re-export or port)

```julia
# Primary construction
crossinterpolate2(f, localdims; tolerance, maxbonddim, ...) → TensorTrain{V,N}

# TCI1 (legacy)
crossinterpolate1(f, localdims; ...) → TensorTrain{V,N}

# Pivot initialization
optfirstpivot(f, localdims; ...) → pivot

# Integration
integrate(f, grid; ...) → scalar
```

## Port Strategy and Long-term Direction

### Monorepo approach

`TensorCI` is ported as a submodule inside `Tensor4all.jl`, not as an independent package. Rationale:

- **Type unification**: Output type is `SimpleTT.TensorTrain{V,N}` directly, eliminating conversion steps
- **Simplified version management**: Splitting submodules into independent packages would require managing C-API distribution and cross-library version compatibility, which adds significant overhead
- **Easier future replacement**: Within a monorepo, internal implementations can be swapped without changing user-facing APIs

### Scope of the full port

All functionality from TensorCrossInterpolation.jl (Matrix CI internals, sweep strategies, batch evaluation, cached function evaluation, etc.) will be ported into `Tensor4all.TensorCI`. The dependency on `TensorCrossInterpolation.TensorTrain{V,N}` is fully eliminated; `SimpleTT.TensorTrain{V,N}` becomes the sole output type.

### Possible future migration to Rust TCI

A Rust-based TCI implementation exists (or is under development) in tensor4all-rs. In the future, the ported Julia TCI implementation may be replaced by the Rust version via C-FFI. However, the timing and scope of this migration are undecided — the fully ported Julia implementation is the source of truth for now.

Design guidelines in light of this possibility:

- **Public API stability is paramount**: Function signatures (`crossinterpolate2`, `optfirstpivot`, etc.) must remain stable regardless of whether the internal implementation is Julia or Rust
- **API compatibility over internal perfection**: Internal refactoring during the port is acceptable, but the public API should maintain compatibility with TensorCrossInterpolation.jl
- **Incremental replacement**: If/when the Rust version becomes available, the switch should happen function-by-function rather than all at once
