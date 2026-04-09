# Julia Frontend QuanticsTCI Support

## Purpose

This document covers the `Tensor4all.QuanticsTCI` submodule.

`Tensor4all.QuanticsTCI` is the quantics-aware interpolation layer in the Julia frontend. It sits above `Tensor4all.QuanticsGrids` and provides convenience entry points for building tensor trains from functions sampled on quantics grids.

## Design Direction

Re-export `QuanticsTCI.jl` as the `Tensor4all.QuanticsTCI` submodule.

This is different from `Tensor4all.TensorCI`, which is the core interpolation layer. `QuanticsTCI` is the grid-aware convenience layer specialized for quantics workflows.

## Why It Sits Above `QuanticsGrids`

`QuanticsTCI` consumes `Grid` objects and quantics/grid conversion helpers. It does not define grid semantics itself.

So the dependency direction should be:

```text
Tensor4all.QuanticsGrids
    ^
    |
    +---- Tensor4all.QuanticsTCI
```

`QuanticsGrids` defines the domain. `QuanticsTCI` defines interpolation on that domain.

## Relationship to `QuanticsTCI.jl`

The current `QuanticsTCI.jl` package is a small convenience layer around:

- `TensorCrossInterpolation.jl`
- `QuanticsGrids.jl`

Its public surface includes:

- `quanticscrossinterpolate`
- `evaluate`
- `sum`
- `integral`
- `cachedata`
- `quanticsfouriermpo`

In the frontend design, this package should be re-exported rather than immediately reimplemented.

## Responsibility Boundary

`Tensor4all.QuanticsTCI` should own:

- quantics-aware interpolation entry points
- overloads that accept quantics grids directly
- quantics-aware evaluation and integration convenience
- convenience wrappers that are already part of `QuanticsTCI.jl`

It should not own:

- generic tensor cross interpolation internals
- base quantics grid definitions
- the final Rust-backed transform operator layer

## Relationship to `Tensor4all.TensorCI`

`Tensor4all.TensorCI` owns the backend-neutral, core TCI algorithms.

`Tensor4all.QuanticsTCI` owns the quantics-facing convenience layer that specializes those ideas to quantics grids and coordinate semantics.

For now, those can remain separate:

- `Tensor4all.TensorCI`: core interpolation layer
- `Tensor4all.QuanticsTCI`: quantics convenience layer, re-exporting the existing Julia package

## Relationship to `Tensor4all.QuanticsTransform`

There is some boundary overlap today, because `QuanticsTCI.jl` already exposes `quanticsfouriermpo`.

For now, this should be treated as an existing convenience surface rather than as the final architectural boundary.

The long-term split should be:

- `QuanticsTCI`: interpolation-oriented convenience
- `QuanticsTransform`: transform-operator-oriented functionality

## Example Surface

```julia
# Grid-aware quantics interpolation
quanticscrossinterpolate(Float64, f, grid; tolerance=1e-8)

# Convenience overloads from axis point sets or sizes
quanticscrossinterpolate(Float64, f, xvals; unfoldingscheme=:interleaved)
quanticscrossinterpolate(Float64, f, size_tuple; unfoldingscheme=:interleaved)

# Evaluate / integrate on the quantics object
evaluate(qtci, i, j, ...)
integral(qtci)
cachedata(qtci)
```

## Open Questions

- Should `QuanticsTCI` stay a permanent re-export, or become a migration layer toward `Tensor4all.TensorCI`?
- Should `quanticsfouriermpo` eventually move under `Tensor4all.QuanticsTransform`?
- If `Tensor4all.TensorCI` standardizes on `SimpleTT.TensorTrain{V,N}`, how should `QuanticsTCI` retarget its outputs over time?
