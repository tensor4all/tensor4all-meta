# Julia Frontend QuanticsGrids Support

## Purpose

This document covers the `Tensor4all.QuanticsGrids` submodule.

`Tensor4all.QuanticsGrids` owns quantics grid semantics in the Julia frontend. It is the layer that defines how physical coordinates, discrete grid indices, and quantics bit-level indices relate to each other.

This document does not cover quantics interpolation helpers or transform operators. Those belong to [julia_ffi_quanticstci.md](./julia_ffi_quanticstci.md) and [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md).

## Design Direction

Re-export `QuanticsGrids.jl` as the `Tensor4all.QuanticsGrids` submodule.

This keeps the frontend aligned with the existing Julia ecosystem and avoids redefining quantics grid vocabulary in multiple places.

## In Scope

- grid types
- layout conventions
- coordinate conversion
- endpoint conventions
- variable ordering and site ordering semantics

## Out of Scope

- tensor cross interpolation itself
- TT/MPS/MPO transforms
- high-level `TTFunction` / `GriddedFunction` semantics

## Relationship to `QuanticsGrids.jl`

The frontend should re-export `QuanticsGrids.jl` as-is for now.

That includes the conceptual surface around:

- `Grid`
- `DiscretizedGrid`
- `InherentDiscreteGrid`
- `localdimensions`
- `grididx_to_quantics`
- `quantics_to_origcoord`
- `grid_step`

The main role of the frontend here is not to redesign these APIs, but to make them the canonical grid vocabulary used by the rest of the Julia frontend.

## Responsibility Boundary

`Tensor4all.QuanticsGrids` should answer:

- what a quantics grid is
- how variables and axes are laid out
- how grid points map to quantics index tuples
- how physical coordinates are reconstructed from quantics indices

It should not answer:

- how a function is interpolated on that grid
- how an MPO transform is built or applied
- how a high-level `TTFunction` stores metadata

## Layer Position

```text
Tensor4all.QuanticsGrids
├── grid definitions
├── layout conventions
├── grid index <-> quantics index conversion
└── quantics index <-> physical coordinate conversion
```

`Tensor4all.QuanticsTCI` and `Tensor4all.QuanticsTransform` should sit above this layer.

## Open Questions

- Which layout should be canonical internally?
- How much layout flexibility should stay public versus be normalized at construction time?
- Should the frontend add any thin aliases or documentation helpers on top of `QuanticsGrids.jl`, or just re-export it verbatim?
