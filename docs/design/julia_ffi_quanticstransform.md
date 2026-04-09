# Julia Frontend QuanticsTransform Support

## Purpose

This document covers the `Tensor4all.QuanticsTransform` submodule.

`Tensor4all.QuanticsTransform` owns transform operators in the Julia frontend: affine coordinate transforms, shifts, flips, phase rotations, cumulative operators, Fourier-like transforms, and related TT/MPO application machinery.

This document does not cover quantics grid semantics or quantics interpolation convenience. Those belong to [julia_ffi_quanticsgrids.md](./julia_ffi_quanticsgrids.md) and [julia_ffi_quanticstci.md](./julia_ffi_quanticstci.md).

## Design Direction

Port the transform/operator-facing parts of `Quantics.jl` into `Tensor4all.QuanticsTransform`, with operator construction backed by `tensor4all-rs`.

The goal is:

- Rust owns operator construction and heavy TT-side kernels
- Julia owns surface API, naming compatibility, and module structure

## Layer Position

```text
Tensor4all.QuanticsGrids
    ^
    |
    +---- Tensor4all.QuanticsTransform
              |
              +---- uses Tensor4all.ITT for operator/state application
```

`QuanticsTransform` depends on:

- `Tensor4all.QuanticsGrids` for grid/layout semantics
- `Tensor4all.ITT` for indexed TT operator/state application

## Backend Flow

1. Julia calls `t4a_qtransform_*`
2. Rust constructs the transform MPO / operator
3. Julia wraps returned tensors as `ITT.ITensorTrain`
4. Application and composition proceed through TT-level frontend APIs

## C-API Transform Constructors

| C-API | Description |
|-------|-------------|
| `t4a_qtransform_affine` | affine `y = A x + b` |
| `t4a_qtransform_shift` | coordinate shift |
| `t4a_qtransform_flip` | flip / reversal |
| `t4a_qtransform_phase_rotation` | phase multiplication |
| `t4a_qtransform_cumsum` | cumulative sum |
| `t4a_qtransform_fourier` | quantics Fourier transform |
| `t4a_qtransform_binaryop` | binary operations on two variables |

## Julia API Direction

The Julia surface should follow existing `Quantics.jl` naming closely where it helps migration:

```julia
# Affine transforms
affinetransformmpo(sites, A, b; bc=1)       -> ITensorTrain

# Coordinate transforms
shiftaxismpo(sites, shift; bc=1)            -> ITensorTrain
flipop(sites; bc=1)                         -> ITensorTrain
reverseaxismpo(sites; bc=1)                 -> ITensorTrain

# Phase / spectral transforms
phase_rotation_mpo(sites, θ)                -> ITensorTrain
fouriertransform_mpo(sites; sign=1)         -> ITensorTrain

# Structural operators
cumsum(sites; includeown=false)             -> ITensorTrain
binaryop_mpo(sites, ...)                    -> ITensorTrain
```

## Application API

```julia
apply(op::ITensorTrain, state::ITensorTrain; method=:zipup, rtol=..., maxdim=...)
    -> ITensorTrain

apply(op::ITensorTrain, state::ITensorTrain, target_sites;
      method=:zipup, rtol=..., maxdim=...)
    -> ITensorTrain
```

- Full application acts on all sites.
- Partial application acts on a subset of sites in a larger quantics state.

## Required Backend Extension

Partial-site application should be handled in Rust rather than by repeated Julia-side site reshuffling.

```c
t4a_treetn_contract_partial(
    const t4a_treetn* op,
    const t4a_treetn* state,
    const size_t* target_sites,
    size_t n_target_sites,
    t4a_contract_method method,
    double rtol, double cutoff, size_t maxdim,
    t4a_treetn** out
)
```

This is especially important for transforms that act on one variable inside a larger multi-variable QTT representation.

## Relationship to `Quantics.jl`

`Quantics.jl` contains the current high-level transform-oriented Julia surface.

The frontend should preserve migration-friendly names where practical, while moving heavy operator construction into the Rust-backed backend.

## Open Questions

- Which existing `Quantics.jl` APIs should be preserved exactly, and which should be reshaped around `ITT.ITensorTrain`?
- Where is the final boundary between `QuanticsTransform` and convenience helpers that currently live in `QuanticsTCI.jl`?
- Which multiresolution or resampling operations belong here versus the still-open high-level layer?
