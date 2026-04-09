# Julia Frontend Quantics Layer

## Purpose

This document covers quantics grid semantics and quantics transform operators in the Julia frontend.

This document does not cover `TTFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## Module Structure

| Module | Source | Description |
|--------|--------|-------------|
| `Tensor4all.QuanticsGrids` | Re-export `QuanticsGrids.jl` | Grid types, coordinate conversion, layout |
| `Tensor4all.QuanticsTransform` | Port from `Quantics.jl`, backed by `t4a_qtransform_*` | Transform operators (MPO construction in Rust, application in Julia) |

## `Tensor4all.QuanticsGrids` (re-export)

Re-exports `QuanticsGrids.jl` as-is:

- named-variable quantics grids
- site layouts and index-table control
- coordinate conversion (grid index ↔ physical coordinate)
- layout conventions: fused, interleaved, grouped
- endpoint conventions

## `Tensor4all.QuanticsTransform`

Transform MPOs are constructed in Rust and copied to Julia as `ITensorTrain`. Subsequent operations (application to states, composition) use Julia-side TT operations.

### Backend Flow

1. Julia calls `t4a_qtransform_*` → Rust constructs MPO → returns `t4a_linop`
2. Extract site tensors via `t4a_linop_get_tensors` → `t4a_tensor[]`
3. Wrap as `ITensorTrain`
4. Application to states via `contract` (Julia-side, from `ITT` module)

### C-API Transform Constructors

| C-API | Description |
|-------|-------------|
| `t4a_qtransform_affine` | affine y = Ax + b (rational coefficients, boundary conditions) |
| `t4a_qtransform_shift` | coordinate shift |
| `t4a_qtransform_flip` | flip/reversal |
| `t4a_qtransform_phase_rotation` | phase multiplication |
| `t4a_qtransform_cumsum` | cumulative sum |
| `t4a_qtransform_fourier` | quantics Fourier transform |
| `t4a_qtransform_binaryop` | binary operations on two variables |

### Julia API (following `Quantics.jl` conventions)

```julia
# --- Affine transforms ---
affinetransformmpo(sites, A, b; bc=1)  → ITensorTrain
# Rational matrix A, rational vector b, boundary condition bc
# Multi-variable affine: y = Ax + b

# --- Coordinate transforms ---
shiftaxismpo(sites, shift; bc=1)       → ITensorTrain
flipop(sites; bc=1)                    → ITensorTrain   # reversal
reverseaxismpo(sites; bc=1)            → ITensorTrain

# --- Phase and spectral ---
phase_rotation_mpo(sites, θ)           → ITensorTrain
fouriertransform_mpo(sites; sign=1)    → ITensorTrain

# --- Structural ---
cumsum(sites; includeown=false)         → ITensorTrain
binaryop_mpo(sites, ...)               → ITensorTrain

# --- Application (delegates to ITT.contract) ---
apply(op::ITensorTrain, state::ITensorTrain; method=:zipup, rtol=..., maxdim=...)
    → ITensorTrain
```

Notes:
- All `*mpo` functions construct the operator in Rust and return `ITensorTrain`.
- Application to states uses `ITT.contract`, not a separate Rust call.
- The Julia API names follow `Quantics.jl` conventions for compatibility with existing user code.

## Multiresolution (future)

- coarsening and averaging
- interpolation and refinement
- layout-preserving embed/resample workflows

## Relationship to Other Docs

- [julia_ffi_tt.md](./julia_ffi_tt.md) covers `ITT.ITensorTrain` and `SimpleTT.TensorTrain{V,N}`.
- [bubbleteaCI.md](./bubbleteaCI.md) covers the higher-level function workflows that consume quantics grids.

## Open Questions

- Which layout should be canonical internally?
- Where should weighted integration and quadrature-like behavior live?
