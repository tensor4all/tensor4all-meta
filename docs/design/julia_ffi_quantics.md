# Julia Frontend Quantics Layer

## Purpose

This document covers quantics grid semantics and quantics-specific transform behavior on the Julia side.

## In Scope

- named-variable quantics grids
- site layouts and index-table control
- coordinate conversion
- layout conventions such as fused, interleaved, and grouped representations
- quantics transform semantics
- multiresolution support

This document does not cover `TTFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## Grid Semantics

- Grid variables should be named, not just positional.
- Layout should be explicit and inspectable.
- Coordinate conversion should be available at the Julia level.
- Endpoint conventions and mixed bases should be represented clearly.

## Layout Conventions

- fused layouts for compact representations
- interleaved layouts for variable/bit interleaving
- grouped layouts for user-defined groupings
- explicit index-table control when a specific unfolding is needed

## Transform Operators

Quantics transforms are constructed in Rust and extracted as `ITensorTrain` for use in Julia.

### Backend Flow

1. Construct a transform in Rust via `t4a_qtransform_*` → returns `t4a_linop`
2. Extract site tensors with `t4a_linop_get_tensors` → returns `t4a_tensor[]`
3. Wrap the result as a Julia `ITensorTrain`
4. Application to a state via `t4a_contract`

### C-API Transform Constructors

- `t4a_qtransform_affine` — affine y = Ax + b (rational coefficients, boundary conditions)
- `t4a_qtransform_shift` — coordinate shift
- `t4a_qtransform_flip` — flip/reversal
- `t4a_qtransform_phase_rotation` — phase multiplication
- `t4a_qtransform_cumsum` — cumulative sum
- `t4a_qtransform_fourier` — quantics Fourier transform
- `t4a_qtransform_binaryop` — binary operations on two variables

### Julia-Side Transform API

These are exposed as high-level Julia functions that hide the C-API flow:

- affine pullbacks and coordinate transforms
- shifts, flips, and reversals
- phase rotation
- cumulative sums
- Fourier-style transforms
- binary operations on two variables

## Multiresolution

- coarsening and averaging
- interpolation and refinement
- layout-preserving embed/resample workflows

## Relationship to Other Docs

- [julia_ffi_tt.md](./julia_ffi_tt.md) covers the backend TT operator layer that can carry these transforms.
- [bubbleteaCI.md](./bubbleteaCI.md) covers the higher-level function workflows that consume quantics grids.

## Open Questions

- Which layout should be canonical internally?
- Which layout forms should be accepted as user-facing construction syntax?
- Where should weighted integration and quadrature-like behavior live?
