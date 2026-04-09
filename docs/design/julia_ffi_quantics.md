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

# --- Application to full or partial site indices ---
apply(op::ITensorTrain, state::ITensorTrain; method=:zipup, rtol=..., maxdim=...)
    → ITensorTrain
# Full application: op and state have the same sites.

apply(op::ITensorTrain, state::ITensorTrain, target_sites;
      method=:zipup, rtol=..., maxdim=...)
    → ITensorTrain
# Partial application: op acts on a subset of state's sites.
# e.g., apply shift operator to "x" variable only in a multi-variable QTT.
# Delegates to Rust-side partial contract C-API.
```

Notes:
- All `*mpo` functions construct the operator in Rust and return `ITensorTrain`.
- Partial-site application requires a Rust-side C-API extension (see below).
- The Julia API names follow `Quantics.jl` conventions for compatibility with existing user code.

### C-API extension needed: partial-site contraction

Multi-variable QTT functions require applying a transform to a subset of sites (e.g., shift only the "x" variable in f(x, y)). This should be handled in Rust for efficiency:

```c
// Contract op (length M) with state (length N, M ≤ N) at specified site positions
t4a_treetn_contract_partial(
    const t4a_treetn* op,           // operator (M sites)
    const t4a_treetn* state,        // state (N sites)
    const size_t* target_sites,     // which N sites the op acts on (length M)
    size_t n_target_sites,
    t4a_contract_method method,
    double rtol, double cutoff, size_t maxdim,
    t4a_treetn** out
)
```

This avoids Julia-side reshuffling of site tensors and index manipulation for every partial application.

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
