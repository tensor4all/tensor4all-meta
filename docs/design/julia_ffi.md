# Julia Frontend Design for `tensor4all-rs`

## Guiding Principles

1. **Rust owns performance-critical kernels** — contraction, factorization, transforms, TCI, solvers, and dense conversion live in `tensor4all-rs`.
2. **Julia is the canonical user-facing frontend** — the Julia package should support both ecosystem integration and practical high-level QTT workflows, not just low-level wrapping.
3. **Lean C-FFI** — expose stable low-level primitives and Rust kernels without pushing semantic bookkeeping or user ergonomics into the FFI layer.
4. **Pure Julia for logic and composition** — index operations, grid semantics, layout bookkeeping, variable-aware contraction setup, and interop are implemented in Julia.

The low-level wrappers below are foundational building blocks for the interface, but they are not the full scope of the Julia frontend.

## Foundational Low-Level Types

### `Index` — Rust opaque pointer wrapper

```julia
struct Index
    ptr::Ptr{Cvoid}  # wraps t4a_index (Rust DynIndex)
end
```

- Wraps Rust's `DynIndex` via C-FFI to ensure consistency with Tensor internals.
- Corresponds to `ITensors.Index{Int}` (no quantum number support — not used in the ecosystem).
- Fields: `dim`, `id`, `tags`, `plev` — accessed via C-API getters.

**C-API surface (existing):**
- Lifecycle: `t4a_index_new`, `t4a_index_clone`, `t4a_index_release`
- Getters: `t4a_index_dim`, `t4a_index_id`, `t4a_index_get_tags`, `t4a_index_get_plev`
- Setters: `t4a_index_set_plev`
- Predicates: `t4a_index_has_tag`

**Pure Julia operations on Index (built on C-API getters):**
- `sim(i)` — create a new Index with same dim but new id
- `prime(i, n)`, `noprime(i)`, `setprime(i, n)` — prime level manipulation
- `hastag(i, tag)` — tag query

**Pure Julia operations on Index collections:**
- `findsites(inds, query)` — find index positions matching criteria
- `commoninds(a, b)`, `uniqueinds(a, b)` — set operations on index collections

### `Tensor` — Rust opaque pointer wrapper

```julia
struct Tensor
    ptr::Ptr{Cvoid}  # wraps t4a_tensor (Rust TensorDynLen)
end
```

- Data storage lives in Rust. Heavy operations (contraction, factorization) are performed in Rust.
- Index metadata is accessed via C-API, then manipulated in Julia.

**C-API surface (existing):**
- Creation: `t4a_tensor_new_dense_f64`, `t4a_tensor_new_dense_c64`, `t4a_tensor_new_diag_f64`, `t4a_tensor_new_diag_c64`
- Lifecycle: `t4a_tensor_clone`, `t4a_tensor_release`
- Getters: `t4a_tensor_get_rank`, `t4a_tensor_get_dims`, `t4a_tensor_get_indices`, `t4a_tensor_get_data_f64`, `t4a_tensor_get_data_c64`
- Compute: `t4a_tensor_contract`

**Pure Julia operations on Tensor (following ITensors.jl API):**

Index manipulation on Tensor — these read indices via C-API, apply pure Julia logic, and create a new Tensor with updated indices:
- `replaceind(T, old, new)`, `replaceinds(T, old => new, ...)` — replace indices
- `prime(T, ...)`, `noprime(T)`, `setprime(T, ...)` — prime level manipulation
- `addtags(T, ...)`, `removetags(T, ...)`, `settags(T, ...)`, `replacetags(T, ...)` — tag manipulation
- `swapind(T, i1, i2)`, `swapinds(T, ...)` — swap indices

Metadata access:
- `inds(T)` — return `Vector{Index}` by reading from Rust
- `rank(T)`, `dims(T)` — delegated to C-API

**C-API delegated operations on Tensor:**
- `contract(a, b)` / `a * b` — tensor contraction in Rust

These low-level types provide the substrate on top of which higher-level TT, TCI, grid, and QTT-function APIs are built.

### `TensorTrain` — Pure Julia struct

```julia
mutable struct TensorTrain
    data::Vector{Tensor}  # each site tensor is a Rust-backed Tensor4all.Tensor
    llim::Int             # left orthogonality limit
    rlim::Int             # right orthogonality limit
end
```

- Pure Julia struct holding a chain of `Tensor` objects (analogous to `SimpleTensorTrains.SimpleTensorTrain` but using `Tensor4all.Tensor` instead of `ITensor`).
- Julia owns the chain structure; individual tensor data lives in Rust.
- No `t4a_treetn` or `t4a_simplett` wrapper — no tree-level C-FFI needed.
- Serves as both MPS and MPO representation (distinguished at runtime by index structure, as in the current design).

**TT-level operations via C-API:**

Whole-chain operations (contraction, compression) are delegated to Rust by passing the site tensors:

- `t4a_contract(tensors_a[], n_a, tensors_b[], n_b, algorithm, ...) → tensors_result[], n_result`
  - Takes arrays of `t4a_tensor` pointers from both TTs
  - Rust reconstructs the TT internally, performs contraction (naive/zipup/fit), returns result as array of `t4a_tensor`
  - Julia wraps result back into `TensorTrain`
  - Name is general enough to support tree connectivity in the future (pass adjacency info)

## Julia Package Dependencies

`Tensor4all.jl` integrates existing Julia packages rather than reimplementing them:

| Submodule | Source | Description |
|-----------|--------|-------------|
| `Tensor4all.TensorTrain` | Absorb `T4ATensorTrain.jl` | TensorTrain type and operations (raw-array variant) |
| `Tensor4all.TensorCI` | Absorb `T4ATensorCI.jl` | TCI2 algorithm |
| `Tensor4all.QuanticsGrids` | Re-export `QuanticsGrids.jl` | Grid types and coordinate conversion |

Users get the full stack with `using Tensor4all`.

## Quantics Transforms

Quantics transform MPOs (affine, shift, flip, phase rotation, etc.) are constructed in Rust and extracted as `Vector{Tensor}` for use in Julia.

**Workflow:**
1. Julia calls a transform constructor via C-API → returns `t4a_linop`
2. Julia extracts site tensors via C-API → returns `t4a_tensor[]`
3. Julia wraps as `TensorTrain`
4. Application to a state via `t4a_contract`

**C-API surface:**

Transform constructors (existing):
- `t4a_qtransform_affine` — affine y = Ax + b (rational coefficients, boundary conditions)
- `t4a_qtransform_shift`, `t4a_qtransform_flip` — coordinate transforms
- `t4a_qtransform_phase_rotation` — phase multiplication
- `t4a_qtransform_cumsum` — cumulative sum
- `t4a_qtransform_fourier` — quantics Fourier transform
- `t4a_qtransform_binaryop` — binary operations on two variables
- All return `t4a_linop`

Tensor extraction (new):
- `t4a_linop_get_tensors(linop, ...) → t4a_tensor[], n` — extract MPO site tensors from a linear operator

This replaces the need for `Quantics.jl` on the Julia side. All carry propagation, rational arithmetic, and boundary condition logic stays in Rust.

## Design Boundary

```
┌──────────────────────────────────────────────────┐
│  Pure Julia                                      │
│                                                  │
│  Index ops: prime, sim, hastag, ...              │
│  Index-collection ops: findsites, commoninds,... │
│  Tensor index ops: replaceind, prime, addtags,...│
│  TensorTrain struct (Vector{Tensor})             │
│  Quantics grids, layouts, variable names         │
│  QTTFunction semantics and contraction setup     │
│                                                  │
│  Absorbed: T4ATensorTrain.jl, T4ATensorCI.jl     │
│  Re-exported: QuanticsGrids.jl                   │
│                                                  │
│  ITensors.jl / HDF5 conversion (extension)       │
│                                                  │
├──────────────────────────────────────────────────┤
│  C-FFI (lean, but sufficient)                    │
│                                                  │
│  Index: new, clone, release, dim, id, tags, plev │
│  Tensor: new, clone, release, data, contract     │
│  TT contraction: t4a_contract                    │
│  Transforms: t4a_qtransform_* → t4a_linop        │
│  Extraction: t4a_linop_get_tensors               │
│                                                  │
├──────────────────────────────────────────────────┤
│  Rust (tensor4all-rs)                            │
│                                                  │
│  DynIndex, TensorDynLen, contraction engine,     │
│  factorization, TCI, transforms, solvers         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## ITensors.jl Compatibility

Interoperability with ITensors.jl is a hard requirement — existing Julia codebases (BubbleTeaCI, etc.) depend on it.

**Bidirectional conversion (via Julia package extension):**
- `Index` ↔ `ITensors.Index{Int}` — ID mapping is direct (both UInt64), tag mapping: comma-separated string ↔ ITensors TagSet.
- `Tensor` ↔ `ITensors.ITensor` — storage mapping: DenseF64 ↔ Dense{Float64}, DenseC64 ↔ Dense{ComplexF64}.

**HDF5 interoperability:**
- `Tensor` can be saved/loaded in ITensors HDF5 format.
- Files written by ITensors.jl can be read by the Julia interface and vice versa.
- This ensures data exchange with external tools and existing workflows.

## Open Questions

- **Index tags representation**: Currently accessed as a comma-separated string from Rust. Should Julia cache tag data to avoid repeated C-API calls?
- **Tensor metadata caching**: Should `Tensor` cache `Vector{Index}` on the Julia side to avoid C-API roundtrips for metadata-only operations?
- **Public naming of the high-level function abstraction**: Should the public API standardize on `QTTFunction`, `TTFunction`, or provide one as an alias of the other?
- **Grid layout normalization**: Which layouts should be canonical internally, and which should be accepted as user-facing construction syntax?
- **Weighted integration boundary**: Which volume-element and quadrature-like semantics should live purely in Julia, and which should be promoted to Rust kernels if performance demands it?
- **Two TensorTrain types**: The absorbed `T4ATensorTrain.jl` uses `Vector{Array}` (no indices). The new `TensorTrain` uses `Vector{Tensor}` (Rust-backed, with indices). Relationship and conversion between these needs to be defined.
- **Packaging boundary of the reusable high-level layer**: Should the refactored generic `TTFunction` layer remain a clearly separated module within `BubbleTeaCI`, or eventually become its own package once stabilized?

## High-Level QTT Julia Layer

### Summary

The high-level QTT functionality required by downstream Julia workflows should not be implemented twice. Rather than copying the current `BubbleTeaCI` `TTFunction` machinery into a second codebase, the plan is to refactor and migrate the reusable high-level layer currently living in `BubbleTeaCI` so that it uses the new `tensor4all-rs`-based Julia frontend instead of `ITensors.jl`.

This migration should also separate generic functionality from application-specific physics code and clean up the reusable layer for stability, readability, flexibility, and user-friendliness. `BubbleTeaCI` is therefore not just a downstream consumer to keep in mind; it is a primary migration target whose reusable core should evolve together with the new frontend.

### Ownership Boundary

The intended split of responsibilities is:

- `tensor4all-rs`: contraction, factorization, transforms, TCI, TT arithmetic, dense conversion, solver kernels
- Julia frontend: low-level wrappers, grid semantics, variable naming, layout bookkeeping, interop, and backend-facing abstractions
- refactored reusable `BubbleTeaCI` core: `TTFunction` / `GriddedFunction` workflows, contraction semantics, embedding, integration, interpolation, and beginner-facing examples
- `BubbleTeaCI` applications: ladder-DGammaA and other physics-specific codes built on the reusable core

### Relationship to `BubbleTeaCI`

- `BubbleTeaCI` is a migration target, not merely an example consumer.
- Generic `TTFunction`-level functionality should be migrated and cleaned up, not reimplemented independently.
- The reusable high-level layer should no longer depend on concrete `ITensors.jl` / `ITensorMPS` storage types in its public design.
- Application-specific physics code should be separated from reusable tensor-network function infrastructure.
- The long-term package split can remain open, but the code boundary between reusable core and applications should be explicit from the beginning.

### Key Additions

The following capabilities are still required in the Julia ecosystem around the new backend. They should be delivered primarily by refactoring, stabilizing, and migrating the reusable high-level functionality currently living in `BubbleTeaCI`, rather than by creating a second independent implementation.

#### 1. Rich quantics-grid abstraction in Julia

Provide a Julia-side grid layer that the migrated high-level `TTFunction` workflows can depend on:

- named variables, not just positional dimensions
- explicit site layout / index-table control
- fused, interleaved, and grouped layouts
- grid-index to physical-coordinate conversion
- support for endpoint conventions and, if feasible, mixed bases

This is essential because real workflows depend on layout and variable semantics, not just raw TT cores.

#### 2. A backend-neutral high-level `QTTFunction` / `TTFunction` type

Refactor the existing `BubbleTeaCI` `TTFunction` concept into a reusable high-level type representing a gridded function in QTT form:

- wraps a TT/MPS-like object plus grid metadata
- supports scalar-, vector-, and matrix-valued functions
- distinguishes continuous-variable legs from discrete/component legs
- avoids direct dependence on concrete `ITensors.jl` storage types in the public API
- remains a Julia semantic wrapper over Rust-owned numerical objects

This should be the main user-facing abstraction for prototyping and should arise from migration and cleanup of existing reusable code, not duplication.

#### 3. Function-to-QTT construction and compression APIs

Provide user-friendly construction paths for:

- compressing Julia functions to QTT
- scalar-, vector-, and matrix-valued function compression
- building constants, zeros, and template-shaped functions
- conversion from TCI/QTCI results into the high-level function type

This is one of the most important usability features and should preserve the strengths of the current `BubbleTeaCI` workflow while improving clarity and robustness.

#### 4. Core QTT function operations

Include:

- point evaluation by grid index
- coordinate lookup such as `grid_coords`
- arithmetic: `+`, `-`, scalar multiplication
- `dot`, `norm`, truncation/compression
- dense materialization / `collect` for small cases
- slicing and component projection

These are central for iterative prototyping and debugging.

#### 5. Variable-aware contraction APIs

Retain and clean up a semantic contraction layer similar in spirit to `BasicContractOrder`:

- contract by variable names or positions
- support multiplication plus contraction in one operation
- support output variable reordering
- support contraction over discrete/component indices
- support weighted contraction / integration options such as volume-element factors

This is a major part of what makes the high-level layer practical.

#### 6. Integration and reduction utilities

Add explicit APIs for:

- integration over selected variables
- summation / partial integration
- reduction over component indices
- weighted integration on quantics grids

These should be first-class user operations.

#### 7. Function transformation utilities

Include high-level transforms for QTT functions, preserving the workflows that already exist around `BubbleTeaCI`:

- affine pullbacks / coordinate transforms
- shift, flip/reverse, phase rotation
- cumulative sum / integral-like transforms
- Fourier-related transforms where supported
- embedding into larger or higher-dimensional grids
- transform APIs that preserve layout semantics where possible

#### 8. Multiresolution utilities

Plan for:

- coarsening / averaging
- refinement / interpolation
- template-based resolution changes
- layout-preserving embed / resample operations

These are especially useful for experimentation and adaptive workflows.

#### 9. Interop and persistence

Retain and expand support for:

- ITensors compatibility
- HDF5 serialization
- conversion between low-level TT objects, QTCI objects, and high-level QTT functions
- dense export for validation and debugging

#### 10. Diagnostics and introspection

Include lightweight inspection APIs for:

- bond dimensions / ranks
- grid metadata and variable layout
- truncation diagnostics
- TCI settings and convergence/error summaries where available

These are important for research use and debugging.

### Cleanup and Stabilization Goals

The migration should not be a line-by-line backend swap. It should also be used to improve the reusable high-level layer:

- decouple generic high-level APIs from concrete backend storage types
- simplify internal abstractions and reduce cross-dependencies
- make variable, layout, and component-leg semantics easier to inspect
- improve naming, error messages, and user ergonomics
- define a stable and well-documented public surface for the reusable `TTFunction` layer

### Tutorial and Onboarding

A beginner-oriented tutorial should be included as part of the migration effort:

- it should be inspired by `basic_operations.jl`, but with much more explanation
- it should introduce grids, function compression, evaluation, contraction orders, transformations, and integration step by step
- it should include small, inspectable examples for scalar-, vector-, and matrix-valued functions
- it should be runnable as executable documentation and ideally serve as part of the regression suite

### Rust-Side Items to Call Out

#### A. Low-level features that should be explicitly exposed through the Julia interface

The plan should include wrapper coverage for useful low-level primitives needed by the migrated high-level layer, such as:

- TT add / scale / dot
- TT reverse / full tensor export / construction from site tensors
- TreeTN add / contract / dense conversion / linear solve
- QTCI options and diagnostics
- complex-number support where available

These belong in the Julia plan as interface coverage work.

#### B. Low-level features that may warrant new Rust support if missing

If profiling shows Julia-side orchestration is not enough, the plan should mention possible Rust additions:

- efficient weighted partial-integration kernels for QTT workflows
- low-level kernels for refinement/coarsening if these become performance-critical
- direct support for vector- and matrix-valued QTT function layouts if Julia-side leg management becomes too costly
- transform kernels beyond affine pullback when repeatedly used in high-level workflows
- specialized contraction helpers for QTT-function semantics if generic TreeTN APIs prove awkward or slow

The default assumption should still be: keep semantic orchestration in Julia unless performance forces a Rust primitive.

### Test Plan

The design should include acceptance tests covering:

- scalar, vector, and matrix function compression to QTT
- evaluation and coordinate lookup on named-variable grids
- arithmetic, norm, and truncation
- variable-aware contraction and output reordering
- integration over subsets of variables and component indices
- affine/shift/flip/Fourier-style transforms
- embedding, interpolation, and averaging across resolutions
- interop round-trips with ITensors and HDF5
- small dense-reference comparisons against direct Julia evaluation
- parity checks for representative `BubbleTeaCI` workflows during migration
- execution of tutorial/example scripts as regression tests where practical

### Assumptions and Defaults

- The Julia interface should prioritize research ergonomics and prototyping speed.
- Heavy numerical kernels should stay in Rust; semantic composition should stay in Julia.
- The first high-level target should be a cleaned-up, backend-neutral `QTTFunction`-style API derived from the reusable `BubbleTeaCI` layer, not a second independent implementation.
- Named variables and layout metadata are mandatory, not optional niceties.
- `BubbleTeaCI` and `ReFrequenTT` usage patterns should drive prioritization more than theoretical completeness.
- No duplicate long-term implementation of the generic high-level `TTFunction` functionality should be maintained.

## Phased Implementation Roadmap

### Phase 1. Low-Level Foundation

Establish the Julia package skeleton and the base FFI layer:

- wrap `Index` and `Tensor` with correct ownership, cloning, and error handling
- expose low-level metadata accessors and tensor construction / contraction
- provide initial `ITensors.jl` conversion hooks for the foundational types
- add smoke tests for creation, destruction, metadata access, and basic contraction

This phase should end with a stable substrate for all later work.

### Phase 2. Core Backend Coverage for TT Workflows

Expose the Rust-side tensor-network primitives required by high-level TT/QTT workflows:

- `TensorTrain` type (`Vector{Tensor}`) with basic operations
- `t4a_contract` for TT-level contraction (pass arrays of `t4a_tensor` pointers)
- Quantics transform constructors via `t4a_qtransform_*` → `t4a_linop` → `t4a_linop_get_tensors`
- Absorb `T4ATensorTrain.jl` and `T4ATensorCI.jl`
- TCI / QTCI construction and diagnostics

This phase should make the backend capable enough to support migration of the reusable high-level layer.

### Phase 3. Backend-Neutral High-Level Interfaces

Define the Julia-facing interfaces that the reusable high-level layer will target:

- backend-neutral abstractions for TT-like objects and operations
- the Julia quantics-grid abstraction and layout semantics
- named variables
- site layout / index-table specification
- fused, interleaved, and grouped representations
- coordinate and index conversion utilities
- validation of layout consistency and endpoint conventions
- Re-export `QuanticsGrids.jl`
- public naming and semantics of `QTTFunction` / `TTFunction`
- an explicit rule that generic high-level APIs should not depend on concrete `ITensors.jl` storage types

This phase should make migration tractable and keep the later refactor focused.

### Phase 4. Separate Reusable `BubbleTeaCI` Core from Applications

Refactor `BubbleTeaCI` so that reusable tensor-network-function code is clearly separated from physics-specific applications:

- isolate `TTFunction` / `GriddedFunction`, contraction-order logic, transforms, integration, interpolation, and related utilities
- keep ladder-DGammaA and other application-specific code outside the reusable core boundary
- preserve existing behavior with tests while improving module structure and readability

This phase should leave a reusable high-level core that can be migrated independently of the applications.

### Phase 5. Migrate the Reusable High-Level Layer to the New Backend

Retarget the separated generic layer from `ITensors.jl` to the new `tensor4all-rs`-based Julia frontend:

- migrate `QTTFunction` / `TTFunction` storage and core operations
- preserve evaluation, `grid_coords`, arithmetic, contraction, integration, embedding, and transform semantics
- support scalar-, vector-, and matrix-valued functions on the new backend
- close any remaining frontend wrapper gaps required by the migrated code

This phase should remove direct `ITensors.jl` dependence from the reusable `BubbleTeaCI` high-level layer.

### Phase 6. Cleanup, Stabilization, and Application Migration

Use the migration to harden the design and bring the application codes across:

- simplify naming and internal data flow in the reusable core
- improve flexibility, diagnostics, and error handling
- document stable public APIs and expected invariants
- ensure transform and multiresolution workflows are fully covered on the new backend, including affine pullbacks, shifts, flips, phase rotations, cumulative sums, Fourier-style operations, embedding, coarsening, refinement, and template-based resampling
- migrate ladder-DGammaA and other applications to the refactored core and new backend

This phase should turn the migrated code into a maintainable ecosystem layer rather than just a backend port.

### Phase 7. Tutorial, Validation, and Performance Closure

Finish the interface by hardening documentation, compatibility, and performance boundaries:

- add a beginner-oriented tutorial in the spirit of `basic_operations.jl`, with more detailed explanations and examples
- keep tutorial and example workflows executable where practical
- complete ITensors and HDF5 round-trip support where it remains useful
- add diagnostics and introspection APIs
- build dense-reference tests and migration-parity tests against representative existing workflows
- profile Julia-side orchestration and identify any Rust-kernel additions that are truly justified

This phase should end with a documented, validated Julia frontend together with a migrated and cleaned-up reusable `BubbleTeaCI` high-level layer.
