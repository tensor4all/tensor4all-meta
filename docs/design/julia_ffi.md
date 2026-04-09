# Julia Frontend Design for `tensor4all-rs`

## Overview

This file is the module index for the Julia frontend. The goal is to organize the design around concrete frontend submodules first, with a clear distinction between:

- backend-facing Julia modules that belong to `Tensor4all`
- the still-open high-level `BubbleTeaCI` layer built on top of them

The Julia frontend owns wrappers, module boundaries, and Julia-facing APIs. `tensor4all-rs` owns kernels, storage, and performance-critical numerics.

## Module Index

| Public module | Exposure | Primary design doc | Responsibility |
|---------------|----------|--------------------|----------------|
| `Tensor4all` / `Tensor4all.Core` | `Core` is re-exported from top-level `Tensor4all` | [julia_ffi_core.md](./julia_ffi_core.md) | `Index`, `Tensor`, ownership, low-level FFI, basic tensor/index helpers |
| `Tensor4all.ITT` | namespaced submodule | [julia_ffi_itt.md](./julia_ffi_itt.md) | indexed TT/MPS/MPO-facing API built from Rust-backed `Tensor` |
| `Tensor4all.SimpleTT` | namespaced submodule | [julia_ffi_simplett.md](./julia_ffi_simplett.md) | raw-array `TensorTrain{V,N}` and TT operations used by TCI |
| `Tensor4all.TensorCI` | namespaced submodule | [julia_ffi_tci.md](./julia_ffi_tci.md) | core TCI construction algorithms targeting `SimpleTT.TensorTrain{V,N}` |
| `Tensor4all.QuanticsGrids` | namespaced submodule | [julia_ffi_quanticsgrids.md](./julia_ffi_quanticsgrids.md) | quantics grids, layouts, coordinates, endpoint conventions |
| `Tensor4all.QuanticsTCI` | namespaced submodule | [julia_ffi_quanticstci.md](./julia_ffi_quanticstci.md) | quantics-aware interpolation convenience layer, re-exporting `QuanticsTCI.jl` |
| `Tensor4all.QuanticsTransform` | namespaced submodule | [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) | quantics transform operators acting on TT objects |

Unsorted for now, with the expectation that they will also be modularized later:

- [bubbleteaCI.md](./bubbleteaCI.md) — high-level `TTFunction` / `GriddedFunction` workflows

## Frontend Overview

```text
tensor4all-rs
    ^
    | C-FFI
    |
Tensor4all
├── Core               # re-exported at top level
├── ITT                # indexed tensor-train layer
├── SimpleTT           # raw-array tensor-train layer
├── TensorCI           # core TCI algorithms
├── QuanticsGrids      # grid/layout/coordinate semantics
├── QuanticsTCI        # quantics-aware interpolation layer
├── QuanticsTransform  # transform operators
└── ...                # further frontend modules may be split out later

BubbleTeaCI
└── still-unsorted high-level layer for now
```

## Per-Module View

### `Tensor4all` / `Tensor4all.Core`

```text
Tensor4all
├── re-export: Index, Tensor, inds, prime, tags, ...
└── Core
    ├── Index
    ├── Tensor
    └── ownership / lifecycle / low-level FFI
```

- This is the stable frontend root.
- Users can start from `using Tensor4all` for core tensor/index work.
- The top-level package should primarily expose the `Core` surface, not flatten every submodule API.

### `Tensor4all.ITT`

```text
Tensor4all.ITT
└── ITensorTrain
    ├── uses Core.Tensor and Core.Index
    ├── supports indexed TT / MPS / MPO workflows
    └── owns contract / truncate / add at the indexed layer
```

- This is the indexed tensor-train layer.
- It is the bridge from low-level `Tensor` to higher-level TT workflows.
- It should stay focused on TT structure, not absorb `TTFunction` semantics.

### `Tensor4all.SimpleTT`

```text
Tensor4all.SimpleTT
└── TensorTrain{V,N}
    ├── stores Vector{Array{V,N}}
    ├── stays free of index metadata
    └── is the natural TT representation for TCI outputs
```

- This is the raw numerical TT layer.
- It is the common representation for TCI/QTCI construction and low-overhead TT manipulation.
- Conversion with `Tensor4all.ITT` is explicit and remains part of the frontend design.

### `Tensor4all.TensorCI`

```text
Tensor4all.TensorCI
├── crossinterpolate1
├── crossinterpolate2
├── optfirstpivot
└── integration / diagnostics / pivot logic
    -> outputs SimpleTT.TensorTrain{V,N}
```

- This module owns TT construction from function evaluation.
- Its output boundary should be `Tensor4all.SimpleTT`, not a separate external TT type.
- Matrix-CI internals may stay internal or be split later, but the public construction API belongs here.

### `Tensor4all.QuanticsGrids`

```text
Tensor4all.QuanticsGrids
├── grid definitions
├── variable/layout conventions
├── coordinate conversion
└── endpoint / indexing semantics
```

- This module owns the discrete geometry of quantics workflows.
- It should define the frontend vocabulary for layouts and coordinates independently of high-level application logic.
- Re-exporting `QuanticsGrids.jl` remains the current direction.

### `Tensor4all.QuanticsTCI`

```text
Tensor4all.QuanticsTCI
├── quanticscrossinterpolate
├── evaluate / sum / integral / cachedata
└── quantics-aware convenience layer
    built on QuanticsGrids
```

- This module sits above `Tensor4all.QuanticsGrids`, because it consumes grid semantics rather than defining them.
- The current direction is to re-export `QuanticsTCI.jl` as the quantics-specific convenience layer.
- It should stay separate from `Tensor4all.TensorCI`: `TensorCI` owns core interpolation algorithms, while `QuanticsTCI` owns quantics-facing entry points and grid-aware wrappers.

### `Tensor4all.QuanticsTransform`

```text
Tensor4all.QuanticsTransform
├── affine / shift / flip / Fourier-like operators
├── Rust-backed MPO/operator construction
└── application through ITT-level TT operations
```

- This module owns transform operators, not grid semantics and not `TTFunction` semantics.
- It depends on both `Tensor4all.QuanticsGrids` and `Tensor4all.ITT`.
- The boundary with `Tensor4all.QuanticsTCI` is still somewhat open where the existing Julia package already exposes helpers such as `quanticsfouriermpo`.
- Partial-site application is part of this module boundary, because it is a transform concern rather than a `BubbleTeaCI` concern.

## Dependency Sketch

```text
Tensor4all.Core
    ^
    |
    +---- Tensor4all.ITT
    |
    +---- Tensor4all.SimpleTT <---- Tensor4all.TensorCI
    |
    +---- Tensor4all.QuanticsTCI <-------- Tensor4all.QuanticsGrids
    |
    +---- Tensor4all.QuanticsTransform <---- Tensor4all.QuanticsGrids
    |
    +---- additional modules may be split later

BubbleTeaCI
    ├── may depend on ITT
    ├── may depend on SimpleTT / TensorCI
    └── may depend on QuanticsGrids / QuanticsTCI / QuanticsTransform
```

## Pending High-Level Layer

`BubbleTeaCI` and everything below the reusable `TTFunction` / `GriddedFunction` layer remain intentionally open in this index.

Those topics stay in [bubbleteaCI.md](./bubbleteaCI.md):

- final public naming such as `TTFunction` vs. `QTTFunction`
- the exact high-level object model above TT primitives
- the clean boundary between reusable core and application-specific workflows
- how multiresolution helpers, embedding, and interpolation should be split between frontend modules and `BubbleTeaCI`

## Unsorted Items

These are intentionally left outside the frontend module table for now. The plan is to eventually give them clearer module boundaries as well.

- [bubbleteaCI.md](./bubbleteaCI.md): high-level reusable layer above the frontend modules

## Ownership Model

- `tensor4all-rs` owns performance-critical kernels, storage, and numerics.
- `Tensor4all` owns low-level wrappers, submodule boundaries, and Julia-facing frontend APIs.
- `BubbleTeaCI` is a separate consumer layer that should build on the frontend modules without redefining their low-level responsibilities.

