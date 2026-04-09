# Julia Frontend Core Primitives

## Purpose

This document covers the low-level Julia frontend primitives that sit directly on top of the `tensor4all-rs` C-FFI. It is intentionally limited to foundational types and their basic Julia-side operations.

## In Scope

- `Index`
- `Tensor`
- C-API ownership and lifecycle
- metadata access
- pure Julia index and tensor manipulation helpers
- low-level ITensors compatibility for `Index` and `Tensor`

This document does not define backend `TensorTrain` support, quantics grid semantics, or `TTFunction` / `GriddedFunction` behavior.

Compatibility for `ITensorMPS.MPS` / `MPO` belongs to [julia_ffi_itt.md](./julia_ffi_itt.md), because that is an `ITT.ITensorTrain` concern rather than a primitive `Index` / `Tensor` concern.

## `Index`

```julia
struct Index
    ptr::Ptr{Cvoid}
end
```

- Wraps Rust `DynIndex` through the C-FFI.
- Matches the `ITensors.Index{Int}` style of indexed tensor work used in the Julia ecosystem.
- Exposes `dim`, `id`, `tags`, and `plev` via getters.

### C-API Surface

- Lifecycle: `t4a_index_new`, `t4a_index_clone`, `t4a_index_release`
- Getters: `t4a_index_dim`, `t4a_index_id`, `t4a_index_get_tags`, `t4a_index_get_plev`
- Setter: `t4a_index_set_plev`
- Predicate: `t4a_index_has_tag`

### Julia Helpers

- `sim(i)` for making a fresh index with the same dimension
- `prime(i, n)`, `noprime(i)`, `setprime(i, n)`
- `hastag(i, tag)`
- `findsites`, `commoninds`, and `uniqueinds` on index collections

### ITensors Compatibility

- `Index` should remain structurally compatible with `ITensors.Index{Int}` workflows.
- Provide bidirectional conversion:
  - `Index` ↔ `ITensors.Index{Int}`
- Preserve the usual Julia tensor-network metadata:
  - `dim`
  - `id`
  - `tags`
  - `plev`

## `Tensor`

```julia
struct Tensor
    ptr::Ptr{Cvoid}
end
```

- Stores tensor data in Rust.
- Keeps Julia responsible for index bookkeeping and metadata manipulation.
- Delegates heavy compute such as contraction to Rust.

### C-API Surface

- Creation: `t4a_tensor_new_dense_f64`, `t4a_tensor_new_dense_c64`, `t4a_tensor_new_diag_f64`, `t4a_tensor_new_diag_c64`
- Lifecycle: `t4a_tensor_clone`, `t4a_tensor_release`
- Getters: `t4a_tensor_get_rank`, `t4a_tensor_get_dims`, `t4a_tensor_get_indices`, `t4a_tensor_get_data_f64`, `t4a_tensor_get_data_c64`
- Compute: `t4a_tensor_contract`

### Julia Helpers

- `inds(T)`
- `rank(T)`, `dims(T)`
- `replaceind`, `replaceinds`
- `prime`, `noprime`, `setprime`
- `addtags`, `removetags`, `settags`, `replacetags`
- `swapind`, `swapinds`

### ITensors Compatibility

- `Tensor` should support bidirectional conversion with `ITensors.ITensor`.
- Conversion should preserve indices and dense/diagonal structure as faithfully as practical.
- The frontend should preserve round-trips between existing Julia `ITensors.jl` tensor workflows and the new Rust-backed primitive layer.

## Design Boundary

- Keep the low-level wrappers small and stable.
- Keep compatibility for primitive `Index` / `Tensor` types here rather than in a separate extensions document.
- Keep `MPS` / `MPO` compatibility in the TT layer.
- Keep semantic composition out of this file.
- Let [julia_ffi_itt.md](./julia_ffi_itt.md), [julia_ffi_simplett.md](./julia_ffi_simplett.md), and `bubbleteaCI.md` build on top of these primitives.

## Open Questions

- Should tag data be cached on the Julia side?
- Should `Tensor` cache `Vector{Index}` for metadata-heavy workflows?
- How should low-level FFI errors be mapped into Julia exceptions?
- Should metadata caching be handled directly by core primitives?
- Are there additional `Index` / `Tensor` conversion paths that should be first-class from day one?
