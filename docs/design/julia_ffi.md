# Tensor4all.jl Redesign: Julia FFI Layer

## Guiding Principles

1. **Rust owns full functionality** — optimal performance is guaranteed in Rust (tensor4all-rs).
2. **Julia wrapper for education and ecosystem support** — for learning QTT/TCI concepts, and for supporting downstream Julia packages (e.g., BubbleTeaCI). Not full-featured.
3. **Minimal C-FFI** — only expose what cannot be implemented in pure Julia. Keeps the FFI surface small and maintainable.
4. **Pure Julia for logic** — index operations, searching, matching, and higher-level composition are implemented in Julia. Rust provides data storage and heavy compute only.

## Core Types

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

## Design Boundary

```
┌──────────────────────────────────────────────────┐
│  Pure Julia                                      │
│                                                  │
│  Index ops: prime, sim, hastag, ...              │
│  Index-collection ops: findsites, commoninds,... │
│  Tensor index ops: replaceind, prime, addtags,...│
│                                                  │
│  ITensors.jl conversion (extension)              │
│                                                  │
├──────────────────────────────────────────────────┤
│  C-FFI (minimal)                                 │
│                                                  │
│  Index: new, clone, release, dim, id, tags, plev │
│  Tensor: new, clone, release, data, contract     │
│                                                  │
├──────────────────────────────────────────────────┤
│  Rust (tensor4all-rs)                            │
│                                                  │
│  DynIndex, TensorDynLen, contraction engine,     │
│  factorization, TCI, full algorithm suite        │
│                                                  │
└──────────────────────────────────────────────────┘
```

## ITensors.jl Compatibility

- Bidirectional conversion between `Tensor4all.Index` and `ITensors.Index{Int}` via a Julia package extension.
- ID mapping is direct (both use UInt64).
- Tag mapping: Tensor4all tags (comma-separated string from Rust) ↔ ITensors TagSet.
- HDF5 I/O in ITensors format for data exchange with external tools.

## Open Questions

- **Index tags representation**: Currently accessed as a comma-separated string from Rust. Should Julia cache tag data to avoid repeated C-API calls?
- **Tensor metadata caching**: Should `Tensor` cache `Vector{Index}` on the Julia side to avoid C-API roundtrips for metadata-only operations?
- **Higher-level structures** (TreeTN, TensorTrain, TCI): To be designed in a later iteration.
