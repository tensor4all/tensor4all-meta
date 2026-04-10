# Design Documents

Repository-scoped design documents (for example, tenferro internals) live in each
implementation repository, such as
[tenferro-rs/docs/design/](https://github.com/tensor4all/tenferro-rs/tree/main/docs/design).

Cross-repository architecture/design documents are stored here.

The Julia frontend design is split into a hub-and-spoke set:

- [julia_ffi.md](./julia_ffi.md) for the overview and index
- [julia_ffi_core.md](./julia_ffi_core.md) for low-level primitives
- [julia_ffi_tensornetworks.md](./julia_ffi_tensornetworks.md) for indexed tensor-network support (`TensorNetworks`: `TensorTrain`, `TreeTensorNetwork`)
- [julia_ffi_simplett.md](./julia_ffi_simplett.md) for raw-array TT support (`SimpleTT.TensorTrain{V,N}`)
- [julia_ffi_tci.md](./julia_ffi_tci.md) for core TCI algorithms (`TensorCI`)
- [julia_ffi_quanticsgrids.md](./julia_ffi_quanticsgrids.md) for `QuanticsGrids`
- [julia_ffi_quanticstci.md](./julia_ffi_quanticstci.md) for `QuanticsTCI`
- [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) for `QuanticsTransform`
- [bubbleteaCI.md](./bubbleteaCI.md) for reusable `TTFunction` logic and migration
