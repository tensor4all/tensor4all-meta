# Design Documents

Cross-repository architecture and design documents are stored here.
Repository-scoped design documents live in each implementation repository:

- tenferro internals: [tenferro-rs/docs/design/](https://github.com/tensor4all/tenferro-rs/tree/main/docs/design)
- Julia frontend: [Tensor4all.jl/docs/design/](https://github.com/tensor4all/Tensor4all.jl/tree/main/docs/design)

## Documents

| Document | Status | Summary |
| --- | --- | --- |
| [core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md) | Proposed (2026-10) | What the tenferro core contains, and why the caller owns CPU threads while every layer borrows them |
