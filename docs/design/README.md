# Design Documents

Cross-repository architecture and design documents are stored here.
Repository-scoped design documents live in each implementation repository:

- tenferro internals: [tenferro-rs/docs/design/](https://github.com/tensor4all/tenferro-rs/tree/main/docs/design)
- Julia frontend: [Tensor4all.jl/docs/design/](https://github.com/tensor4all/Tensor4all.jl/tree/main/docs/design)

## Documents

| Document | Status | Summary |
| --- | --- | --- |
| [core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md) | Proposed (2026-10) | What the tenferro core contains, and why the caller owns CPU threads and GPU streams while every layer borrows them |
| [cpu-placement-token.md](./cpu-placement-token.md) | Explainer (2026-10) | The CPU placement token (`strided_exec::Exec`) with figures: what it is, how it is split, how the rules are checked |
