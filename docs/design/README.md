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
| [next-generation-dense-linear-algebra.md](./next-generation-dense-linear-algebra.md) | Design under review (2026-11) | colmat's double-double design, including independent kernel specialisations, SIMD, caller-owned teams and workspace, and the questions to resolve before implementation |
| [rocm-backend.md](./rocm-backend.md) | Detailed design, proposed (2026-11) | AMD GPU backend built together with CUDA: additive `cuda`/`rocm` features, run-time loading only, our own `t4a-hip-sys`/`t4a-hip-runtime`, one kernel source with vendor-prefixed C ABIs (`t4a_cuda_*` / `t4a_hip_*`), and coexisting C FFI |
| [threading-substrate.md](./threading-substrate.md) | Decision record (2026-10) | What runs underneath threadteam (Rayon now, own threads kept on `develop`), the evidence, what Rayon cannot do for HPC kernels, how the two interoperate, and what must hold before the own-thread version ships |
