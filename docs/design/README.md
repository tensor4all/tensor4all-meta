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
| [metal-backend.md](./metal-backend.md) | Detailed design, proposed (2026-11) | Apple GPU backend over a vendor-neutral shared allocation domain: f32/c32 on Metal with explicit `Unsupported` for f64, the paired CPU engine reading the same tensors with automatic waiting, run-time loading only (`t4a-metal-sys`/`t4a-metal-runtime`), MSL sources in t4a-gpu-kernels, MPS GEMM |
| [next-generation-dense-linear-algebra.md](./next-generation-dense-linear-algebra.md) | Design under review (2026-11) | colmat's double-double design, including independent kernel specialisations, SIMD, caller-owned teams and workspace, and the questions to resolve before implementation |
| [pencil-rs-adoption.md](./pencil-rs-adoption.md) | Decision record (2026-11) | pencil-rs joins tensor4all and becomes its distributed layer, depending on tenferro (pencil-rs → tenferro); the plan for removing duplicate implementations by extending tenferro-fft, tenferro-hdf5 and tenferro-mpi, the stages and gates, and the longer-term plan for distributed linear algebra on colmat |
| [rocm-backend.md](./rocm-backend.md) | Detailed design, proposed (2026-11) | AMD GPU backend built together with CUDA: additive `cuda`/`rocm` features, run-time loading only, our own `t4a-hip-sys`/`t4a-hip-runtime`, one kernel source with vendor-prefixed C ABIs (`t4a_cuda_*` / `t4a_hip_*`), and coexisting C FFI |
| [symmetric-tensors-and-symtn.md](./symmetric-tensors-and-symtn.md) | Decision record (2026-11) | tensor4all-rs has no quantum numbers; symtn owns symmetry; tensor4all provides symmetry-agnostic kernels (irregular-batch GEMM, batched permute and factorization) outside the tenferro core, on dense payloads that tenferro's AD differentiates; candidates from symtn, lessons from the ITensor rewrite, and symtn's tenferro workarounds that can now go |
| [tenferro-sci.md](./tenferro-sci.md) | Proposed (2026-11) | A SciPy-like algorithm collection on tenferro: repository `tenferro-sci-rs`, one crate per module (`tenferro-krylov`, `-optimize`, `-special`, `-integrate`, `-interpolate`) released at one version, independent algorithm cores such as `krylov` wrapped with tenferro tensors and AD rules, validation against SciPy |
| [threading-substrate.md](./threading-substrate.md) | Decision record (2026-10) | What runs underneath threadteam (Rayon now, own threads kept on `develop`), the evidence, what Rayon cannot do for HPC kernels, how the two interoperate, and what must hold before the own-thread version ships |
| [threadteam-roadmap.md](./threadteam-roadmap.md) | Proposal for review (2026-11) | Problems of the own-thread version beyond the decision record, a survey of how BLIS, OpenBLAS, MKL, PLASMA, FFTW, PyTorch, Eigen, LAMMPS and GROMACS use threads, the features threadteam lacks, and a staged plan |
