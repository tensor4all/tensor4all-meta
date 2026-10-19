# tensor4all-meta

Meta-information for the [tensor4all](https://github.com/tensor4all) project.

## Rust stack guides

Published at <https://tensor4all.org/tensor4all-meta/stack/> from [`site/`](site/)
by [`.github/workflows/pages.yml`](.github/workflows/pages.yml):

- [Stack overview](https://tensor4all.org/tensor4all-meta/stack/): layer map, repositories, release status
- [For tensor network users](https://tensor4all.org/tensor4all-meta/stack/users.html): QTT, TCI, tree tensor networks
- [For backend developers](https://tensor4all.org/tensor4all-meta/stack/backend.html): tenferro, cpueinsum, tcontract, tlinalg, strided-rs, hataori

## Design documents

- [Cross-repository design](docs/design/): decisions that span several repositories,
  such as the [tenferro core boundary and CPU execution ownership](docs/design/core-boundary-and-execution-ownership.md)
- [tenferro-rs design index](https://github.com/tensor4all/tenferro-rs/blob/main/docs/design/index.md): tenferro internals
- [Tensor4all.jl design](https://github.com/tensor4all/Tensor4all.jl/tree/main/docs/design): the Julia frontend

## Essays

Hypothetical explorations, not implementation plans:

- [Why Rust for Julia tensor network developers](docs/why_rusty_julia.md)
- [Why rewrite first-principles codes in Rust](docs/why_rusty_openmx.md), with the [OpenMX code analysis](docs/openmx_analysis.md)
