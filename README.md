# tensor4all-meta

Meta-information for the [tensor4all](https://github.com/tensor4all) project.

## Rust stack guides

Published at <https://tensor4all.org/tensor4all-meta/stack/> from [`site/`](site/)
by [`.github/workflows/pages.yml`](.github/workflows/pages.yml):

- [Stack overview](https://tensor4all.org/tensor4all-meta/stack/): layer map, repositories, release status
- [For tensor network users](https://tensor4all.org/tensor4all-meta/stack/users.html): QTT, TCI, tree tensor networks
- [For backend developers](https://tensor4all.org/tensor4all-meta/stack/backend.html): tenferro, cpueinsum, tcontract, tlinalg, strided-rs, hataori

## Logos

Published at <https://tensor4all.org/tensor4all-meta/logos/> from [`site/logos/`](site/logos/), with the files under
[`branding/`](branding/): the [tensor4all logo](branding/logo/README.md) and the [tenferro logo](branding/tenferro/README.md),
both proposals.

## Contribution rules

- [Contribution rules for the Rust repositories](docs/contributing/rust-repositories.md): the rules that hold in every
  Rust repository here — issue-first and claimed work, one decision per pull request, evidence in the body, red-first
  tests, the gates each repository runs, the review requirement, and the version policy. A repository's own
  `CONTRIBUTING.md` records only its local facts (MSRV, feature matrix, platforms) and points at that page.
- [`scripts/check-contribution.py`](scripts/check-contribution.py) reports which repositories carry the pointer.
  Forks are exempt: an upstream project's own contribution rules apply there.

## Design documents

- [Cross-repository design](docs/design/): decisions that span several repositories,
  such as the [tenferro core boundary and CPU execution ownership](docs/design/core-boundary-and-execution-ownership.md)
- [tenferro-rs design index](https://github.com/tensor4all/tenferro-rs/blob/main/docs/design/index.md): tenferro internals
- [Tensor4all.jl design](https://github.com/tensor4all/Tensor4all.jl/tree/main/docs/design): the Julia frontend

## Benchmarks

- [Benchmarks of the tensor4all stack](docs/benchmarks/README.md): the campaigns and their sites, the weekly full CPU run and its
  [agent procedure](skills/weekly-cpu-benchmark/SKILL.md), and how to recreate the monitoring cron

## Essays

Hypothetical explorations, not implementation plans:

- [Why Rust for Julia tensor network developers](docs/why_rusty_julia.md)
- [Why rewrite first-principles codes in Rust](docs/why_rusty_openmx.md), with the [OpenMX code analysis](docs/openmx_analysis.md)
