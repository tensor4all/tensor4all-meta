<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to <repository>

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `<repository>`:

## Local facts

- **MSRV**: <!-- the rust-version in Cargo.toml, and the toolchain CI pins -->
- **Feature combinations CI builds**: <!-- e.g. default, no_std, and the BLAS injection matrix -->
- **Platforms claimed**: <!-- the targets that are tested, and the ones that are only built -->
- **Local gate**: <!-- the command that runs what CI runs, e.g. `cargo test --workspace` -->

## Anything here that departs from the shared rules

<!-- If this repository cannot follow a rule above, say which one and why. Silence means it follows them. -->
