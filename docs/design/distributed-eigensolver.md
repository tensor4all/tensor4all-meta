# Distributed Hermitian eigensolver: the first piece of a distributed colmat

**Status:** Proposal (2026-11-28). Nothing is implemented. The crate name, the repository and the open questions in
section 9 are not decided.
**Scope:** a distributed-memory (MPI) dense Hermitian eigensolver on top of [colmat-rs](https://github.com/tensor4all/colmat-rs)
(and later [colmat-next-rs](https://github.com/tensor4all/colmat-next-rs)), with
[sphaira-rs](https://github.com/tensor4all/sphaira-rs) as its first user.
**Checked against:** the sphaira-rs specification book
([ab-initio-common-rust-substrate](https://github.com/shinaoka/ab-initio-common-rust-substrate), `design/data-structures.md`
and `modules/generalized-eigen.md`) and the LICENSE files of the candidate libraries, read on 2026-11-28.

## 1. Decision

1. **Distributed colmat starts with the eigensolver.** [pencil-rs-adoption.md](./pencil-rs-adoption.md) §8 staged
   distributed linear algebra by difficulty (GEMM, then Cholesky and QR, eigensolvers last). This record changes the order
   of *user-visible* deliverables: the eigensolver comes first, because it is what sphaira-rs needs, and it is what HPC users
   ask for first ("can it replace ScaLAPACK/ELPA?"). It still brings only the infrastructure the eigensolver needs (layout,
   redistribution, distributed GEMM), and in that order.
2. **The crate stays outside colmat**, as pencil-rs-adoption §8 already says, so colmat gains no MPI dependency. Each rank
   uses colmat for its local tiles, on its own `threadteam` team.
3. **Port from BSD-licensed libraries only** (section 4): DLA-Future is the main source, ScaLAPACK the starting point and
   layout reference, ChASE a later accelerator for SCF. **ELPA's code is not ported** (LGPL-3.0); its papers may be read.

## 2. What sphaira-rs needs

From the specification book, at the commit sphaira-rs pins (`SPEC_VERSION`):

| Requirement | Source | Consequence for the solver |
|---|---|---|
| Generalized problem $Hc = \varepsilon Sc$, solved by diagonalizing $S = U\Lambda U^\dagger$, regularizing $\Lambda$ (rules A, B, C, D), $X = U\tilde\Lambda^{-1/2}$, then the **standard** problem $H' = X^\dagger H X$ | `generalized-eigen.md` | The core is a standard Hermitian eigensolver, called twice per k point (once for $S$, once per spin for $H'$), plus distributed GEMM for $X^\dagger H X$ and $C = XD$. No Cholesky reduction is needed. Rule C removes columns, so $n$ of $H'$ may shrink |
| Only the $n_{\mathrm{ev}}$ lowest eigenpairs of $H'$; all eigenpairs of $S$ | same | Partial-spectrum interface (index range) from the start |
| Real symmetric (Cluster, Γ only) and complex Hermitian (Band, noncollinear) | `data-structures.md` §6.4 | `f64` and `c64` first |
| $X$ cached per k point for the whole SCF; spins of one k point on the same rank group | §6.4 | The solver works on a sub-communicator (a "k group") and must not assume `MPI_COMM_WORLD` |
| Eigenvalue stage is replicated per k group today; "a distributed eigensolver later replaces it with a block-cyclic redistribution without changing the other stages" | §7.2 | Input/output via a redistribution from sphaira's row-distributed `BlockMatrix` to the solver's layout and back |
| Bitwise reproducible results for a fixed rank count; ordered reductions only, never an MPI reduction of unspecified order; agreement within tolerance across rank counts | §7.3, sphaira-rs `REPOSITORY_RULES.md` | Every reduction in the ported algorithms goes through an ordered reduction (section 6) |
| Drivers generic over a communicator trait with `SelfComm`, `SimComm` (P ranks as threads in one process) and `MpiComm` (rsmpi-rt) | §7.1, §9 | The solver is generic over the same trait, so it is tested with `SimComm` without `mpirun` |

## 3. Prior art in Rust

A search on 2026-11-28 found no Rust crate with a distributed dense eigensolver. The Rust MPI layer exists
([rsmpi](https://github.com/rsmpi/rsmpi), and tensor4all's runtime-loading fork [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt));
distributed solvers in Rust are iterative linear solvers (for example `kryst`). This would be the first.

## 4. Port sources and licences

Licences were read from each repository's LICENSE file on 2026-11-28 (GitHub API), except EigenExa.

| Library | Licence (as read) | Role here | Why |
|---|---|---|---|
| [DLA-Future](https://github.com/eth-cscs/DLA-Future) (ETH Zurich / CSCS) | **BSD-3-Clause**. Every source file carries `SPDX-License-Identifier: BSD-3-Clause`, checked in CI; no other licence found in the tree | **Main port source** | Modern C++ (templated per scalar), tile layout, two-stage reduction (full → band → tridiagonal), distributed tridiagonal divide and conquer, generalized eigensolver, GPU. Used by CP2K as an ELPA replacement |
| [ScaLAPACK](https://github.com/Reference-ScaLAPACK/scalapack) | BSD-3-style with an extra non-infringement disclaimer (GitHub reports `NOASSERTION`; the exact SPDX identifier is to be confirmed) | **Starting point and layout reference; test oracle** | One-stage path (`pdsytrd`/`pzhetrd`, `pdstedc`, `pdormtr`/`pzunmtr`) is simple to follow; its 2D block-cyclic descriptor is the de facto interchange format |
| [ChASE](https://github.com/ChASE-library/ChASE) (Forschungszentrum Jülich; GitHub is a mirror of GitLab) | **BSD-3-Clause** | **Later: SCF accelerator** | Chebyshev-filtered subspace iteration for sequences of eigenproblems; needs only HEMM, QR and Rayleigh–Ritz. Pays when $n_{\mathrm{ev}} \ll n$ and a previous solution exists; the first SCF step still needs a direct solver |
| [SLATE](https://github.com/icl-utk-edu/slate) (University of Tennessee) | **BSD-3-Clause** | Secondary reference | Same family as DLA-Future (tiles, two-stage), larger and more general; OpenMP tasks |
| [Elemental](https://github.com/elemental/Elemental) | Mostly BSD-2-Clause; `cmake/` and `external/` mix Boost, BSD-3, LGPL-2.1 and MIT (`debian/copyright`) | Not used | Per-file licence checks needed; development stopped |
| [ELPA](https://github.com/marekandreas/elpa) | **LGPL-3.0**; the LICENSE also asks that derivatives stay under LGPL and that changes go back to ELPA | **Papers only; optional backend and oracle** | Translating its code would make a derived work under LGPL |
| EigenExa (RIKEN R-CCS) | BSD-2-Clause according to its user manual; the LICENCE file in the 2.12 tarball was **not checked** | Not decided | Distributed as tarballs only; heavily tuned Fortran. More valuable as a collaboration than as a port source |

### Rules when porting BSD code

These follow [tensor4all-agent-rules `common/provenance.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/provenance.md).

- A close translation (reading the upstream source function by function) is a derived work. Keep the upstream copyright
  notice, the conditions and the disclaimer in each derived file, ship the upstream LICENSE text in the crate, and record
  repository, path and commit in a file-level comment at writing time.
- Package licence metadata: `MIT AND BSD-3-Clause` (plus ScaLAPACK's identifier if its code is translated).
- BSD-3 clause 3: the names of ETH Zurich, Forschungszentrum Jülich or the University of Tennessee must not be used to
  endorse or promote the crate. Stating "ported from DLA-Future" as a fact is fine.
- Credit beyond the licence: cite the DLA-Future, ChASE and ELPA papers (and the original algorithm papers) in the crate's
  citation policy; the user decides the wording.

## 5. Algorithm and stages

The standard eigensolver is the usual three steps: reduce to tridiagonal, solve the tridiagonal problem, back-transform.

| Stage | Content | Port source | Done when |
|---|---|---|---|
| **0. Infrastructure** | Distributed matrix on a 2D block-cyclic layout (ScaLAPACK-descriptor compatible), process grid on a sub-communicator, redistribution from and to sphaira's row-distributed blocks, distributed GEMM (SUMMA), ordered reductions | ScaLAPACK (layout), DLA-Future (`Matrix`, communication patterns) | Redistribution round-trips bitwise; GEMM matches colmat at `P = 1` and agrees within tolerance for `P = 2, 3, 4, 6` on `SimComm` |
| **1. One-stage solver** | Blocked Householder tridiagonalization; tridiagonal problem replicated on every rank of the group (O(n) data) and solved locally, each rank computing its own index range of eigenvectors; back-transformation with compact WY (BLAS 3) | ScaLAPACK one-stage path | `f64`/`c64`, full and partial spectrum; residual $\lVert HV - V\Lambda\rVert$ and orthogonality $\lVert V^\dagger V - I\rVert$ within LAPACK-test bounds; bitwise repeatable for fixed P; agrees with colmat's `eigh` at `P = 1` and with ScaLAPACK as an oracle |
| **2. Two-stage solver** | Full → band (BLAS 3), band → tridiagonal (bulge chasing), distributed tridiagonal divide and conquer, two back-transformations | DLA-Future | Faster than stage 1 above a measured size, with the same accuracy and determinism gates; benchmarks against ELPA2 and DLA-Future |
| **3. SCF accelerator** | Chebyshev-filtered subspace iteration warm-started from the previous SCF step | ChASE | Fewer seconds per SCF iteration than stage 2 on sphaira-rs target systems, falling back to stage 2 when it does not converge |

Stage 1 is deliberately the simple one: its Householder tridiagonalization is half BLAS 2 and does not scale far, but it
fixes the interfaces, the layout, the determinism rules and the tests that stage 2 must keep.

Execution model: stages 1 and 2 are first ported **bulk-synchronously** (every rank steps through the panels together).
DLA-Future's task graph with look-ahead (pika) is reproduced later, once threadteam has dependent tasks with priority
([threadteam-roadmap.md](./threadteam-roadmap.md), G13).

## 6. Determinism

- Fixed rank count and thread count ⇒ bitwise identical eigenvalues and eigenvectors.
- Every reduction that feeds a result (norms in Householder vectors, dot products, partial sums in distributed GEMM)
  goes through the communicator's ordered reduction (gather, then add in increasing rank). The ported libraries use
  `MPI_Allreduce`; each such call is replaced, and the cost is measured.
- Eigenvector phase: each eigenvector is normalized to a fixed phase convention (for example, the largest-magnitude
  component real and positive), so repeated runs and different solvers can be compared column by column outside
  degenerate subspaces. Within a degenerate subspace, tests compare subspaces, not vectors.
- Across rank counts: agreement within tolerance only, as in sphaira-rs.

## 7. Layers and dependencies

```
sphaira-rs (k groups, BlockMatrix rows)
      │  redistribution (rows ↔ 2D block-cyclic)
distributed eigensolver crate  ── generic over the communicator trait (SelfComm / SimComm / MpiComm)
      │  local tiles                      │  MPI
colmat (+ threadteam team per rank)      rsmpi-rt
```

- The layout type should become one case of pencil-rs's generalized layouts (pencil-rs-adoption §8: each axis undistributed,
  block, or block-cyclic). Stage 0 starts with a 2D block-cyclic type and keeps it convertible, rather than waiting for the
  generalization.
- No dependency on tenferro. Whether sphaira-rs calls this crate directly or through a tenferro wrapper is open (section 9).

## 8. Testing

- Most tests run on `SimComm` with `P = 1, 2, 3, 4, 6` and non-square process grids, including ranks with no local rows.
- Accuracy against colmat's `eigh` at `P = 1`; ScaLAPACK and ELPA as optional, feature-gated oracles (never default
  dependencies).
- Small fixed-seed matrices with exact or tight asserts (graded spectra, clustered and degenerate eigenvalues, the nearly
  singular $S$ of sphaira's regularization rules); no tests of tens of seconds.
- A few `MpiComm` cases under `mpirun` in a separate CI job.

## 9. Open questions

1. **Name and home.** A new repository (for example `colmat-dist-rs`), or a crate inside pencil-rs (its record says
   pencil-rs is tensor4all's distributed layer)?
2. **Where the communicator trait lives.** sphaira-rs has `sphaira-comm`; tenferro has `tenferro-mpi`; pencil-rs has its
   own. The solver and sphaira-rs must share one trait (including `SimComm`) so that sphaira's tests run the solver without
   `mpirun`.
3. **Local tridiagonal solver.** Stage 1 needs a tridiagonal solver that returns a chosen index range of eigenvectors
   (MRRR-type or bisection plus inverse iteration). Check what colmat provides and whether it is deterministic.
4. **colmat-rs or colmat-next-rs underneath.** Starting on colmat-rs gives results sooner; next's per-primitive kernel
   selection and caller-owned workspace (colmat-next-rs#45) fit a distributed driver better. Proposed: colmat-rs now,
   behind a narrow local-kernel interface that next can implement later.
5. **sphaira-rs's rule that dense linear algebra goes through tenferro-rs.** Either the distributed solver is exempt (it
   is tensor-free and below pencil-rs), or tenferro gains a thin wrapper. To be decided with sphaira-rs.
6. **ScaLAPACK's SPDX identifier and EigenExa's LICENCE file**, before any code from either is translated.
7. **GPU.** DLA-Future supports it; colmat-next is CPU-only by design. Out of scope until the CPU path passes stage 2.
