# Why pure Rust

**Status:** position paper (2026-11). Numbers are quoted from the benchmark
and design documents linked next to them, with the hardware and the commit
they were measured at. Re-check them before quoting them elsewhere.

The tenferro stack (strided-rs, threadteam, colmat, tcontract, tlinalg,
tenferro-rs, tenferro-sci-rs) keeps its numerical kernels in Rust instead of
calling C, C++ or Fortran libraries such as MKL, OpenBLAS, AOCL or LAPACK.
This note explains why. It does not argue for Rust over Julia, C++ or Python;
for that, see [why_rusty_julia.md](why_rusty_julia.md) and
[why_rusty_openmx.md](why_rusty_openmx.md).

**In short:** owning the kernels lets us control performance, placement and
verification across the whole stack. The price is the work of writing and
validating them. That work has been done for dense linear algebra, and colmat
measures on par with vendor libraries on the hardware we have tested.

## 1. What "pure Rust" means here

- **Build time:** `cargo` alone. No C, C++ or Fortran is compiled, and there is
  no `build.rs` that compiles native code, no CMake and no pkg-config.
- **Run time exceptions:** vendor libraries that only a vendor can provide are
  loaded at run time, not linked at build time: CUDA, Metal, HDF5 and MPI
  (for example [hdf5-rt](https://github.com/tensor4all/hdf5-rt) and
  [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt)). GPU kernel source
  compiled at run time (NVRTC) may include upstream C++ headers, such as
  SciPy's xsf.
- **Vendor CPU libraries stay optional:** they are benchmark baselines and
  migration targets, not dependencies.

## 2. What we do not lose

The usual objection is that hand-written kernels cannot match decades of
vendor tuning, and cannot be trusted as much as LAPACK. colmat, the stack's
dense linear algebra (derived from [faer](https://codeberg.org/sarah-quinones/faer)),
is the test of that objection.

**Speed.** From [colmat-benchmark](https://tensor4all.org/colmat-benchmark/) at
colmat `93da621` (0.34.0), f64/c64, sizes from 32 up to 768, 1 to 8 threads, on
a Zen 5 host (Ryzen AI 9 HX 470, 12 cores; marked provisional), 720 cells per
vendor:

| against | colmat faster | median vendor time / colmat time |
| --- | --- | --- |
| Intel MKL 2026.2 | 615 / 720 (85%) | 1.32 |
| AMD AOCL 6.0.0 | 392 / 720 (54%) | 1.01 |
| OpenBLAS 0.3.32 | 620 / 720 (86%) | 1.23 |

On Zen 2 (EPYC 7702P, 64 cores, MKL only, 16 to 60 threads) colmat wins
141 of 180 cells. Where it loses is recorded as well:

- small `eigh`, `qr` and multi-threaded small `gemm` against MKL;
- small and tall `gemm` against AOCL;
- Intel Cascade Lake (Xeon Platinum 8280), where MKL wins most cells and `lu` and
  `qr_factor` are the gap (a short run, below the timing baseline).

colmat's own goal is parity plus control, not to beat vendor libraries
everywhere.

**Correctness.** colmat runs a Rust adaptation of the LAPACK 3.12.0 test suite
over 16 routine families every night
([lapack-validation.md](https://github.com/tensor4all/colmat-rs/blob/main/docs/lapack-validation.md)).
Results are bitwise reproducible on the same machine and build at a fixed
thread count
([reproducibility.md](https://github.com/tensor4all/colmat-rs/blob/main/docs/reproducibility.md)).
Per-routine tests check that steady-state calls do not allocate.

## 3. What we gain

### 3.1 One owner of the threads

A vendor library keeps its own thread pool. When a caller is itself parallel,
the two pools compete:

- OpenBLAS inside an OpenMP region runs on one thread, and may deadlock without
  its OpenMP build;
- two spinning pools "hog the CPU" at every hand-off;
- parallel tasks that each enter a parallel library oversubscribe the machine
  quadratically.

[threadteam-roadmap.md](design/threadteam-roadmap.md), section 3, surveys these
failure modes in BLIS, OpenBLAS, oneMKL, PLASMA, FFTW, PyTorch, LAMMPS and
GROMACS.

In the tenferro stack, threadteam is the one crate that starts CPU threads.
Every kernel above it takes the caller's execution token (`Exec`): strided,
tcontract, tlinalg, colmat and tenferro. So a contraction in tcontract that
calls GEMM in colmat runs on the same team, with the same placement. This
holds for every routine because the kernels are ours. It cannot be had for
a vendor library, short of the per-call hooks a few of them offer (FFTW's
thread callback, MKL's thread-local count).

Placement matters, measured in
[threadteam placement.md](https://github.com/tensor4all/threadteam-rs/blob/main/docs/design/placement.md):

- **Pinning removes slow runs.** On EPYC 7702P, colmat `qr_factor` f64 n=512 at
  16 threads had slow processes in 11 of 20 runs unpinned and 0 of 20 pinned.
- **The leader's placement counts.** On a Ryzen AI MAX+ 395, LU n=512 took
  231 ms when only the members were pinned, against 0.9 ms when the leader was
  placed too.
- **NUMA placement is visible.** On a 2-socket Xeon Platinum 9242 (4 NUMA
  nodes), reported in threadteam#57, packed placement was 1.57x slower than
  unpinned on fresh memory and 1.33x faster on warm memory.
- **Holding a team is cheap; making one is not.** On EPYC 7702P at width 64, a
  new pool per call cost 15.4 ms, against 0.02 to 0.04 ms with a held team
  ([threading-substrate.md](design/threading-substrate.md)).

**Not measured yet:** an end-to-end case where nesting (tcontract calling GEMM
on one team), or a run on a multi-socket node, beats the same work on a vendor
library with its own pool. The tcontract campaigns so far are single-socket, at
1 and 4 threads. A multi-socket comparison at 1, 2 sockets and several thread
counts is the next measurement this note needs.

### 3.2 Reproducibility and no hidden state

Bitwise reproducibility at a fixed thread count, no allocation in steady state
and no implicit global pool are properties of the code we own, and are tested
there. With a vendor library, they hold only as far as its documentation says.
For example, MKL needs its CNR mode for reproducibility and charges up to 2x
in its strict mode.

### 3.3 Distribution

A pure Rust crate builds with `cargo` alone on every target Rust supports. The
same code runs on x86 Linux servers and on Apple Silicon, where t4a-gemm has
SME kernels. It goes into Python wheels
and Julia JLL artifacts without a system BLAS, and two installed copies do not
fight over `libgomp` and `libiomp`.

### 3.4 Verification at every layer

Because every layer is Rust in one workspace graph, the same checks apply at
every level:

- reference values from LAPACK, mpmath and SciPy, with a stated tolerance;
- autodiff checked against finite differences;
- allocation counts;
- loom models for the threading core.

A bug can be followed from a tenferro operation down to a colmat kernel
without crossing a language or a binary boundary.

### 3.5 Maintenance by people and agents

The C and Fortran boundary was the expensive part of earlier two-language
stacks: build systems, ABI details and knowledge held by one maintainer (the
bus factor; see [why_rusty_julia.md](why_rusty_julia.md)). Without it, a
contributor or a coding agent can read, change and test the whole path from the
user-facing operation to the kernel with one toolchain. The compiler and the
test suites check that work.

## 4. What it costs, and how the cost is paid

- **Porting effort.** LAPACK-level algorithms, special functions (SciPy's xsf)
  and quadrature rules have to be written or ported. We start from proven code
  (faer, xsf, LAPACK's own tests) and port the upstream tests before the
  implementation.
- **Risk of porting errors.** New code has not been run by many users. This is
  answered by validation against references (section 2) and by public
  benchmarks keyed to commits and hardware. Defects found in upstream libraries
  are recorded with their evidence.
- **Gaps that remain.** Some shapes and hosts still lose (section 2). These are
  tracked as issues in colmat-rs and colmat-benchmark, not hidden.

## 5. Where the stack stands

| layer | crate | pure Rust | notes |
| --- | --- | --- | --- |
| strided views and kernels | strided-rs | yes | |
| threads | threadteam | yes | one owner of CPU threads |
| dense linear algebra | colmat (t4a-gemm for GEMM) | yes | vendor libraries only as benchmark baselines |
| contraction | tcontract | yes | TAPP C ABI offered for other languages |
| factorizations for tensors | tlinalg | yes | on colmat |
| tensors, autodiff | tenferro-rs | yes on CPU | CUDA and Metal loaded at run time; XLA through a PJRT plugin loaded at run time |
| numerical algorithms | tenferro-sci-rs (planned) | yes | independent pure Rust cores; xsf headers only in GPU kernels |
