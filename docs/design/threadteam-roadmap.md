# threadteam roadmap: the own-thread version and the features it still lacks

**Status:** Proposal for review (2026-11). Nothing here is decided.
**Scope:** [threadteam-rs](https://github.com/tensor4all/threadteam-rs), the one crate in the stack that starts CPU tasks, and the crates that take its `Exec` (strided-rs, tcontract-rs, cpueinsum-rs, tlinalg-rs, tenferro-rs, t4a-faer, colmat-rs, tensor4all-rs).
**Checked against:** threadteam-rs `main` `bbf75ba` (0.6.0 plus fixes) and `develop` `41370f3`, on 2026-11-14.
**Reads first:** [threading-substrate.md](./threading-substrate.md) is the decision record: Rayon now, own threads kept on `develop`, and the gates before 0.3.0. This document is the forward plan. It does not repeat that record's evidence.

The document answers three questions for reviewers:

1. What is wrong with the own-thread version today, beyond what the decision record lists (section 2)?
2. Which threading features do well-known numerical codes rely on, and which of them does threadteam lack (sections 3 and 4)?
3. In what order should the missing pieces land, and which of them need own threads at all (section 5)?

Section 6 lists the decisions the maintainers are asked to make.

## 1. Where things stand

`main` and `develop` have split since 2026-10-29:

| | `main` | `develop` |
| --- | --- | --- |
| version | 0.6.0 (crates.io), plus fixes up to #72 | 0.2.0 of the own-thread line (`302c37d`), never published |
| substrate | private Rayon pool per width, reused across pools | `std::thread` workers in a process-wide reserve, leased per `Pool` |
| design notes | `placement.md`, `thresholds.md`, `waiting.md` | the same at their 2026-10-29 state, plus `own-threads.md` and `hybrid-scheduling.md` |
| commits since the split | 38 | 1 (the hybrid-scheduling design study, #23) |

What `main` added after the split, none of which is on `develop`:

* **Placement.** Participant pinning is on by default (0.4.0); the leader pin belongs to the `Team` (0.5.0); pinned teams spread round-robin over NUMA nodes (0.6.0, #57); `PinStrategy` is closed (#60); a nested team inside `external` places its leader (#54).
* **Entries.** `Exec::fork_sized` for uneven shares (#24) and `fork_sized_slots` with an exclusive `&mut` state slot per child (#37); `TeamExec::external` (#45).
* **Width model.** `WidthCosts` anchors to width 64 (#6), the transfer cost of a caller-resident source (#27, #29), an installable multi-node term (#68, #70), and the CPU count memoised per mint (#62).
* **Hot-path fixes.** `claim_cold` iterates claimed slots only (#71); a member's spin budget is stored only when it changes (#67, #72).

All twelve dependents were migrated to 0.4.0 (#39). So the code that downstream runs today is `main`, and `develop` is a design branch whose code is a month behind.

## 2. Problems of the own-thread version

The decision record (section 6 there) lists eight gates before 0.3.0. They still hold. The review for this document found the following on top of them, ordered by how much they block.

### 2.1 No kernel needs it yet

The decision record says a move needs "a named kernel and a measurement that the Rayon version's layer is not enough" (its section 7, question 1). None has been named since. faer#107, the problem that started the own-thread work, is fixed on `main` by reusing the Rayon pool, and the acceptance runs could not tell the two substrates apart except in the one-call acquire, where Rayon was faster (2.2 ms against 11.5 ms at width 64).

The features that do need own threads are those the decision record marks "conflicts" with Rayon:

* dependent tasks with priority (look-ahead LU, tiled Cholesky);
* a member that runs tasks while it waits, without Rayon's unrestricted helping (`hybrid-scheduling.md`, section 5);
* stealing restricted to a domain, so finished siblings' CPUs flow to the rest (`hybrid-scheduling.md`, section 6);
* a wait policy that also governs the idle workers, not only team members.

Each is reachable only through the hybrid-scheduling stages 1 to 4, and none has a measured kernel behind it. **Until one does, the own-thread version has no user, and its cost (section 2.3) buys nothing.**

### 2.2 `develop` cannot be rebased cheaply

The own-thread change touched 16 files under `src/` (+2078 / -683). `main` has since changed 18 files under `src/` (+4104 / -2123). Fourteen files are in both sets:

`awake.rs`, `exec.rs`, `ledger.rs`, `lib.rs`, `park.rs`, `pin.rs`, `policy.rs`, `pool.rs`, `reserve.rs`, `sched.rs`, `team.rs`, `worker.rs`, `tests/mod.rs`, `tests/placement.rs`.

These are the synchronization core (`awake.rs`, `park.rs`), placement (`pin.rs`) and the entries (`exec.rs`, `team.rs`). A textual rebase would be a re-implementation of the 0.3 to 0.6 placement work on the own-thread worker loop, and every conflict resolution in `awake.rs` and `park.rs` is a change to model-checked code (the repository rule: "any change to the park/wake core ... must pass the loom models").

**Proposal.** Do not rebase. Put a crate-internal substrate boundary into `main` first (section 5, stage B), so that the Rayon substrate and an own-thread substrate implement the same internal interface, and port the own-thread worker loop onto it as a second implementation behind a non-default feature. The public API then does not fork, and the two substrates can be measured from one commit.

### 2.3 Its correctness cost is high and only partly paid

From the decision record: 43 new `unsafe` blocks, functions and impls (this review counts 45 added lines containing `unsafe` in `git diff 9eeb0b4^ 302c37d -- src`), an intrusive queue, a broadcast list, five new loom models, and three deadlocks, an endless wake-up and an affinity regression found by review rather than by the models. The gates in the decision record (TLA+ model of lease, detach, batch claim and broadcast; shuttle; Miri in CI; optionally Verus for the slot state word) are not started.

The hybrid-scheduling study adds a per-slot deque, domains, a descendant rule, a draining barrier and prepared graphs on top of that loop (its invariants I1 to I10). The verification cost grows with each stage; it should be planned per stage, not once for 0.3.0.

### 2.4 `external_rayon` oversubscribes the pool's CPUs

On `develop`, `Exec::external_rayon` runs its callback on `pool.rayon()`, a private Rayon pool built lazily with `num_threads(self.width.get())` (`src/pool.rs:105`, `src/exec.rs:702`). Two defects:

* It is sized by the **pool's** width, not the token's budget, so a token of budget 4 on a pool of 64 gets a 64-thread Rayon pool (already noted in `hybrid-scheduling.md`, section 14, question 4).
* Those threads are **in addition** to the pool's own workers, on the same CPU set. While a callback runs, the pool's idle workers spin for up to `spin_polls` and the Rayon workers search for 32 rounds, so the CPU set briefly carries up to twice its width. That is the same oversubscription the two-pool option must avoid by phase exclusion (`hybrid-scheduling.md`, section 11, item 1), here inside one pool.

Hosting the Rayon pool on leased threads (`spawn_handler`, decision record section 5.1) fixes both, and it is not built.

### 2.5 The one-call acquire is not explained

The acquire phase of a one-call pool at width 64 is 11.5 ms on own threads against 2.2 ms on the reused Rayon pool (decision record, section 3). The cause has not been looked for. A one-call pool is the shape faer's `with_team` used; if a downstream crate builds pools per call again, the own-thread version is the slower one.

### 2.6 Process-wide state on both substrates

Both substrates keep idle threads process-wide (a reserve, or a per-width cache of Rayon pools). Whether that is compatible with "no implicit global pool" is an open reading (decision record, section 7, question 2). Section 3 below shows that every production runtime surveyed keeps such a process-wide set, and that the problems it causes in practice are oversubscription between runtimes and fork safety, not the cache itself. Those two problems are not handled on either substrate (section 4).

## 3. How well-known numerical codes use threads

The survey covered the projects below. Each row names what the project relies on and the lesson for threadteam. **[v]** marks a claim checked against the linked source or documentation for this document; everything else is from the surveyor's knowledge and is listed in section 7.

| project | what it relies on | lesson for threadteam |
| --- | --- | --- |
| [BLIS](https://github.com/flame/blis/blob/master/docs/Multithreading.md) | One team per call, split into nested sub-teams that follow the GEMM loops (`BLIS_JC_NT`, `IC`, `JR`, `IR`; the `pc` loop is never split because it would need a reduction) **[v]**. Per-call settings in a `rntm_t` value, which override the global ones and are thread-safe **[v]**. Backend chosen at run time (`openmp`, `pthreads`, `hpx`, `single`) **[v]**; pinning left to the OpenMP runtime **[v]**. Internally a sub-team communicator with a barrier and a broadcast of the packed-buffer pointer. | Hierarchical sub-teams with their own barriers plus a **one-value broadcast inside a team** are enough for a fast GEMM. Configuration travels with the call, as threadteam's `Exec` already does. |
| [OpenBLAS](http://www.openmathlib.org/OpenBLAS/docs/faq/) | Its own pthread server, or an OpenMP build. Inside an OpenMP region it runs on one thread; without the OpenMP build, calls from OpenMP regions risk deadlock **[v]**. Idle workers busy-wait; the FAQ advises `OMP_WAIT_POLICY=passive` and a shorter `OPENBLAS_THREAD_TIMEOUT`, or both pools "hog the cpu" **[v]**. `openblas_set_num_threads_local` since 0.3.27 **[v]**. | "Am I already inside a parallel region?" decides nested calls. Two spinning pools cost CPU at every hand-off. |
| [oneMKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/techniques-to-set-the-number-of-threads.html) | Thread-local count `mkl_set_num_threads_local`; global setters change in-flight calls and are reported as races **[v]**. Threading layer chosen at link or load time (Intel OpenMP, GNU OpenMP, TBB, sequential). [CNR](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/obtaining-numerically-reproducible-results.html): bitwise reproducibility needs a constant thread count and aligned memory; a strict mode drops the thread-count condition for some routines at up to 2x cost **[v]**. | Per-call or thread-local counts, never global setters. Reproducibility is a property the runtime has to offer explicitly. |
| [PLASMA](https://www.netlib.org/utk/people/JackDongarra/PAPERS/toms-plasma-2019.pdf), SLATE, Chameleon/StarPU | Tile algorithms as OpenMP tasks with `depend`; task `priority` on the panel closes much of the gap to MKL for LU **[v]**, and priority alone does not deepen look-ahead ([HPC Asia 2020 poster](https://sighpc.ipsj.or.jp/HPCAsia2020/hpcasia2020_posters/poster_12.pdf)) **[v]**. SLATE runs a multithreaded panel as a nested parallel region inside one task and sets BLAS to one thread for tile kernels. StarPU infers dependencies from data access modes. | Dense factorizations need dependencies, priority and explicit look-ahead, **and a team inside a task** for the panel. |
| [FFTW](https://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html) | `fftw_plan_with_nthreads`; `fftw_threads_set_callback(parallel_loop, data)` lets the host run FFTW's `njobs` jobs on its own threads **[v]**. Plan execution is thread-safe, plan creation is not **[v]**. | The cleanest interop seam found: a flat "run these `n` jobs" hook that the host implements. threadteam can implement it in a few lines over `for_ranges`. |
| [PyTorch ATen](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Parallel-inl.h) | `parallel_for(begin, end, grain, f)` runs serially when the range fits one grain, when already in a parallel region, or with one thread **[v]**. The OpenMP backend splits statically into `min(team, ceil(n / grain))` contiguous blocks and avoids the `num_threads` clause because of a libgomp bug ([ParallelOpenMP.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/ParallelOpenMP.h)) **[v]**. The first exception is kept and rethrown after the join **[v]**. `parallel_reduce` keeps one partial per thread and combines them in thread order **[v]**. Separate intra-op and inter-op pools. | Nested calls run serially; grain caps how many members wake; a panic reaches the caller; a fixed-order reduction is cheap to offer. |
| [Eigen NonBlockingThreadPool](https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/ThreadPool/NonBlockingThreadPool.h) (TensorFlow) | A bounded deque per worker; a full deque runs the task inline; steal within a partition, then globally; **at most one spinning thread**, the others park on an event count; `SetStealPartitions` limits stealing to, for example, a NUMA node; `CurrentThreadId()` is -1 outside the pool **[v]**. | Even a general pool caps spinning threads and offers NUMA-scoped stealing. "Which pool am I a worker of" is a core primitive. |
| [LAMMPS OPENMP](https://docs.lammps.org/Speed_omp.html) | Ranks times threads at most the physical cores; bind ranks to sockets, not cores **[v]**. Per-thread force arrays and accumulators, an explicit reduction over them, optional per-thread timers ([`thr_data.h`](https://github.com/lammps/lammps/blob/develop/src/OPENMP/thr_data.h)) **[v]**. | Scatter-add kernels use thread-private buffers plus an ordered reduction, not atomics. Per-thread timing is built in. |
| [GROMACS](https://manual.gromacs.org/current/user-guide/mdrun-performance.html) | Thread-MPI plus OpenMP; `-pin auto/on/inherit/off`, where `inherit` pins only inside the mask a launcher set; `-pinoffset` and `-pinstride` so several runs share a node; without pinning performance "will degrade dramatically" **[v]**. | Pinning has to respect an inherited mask, and several processes on one node need an offset. |
| Julia | Task-based; `@threads :static` is not allowed in nested tasks; tasks migrate, so thread-indexed scratch is wrong; BLAS has its own pool and is usually set to one thread under Julia threads. | Stable member numbers are a team property, not a thread property. Relevant to Tensor4all.jl, which calls into the stack from Julia threads. |

Knobs that applications set in practice: `OMP_PROC_BIND` and `OMP_PLACES` (BLIS recommends its OpenMP backend for them **[v]**); `OMP_WAIT_POLICY`, `KMP_BLOCKTIME`, `GOMP_SPINCOUNT`; `OMP_MAX_ACTIVE_LEVELS`, whose default of no nesting is why BLAS inside a parallel region runs serially; and, in Python, [`threadpoolctl`](https://github.com/joblib/threadpoolctl), which finds the BLAS and OpenMP pools loaded in a process and limits them. Known failure modes: libgomp and libiomp loaded together, a child that hangs after `fork()` in a parent that used OpenMP, and quadratic oversubscription when parallel tasks each enter a parallel library (the SciPy 2016 and 2018 papers, already summarized in `hybrid-scheduling.md`, section 11, item 6).

Three observations from the survey bear on the substrate question:

1. **Most production users of OpenMP use only what threadteam already has.** BLIS, PyTorch, LAMMPS and GROMACS use static teams, sub-teams, barriers, a grain, pinning and a wait policy. They do not use tasks.
2. **The users of tasks are the tile factorizations** (PLASMA, SLATE, Chameleon). That is the one class of kernel that needs what Rayon "conflicts" with: dependencies plus priority, with a team inside a task.
3. **The recurring operational problems are between runtimes**, not inside one: nested calls, two spinning pools, fork, and processes that share a node. Neither substrate addresses them today.

## 4. Features threadteam lacks

Status on `main` at `bbf75ba`. "Substrate" says whether the feature needs the own-thread version; "any" means it can land on the Rayon version now. Priority 1 is proposed for the next releases, 2 for after that, 3 when a user asks.

| # | feature | used by | on `main` | substrate | priority |
| --- | --- | --- | --- | --- | --- |
| G1 | **"Which pool, and which member, is this thread"** query, and a documented policy for nested calls (serial, or a sub-team of the current team) | OpenBLAS, MKL, PyTorch, Eigen | missing. `Exec::carries_team` says whether a token holds a team, not where the thread is. tensor4all-rs tests `rayon::current_thread_index()` instead (decision record, section 5.3), which breaks on own threads. Decision record gate 6. | any | 1 |
| G2 | **Fork safety**: a child process after `fork()` must not wait for workers that do not exist in it | OpenBLAS and PyTorch (`pthread_atfork`, from the surveyor's knowledge) | missing. The Rayon pool cache and the own-thread reserve both hold threads that do not survive `fork()`. What a `Pool` built before the fork does when the child uses it is untested; a wait for members that do not exist is the expected failure. Python `multiprocessing` with the fork start method and Julia `Distributed` on some setups reach this. | any; the reserve makes it more urgent | 1 |
| G3 | **Reproducible reduction** entry: fixed chunks by data size, combined through a fixed tree | MKL CNR, PyTorch `parallel_reduce`, LAMMPS | missing. Designed in `hybrid-scheduling.md`, section 8, independent of the substrate. Needed by any test that compares bits across runs, and by colmat's double-double accumulations. | any | 1 |
| G4 | **One-value broadcast and `single` inside a team** (a leader publishes a pointer, members read it after the barrier) | BLIS `thrcomm` | partial. `Participant::index` and `barrier` let a kernel build it with its own cell. A primitive removes the `unsafe` each kernel writes for it. | any | 2 |
| G5 | **Process-wide admission**: a cap on threads running across pools, or an exclusive mode | Intel OpenMP `KMP_COMPOSABILITY`, TBB IPC server, LAMMPS and GROMACS rules for ranks times threads | partial. A root scope that has more participants than CPUs paces its waits (`pin::oversubscribed`), but two `Pool`s of width 64 on 64 CPUs still run 128 threads. | any | 2 |
| G6 | **Control of foreign pools** (vendor BLAS to one thread or to the share's width inside a team) | SLATE, `threadpoolctl`, MKL and OpenBLAS `_local` setters | missing, and probably not threadteam's job: t4a-faer's vendor path and the engine own those calls (threadteam-rs#42). threadteam can expose the width a share has, which it does (`Exec::budget`). | engine | 2 |
| G7 | **Placement below NUMA**: places by L3 domain, core and SMT sibling; offset and stride for processes that share a node | OpenMP places, GROMACS `-pinoffset` and `-pinstride` | partial. `PinStrategy::{Spread, Pack}` lays a team over NUMA nodes inside the inherited mask, which is GROMACS's `inherit`. No L3 or SMT level: on a 64-core Zen 2 part the L3 domain is four cores (decision record, item 5). An offset is expressible only by the caller narrowing the mask. | any | 2 |
| G8 | **Dynamic or guided work-sharing inside a team** (an atomic chunk counter shared by the members) | OpenMP `schedule(dynamic)`, Eigen | missing as an entry; kernels write it with an atomic. `for_ranges` is static. | any | 3 |
| G9 | **FFTW-style callback adapter** (C ABI `parallel_loop(work, data, elsize, njobs)`) | FFTW, and libraries with the same hook | missing. A thin function over `for_ranges`; belongs in a C API crate, not in threadteam itself. | any | 3 |
| G10 | **Pause and resume**: release every idle thread, for example before `fork()` or to hand CPUs to another runtime | `omp_pause_resource` | partial. `Pool::release_idle` parks idle members. Nothing ends the cached or reserved threads, so a process cannot give its threads back without exiting. | any | 3 |
| G11 | **Production hooks**: region start and end, barrier, park, with per-member timing | OMPT, LAMMPS timers | partial. `trace` (a test feature), `WaitCounters`, `pin_skips`. No hook a profiler can attach to in a release build. | any | 3 |
| G12 | **A cap on spinning threads** | Eigen (one spinner) | missing. Every idle member spins its `spin_polls`. Relevant on shared hosts, where faer#107 came from. | any | 3 |
| G13 | **Dependent tasks with priority, replayed as a prepared graph** | PLASMA, SLATE, StarPU | missing. `hybrid-scheduling.md`, stage 4. | own threads, or the two-pool baseline with hand-built counters on Rayon `scope` | gated by a kernel (section 5, stage C) |
| G14 | **Stealing within a domain, and a member that runs descendants while it waits** | Rayon, oneTBB `isolate`, Eigen partitions | missing. `hybrid-scheduling.md`, stages 1 to 3. | own threads | gated by a kernel |

What `main` already has that the surveyed libraries treat as essential, for completeness: sub-teams with barriers (`TeamExec::split`), a per-call configuration value (`Exec`, the shape BLIS and MKL moved to), a grain that caps the woken members (`for_ranges`), panic propagation from team sections (`team.rs`, `catch_unwind` and `resume_unwind`), a team inside a task (any child `Exec` may call `team`), pinning inside the inherited mask with a NUMA spread, a per-pool and per-team wait policy (`spin_polls`, `WaitOverride`), and a width chosen from a cost model (`Exec::width_for`), which goes further than PyTorch's grain or BLIS's automatic factoring.

Configuration from environment variables (an `OMP_NUM_THREADS` analogue) is deliberately absent: the caller owns the threads (core-boundary document, section 1), so a process default belongs to the top-level engine, as tensor4all-rs#1054 proposes for the wait policy.

## 5. Proposed order of work

The order puts what any substrate needs first, and keeps the own-thread version on a track where it is measured against `main` from one commit instead of a month-old branch.

### Stage A: on `main`, independent of the substrate (now)

1. G1, the "which pool and member" query, then replace the `rayon::current_thread_index()` guard in tensor4all-rs with it (decision record gates 6 and 7).
2. G2, fork safety: mark pools and the cache or reserve invalid in the child, make every entry in the child run serially or return `Refused`, and test it with a forking test.
3. G3, `Exec::reduce_reproducible` with the API of `hybrid-scheduling.md`, section 8.
4. The wait-policy layering of decision record gate 8, and the narrowing of the unused tuning surface (threadteam-rs#4), so that the public API is small before a second substrate has to implement it.

### Stage B: a substrate boundary inside the crate

1. Separate, inside threadteam, the code that depends on the thread provider (worker spawn and loop, park and wake of idle workers, `external`) from the code that does not (tokens, splits, team sections, barriers, placement, cost model). The boundary is crate-internal; no new public type.
2. Port the own-thread worker loop from `develop` onto that boundary as a second implementation behind a non-default feature. `develop` is then retired, and its design notes move to `main`.
3. Run decision record gates 1 to 5 (one-call acquire, TLA+, shuttle, Miri, optionally Verus) on the feature build, and fix `external_rayon` (section 2.4) by hosting the Rayon pool on leased threads or by removing it.
4. Run the acceptance criteria (a) to (d) of the decision record on both builds from the same commit.

**Exit:** the own-thread build meets the criteria and the gates, or the stage stops and the feature is removed.

### Stage C: name the kernel that needs tasks

1. Pick one tile factorization as the first consumer: tiled Cholesky or look-ahead LU in tlinalg-rs or colmat-rs.
2. Build it on the two-pool baseline (`hybrid-scheduling.md`, stage 0: Rayon `scope` with hand-built dependency counters, a threadteam team for the panel).
3. Measure it against vendor LAPACK and against a level-synchronous version (`hybrid-scheduling.md`, section 13, measurement 5).

**Exit:** a measured gap that priority or domain stealing would close. Without it, stages D and E do not start, and G13 and G14 stay unbuilt.

### Stage D: switch the default substrate

Only when stage B exited with the gates met and stage C showed a gain. The release number follows `main`; the "0.3.0" in the decision record is stale, because `main` used 0.3.0 to 0.6.0 for the Rayon line.

### Stage E: hybrid scheduling

`hybrid-scheduling.md`, stages 1 to 5, each against the kernel named in stage C and against the two-pool baseline, with the per-stage verification of section 2.3.

### Independent of the stages

G4, G5, G7 to G12 can land on whichever substrate is current, when a user asks. G6 is for the engine and t4a-faer, not threadteam.

## 6. Decisions requested

1. **Is the own-thread version still the destination** (decision record, section 1: "very likely the future")? Or is the two-pool engine acceptable as the long-term shape (`hybrid-scheduling.md`, section 14, question 1), with threadteam staying on Rayon? This document's stages A to C are useful under either answer.
2. **Stage B instead of a rebase of `develop`.** Accept the crate-internal substrate boundary, and retire `develop` once its code is ported.
3. **The first task kernel** for stage C, and who owns it (tlinalg-rs or colmat-rs).
4. **Fork policy** (G2): in the child, entries run serially, or the pool is rebuilt lazily on first use?
5. **Admission control** (G5): a process-wide cap inside threadteam, or a rule for engines ("one engine per CPU set")? The second keeps threadteam free of process-wide state beyond the thread cache, which bears on decision record section 7, question 2.
6. **Foreign BLAS control** (G6): confirm that it belongs to the engine and t4a-faer, not threadteam.

## 7. Not verified

* The survey rows without **[v]** in section 3 are from the surveyor's knowledge and were not re-read: BLIS's internal communicator, OpenBLAS's `NUM_PARALLEL` and `pthread_atfork` handler, `MKL_DYNAMIC` behaviour and the spelling of `MKL_CBWR` strict mode, SLATE's nested panel and BLAS thread setting, StarPU and Chameleon details, PyTorch's `pthread_atfork` reset, Eigen's `setNbThreads` scope, GROMACS's use of `schedule(static)` and its back-off on `KMP_AFFINITY`, the Quantum ESPRESSO, VASP and CP2K remarks, the default values of `KMP_BLOCKTIME` and `GOMP_SPINCOUNT`, the real-world use of `omp_pause_resource`, the libgomp and libiomp coexistence and fork-hang reports, and every Julia statement.
* "Most production users of OpenMP do not use tasks" (section 3) is a reading of the projects surveyed, not a census.
* That Python `multiprocessing` or Julia `Distributed` reaches a `fork()` after a pool was used (G2) is not reproduced; a forking test is the first step of stage A, item 2.
* The size of the stage B boundary (section 5) is not estimated; the file overlap in section 2.2 suggests it touches most of `pool.rs`, `awake.rs` and `exec.rs`.
* The survey was done by an AI research agent in one session; the `[v]` marks mean it opened the source, not that a maintainer did.
