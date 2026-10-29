# The threading substrate under threadteam

Status: Decision record (2026-10). Scope: [threadteam](https://github.com/tensor4all/threadteam-rs), the only crate in the tensor4all stack that starts tasks, and the choice of what runs underneath it. Read [core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md) first for who owns threads; this record covers which threads they are.

## 1. Decision

Maintainer, 2026-10-30: the own-thread version of threadteam is very likely the future, and the timing is not settled.

* threadteam `main` is the Rayon version, released as 0.2.0. It runs on a private Rayon pool, waits the way libgomp does (`PoolBuilder::spin_polls`, a poll count like `GOMP_SPINCOUNT`), and keeps a process-wide, per-width cache of the private Rayon pool, so a pool built per call reuses running threads ([threadteam-rs#21](https://github.com/tensor4all/threadteam-rs/pull/21)).
* The own-thread version (`std::thread` workers in a process-wide reserve) is kept on the [`develop` branch](https://github.com/tensor4all/threadteam-rs/tree/develop). If it ships, it ships as 0.3.0 ([#16](https://github.com/tensor4all/threadteam-rs/pull/16), the design in [own-threads.md](https://github.com/tensor4all/threadteam-rs/blob/develop/docs/design/own-threads.md)).
* The team keeps publishing the Rayon version while it works out what Rayon cannot do for HPC kernels (sections 4 and 7).

One rule holds on both versions: **threadteam is the one owner of CPU threads in the stack, and the pools are owned by the caller.** No other crate starts a task. What runs underneath threadteam, Rayon or its own threads, is an implementation choice behind the public API. The API names no Rayon type by default (`scripts/check-no-public-rayon.sh` in threadteam-rs enforces it), so the choice can change without a source change in callers.

## 2. The problem the current version solves

[tensor4all/faer#107](https://github.com/tensor4all/faer/issues/107) reported a per-process slow mode: some processes ran every repetition of a kernel at about the one-thread time, for the whole life of the process.

Cause, from the acceptance runs on a 64-core EPYC 7702P host (Zen 2) with other users' processes on it:

* faer's [`with_team`](https://github.com/tensor4all/faer/blob/main/faer/src/utils/team.rs) builds a `threadteam::Pool` per call, so it starts a set of fresh threads per call (64 at width 64).
* Under foreign load the scheduler places a new thread before it knows its load, so several of them start stacked on one CPU. The team's spinning waiters then keep every CPU busy, no idle CPU pulls a stacked thread away, and each short section of the call waits for a descheduled member.
* A pool that is held across calls does not show the mode: in the same run, `into_team_per_call` (pool held, members reserved per call) took 0.03 ms where `fresh_pool` (a new pool per call) took 15.99 ms ([own-threads.md, "Why: the one-call pool"](https://github.com/tensor4all/threadteam-rs/blob/develop/docs/design/own-threads.md)).

Tuning the waits did not remove it. [#14](https://github.com/tensor4all/threadteam-rs/pull/14) iterated on the wait policy on the Rayon base (a throttle for oversubscribed pools, libgomp's spin, yield and sleep). The throttle fixed the slow mode and cost the loaded case. libgomp's waits met the quiet and loaded cases and left the fresh pool at about 11 ms. Reusing the threads fixed it.

**Rule.** When a cost appears per call, first ask why it is per call. Here the per-call cost was thread creation, and the waiting policy only changed how that cost showed up.

## 3. Evidence

Acceptance criteria, set on [#16](https://github.com/tensor4all/threadteam-rs/pull/16) and applied to both candidates on [#19](https://github.com/tensor4all/threadteam-rs/pull/19):

* (a) no slow mode in any faer#107 arm;
* (b) on a quiet host, held teams and pools within 10 % of 0.1.3;
* (c) under foreign load, held teams within libgomp's time;
* (d) the first section after a 10 ms gap within 10 % of 0.1.3.

Three sides were run interleaved: base 0.1.3 (`a3a9320`), Rayon with pool reuse (#19 head `9dfcbdb`, the code of 0.2.0) and own threads (`302c37d`). Criteria (a) to (c) are from the 64-core EPYC 7702P host at width 64; (b) and (d) were also run on a quiet 8-CPU measurement host (boost off, measurement pinned to 8 CPUs), where the two candidates differ by 1 to 2 % and neither approaches a criterion. All numbers are from the [#19 comments](https://github.com/tensor4all/threadteam-rs/pull/19).

| criterion | base 0.1.3 | Rayon, pool reuse (`9dfcbdb`) | own threads (`302c37d`) |
| --- | --- | --- | --- |
| (a) `fresh_pool` and `with_team`, `gemm128` | 15.38 ms (6 of 6 processes slow) | 0.04 ms | 0.02 ms |
| (a) `fresh_pool` / `with_team`, `chol2048` | 151 / 135 ms | 18 / 23 ms | 20 / 18 ms |
| (a) held arms | 0.02 ms (`gemm128`), 13.0 ms (`chol2048`) | 0.02 ms, 13.9 ms | 0.02 ms, 13.4 ms |
| (b) quiet `held_team`, per call | 1548 us | 1494 us (-3.5 %) | 1484 us (-4.1 %) |
| (b) quiet `held_pool`, per call | 1539 us | 1648 us (+7.1 %) | 1605 us (+4.3 %) |
| (c) loaded `held_team`, per call (libgomp 1680 us) | 1502 us | 1533 us (0.91 of libgomp) | 1523 us (0.91 of libgomp) |
| (d) 10 ms gap, first section, quiet 8-CPU host | 5.9 to 6.1 us | 0.99 to 1.01 of base | 0.99 to 1.00 of base |
| `fresh_pool` total per call, quiet | 150 459 us | 4005 us | 13 671 us |
| one-call pool acquire, width 64 | not measured | 2.2 ms | 11.5 ms |

Both candidates meet (a) to (d). The one measured difference is the acquire phase of a pool that serves a single call: 2.2 ms with the reused Rayon pool against 11.5 ms with own threads (3.2 ms against 13.9 ms under load). The held arms, the gap rows and the faer#107 arms agree within noise.

Cost of the two candidates, as the PR measured them:

| | Rayon with pool reuse (`625a593..9dfcbdb`, `src/`) | own threads (`625a593..302c37d`, `src/`) |
| --- | --- | --- |
| size | +190 / -31 lines in 4 files | +1562 / -543 lines in 12 files, about 1040 code lines |
| new `unsafe` | none | 43 blocks, functions and impls |
| new synchronization | a `Mutex<Vec<_>>` cache touched in `build` and `drop`, one `fetch_max`, a wait in `drop` | slot state word, intrusive batch queue, broadcast list, cooperative waits |
| new loom models | none (the 7 existing ones pass) | five |

An implementation review of the own-thread version found three deadlocks, an endless wake-up and an affinity regression. All were fixed with regression tests ([#16](https://github.com/tensor4all/threadteam-rs/pull/16), "Post-review resolutions" in own-threads.md). That review is the reason section 6 asks for more than loom before shipping.

## 4. What HPC kernels need from a thread substrate

The table compares OpenMP 6.0 (November 2024) with Rayon 1.x for dense and sparse CPU kernels. "Native" means Rayon provides the feature. "Buildable" means it can be written on Rayon's public API with a known cost. "Conflicts" means the feature needs a guarantee Rayon's design withholds. "Out of scope" means a thread crate does not provide it.

The Rayon column rests on the Rayon documentation ([`ThreadPool`](https://docs.rs/rayon-core/latest/rayon_core/struct.ThreadPool.html), [`ThreadPoolBuilder`](https://docs.rs/rayon-core/latest/rayon_core/struct.ThreadPoolBuilder.html)). The OpenMP column is from the maintainers' working notes and has not been checked clause by clause against the [6.0 specification](https://www.openmp.org/specifications/): **clause names, defaults and section numbers are to be verified against the 6.0 spec** before a feature is adopted on the strength of this table.

Rayon schedules jobs that run "sometime", and a task is not guaranteed to run concurrently with another. Every "conflicts" row below follows from that one property.

| OpenMP feature | Group | Rayon 1.x |
| --- | --- | --- |
| nested `parallel` that composes without multiplying threads | native | One pool, work stealing, nested `par_iter` and `join` add no threads. Rayon's strongest point. |
| `omp_get_thread_num`, `omp_get_num_threads` | native | `current_thread_index`, `current_num_threads`, `BroadcastContext::index`. |
| `for` with `dynamic`, `guided`, `auto` schedules; `taskloop` | native | Default splitting, `with_min_len`, `with_max_len`, `par_chunks`. |
| `task`, `taskwait`, `taskgroup`, `sections` | native | `scope` and `spawn`, `join`. |
| `critical`, `atomic`, `flush`, locks, `reduction` | native | `std` and `parking_lot` primitives, `reduce`, `fold`. |
| `thread_limit`, `OMP_STACKSIZE`, `OMP_NUM_THREADS` | native | Pool size, `stack_size`, `RAYON_NUM_THREADS`. |
| `cancel` | native for search | `try_for_each`, `find_any`, `any`; otherwise an `AtomicBool`. |
| free-agent threads (6.0) | native | Rayon workers are not bound to a team; any worker runs any job. |
| `if`, `collapse`, `num_threads` per region | buildable | A serial branch, an index split, one pool per width with `install`. |
| `schedule(static)` by a stable thread index | buildable | `broadcast` with a hand-computed range, on a pool of fixed size. |
| `ordered`, `nowait`, `scan`, task reductions | buildable | Hand-built on `scope`, atomics or chunked passes. |
| `depend`, `depobj`, `taskgraph` (6.0), `detach` | buildable | Dependency counters and successor lists, a graph built once and replayed. |
| `taskyield`, `threadprivate` | buildable | `yield_now`, `yield_local`, `thread_local!`. |
| `OMP_PLACES`, `OMP_PROC_BIND`, `proc_bind` | buildable, partly | `start_handler` can pin each worker once. Rayon has no place or binding concept and no per-region rebinding. |
| `barrier` among the members of a team | conflicts | No guarantee that N jobs run at the same time, so N jobs waiting for each other can deadlock. |
| `parallel` as a concurrent team, with `single` and `masked` | conflicts | `broadcast` runs one closure per worker but gives no team, no leader role and no barrier. |
| sub-teams (nested `parallel num_threads(k)`) with their own barriers | conflicts | `join` splits independent work; a barrier inside the half has the problem of `barrier`. |
| bitwise reproducible reductions | buildable, not by default | Rayon's own `reduce` splits adaptively, depending on stealing, so a floating-point sum can differ in its last bits between runs at one thread count. A reproducible sum is built on the static partition above: fixed chunks owned by a stable thread index, combined in a fixed order. |
| low-latency wake and `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT`, `KMP_BLOCKTIME` | conflicts | Workers spin a fixed number of rounds, then sleep, and the policy is not adjustable. threadteam's own `spin_polls` waits are layered on top for team members only. |
| `priority` on tasks | conflicts | No priority queue. `spawn_fifo` gives order, not priority. |
| `affinity` clause, `omp_get_place_num` | conflicts | Stealing is indifferent to where a job runs. |
| `omp_pause_resource`, `OMP_DYNAMIC` | conflicts, partly | A global pool cannot be stopped; a private pool ends with its drop. |
| `teams`, `distribute` on the host | conflicts | One pool per NUMA domain is the closest shape, and it is a convention, not a construct. |
| allocators and memory spaces, `simd`, loop transformations, `target` offload, `workshare` | out of scope | Memory placement, vector code, compiler transforms and devices are not a threading concern. |
| OMPT and OMPD tool interfaces | out of scope | Profiling goes through `tracing` or `perf`. |

Dense linear algebra needs these first, in this order of how often a kernel hits them:

1. **A concurrent team with barriers.** BLIS-style GEMM and panel factorizations run a team through several phases with a barrier between them.
2. **Static partition by a stable thread index.** The same thread owns the same rows every call, which keeps cache and first-touch placement, and it is the precondition for bitwise reproducible reductions.
3. **Sub-teams.** Multi-level GEMM loops and 2D splits give each group its own barrier.
4. **Low-latency wake.** Sections of 5 to 50 us follow each other, so a wake through a futex costs as much as the section.
5. **Pinning and places.** A 64-core Zen 2 part has L3 domains of four cores; where a member runs decides which cache it shares.
6. **Wait policy.** The caller decides how long idle members spin, because the right value differs between a solver loop and a service that shares the host.
7. **Dependent tasks.** Look-ahead LU and tiled Cholesky are DAGs with a critical path. `taskgraph` (OpenMP 6.0) replays one such graph per iteration, and task priority puts the panel first. The Rayon column has no priority.

threadteam provides items 1 to 6 today on the Rayon version, as a layer above Rayon's threads: a team is a reservation of members that spin on a generation counter, with `Barrier`, sub-team splitting, per-member pinning and `spin_polls`. The members are Rayon workers occupied for the lifetime of the reservation, so they are not available to Rayon jobs meanwhile. The own-thread version provides the same through its own loop and no longer takes workers from Rayon. Item 7 is provided by neither.

OpenMP 6.1 is not yet published as a final version. The TR15 preview (July 2026) is reported to extend thread affinity control. It is not read here, and the affinity design in [placement.md](https://github.com/tensor4all/threadteam-rs/blob/develop/docs/design/placement.md) is not changed by it.

## 5. Interoperation of Rayon and threadteam

Code in the stack uses Rayon in two ways: a library or user closure calls `par_iter`, and `tensor4all-rs` hands a `rayon::ThreadPool` to some of its interpolation code. Both directions have to work, because neither side can ask the other to give up its runtime.

### 5.1 threadteam lends to Rayon

* **Rayon version.** The pool's threads are a private Rayon pool. `Exec::external(f)` runs `f` inside that pool's `install`, so a `par_iter` in `f` runs on the pool's own threads, inside the CPU set the pool was built with. A pool of width `w` is one Rayon pool of `w` threads, and the one-owner rule holds because threadteam made that pool.
* **Own-thread version.** `Exec::external` runs the callback on a pool worker and lends members to the pool's own queue. A Rayon pool for user code is hosted on threadteam's threads with `rayon::ThreadPoolBuilder::spawn_handler`: the handler runs each Rayon worker loop on a thread leased from the reserve, so the Rayon pool's size and CPU set come from the lease. This is a design, not an implementation, and nothing in it has been measured. The lease is not returned until the Rayon pool is dropped. On `develop`, `Exec::external_rayon` behind the `rayon` feature takes the simpler route of a separate private Rayon pool, on separate threads.

### 5.2 threadteam borrows from a caller's Rayon pool

A caller that already has a Rayon pool can have threadteam run on it, using public Rayon API only:

* `ThreadPool::spawn_broadcast` puts a member loop on every worker; `current_thread_index` tells a member which worker it is.
* A waiting member calls `yield_now` or `yield_local` so it can serve Rayon jobs while it has nothing to do for the team.
* Each worker sets its own CPU affinity inside the broadcast job and restores it on return. threadteam 0.1 worked this way.

The limits, without support from Rayon itself:

* **No partial lease.** A broadcast reaches every worker of the pool, so a team of four on a pool of sixteen occupies sixteen.
* **No reservation and no bound on assembly time.** A worker that is busy joins the team only after its current job ends, so the time to assemble a team has no upper bound, and a section that starts before all members have arrived waits for the slowest.
* **No direct wake and no wait-policy control after return.** Once the member loop is running inside a Rayon job, Rayon owns when the worker wakes and sleeps.

Hazards:

* A nested `install` into the lent pool deadlocks when every worker is a member waiting for a barrier and none runs the installed job. A waiting member has to run Rayon jobs while it waits.
* Affinity has to be restored when the lease ends. A worker that keeps its team mask changes where later Rayon jobs run.
* While a pool is lent, Rayon owns the threads. The one-owner rule holds inside the lease only, and the lease has to be visible to the caller as a scope.

Any such API sits behind the `rayon` feature, because no Rayon type is public by default.

### 5.3 The two couplings to Rayon in tensor4all-rs

* `Parallelism::Pool { pool: &rayon::ThreadPool }` in [`crates/tensor4all-treetn/src/interpolation.rs`](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-treetn/src/interpolation.rs) takes a Rayon pool from the caller.
* The re-entry guard in [`crates/tensor4all-tensorbackend/src/context.rs`](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-tensorbackend/src/context.rs) tests `rayon::current_thread_index()`. On the Rayon version a threadteam worker is a Rayon worker, so the guard sees it. On the own-thread version it would stop detecting threadteam workers and not report an error.

## 6. What must hold before the own-thread version ships as 0.3.0

1. **The one-call acquire.** The acquire phase of a one-call pool at 64 threads is 11.5 ms against 2.2 ms on Rayon. The cause is to be found and the cost brought to the Rayon figure, on the same host and the same arms as in section 3.
2. **A protocol model.** Loom covers small interleavings (the slot state word, park and wake, a join, a barrier round). The protocol of lease, detach, batch claim and broadcast needs a model checked in TLA+ for deadlock freedom and liveness, because three deadlocks were found by review and not by the models.
3. **Larger randomized runs** with shuttle, on the batch queue and the broadcast list.
4. **Miri in CI** for the unsafe hand-offs (the stack-allocated batch published to the intrusive queue, the broadcast shares).
5. **Optionally, Verus** for the slot state word, whose four states and transitions are small enough to prove.
6. **A public query for "which pool is this thread a worker of".** The re-entry guard in section 5.3 needs it, and it replaces thread-local tests in callers.
7. **Both tensor4all-rs couplings replaced** (section 5.3): the pool type in `interpolation.rs` becomes a threadteam `Exec`, and the guard asks the query of item 6.
8. **Wait-policy layering.** threadteam owns `spin_polls` and `idle_spin`; tenferro passes the policy through; tensor4all-rs exposes `WaitPolicy::{Active, Passive, SpinPolls(n)}` on `CpuBackend` and a process default ([tensor4all-rs#1054](https://github.com/tensor4all/tensor4all-rs/issues/1054), [threadteam-rs#9](https://github.com/tensor4all/threadteam-rs/issues/9)). The layering does not depend on the substrate and can land on the Rayon version first.

The Rayon version of the contract change ships in 0.2.0: dropping a pool with default thread settings caches its Rayon pool instead of joining its threads, and pools with a name, an `on_start` hook, a stack size or a QoS keep dedicated threads that the drop joins.

## 7. Open questions to settle while staying on Rayon

1. **Which "conflicts" features does a concrete kernel need now?** Sub-teams, pinning per lease and dependent tasks are the candidates. Each needs a named kernel and a measurement that the Rayon version's layer is not enough, because a feature without a caller does not justify the move.
2. **Is the process-wide reuse cache acceptable under the "no implicit global pool" rule?** The cache holds idle threads and accepts no work without a pool, and the first explicit `Pool::build` creates it. The core-boundary document says no layer starts a hidden global pool or keeps a process-wide registry of engines. Whether a cache of idle threads counts as such a registry is a reading of that rule, and the cache's size cap (the larger of the available parallelism and the widest shared pool built) is part of the answer.
3. **Does hosting a Rayon pool on leased threads work** (section 5.1)? It has to be built and measured before it is relied on.

## 8. Claims not verified

* The OpenMP 6.0 clause details in section 4 are from working notes, not from the specification text.
* The TR15 content of OpenMP 6.1 is taken from news reports, and the final 6.1 specification is not out.
* That Rayon workers spin a fixed number of rounds before sleeping, and that the policy is not adjustable, is from memory of the Rayon sources and not re-read.
* The `spawn_handler` hosting in section 5.1 and the lending of a caller's Rayon pool in section 5.2 are designs; the 0.1 behavior is described from the maintainers' notes, not re-run.
