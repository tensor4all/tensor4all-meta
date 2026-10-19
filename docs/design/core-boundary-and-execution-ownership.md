# tenferro core boundary and CPU execution ownership

**Status:** Proposed future design. Target: implementation starts around 2026-10-27.
**Scope:** tenferro-rs, the provider crates (tcontract, cpueinsum, tlinalg, t4a-faer), hataori-rs, and downstream libraries that write their own kernels (symtn, tensor4all-rs).
**Checked against:** tenferro-rs main `14bc1ab9`, tcontract-rs main, symtn main `88e5895b` (2026-10-20).

This document fixes three cross-repository decisions. The first is what belongs
in the tenferro core and what users of tenferro build themselves. The second
is who owns CPU threads. The third is who owns the mutable state used while
executing. Repository-internal details (type names, migration
steps) belong in tenferro-rs issues and `docs/design/`.

## 1. Policy in one sentence each

1. The tenferro core stays small. Batches whose items do not share one shape
   are handled by the caller, with a dedicated kernel written against the
   provider crates.
2. The caller owns the threads. Every layer, tenferro included, borrows the
   pool it is given and never decides how the machine is partitioned.
3. The caller owns the execution resources. Runtimes keep the computation
   (graphs, compiled programs, values and gradients); the workspaces and
   caches that execution uses travel with the engine the caller passes in.

## 2. Layers

![Layers: downstream, tenferro core and extensions, provider crates](figures/layers.svg)

Downstream kernels may call the provider crates directly (the arrow on the
right). A symtn kernel that
runs one GEMM or one factorization per block does exactly that, so the
providers' borrowed execution tokens (`tcontract_exec::Exec`, the tlinalg
`Parallel` token) are part of the boundary, not an internal detail.

### Batch axes

A batch axis appears in both operands and survives into the output without
being summed:

```text
C[b, i, k] = sum_j A[b, i, j] * B[b, j, k]      (b is a batch axis)
```

![Regular batch in one call vs irregular batch in the caller's kernel](figures/batches.svg)

A batch axis is never required: one can call the contraction once per value
of `b`. It exists so that a regular batch runs as one fast call. The core
supports regular batches through `dot_general` and the batched linear algebra
entry points. Irregular batches (different shapes per item, as with symmetry
blocks) are out of scope for the core.

## 3. CPU execution ownership

### Principle

Threads and scratch memory come from objects the caller creates and holds.
No layer starts a hidden global pool, keeps a process wide registry of
engines, or arbitrates between callers. This is the rule the provider crates
already follow, extended to tenferro.

![Thread ownership today vs in this design](figures/ownership.svg)

On the left, tenferro decides who may use which worker and must check those
decisions at run time. On the right, there is nothing to check: whoever holds
the engine uses its pool.

### Background for readers new to the CPU stack

This section explains the words used below without assuming knowledge of
tenferro's internals.

* **Thread pool.** A fixed set of worker threads that wait for parallel work.
  tensor4all uses [Rayon](https://docs.rs/rayon) pools. Creating a pool is
  cheap compared with a large computation but not free, and how many pools
  exist and on which CPU cores they run decides how the machine is shared.
* **Workspace.** Everything a numerical routine keeps between calls so the next
  call is faster: recycled buffers, cached contraction plans, cached analysis
  of matrix-multiplication shapes, scratch for multi-operand contractions. A workspace is
  mutable and is only safe to use from one computation at a time.
* **Engine.** A `CpuEngine` bundles one pool with one workspace. It is the
  unit of "computing power plus memory" that one caller uses at a time, and
  it is what every operation takes as its argument. A GPU engine bundles a
  device stream with its memory pools in the same way. There is no separate
  session object (see "One concept: the engine" below).
* **Task lane and worker.** A *task lane* is one stream of work in the
  caller's scheduler, for example one of the `k` parallel tasks hataori runs.
  A *worker* is one thread of an engine's pool. Each task lane gets its own
  engine, and inside that lane an operation may use the `m` workers of that
  engine's pool.
* **Runtime.** tenferro's eager runtime (immediate execution with automatic
  differentiation) and its traced runtime (record a graph, compile it once,
  run it many times). A runtime stores what was *prepared*: graphs, compiled
  programs, differentiation rules, gradients being accumulated.

An analogy that matches the design: a runtime holds recipes, an engine is a
kitchen with its staff (the pool) and its counters and utensils (the
workspace). Several kitchens can cook from the same recipe book at the same
time. A recipe book that owned one kitchen would force everyone to queue for
it, and a kitchen whose staff were hired and assigned by the recipe book
would leave the restaurant owner unable to decide how many cooks work where.
Today tenferro does both; this design undoes both.

### Why the caller, not tenferro, owns the threads

The programs that use tenferro already have a scheduler: hataori distributes
work over MPI ranks and over lanes inside a rank, and tensor4all-rs fans out
over patches and tree nodes. Only that scheduler knows how many tasks run at
once and how the machine should be split between them. When tenferro also
builds and arbitrates threads, two schedulers compete for the same cores, and
tenferro has to rent workers back to the real scheduler under rules it can
verify at run time. That is where the arbiter, the held sessions, the phase
lease and the child tickets listed in section 4 came from. Each was a
reasonable fix; together they are the cost of the ownership being upside
down, not of numerical work.

When the caller owns the threads, none of those rules are needed: the caller
gives each task its own engine, and an engine is used by whoever holds it.

### The CPU engine

A `CpuEngine` is a plain value:

| Part | Content |
| --- | --- |
| pool | a Rayon pool built by the caller, or none for a serial engine |
| workspace | buffer pool, GEMM analysis cache, indexed plan cache, N-ary contraction workspace, extension plan caches, graph execution scratch |

tenferro never builds threads. There is one way to make a parallel engine,
from a pool the caller built, and one way to make a serial engine, with no
pool at all. A serial engine runs everything on the calling thread and never
touches a global Rayon pool, so it is safe to use from inside any other pool
(for example inside a task of the caller's own scheduler).

tenferro offers one helper, a `rayon::ThreadPoolBuilder` preset with the
worker stack size that the linear algebra providers need (deep recursion in
some LU implementations overflows Rayon's default stack). It returns the
ordinary Rayon builder; the caller sets the thread count, pins workers if it
wants to, and builds the pool. The helper is a convenience, not a second
construction mode.

Rules:

1. **The engine is passed to every execution.** Runtimes and tensors never
   store an engine. Eager operations, the backward pass and traced program
   runs all take `&mut` of the engine they should run on. A run that
   needs several engines (for example CPU and GPU with transfers between
   them) receives all of them for that call.
2. **One computation per engine at a time.** The engine is used through
   `&mut`, so Rust's borrow rules guarantee this without run time locks. The
   one exception is buffer return: a tensor may be dropped on any thread, and
   its memory then goes back to the buffer pool of the engine that allocated
   it under a small internal lock. If that engine no longer exists, the memory
   is simply freed.
3. **Every operation may use the whole pool of its engine.** Throttling for
   small work happens inside the providers, which already choose how many
   workers to wake from a work model, up to the budget they are given. A caller that
   wants fewer threads builds a smaller pool.
4. **The engine hands out the provider token for its pool** (sketch:
   `engine.exec(budget) -> tcontract_exec::Exec<'_>`). The token is a
   borrowed permission to run provider code on the engine's pool and
   workspace, and its budget is the largest number of workers that code may
   use. A downstream kernel that calls a provider directly passes this token,
   so it uses the same threads and the same budget as tenferro.
5. **The caller of an operation must not be a worker of that engine's pool.**
   tenferro checks this against the engine's own pool and reports an error
   rather than silently running serially.

How one operation runs:

![How one operation runs through CpuEngine and a provider](figures/operation.svg)

A downstream kernel takes the same path from the third step on, by calling
`engine.exec(budget)` itself.

Sketch (names are not final):

```rust
let pool = tenferro_cpu::thread_pool_builder().num_threads(4).build()?;
let mut engine = CpuEngine::new(Arc::new(pool));   // parallel, uses all 4 threads
let mut serial = CpuEngine::serial();              // no pool at all

let c = dot_general(&mut engine, &a, &b, &dims)?;          // eager operation
let grads = loss.backward_in(&mut engine)?;                // backward pass on the same engine
let out = runtime.run_prepared(&mut engine, &program, &inputs)?; // compiled graph, same engine
let d = dot_general(&mut serial, &a, &b, &dims)?;          // same code, serial engine
```

### One concept: the engine

Code that runs work names exactly one kind of object, the engine. Generic
code takes `&mut dyn Engine` (or `&mut E` with `E: Engine`), a
backend-neutral trait that `CpuEngine` and the GPU engines implement.

tenferro used to have a second concept, the session: a handle opened from a
backend for the duration of a closure, through which every operation ran.
The session existed to do work that this design removes:

* admission against the process wide arbiter, and the reentry and
  contention errors that came with it;
* narrowing the calling thread's CPU mask while the closure ran and
  restoring it afterwards;
* holding an exclusive lease on a backend that several callers could reach.

Without those, a session would only repeat what the engine already gives:

* **Exclusivity** comes from `&mut engine`. A caller that really wants to
  share one engine between threads wraps it in a `Mutex`; the guard is then
  the lease, written in ordinary Rust.
* **The check of rule 5** (caller not a worker of the pool) runs at the
  start of each operation. It is one thread-local read, negligible against
  any operation that reaches a provider.
* **Provider state that should live across operations** (the
  `tcontract_exec::Pool` wrapper with its arena and its SPMD gate) is owned
  by the engine, which builds it once with `Pool::shared` from its
  `Arc<rayon::ThreadPool>`. Nothing has to be rebuilt per operation or per
  closure.
* **GPU work boundaries.** A GPU engine owns its stream. Submitting queued
  work and waiting for the device are explicit engine methods (sketch:
  `engine.submit()`, `engine.synchronize()`), and dropping the engine
  synchronizes. These are points in time, not a scope, so they need no
  closure.

A run that needs several engines (CPU and GPU with transfers, rule 1)
receives them as a set of borrowed engines for that one call. That set is
an argument, not a stored object, and it is not called a session.

### Runtimes hold what was prepared, engines hold what is mutable

![Runtimes are shared; each lane brings its own engine](figures/runtime_split.svg)

Compiling a traced graph or setting up automatic differentiation is expensive
and gives the same result for every lane, so it should happen once and be
shared. Executing needs scratch memory and caches that change on every run,
so those must not be shared. The design therefore splits every runtime into
two parts:

* **Kept in the runtime and shared:** what describes the computation and its
  results: graphs, compiled programs and their cache, differentiation rules,
  recorded values and gradients. None of this depends on which engine runs
  the work. (The eager graph grows and gradients accumulate as operations
  run; how concurrent updates to them are synchronized is a property of the
  runtime and is not changed by this design. Separate engines do not give
  separate gradients.)
* **Moved to the engine:** the execution resources: buffers, plan and
  analysis caches, extension plan caches (for example FFT plans), and the
  scratch slots a compiled graph uses while it runs.

With this split, `k` task lanes run the same compiled program at the same
time, each on its own engine, and the program is compiled once. Today a
runtime keeps one copy of the execution resources behind a lock, so runs on
one runtime cannot overlap: a second run either waits or is rejected as busy.
Task lanes that want to run in parallel must each build and compile their own
runtime.

The cost is that caches filled during execution are per engine: each lane
warms up its own plan caches, and memory for them grows with the number of
engines (see section 5).

### Nesting

Nested parallelism is expressed by creating several engines, not by
splitting one. hataori (or any other scheduler) builds `k` pools of `m`
threads, wraps each in an engine, and gives one engine to each task lane.
The lane thread is not a worker of its engine's pool, so the operations
inside the lane are free to use that pool in full.

![Nesting: k lanes, one CpuEngine of m threads each](figures/nesting.svg)

Each lane runs its operations on its own `m` threads. Keeping the CPU sets
disjoint, as drawn, is the caller's choice; tenferro neither enforces nor
checks it.

### Responsibilities of the caller

* **Machine layout.** Whether the pools of different engines overlap on CPUs,
  whether the total thread count oversubscribes the machine, and whether
  workers are pinned to cores are the caller's decisions. tenferro does not
  check them.
* **One pool, one active engine.** A Rayon pool backs at most one engine in
  use at a time. Lanes that run at the same time need separate pools. (The
  provider crates keep per-pool coordination state inside the engine, so two
  engines on one pool would not see each other.)
* **The calling thread.** Parts of an operation, including vendor BLAS and
  LAPACK calls, run on the thread that called it. Its CPU affinity and stack
  size are the caller's, exactly like the pool's.
* **Asynchronous execution.** tenferro does not start background threads to
  run work later. A caller that wants a computation in the background starts
  its own thread and runs it there.
* Threads started by a vendor BLAS or LAPACK are outside tenferro's control,
  as before.

## 4. What this removes from tenferro

The following exist only to let tenferro own, subdivide or hide threads and
execution state on behalf of its callers, and are removed:

* every constructor that builds threads: thread counts and CPU sets on the
  backend, sizing from `RAYON_NUM_THREADS`, NUMA placement, the worker stack
  setting
* the process global resource arbiter and its CPU set conflict detection
* the per placement engine registry
* owner thread locals used as admission keys, and owner inheritance
* the held CPU session, the per worker phase lease, and the child execution
  ticket, together with their error types
* the session layer itself: opening a session from a backend and running
  operations through a closure; the backend-neutral session trait becomes the
  `Engine` trait implemented by the engines directly
* tenferro's own thread pinning, including narrowing the calling thread's CPU
  mask during a session
* backends and execution state stored inside runtimes, and the locks that
  serialize execution on one runtime
* the detached submission API that runs a compiled graph on a thread tenferro
  starts

The work these served moves to the caller: per lane execution becomes one
engine per lane, and lanes are no longer forced to run serially inside.

## 5. Costs we accept

* Workspaces and execution caches are per engine, so memory use and warm up
  grow with the number of engines. Each engine has its own retention limit
  for recycled buffers.
* A bad CPU layout shows up as lost performance, not as an error.
* Calling an engine from a worker of its own pool is still possible and is
  detected at run time.
* Every execution names its engine, so code is more explicit than with a
  runtime that hides its backend. This is intended: the engine in the call is
  exactly what decides where the work runs.

## 6. Consistency with other documents

* The [backend developer guide](https://tensor4all.org/tensor4all-meta/stack/backend.html)
  already describes hataori passing each task the pool it may use. This
  design makes tenferro match that description.
* tenferro-rs design documents for the held session, the phase lease, the
  child ticket and the shared execution scope are superseded by this document
  and should be retired when the implementation lands.

## 7. Plan

1. Implement section 3 and the removals of section 4 in tenferro-rs
   ([tenferro-rs#2206](https://github.com/tensor4all/tenferro-rs/issues/2206)),
   without a compatibility layer. Internal details (how an engine proves it
   is compatible with a compiled program, the exact public names) are settled
   there.
2. Port the downstream users to one engine per lane, each in its own change:
   tensor4all-rs (adaptive interpolation, tests, the C API's default engine),
   the hataori tenferro adapter, tenferro-benchmark, gaugefields-rs.
3. Update the backend guide and retire the superseded tenferro design notes.

## Settled questions

1. *Should the workspace be separable from the pool?* Not in the first
   version. An engine owns its workspace; two engines never share one.
2. *How should pinning behave on macOS?* tenferro does no pinning, so the
   question moves to whoever builds the pools.
3. *Should tenferro keep a mode that builds its own pool?* No. One helper
   presets the Rayon builder; the caller always builds the pool.
4. *Should runtimes own their engine?* No. Engines are passed per call, and
   runtimes keep only what was prepared.

5. *Are "engine" and "session" two concepts or one?* One. Operations take
   `&mut` of the engine through a backend-neutral `Engine` trait, and the
   session is removed (see "One concept: the engine"). The word "backend"
   also leaves the public API: `CpuBackend` becomes `CpuEngine`.

## Open questions (to mature before implementation)

1. *How does an engine prove it is compatible with a compiled
   program?* Backend kind alone is too weak (CPU allocation domains, GPU
   device and stream state). Settled in tenferro-rs#2206.
2. *What does a GPU engine cost per operation without a session scope?*
   The session bound the device stream to the calling thread once per
   closure. An engine binds it per operation, or remembers which thread it
   is bound to and skips the binding when nothing changed. Measure before
   choosing.

The figures are generated by `figures/make_figures.py` (standard library
Python). Edit the script and rerun it rather than editing the SVG files.
