# tenferro core boundary and execution ownership

**Status:** Proposed future design. Target: implementation starts around 2026-10-27.
**Scope:** tenferro-rs, the provider crates (tcontract, cpueinsum, tlinalg, t4a-faer), hataori-rs, and downstream libraries that write their own kernels (symtn, tensor4all-rs).
**Checked against:** tenferro-rs main `14bc1ab9`, tcontract-rs main, symtn main `88e5895b` (2026-10-20).

This document fixes three cross-repository decisions. The first is what belongs
in the tenferro core and what users of tenferro build themselves. The second
is who owns CPU threads and GPU streams. The third is who owns the mutable state used while
executing. Repository-internal details (type names, migration
steps) belong in tenferro-rs issues and `docs/design/`.

## 1. Policy in one sentence each

1. The tenferro core stays small. Batches whose items do not share one shape
   are handled by the caller, with a dedicated kernel written against the
   provider crates.
2. The caller owns the threads. Every layer, tenferro included, uses the
   pool it is given and never decides how the machine is partitioned.
3. The caller owns the execution resources. Runtimes keep the computation
   (graphs, compiled programs, plans, values and gradients); the memory and
   scratch that execution writes travel with the engine the caller passes in.

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

tenferro is handed a thread pool and computes on it; that is the whole
model. Threads and scratch memory come from objects the caller creates and
holds. No layer starts a hidden global pool, keeps a process wide registry of
engines, or arbitrates between callers. This is the rule the provider crates
already follow, extended to tenferro.

![Thread ownership today vs in this design](figures/ownership.svg)

On the left, tenferro decides who may use which worker and must check those
decisions at run time. On the right, there is nothing to check: whoever holds
the engine uses its pool.

### Background for readers new to the CPU stack

This section explains the words used below without assuming knowledge of
tenferro's internals.

* **Thread pool and worker.** A thread pool is a fixed set of threads, its
  *workers*, that wait for parallel work.
  tensor4all uses [Rayon](https://docs.rs/rayon) pools. Creating a pool is
  cheap compared with a large computation but not free, and how many pools
  exist and on which CPU cores they run decides how the machine is shared.
* **Workspace.** The memory a computation writes while it runs and keeps for
  the next call: recycled buffers and scratch areas. A workspace is mutable
  and is only safe to use from one computation at a time.
* **Plan.** A decision worked out once and reused, such as how to split a
  contraction or which kernel to use for a matrix-multiplication shape. A
  plan is never changed after it is made, so any number of computations can
  read it at the same time.
* **Engine.** A `CpuEngine` bundles one pool with one workspace. It is the
  unit of "computing power plus memory" that one caller uses at a time, and
  it is what every operation takes as its argument. A GPU engine bundles a
  device stream with its memory pools in the same way. There is no separate
  session object (see "One concept: the engine" below).
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

### One engine

The basic case is one pool and one engine, and it is simple: the caller
builds a Rayon thread pool and hands it to tenferro, tenferro wraps it into
a `CpuEngine`, and every operation on that engine computes on that pool.
tenferro does not create the threads, choose their number, or decide who
else may use them. Everything else in this section builds on this case.

Sketch (names are not final):

```rust
let pool = tenferro_cpu::thread_pool_builder().num_threads(4).build()?;
let mut engine = CpuEngine::new(pool);             // takes the pool; uses all 4 threads
let mut serial = CpuEngine::serial();              // no pool at all

let c = dot_general(&mut engine, &a, &b, &dims)?;          // eager operation
let grads = loss.backward_in(&mut engine)?;                // backward pass on the same engine
let out = runtime.run(&mut [&mut engine], &prepared, &inputs)?; // prepared graph, same engine
let d = dot_general(&mut serial, &a, &b, &dims)?;          // same code, serial engine
```

A `CpuEngine` is a plain value:

| Part | Content |
| --- | --- |
| pool | a Rayon pool built by the caller and handed over to the engine, or none for a serial engine |
| workspace | buffer pool, N-ary contraction scratch, graph execution scratch, and accounting of memory still in use by unfinished work |

Plans (contraction plans, matrix-multiplication analysis, extension plans
such as FFT plans) are immutable, so they can be shared. A prepared program
carries the plans it needs. Operations called directly on an engine find
their plans in a `PlanCache`, a value of its own that an engine refers to:
by default each engine gets a private one, and a caller that wants several
engines to share planning gives them the same `PlanCache`. A lookup hands
out a shared, read-only plan and releases the cache before the computation
starts, so engines sharing a cache never wait for each other's work. The
cache keeps today's bounds, statistics and clearing. Plans tied to one
engine's device handles (for example cuTENSOR plans on a GPU) stay in that
engine.

tenferro never builds threads. There is one way to make a parallel engine,
from a pool the caller built and hands over, and one way to make a serial
engine, with no pool at all. The engine takes the pool by value, so one pool
can never back two engines; `engine.pool()` gives the caller read access to
it, for example to ask its size. A serial engine runs everything on the calling thread and never
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
   it under a small internal lock. On the CPU, if that engine no longer
   exists, the memory is simply freed. On a GPU, memory that unfinished work
   may still use is never freed or reused early; section 4 gives the rule.

   **Unfinished work is bounded.** On a GPU, work may still be running after
   a call returns (see "When run returns"). The engine counts the memory
   that its unfinished work holds: temporaries, outputs being written, and
   memory waiting for other engines to finish with it. This is separate from
   tensors the caller still holds and from idle recycled memory. When the
   count passes a limit the caller can set, the engine waits for the work
   that holds that memory, including other engines' work, before submitting
   more. A single operation that needs more than the limit by itself is
   rejected before it is submitted.

   **Broken engines and broken devices.** If an operation panics, the engine
   marks itself broken: later operations on it return an error, while
   waiting for and cleaning up its earlier work still happens. The caller
   replaces the engine. Some GPU errors (an illegal memory access, a failed
   device assertion) break the whole device context, not one engine; then
   every engine of that `CudaDevice` reports the error, and creating another
   engine on the same device is not a recovery. Recovering from such an
   error may need a new process.
3. **Every operation may use the whole pool of its engine.** Throttling for
   small work happens inside the providers, which already choose how many
   workers to wake from a work model, up to the budget they are given. A caller that
   wants fewer threads builds a smaller pool.
4. **The engine hands out the provider token for its pool** (sketch:
   `engine.exec(budget) -> tcontract_exec::Exec<'_>`). The token is a
   borrowed permission to run provider code on the engine's pool and
   workspace, and its budget is the largest number of workers that code may
   use. A downstream kernel that calls a provider directly passes this token,
   so it uses the same threads and the same budget as tenferro. Getting the
   token takes `&mut engine` and applies the check of rule 5; what the kernel
   then does with the token is that kernel's responsibility, as with any
   direct provider call.
5. **An operation must not be called from one of its engine's own workers.**
   For example, `pool.install(|| dot_general(&mut engine, ...))`, where
   `engine` was built from `pool`, is an error. In that call the operation
   runs on one of the pool's own workers, and some provider strategies
   cannot start from there: a strategy that needs every worker of the pool
   at once (a team that meets at a barrier) is refused, while others would
   run with less parallelism than rule 3 promises. Rather than let the
   outcome depend on which strategy a provider picks, tenferro checks this
   case at the start of the operation and returns an error. Calling from a
   worker of a *different* pool is allowed, with one caveat described under
   "The calling thread" below, and a serial engine has no pool to check.

How one operation runs:

![How one operation runs through CpuEngine and a provider](figures/operation.svg)

A downstream kernel takes the same path from the third step on, by calling
`engine.exec(budget)` itself.

### One concept: the engine

Code that runs work names exactly one kind of object, the engine. Generic
code takes `&mut dyn Engine` (or `&mut E` with `E: Engine`), a
backend-neutral trait that `CpuEngine` and the GPU engines implement.

tenferro used to have a second concept, the session: a handle opened from a
backend for the duration of a closure, through which every operation ran.
Its main reasons to exist are removed by this design:

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
  by the engine, which builds it once around the pool it owns. Nothing has
  to be rebuilt per operation or per closure.
* **GPU work boundaries.** A GPU engine owns its stream. Submitting queued
  work and waiting for the device are explicit engine methods (sketch:
  `engine.flush()`, `engine.synchronize()`), and dropping the engine
  synchronizes. These are points in time, not a scope, so they need no
  closure.

The session also did cleanup at the end of its scope, and that cleanup
stays, attached to a narrower scope instead of a public object:

* **Recovery after a panic.** Buffer accounting is restored when an
  operation unwinds. This becomes a guard around each operation.
* **Unfinished GPU work.** Work that a run queued but the GPU has not
  finished, with the temporary memory and transfer buffers it still uses,
  is kept by the engine until it completes, including after an error or a
  panic. (There is no separate deferred-run object any more; see "Running a
  compiled graph".)
* **GPU context restoration.** Native GPU code that changes the device
  context restores it on exit, including on unwind. This becomes a guard
  around each such call.

**GPU engine drop.** Dropping a GPU engine may block while its queued work
is retired, and it cannot report an error; if retirement fails, the memory
that might still be in use is kept rather than freed, as today. A caller
that needs to know the work finished, or wants its error, calls
`synchronize()` first.

A run that needs several engines (CPU and GPU with transfers, rule 1)
receives them as a set of borrowed engines for that one call. That set is
an argument, not a stored object, and it is not called a session.

### Several engines

tenferro makes one engine from one pool it is given. A caller with several
pools makes several engines, one from each. tenferro does not know how many
engines exist, how the caller distributes work over them, or which thread
uses which engine; it never subdivides a pool and has no notion of parallel
tasks above the engine. Parallelism above one operation is therefore
expressed by holding more engines, and coordinating them is the caller's
job (see "Responsibilities of the caller"). An example is at the end of this section.

Say the caller wants `k` computations at the same time, each on its own
pool of `m` threads. Using `k` engines at the same time takes `k` calling
threads, one per engine, and none of them may be a worker of a compute pool it calls into
(rule 5). The calling threads are usually a pool of the caller's own, for
example `k` threads that run the caller's tasks; that pool is separate from
the `k` compute pools, and tenferro never sees it. The process then has `k`
calling threads plus `k * m` workers. A calling thread mostly waits while its
operation runs on the workers, but it does run the serial parts of an
operation (see "The calling thread" below).

### Runtimes hold what was prepared, engines hold what is mutable

![Runtimes are shared; each concurrent run brings its own engine](figures/runtime_split.svg)

Compiling a traced graph or setting up automatic differentiation is expensive
and gives the same result for every engine, so it should happen once and be
shared. Executing needs scratch memory and caches that change on every run,
so those must not be shared. The design therefore splits every runtime into
two parts:

* **Kept in the runtime and shared:** what describes the computation and its
  results: graphs, compiled programs and their cache, prepared programs with
  their plans, differentiation rules, recorded values and gradients. (Plans
  for operations called directly on engines live in `PlanCache` values,
  which several engines may share; see "One engine".) None of this depends on which engine runs
  the work. (The eager graph grows and gradients accumulate as operations
  run; how concurrent updates to them are synchronized is a property of the
  runtime and is not changed by this design. Separate engines do not give
  separate gradients.)
* **Moved to the engine:** what execution writes: buffers, scratch, the
  slots a compiled graph uses while it runs, and plans that are tied to the
  engine's own device handles.

With this split, several callers run the same compiled program at the same
time, each on its own engine, and the program is compiled once. Today a
runtime keeps one copy of the execution resources behind a lock, so runs on
one runtime cannot overlap: a second run either waits or is rejected as busy.
Callers that want to run in parallel must each build and compile their own
runtime.

Engines that share prepared programs or a `PlanCache` do not repeat the
work of planning; each engine warms up its own buffers and scratch, and any
plans tied to its device handles (see section 6).

### Running a compiled graph: compile, prepare, run

The traced runtime turns a recorded computation into something that can be
run many times. This happens in three steps, and each step exists for a
different reason:

| Step | Input | Output | Done how often |
| --- | --- | --- | --- |
| **compile** | a recorded graph | a `CompiledGraph`: the computation in a hardware-independent form, keeping the shape constraints that were recorded (fixed sizes stay fixed; only symbolic sizes may vary) | once per computation |
| **prepare** | a compiled graph, the inputs' shapes, types, layouts and kinds of memory, and a description of the engines it will run on | a prepared program: an execution plan for those inputs, including fusion chosen for the target hardware, and the requirements an engine must meet to run it (device, memory kind, supported operations) | once per input signature |
| **run** | a prepared program, the engines and the inputs | the outputs | as often as needed |

Prepare has to know what it is preparing for: which kind of engine runs
each part, on which device, with which capabilities. It cannot always tell
from the inputs (a graph may have no inputs, or move data between a CPU and
a GPU), so the caller passes a description of the target engines. Every
engine can produce such a description, `engine.target()`: an immutable value
naming the provider, the device, the memory the engine can read and the
operations it supports. It holds no threads, streams or memory.

Prepare gives each target a numbered slot, decides which slot runs each
part of the program and which transfers move data between slots, and
checks that the memory each slot will read is compatible. A prepared program
is specialized for its input signature together with these target
requirements. It records requirements, not engines: it keeps no stream,
event or memory of the engines that will run it.

Prepare does the expensive checks and decisions once, so that run only
checks that the inputs still match (shapes, types, strides, alignment and
memory domain, as today) and that the engines it is given match the
prepared targets, binding slot `i` to the `i`-th engine in the list. Any engine
whose description matches the target, with the same capabilities and
compatible memory, may run it, so several callers can run one prepared
program at the same time, each on its own engine.

There is exactly one way to prepare and one way to run:

```rust
let prepared = runtime.prepare(&compiled, &inputs, &[engine.target()])?;
let outputs = runtime.run(&mut [&mut engine], &prepared, &inputs)?;
```

`run` takes the engines as a list because a prepared program may span
several of them, for example a CPU engine and a GPU engine with transfers
between them; a program on one engine gets a list of one.

**Inputs and outputs.** Inputs are borrowed views of tensors. Each output
is one of two kinds:

```rust
enum Output<'a> {
    Owned(Tensor),            // a new tensor, stored compactly
    Borrowed(TensorView<'a>), // an input, or a view of one, without a copy
}
```

An output computed by the graph is a new tensor with a compact layout: for
`transpose(neg(x))` the result is materialized in its transposed order, not
kept as a lazy view. An output that is just an input, or a view of one (for
example `transpose(x)`), is returned as a view of that input instead of a
copy; Rust's borrow rules then keep the input alive as long as the output
is used. A caller that needs an owned tensor converts the output, which
copies only in the borrowed case.

**When run returns.** On the CPU, run returns when the work is done. On a
GPU, run returns once the work is queued on the engine's stream and once it
no longer reads memory that the caller could change or free (for example,
host data being uploaded). The GPU may still be computing. Outputs can be
used right away by later operations on the same engine, because the stream
runs work in order; reading them on the host, or calling
`engine.synchronize()`, waits for the work. Using inputs or outputs on
another engine is safe without extra calls (section 4, rule 2).

**What this replaces.** tenferro had several entry points for this: one-shot
variants that entered preparation on every call, a variant that kept lazy
views in its outputs, a variant for borrowed inputs, a variant where the
runtime opened sessions itself and one where the caller supplied a session,
a deferred variant that batched runs and waited at the end, and a detached
submission on a background thread. They differed in who owned the session,
when completion was awaited and how outputs were returned. With engines
passed per call, completion handled by the engine's stream, and one output
type, those differences disappear.

### Why the caller, not tenferro, owns the threads

The programs that use tenferro already have a scheduler: hataori distributes
work over MPI ranks and over parallel tasks inside a rank, and tensor4all-rs fans out
over patches and tree nodes. Only that scheduler knows how many tasks run at
once and how the machine should be split between them. When tenferro also
builds and arbitrates threads, two schedulers compete for the same cores, and
tenferro has to rent workers back to the real scheduler under rules it can
verify at run time. That is where the arbiter, the held sessions, the phase
lease and the child tickets listed in section 5 came from. Each was a
reasonable fix; together they are the cost of the ownership being upside
down, not of numerical work.

When the caller owns the threads, none of those rules are needed: the caller
gives each task its own engine, and an engine is used by whoever holds it.

### Responsibilities of the caller

* **Machine layout.** Whether the pools of different engines overlap on CPUs,
  whether the total thread count oversubscribes the machine, and whether
  workers are pinned to cores are the caller's decisions. tenferro does not
  check them.
* **One pool per engine** is enforced by the type: the engine takes its pool
  by value. (The provider crates keep per-pool coordination state inside the
  engine, so two engines on one pool would not see each other; taking the
  pool by value rules this out.)
* **Exclusion between engines.** Inside one thread, `&mut` keeps an engine
  to one computation. Which of the caller's threads or tasks holds which
  engine, and that two of them never hold the same engine at once, is
  decided by the caller's scheduler. A caller that shares an engine between
  threads wraps it in its own lock.
* **The calling thread.** Parts of an operation, including vendor BLAS and
  LAPACK calls, run on the thread that called it. Its CPU affinity and stack
  size are the caller's, exactly like the pool's.
* **Calling threads that are Rayon workers.** When a worker of one Rayon
  pool calls an operation on another pool's engine, Rayon lets that worker
  run other jobs of its own pool while it waits. A second task can then
  start on the same thread before the first operation returns. That is
  harmless if each task carries its engine as an argument. It deadlocks if
  tasks find their engine through a lock that the waiting task already
  holds, and it fails if they look it up per thread. Plain threads, or a
  scheduler that gives each task its engine explicitly, avoid this.
* **Asynchronous execution.** tenferro does not start background threads to
  run work later. A caller that wants a computation in the background starts
  its own thread and runs it there.
* Threads started by a vendor BLAS or LAPACK are outside tenferro's control,
  as before.

### Example: nested parallelism in hataori

This example uses hataori's vocabulary, which tenferro does not share. A
hataori *lane* is one of the `k` streams in which hataori runs its tasks,
one task after another; tasks are its units of work, such as one patch.
hataori runs its lanes on `k` threads of its own, separate from the
compute pools. It builds `k` compute pools of `m` threads, makes one engine
from each, and gives each lane thread its own engine for the whole run. An
operation inside a task is called from the lane thread and may use all `m`
workers of that lane's engine. The lane thread is not a worker of that
pool, so rule 5 holds.

![Nesting in hataori: k lanes, one CpuEngine of m threads each](figures/nesting.svg)

Keeping the CPU sets disjoint, as drawn, is hataori's choice; tenferro
neither enforces nor checks it. From tenferro's side this is simply `k`
engines used at the same time by different threads.

## 4. GPU execution ownership

The GPU side follows the same model: the caller creates and holds what runs
the work, operations take `&mut` of an engine, and nothing is stored in
runtimes or opened as a session. This section explains what differs.

### Background: devices and streams

* **Device.** One GPU. Opening it sets up a driver context and the memory
  that tensors on that GPU live in. A `CudaDevice` is the caller's handle to
  one opened GPU; cloning the handle is cheap and refers to the same GPU
  state.
* **Stream.** A GPU does not run work when the program asks for it; the
  program puts work into a queue, the *stream*, and the GPU runs the queue in
  order. Work in two different streams may run at the same time and in any
  order relative to each other, unless one stream is told to wait for the
  other.
* **GPU engine.** A `CudaEngine` is made from a device and plays the role of
  a `CpuEngine`: it has its own stream (the GPU counterpart of a thread
  pool) and its own workspace (vendor library handles, plan caches,
  scratch, queues of memory waiting to be released).

| CPU | GPU | role |
| --- | --- | --- |
| Rayon pool, built by the caller | `CudaDevice`, opened by the caller | the shared hardware |
| `CpuEngine::new(pool)` | `CudaEngine::new(&device)` | what operations run on, used through `&mut` |
| workers of the pool | the engine's stream | where the work runs |

```rust
let device = CudaDevice::open(0)?;              // GPU 0
let mut gpu_a = CudaEngine::new(&device);       // two engines on one GPU,
let mut gpu_b = CudaEngine::new(&device);       // each with its own stream

let c = dot_general(&mut gpu_a, &a, &b, &dims)?; // queued on gpu_a's stream
gpu_a.synchronize()?;                            // wait for gpu_a's work and report errors
```

### Rules

1. **Engines from the same device share memory.** A tensor made by one engine
   can be read by another engine of the same device (the same `CudaDevice`
   value or its clones) without a copy. Engines from different devices need
   an explicit transfer, as today.
2. **Sharing tensors between engines is safe without extra calls.** Two
   engines have two streams, so engine B could start reading a tensor before
   engine A has finished writing it, or A could reuse memory that B is still
   reading. tenferro prevents both by tracking use across engines:
   * The record belongs to the memory allocation, not to a tensor object:
     every tensor and view that shares one allocation shares one record. The
     record holds the last write and the reads since then that have not yet
     finished, and finished reads are removed from it.
   * Before an engine reads, it makes its stream wait for the last write if
     another engine made it. Before an engine writes, including writes into
     an existing tensor, it waits for the last write and for every
     unfinished read by other engines.
   * The record lives as long as any of this work is unfinished, regardless
     of what happens to the tensors, views or outputs: dropping or forgetting
     them does not remove the protection.
   * Memory is reused, evicted from a recycling pool, or released only after
     every use in its record has finished, even if the engine that allocated
     it has been dropped. If finishing cannot be confirmed (for example after
     a device error), the memory is set aside and never reused.

   Within one engine no GPU wait is added, because one stream already runs
   in order; the bookkeeping itself has a small cost. Across engines, each
   dependency is one GPU-side wait. The program does not block, although
   handing queued work to the GPU before recording a wait may briefly block
   on today's CubeCL runtime.
3. **Paths outside this tracking are `unsafe`.** Code that reaches device
   memory directly (raw CUDA pointers for a caller's own kernels or vendor
   calls) takes the device memory through an `unsafe` function whose
   contract states which stream may use it and until when.
4. **Flush and synchronize are points in time.** `engine.flush()` hands queued
   work to the GPU; `engine.synchronize()` waits for the engine's stream and
   reports the errors the driver reports there. Neither needs a scope.
   Errors that belong to a particular output buffer still surface when that
   buffer is read.
5. **Dropping an engine waits for its work, best effort.** It may block, it
   cannot report errors, and memory whose completion cannot be proven is kept
   rather than freed. Pending transfers keep their own completion state, so
   a transfer may outlive the engine that started it.
6. **Per operation, an engine only selects its stream.** Driver context
   switches and flushes happen only at the boundaries that need them: raw
   CUDA and vendor library calls (cuBLAS, cuSOLVER, cuFFT, cuTENSOR, a
   caller's own kernels), and explicit submission, waiting and completion
   points. Ordinary kernels pay neither.

WebGPU follows the same shape (`WgpuDevice`, `WgpuEngine`); its queues are
not independent hardware queues, so two WebGPU engines on one device run one
after another.

### What is not yet under tenferro's ownership: CubeCL's runtime layer

tenferro's GPU kernels are built with [CubeCL](https://github.com/tracel-ai/cubecl)
(through the `t4a-cubecl` fork). CubeCL has two layers:

* a **code generation layer** that turns `#[cube]` Rust functions into GPU
  kernels, and
* a **runtime layer** that keeps one process wide server per GPU with its own
  thread, chooses streams by thread-local state, and owns the device memory
  pool.

The runtime layer contradicts this design: it is a hidden per-device global
with its own thread. Two consequences remain in the first version and are
documented, not hidden:

* An engine's stream is a *logical* stream. CubeCL maps logical streams onto
  a fixed set of physical streams, so two engines may share one physical
  stream and then run one after another instead of in parallel.
* CubeCL orders reads only against the stream that allocated a buffer, so
  the tracking of rule 2 is tenferro's own and sits on top of CubeCL.

**Direction.** tenferro takes ownership of the GPU runtime layer in stages:

1. *Now (tenferro-rs#2206).* No CubeCL type appears in the public API. A
   `CudaEngine` is defined by what it owns (a stream, a workspace), so the
   layer below can change without changing the API.
2. *Next.* Replace CubeCL's runtime layer with tenferro's own, built on the
   CUDA driver API (through `cudarc`, which tenferro already uses for its
   vendor libraries): one real stream per engine, a stream-ordered memory
   pool per engine, events for hand-offs, and no device server thread. This
   removes the first consequence above and lets the tracking of rule 2 move
   into tenferro's own allocator. It also removes a blocking
   hand-off to the server thread that every flush pays today.
3. *Later.* Decide separately whether CubeCL stays as a kernel compiler only,
   and what happens to WebGPU, which runs only through CubeCL today.

## 5. What this removes from tenferro

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
* the other ways to run a compiled graph (one-shot, lazy-output, scoped,
  caller-session and deferred variants); only `prepare` and `run` remain
* on the GPU: `CudaBackend` (split into `CudaDevice` and `CudaEngine`), the
  scoped and held CUDA sessions with their reservation and statistics, the
  session thread-locals, the per-runtime table that maps streams to vendor
  handles, and backend clones kept by runtimes

The work these served moves to the caller: a scheduler that runs work in
parallel builds one pool and one engine for each parallel part, and each part
may use its whole pool.

## 6. Costs we accept

* Buffers and scratch are per engine, so memory use and warm up grow with
  the number of engines. Each engine has its own retention limit for
  recycled buffers and its own limit on memory held by unfinished work.
  Plans are shared when engines share prepared programs or a `PlanCache`.
* A bad CPU layout shows up as lost performance, not as an error.
* Using a tensor on another GPU engine adds GPU-side waits and some
  bookkeeping, and two GPU engines may share a physical stream until
  tenferro owns the GPU runtime layer (section 4).
* Calling an engine from a worker of its own pool is still possible and is
  detected at run time.
* Every execution names its engine, so code is more explicit than with a
  runtime that hides its backend. This is intended: the engine in the call is
  exactly what decides where the work runs.

## 7. Consistency with other documents

* The [backend developer guide](https://tensor4all.org/tensor4all-meta/stack/backend.html)
  already describes hataori passing each task the pool it may use. This
  design makes tenferro match that description.
* tenferro-rs design documents for the held session, the phase lease, the
  child ticket and the shared execution scope are superseded by this document
  and should be retired when the implementation lands.

## 8. Plan

Work proceeds from the bottom of the stack up. Downstream libraries are not
changed until tenferro and the layers below it are finished.

Each repository's part is developed on one working branch and opened as a
single pull request when it is finished, not as a series of partial ones.

1. Fix the layers below tenferro first: for example, a serial engine must
   not start Rayon's global pool, which needs a change in strided-rs.
2. Implement sections 3 and 4 and the removals of section 5 in tenferro-rs
   ([tenferro-rs#2206](https://github.com/tensor4all/tenferro-rs/issues/2206)),
   without a compatibility layer. Internal details (how an engine proves it
   is compatible with a prepared program, the exact public names) are settled
   there.
3. Only then port the downstream users to engines built from their own
   pools, each in its own change: tensor4all-rs (adaptive interpolation,
   tests, the C API's default engine), the hataori tenferro adapter,
   tenferro-benchmark, gaugefields-rs.
4. Update the backend guide and retire the superseded tenferro design notes.

## Settled questions

1. *Should the workspace be separable from the pool?* Not in the first
   version. An engine owns its workspace; two engines never share one.
2. *How should pinning behave on macOS?* tenferro does no pinning, so the
   question moves to whoever builds the pools.
3. *Should tenferro keep a mode that builds its own pool?* No. One helper
   presets the Rayon builder; the caller always builds the pool and hands it
   to the engine by value.
4. *Should runtimes own their engine?* No. Engines are passed per call, and
   runtimes keep only what was prepared.

5. *Are "engine" and "session" two concepts or one?* One. Operations take
   `&mut` of the engine through a backend-neutral `Engine` trait, and the
   session is removed (see "One concept: the engine"). The word "backend"
   also leaves the public API: `CpuBackend` becomes `CpuEngine`.

6. *How are tensors handed between GPU engines on one device?* tenferro
   tracks use across engines and inserts the GPU waits itself, so safe code
   cannot race (section 4, rule 2). Direct device memory access is `unsafe`.
7. *Where do plans live?* In prepared programs, and in `PlanCache` values
   that engines refer to and may share. Engines keep what execution writes,
   plus plans tied to their own device handles.
8. *How does prepare know where a program will run?* The caller passes
   `engine.target()` descriptions; run checks that the engines it is given
   match them.

## Open questions (to mature before implementation)

1. *What does a GPU engine cost per operation without a session scope?*
   The session bound the device stream to the calling thread once per
   closure. An engine binds it per operation, or skips the binding when it
   can prove nothing changed; the thread alone is not enough proof, since
   another engine or stream may have been bound on the same thread in
   between. Measure before choosing.

The figures are generated by `figures/make_figures.py` (standard library
Python). Edit the script and rerun it rather than editing the SVG files.
