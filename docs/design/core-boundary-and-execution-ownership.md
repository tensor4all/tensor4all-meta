# tenferro core boundary and CPU execution ownership

**Status:** Proposed future design. Target: implementation starts around 2026-10-27.
**Scope:** tenferro-rs, the provider crates (tcontract, cpueinsum, tlinalg, t4a-faer), hataori-rs, and downstream libraries that write their own kernels (symtn, tensor4all-rs).
**Checked against:** tenferro-rs main `2ecd7911`, symtn main `88e5895b` (2026-10-19).

This document fixes two cross-repository decisions. The first is what belongs
in the tenferro core and what users of tenferro build themselves. The second
is who owns CPU threads. Repository-internal details (type names, migration
steps) belong in tenferro-rs issues and `docs/design/`.

## 1. Policy in one sentence each

1. The tenferro core stays small. Batches whose items do not share one shape
   are handled by the caller, with a dedicated kernel written against the
   provider crates.
2. The caller owns the threads. Every layer, tenferro included, borrows the
   pool it is given and never decides how the machine is partitioned.

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

### The CPU engine

A `CpuEngine` is a plain value:

| Part | Content |
| --- | --- |
| pool | a Rayon pool, either owned by the engine or borrowed from the caller |
| workspace | buffer pool, GEMM analysis cache, indexed plan cache, N-ary contraction workspace |

Rules:

1. The engine is used through `&mut`, so one caller uses it at a time.
2. Every operation may use the whole pool of its engine. Throttling for
   small work happens inside the providers, which already choose the number
   of lanes from a work model within the budget they are given.
3. The engine hands out the provider token for its pool (sketch:
   `engine.exec(budget) -> tcontract_exec::Exec<'_>`), so a downstream kernel
   uses the same threads and the same budget as tenferro.
4. The caller of an operation must not be a worker of that engine's pool.
   tenferro reports this as an error rather than silently running serially.

How one operation runs:

![How one operation runs through CpuEngine and a provider](figures/operation.svg)

A downstream kernel takes the same path from the third step on, by calling
`engine.exec(budget)` itself.

Sketch (names are not final):

```rust
let pool_a = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
let mut engine_a = CpuEngine::from_pool(&pool_a);      // borrowed
let mut engine_b = CpuEngine::builder().threads(4).build()?; // owned

let c = dot_general(&mut engine_a, &a, &b, &dims)?;    // uses all of pool_a
```

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

* Whether the pools of different engines overlap on CPUs, and whether the
  total thread count oversubscribes the machine, is the caller's
  responsibility. tenferro does not check it. The engine builder accepts a
  thread count and an optional CPU set for pinning, so that a disjoint layout
  is easy to build.
* Threads started by a vendor BLAS or LAPACK are outside tenferro's control,
  as before.

## 4. What this removes from tenferro

The following exist only to let tenferro own and subdivide threads on behalf
of its callers, and are removed:

* the process global resource arbiter and its CPU set conflict detection
* the per placement engine registry
* owner thread locals used as admission keys, and owner inheritance
* the held CPU session, the per worker phase lease, and the child execution
  ticket, together with their error types

The work these served moves to the caller: per lane execution becomes one
engine per lane, and lanes are no longer forced to run serially inside.

## 5. Costs we accept

* Workspaces are per engine, so memory use and warm up grow with the number
  of engines.
* A bad CPU layout shows up as lost performance, not as an error.
* Calling an engine from a worker of its own pool remains possible with a
  borrowed pool, and must be detected at run time.

## 6. Consistency with other documents

* The [backend developer guide](https://tensor4all.org/tensor4all-meta/stack/backend.html)
  already describes hataori passing each task the pool it may use. This
  design makes tenferro match that description.
* tenferro-rs design documents for the held session, the phase lease, the
  child ticket and the shared execution scope are superseded by this document
  and should be retired when the implementation lands.

## 7. Plan

1. Open a tenferro-rs issue that implements section 3 and lists the removals
   of section 4, without a compatibility layer.
2. Port the downstream users (tensor4all-rs adaptive interpolation, the
   hataori tenferro adapter) to one engine per lane.
3. Update the backend guide and retire the superseded tenferro design notes.

## Open questions

1. Should the workspace be separable from the pool, so that two engines over
   the same pool at different times can share caches?
2. How should the pinning option behave on macOS, where thread affinity is
   advisory?

The figures are generated by `figures/make_figures.py` (standard library
Python). Edit the script and rerun it rather than editing the SVG files.
