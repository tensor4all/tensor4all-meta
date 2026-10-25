# The CPU placement token, with figures

This is a walkthrough of "Where placement is decided" in
[core-boundary-and-execution-ownership.md](./core-boundary-and-execution-ownership.md).
That document holds the rules; this one explains them with pictures. The token
lives in `strided-exec` (strided-rs, tensor4all/strided-rs#331).

## What the token is

A token is permission to run work on a pool, with a limit on how many tasks may
run at once. It holds two things: a borrowed reference to a pool, and a budget.
It does not own the pool. The engine owns the pool and makes a fresh token for
each operation.

![CpuEngine owns its pool; Exec borrows it and carries a budget](figures/token_anatomy.svg)

* `Pool` is tensor4all's own type. It wraps a Rayon pool, but no Rayon type
  appears in any public signature, so the scheduler behind it can be replaced
  inside `strided-exec` alone.
* A token is neither `Clone` nor `Copy`. Holding a token with budget 4 means
  holding 4 slots of that pool, and slots can be divided but never duplicated.
* Only the pool's owner can make a token, because `Exec::new` needs the pool
  itself and a token never gives its pool back out.
* A token without a pool is serial: everything runs on the calling thread and
  no global pool is ever started.

## How one operation uses it

The engine makes the token and passes it down by value through tenferro's
dispatch to the provider crates (tcontract, strided, tlinalg). The pool itself
is never split or copied on the way; the layers pass the same reference along.
Nothing stores the token.

A kernel first looks at the problem size. A small operation runs where it is:
a 1 KiB elementwise call costs about 40 ns, and entering a pool costs
microseconds. A large one calls one of the parallel entries, and each entry
consumes the token:

| Entry | What it does |
| --- | --- |
| `fork(jobs, \|i, child\| ...)` | runs `jobs` jobs on at most `budget` tasks; each job gets a child token on the same pool holding its share |
| `for_ranges(len, grain, \|r\| ...)` | data-parallel loop: at most `budget` tasks take ranges from a shared counter; each range runs sequentially |
| `external(\|w\| ...)` | enters the pool and hands a plain width to a library outside tensor4all (faer, a vendor BLAS) |
| `broadcast(\|t\| ...)` | barrier-bearing kernels: all `budget` participants run at once, or the call is refused |

Two more operations keep the slot count without starting tasks. `reborrow`
lends the token to one phase while the caller keeps it for the next, so a
kernel that fills and then copies passes `exec.reborrow()` to each step.
`limit(max)` narrows the budget, for example to the team size a work model
picked. `broadcast` is accepted only for a token from `new` (or its reborrows
and limits); a child of `fork` is refused, because its slots may share workers
with its siblings, and the caller then takes its barrier-free path.

Splitting can be nested to any depth, and at every level the children's budgets
add up to the parent's:

![A budget of 4 forked into 2 + 2; one child loops, the other calls a library](figures/token_split.svg)

```rust
// with a token of budget 4, run two independent parts in parallel
exec.fork(2, |i, child| {                 // child.budget() == 2
    match i {
        0 => child.for_ranges(len, grain, |r| add(r)),          // at most 2 tasks
        _ => child.external(|w| matmul(.., faer::Par::Rayon(w))), // width 2
    }
});
```

Because a token cannot be copied and every entry consumes it, no code can hand
the same budget to two children. Because only `strided-exec` may depend on
Rayon (`cargo deny`), no code can start a task without a token. Together these
make "at most `budget` tasks of one token run at any moment" true by
construction.

## Entering a pool

Every parallel entry decides in the same way whether to enter the pool:

![Serial runs inline; a worker of the pool spawns in place; anyone else enters once](figures/token_entry.svg)

## Splitting across engines and inside one operation

There are two kinds of splitting, and they use different tools.

**Across engines, use separate pools.** A caller such as hataori that runs `k`
lanes at once builds `k` pools of `m` threads and makes one engine from each.
Each lane thread holds its engine for the whole run.

![k lanes, one CpuEngine of m threads each](figures/nesting.svg)

```rust
let engines: Vec<CpuEngine> = (0..k)
    .map(|_| Ok(CpuEngine::new(Pool::builder().threads(m).build()?)))
    .collect::<Result<_>>()?;
```

**Inside one operation, split the token.** An operation with independent parts
calls `fork` on its token. The children run on the same pool, within the
parent's budget.

Two engines never share one pool. A budget is a cap, not a reservation of
particular workers, so two engines on one pool would compete for the same
queue, and each engine's per-pool state could not see the other. The engine
therefore takes its pool by value.

![Two engines on one pool is not allowed; give each engine its own pool](figures/shared_pool.svg)

## How the rules are checked

| Mechanism | What it prevents | Check |
| --- | --- | --- |
| tokens cannot be copied | handing one budget to two children | compile-fail doctest: `Exec` is not `Clone` or `Copy` |
| only the pool's owner makes tokens | a kernel minting a full-budget token | `new` takes `&Pool`; no accessor returns the pool |
| splits keep the total | a split that grows the budget | property test: shares add up to the parent's budget |
| only `strided-exec` depends on Rayon | starting tasks without a token | `cargo deny` with `wrappers` |
| no Rayon type is public | a scheduler swap breaking callers | `cargo public-api` output contains no `rayon::` |
| the ledger | anything the above missed | per root token, the peak number of live tasks stays within the budget in every test |

What the token does not bound: a call into faer or a vendor BLAS is a leaf whose
internal concurrency belongs to that library, and inside one pool a task's
children can run on any worker. The main document explains both.

## faer

`strided-exec` knows nothing about faer. It hands over a pool and a width
through `external`; t4a-faer turns the width into faer's own scheduling, which
is spindle. spindle is confined to t4a-faer: no other tensor4all crate
depends on it, and a batched caller such as tlinalg splits its batch with the
token (`for_ranges` with `Par::Seq` items when the batch is at least the
budget, otherwise `fork` with one `external` per item). How spindle works, and
why the region it needs is opened inside t4a-faer, is in
[spindle.md](./spindle.md).

## Planned: one pool per cache domain

Many machines have several shared-cache domains. Inside one Rayon pool any
worker may steal any task, so a pool that spans domains moves data between
caches. The plan is one Rayon pool per domain, bundled by `Pool`; a `fork`
splits at domain boundaries first, and each child receives one domain.

![An M5 Max has three cache domains; fork gives each child one domain's pool](figures/cache_domains.svg)

Separate Rayon pools never share a worker, so a child cannot leave its domain at
any depth. Because no Rayon type is public, this is a change inside
`strided-exec`.
