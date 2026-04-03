# v2 computegraph-rs Design

**Date:** 2026-04-03
**Status:** Draft
**Repo:** computegraph-rs (new)
**Parent:** `README.md`

---

## I. Purpose

`computegraph-rs` is a general-purpose computation graph engine in Rust. It
provides the data structures and transforms for building, resolving, flattening,
compiling, and evaluating computation graphs.

It is **AD-agnostic**. Automatic differentiation is not part of this crate.
The graph infrastructure is equally usable for:

- differentiable programming (via tidu-rs)
- multi-tensor einsum (graph of binary contractions)
- any DAG-structured computation

---

## II. Core Traits

### Operand

`Operand` is the runtime value type. Scalars are rank-0 tensors.

```rust
trait Operand: Clone + Send + Sync + 'static {
    fn zero(shape: &[usize]) -> Self;
    fn one(shape: &[usize]) -> Self;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn reduce_sum(&self, axes: &[usize]) -> Self;
    fn dot_general(&self, other: &Self, ...) -> Self;
    fn conj(&self) -> Self;
}
```

`zero` is required for sparse propagation. `one` is required for seeding
(e.g. reverse-mode AD seeds `ct_y = one`).

**Note:** `Operand` contains tensor-specific methods (`dot_general`, `reshape`,
`broadcast_in_dim`, etc.). This is intentional -- computegraph-rs is designed as
a **tensor computation graph engine**, not a fully generic DAG engine. The
tensor-oriented interface ensures that graph transforms (e.g. AD in tidu-rs) can
reason about tensor structure without depending on concrete primitive types.

### GraphOp

`GraphOp` is the operation node trait. It defines evaluation and arity.
`computegraph` is fully generic over this trait and never references specific
primitives.

```rust
trait GraphOp: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    type Operand: Operand;
    type Context;
    type InputKey: Clone + Debug + Hash + Eq + Send + Sync + 'static;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

`InputKey` is an associated type so that downstream implementors can choose
the representation (e.g. `String`, interned `u64`, domain-specific struct).
```

`Context` is an associated type so that backends can inject execution state
(e.g. `CpuContext`, `CudaContext`).

---

## III. Data Structures

### Fragment

A **Fragment** is the unit of graph construction. It owns only local nodes and
may reference values in other fragments through external references.

```rust
type LocalValId = usize;
type LocalOpId = usize;

enum ValRef<Op: GraphOp> {
    Local(LocalValId),
    External(GlobalValKey<Op>),
}

struct ValNode<Op> {
    key: GlobalValKey<Op>,
    producer: Option<(LocalOpId, usize)>,
}

struct OpNode<Op> {
    op: Op,
    inputs: Vec<ValRef<Op>>,
    outputs: Vec<LocalValId>,
    mode: OpMode,
}

struct Fragment<Op: GraphOp> {
    vals: Vec<ValNode<Op>>,
    ops: Vec<OpNode<Op>>,
    inputs: Vec<LocalValId>,
    outputs: Vec<LocalValId>,
    parents: Vec<Arc<Fragment<Op>>>,
}
```

`parents` are the lookup base used by `resolve`, not eager ownership.

### Global Identity

`GlobalValKey` is the cross-fragment identity mechanism. It is structural:

```rust
enum GlobalValKey<Op: GraphOp> {
    Input(Op::InputKey),
    Derived {
        op: GlobalOpKey<Op>,
        output_slot: u8,
    },
}

struct GlobalOpKey<Op> {
    primitive: Op,
    inputs: Vec<GlobalValKey<Op>>,
    mode: OpMode,
}

enum OpMode {
    Primal,
    Linear { active_mask: Vec<bool> },
}
```

`GlobalValKey` enables:

- external reference resolution
- cross-fragment CSE
- logical reachability analysis

### Key Interning

`GlobalValKey` is recursive and grows with graph depth. To keep equality
comparison O(1) and avoid storing duplicate substructure, all keys are
interned in a global `KeyInterner`:

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ValKeyId(u32);

struct KeyInterner<Op: GraphOp> {
    map: HashMap<GlobalValKey<Op>, ValKeyId>,
    keys: Vec<GlobalValKey<Op>>,  // id → full key (reverse lookup)
}
```

Fragment construction registers every new `GlobalValKey` with the interner
and stores the resulting `ValKeyId`. Cross-fragment equality, CSE, and
compilation cache lookups all use `ValKeyId` comparison (O(1)).

The interner is global (shared across all fragments). This is appropriate
because the primary use case is cross-fragment identity. Thread-safe access
can be added later (`DashMap` or `RwLock<HashMap>`) if parallel fragment
construction becomes necessary.

`parents` in `Fragment` form an acyclic structure: a fragment can only
reference fragments that existed at its construction time, so cycles are
structurally impossible.

### ResolvedView

`resolve` builds a logical lookup view over fragments without copying nodes.

```rust
trait Resolver<Op> {
    fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>>;
}

struct ResolvedView<Op> {
    roots: Vec<Arc<Fragment<Op>>>,
    resolver: Arc<dyn Resolver<Op>>,
}
```

### MaterializedGraph

The fully flattened graph produced by `materialize_merge`:

```rust
struct MaterializedGraph<Op> {
    vals: Vec<MaterializedVal<Op>>,
    ops: Vec<MaterializedOp<Op>>,
    inputs: Vec<GlobalValKey<Op>>,
    outputs: Vec<GlobalValKey<Op>>,
}
```

---

## IV. Transforms

### `resolve`

```rust
fn resolve<Op: GraphOp>(roots: Vec<Arc<Fragment<Op>>>) -> ResolvedView<Op>;
```

Cheap and logical. Prepares a traversal view over fragment parents and
external references. Does not copy nodes or run CSE.

### `materialize_merge`

```rust
fn materialize_merge<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
) -> MaterializedGraph<Op>;
```

Walks the resolved logical DAG, collects reachable definitions, deduplicates
by `GlobalValKey`/`GlobalOpKey`, and computes one concrete topological order.

### `compile`

```rust
fn compile<Op: GraphOp>(graph: &MaterializedGraph<Op>) -> CompiledProgram<Op>;
```

Produces an SSA-form instruction sequence. Each slot is written exactly once.

```rust
struct CompiledProgram<Op: GraphOp> {
    instructions: Vec<Instruction<Op>>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
}

struct Instruction<Op> {
    op: Op,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}
```

### `eval`

```rust
impl<Op: GraphOp> CompiledProgram<Op> {
    fn eval(&self, ctx: &mut Op::Context, inputs: &[&Op::Operand]) -> Vec<Op::Operand>;
}
```

Compile once, eval many times.

---

## V. Compilation Cache

`computegraph` caches compiled programs keyed by graph structure
(`GlobalValKey`-based identity) to avoid recompilation when the same graph
is submitted multiple times with different input values.

---

## VI. Logical Traversal

All walkers operate on the same rule:

```text
Local(LocalValId)       -> follow the local producer
External(GlobalValKey)  -> ask the resolver for the defining op
```

This works recursively through any number of fragment boundaries.
Topological traversal is logical, not physical:

- visitation is keyed by `GlobalValKey`
- ordering is computed on the resolved logical DAG
- local ids only matter inside the fragment currently being built

---

## VII. Design Boundaries

```text
computegraph owns:
  - GraphOp, Operand traits
  - Fragment, ResolvedView, MaterializedGraph
  - resolve, materialize_merge, compile, eval
  - compilation cache

computegraph does NOT own:
  - AD transforms (differentiate, transpose) → tidu-rs
  - AD traits (linearize, transpose_rule) → chainrules-rs
  - concrete primitives → downstream (tenferro-rs)
  - backend-specific lowering (StableHLO) → downstream
```
