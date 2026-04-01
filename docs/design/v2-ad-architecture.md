# v2 AD Architecture

**Date:** 2026-04-01
**Status:** Draft
**Repos:** chainrules-rs, tidu-rs, tenferro-rs

---

## I. Vision

Build a differentiable programming stack in Rust where:

- **`differentiate` is the only derivative operation.** It is a graph-to-graph
  transformation that adds tangent nodes by linearizing each Op. Applying it
  n times gives the n-th derivative. All derivative information comes from
  this single operation.
- **`transpose` is not differentiation — it is an evaluation strategy.** It
  rewrites a linear map to flow in the reverse direction (JVP → VJP). It is
  always optional, always at the end, and does not add derivative information.
- **Forward and backward are unified.** Both are graph transformations on the
  same DAG. Evaluation is always upstream to downstream. No special backward
  evaluation mode.
- The same graph can be **compiled to CPU execution or lowered to StableHLO**
  for GPU/TPU.

| Goal | Transformations |
|------|----------------|
| JVP (1st derivative, forward) | `differentiate` |
| VJP (1st derivative, backward) | `differentiate` → `transpose` |
| n-th derivative | `differentiate` × n |
| n-th derivative as gradient | `differentiate` × n → `transpose` |

Three crates, strictly layered:

```
chainrules2    PrimitiveOp trait (op contract)
    ↓
tidu2          Graph engine (build, differentiate, transpose, compile, eval)
    ↓
tenferro2      Tensor primitives + StableHLO lowering
```

---

## II. Computation Graph

### Data structure

A computation is a directed acyclic graph (DAG) with two node types:

- **Val node** (circles): a value. External input or one output slot of an Op.
- **Op node** (boxes): a primitive operation. Fixed input/output arity.

```
f(x) = exp(a * x):

  (v0:x)  (v1:a)
     \     /
     [Mul]          Op0
       |
     (v2:a*x)
       |
     [Exp]          Op1
       |
     (v3:exp(a*x))
```

Val nodes `( )` carry values. Op nodes `[ ]` carry operations.
A Val can fan out to multiple Ops (e.g., `v3` referenced by derivative nodes).

```rust
type ValId = usize;
type OpId = usize;

struct ValNode {
    producer: Option<(OpId, usize)>,  // None for inputs
}

struct OpNode<Op> {
    op: Op,
    inputs: Vec<ValId>,
    outputs: Vec<ValId>,
}

struct Graph<Op: PrimitiveOp> {
    vals: Vec<ValNode>,
    ops: Vec<OpNode<Op>>,
    op_cache: HashMap<(Op, Vec<ValId>), OpId>,
}
```

### No constant/variable distinction

All leaf nodes are `Input`. Whether an input is a "constant" or a "differentiable variable" is decided at differentiation time (`wrt` argument), not in the graph structure.

### Content-addressable CSE

`OpId` is determined by `(op_kind, input_val_ids...)`. Adding the same op with the same inputs returns the existing node. Duplicate computation is structurally impossible.

### Single graph for all levels

Primal, tangent, cotangent, and higher-order derivative nodes all live in the **same graph**. Shared values (e.g., `exp(x)`) are computed once and referenced by all levels.

---

## III. Differentiation

### Two graph transformations

| Operation | Direction | What it does |
|-----------|-----------|--------------|
| `differentiate` | forward (upstream → downstream) | Linearize each Op, add tangent nodes |
| `transpose` | backward (downstream → upstream) | Reverse the linear map, add cotangent nodes |

Both are **graph → graph** transformations. They add new nodes; they do not evaluate anything.

### differentiate (forward linearize)

Walks Ops in topological order. For each Op, calls `op.linearize()` which adds tangent nodes to the graph. Primal output vals are referenced as coefficients (not recomputed).

```rust
fn differentiate(
    &mut self,
    outputs: &[ValId],
    wrt: &[(ValId, ValId)],    // (Input node, tangent Input node)
) -> Vec<Option<ValId>>        // tangent per output
```

`wrt` restricted to Input nodes. Inputs not in `wrt` have tangent = zero → zero propagation skips downstream nodes.

### transpose (backward)

The linearized computation is a linear map `L: tangent_in → tangent_out`. Transpose produces `Lᵀ: cotangent_out → cotangent_in`.

Transpose is also a graph transformation using IR primitives:

```
Add  ↔  Dup        (sum ↔ broadcast)
Scale(c) ↔ Scale(c)  (self-adjoint)
Mul(a, ·) ↔ Mul(a, ·)  (self-adjoint)
```

With a sufficiently rich primitive set, all transposes are expressible as IR ops. No special backward evaluation mode needed.

### Higher-order AD = repeat `differentiate`

n-th derivative = apply `differentiate` n times. `transpose` is optional (evaluation choice).

```
1st derivative:   differentiate
2nd derivative:   differentiate → differentiate
3rd derivative:   differentiate → differentiate → differentiate
```

To evaluate any of these as a gradient (backward), append `transpose`:

```
gradient:              differentiate → transpose
Hessian-vector product: differentiate → differentiate → transpose
```

FoR, RoR etc. are just different orderings of `differentiate` and `transpose`.
The derivative content is identical — only the evaluation direction differs.

### Zero propagation

During differentiation, if all tangent inputs to a node are zero (node doesn't depend on `wrt`), skip it. Essential for efficiency in higher-order differentiation.

### Concrete example: `f(x) = exp(a * x)`

**Primal graph:**

```
  (v0:x)  (v1:a)
     \     /
     [Mul]
       |
     (v2)
       |
     [Exp]
       |
     (v3:y)              y = exp(a*x)
```

**After differentiate (JVP w.r.t. x):**

New nodes added (right side). `v3` is referenced from primal graph.

```
  (v0:x)  (v1:a)         (v4:t_x)  (v1:a)
     \     /                  \     /
     [Mul]                    [Mul]
       |                        |
     (v2)                     (v5)        a * t_x
       |                        |
     [Exp]               (v3)--[Mul]
       |                 /      |
     (v3)───────────────'     (v6)        exp(a*x) * a * t_x
```

**After transpose (VJP):**

Cotangent flows backward through the linearized map.

```
                              (v7:ct_y)
                                |
                         (v3)--[Mul]
                         /      |
  (v0:x)  (v1:a)       /     (v8)        exp(a*x) * ct_y
     \     /           /        |
     [Mul]            /   (v1)--[Mul]
       |             /    /     |
     (v2)           /    /    (v9:ct_x)   a * exp(a*x) * ct_y
       |           /    /
     [Exp]        /    /
       |         /    /
     (v3)───────'    /
     (v1)───────────'
```

**Full graph after FoR (2nd derivative):**

```
v0  = Input(x)
v1  = Input(a)
v2  = Mul(v0, v1)                   // a*x                [primal]
v3  = Exp(v2)                       // exp(a*x)            [primal]
v4  = Input(t_x)
v5  = Mul(v4, v1)                   // a*t_x               [fwd linearize]
v6  = Mul(v3, v5)                   // exp(a*x)*a*t_x      [fwd linearize]
v7  = Input(ct_y)
v8  = Mul(v3, v7)                   // exp(a*x)*ct_y       [bwd transpose]
v9  = Mul(v1, v8)                   // a*exp(a*x)*ct_y     [bwd transpose]
v10 = Input(dt_x)
v11 = Mul(v1, v10)                  // a*dt_x              [FoR linearize]
v12 = Mul(v3, v11)                  // exp(a*x)*a*dt_x     [FoR linearize]
v13 = Mul(v12, v7)                  // exp(a*x)*a*dt_x*ct_y [FoR linearize]
v14 = Mul(v1, v13)                  // a²*exp(a*x)*dt_x*ct_y [FoR linearize]
```

`v3 = exp(a*x)` computed once, referenced by v6, v8, v12. All methods give `d²f/dx² = a² exp(ax)` ✓

---

## IV. Primitive Set

### PrimitiveOp trait (defined in chainrules2)

```rust
pub trait PrimitiveOp: Clone + Hash + Eq + Send + Sync + 'static {
    type Operand: Clone + Send + Sync + 'static;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;

    fn eval(&self, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;

    fn linearize(
        &self,
        graph: &mut Graph<Self>,
        primal_in: &[ValId],
        primal_out: &[ValId],
        tangent_in: &[Option<ValId>],
    ) -> Vec<Option<ValId>>
    where
        Self: Sized;
}
```

Linearize rules are standalone functions; the trait impl is a `match` dispatch:

```rust
impl PrimitiveOp for ScalarOp {
    fn linearize(&self, g: &mut Graph<Self>, p_in: &[ValId], p_out: &[ValId], t_in: &[Option<ValId>]) -> Vec<Option<ValId>> {
        match self {
            Exp => linearize_exp(g, p_in, p_out, t_in),
            Mul => linearize_mul(g, p_in, p_out, t_in),
            // ...
        }
    }
}
```

### Design principle

A primitive's `linearize` must be **linear in the tangent inputs**. It may:
- Reference primal output vals as coefficients (e.g., `Exp` references its own output `exp(x)`)
- Emit other primitives (e.g., `Exp` emits `Mul`)
- NOT apply nonlinear ops to tangent inputs

### Two tiers (defined in tenferro2)

**Tier 1 — Semiring Core**: sufficient for einsum-based computation. Compatible with custom algebraic backends (tropical, p-adic, polynomial rings, etc.).

- Elementwise: `Add`, `Mul`, `Scale`, `Neg`
- Tensor: `Einsum`, `Transpose`, `Reshape`, `Broadcast`, `Dup`
- Reduction: `Sum`, `Prod`

**Tier 2 — Standard = Core + JAX prims**: full JAX-compatible set for general-purpose differentiable programming.

- Transcendental: `Exp`, `Log`, `Sin`, `Cos`, ...
- Linalg: `SVD`, `Cholesky`, `QR`, `Eig`, `Solve`
- Indexing: `Gather`, `Scatter`, `Slice`
- Control: `Cond`, `Scan`, `While`
- Misc: `Sort`, `FFT`, `Conv`, ...

Each tier 2 primitive's `linearize` is expressed in terms of tier 1 + tier 2 primitives.

### Operand type

`PrimitiveOp::Operand` is the type of data flowing through the graph. tidu2 does not inspect it.

```
Tropical:  Operand = Tensor<f64>    (Add=max, Mul=plus)
Standard:  Operand = Tensor<f64>    (normal arithmetic)
Standard:  Operand = DynTensor       (scalar type erased)
Scalar:    Operand = f64             (Phase 1 testing)
```

---

## V. Compilation & Execution

### Two-stage compilation

```
                     ┌─────────────────────────┐
                     │  Graph (DAG)             │
                     │  Val nodes + Op nodes    │
                     └────────┬────────────────┘
                              │ differentiate / transpose
                              │ (graph → graph)
                              ▼
                     ┌─────────────────────────┐
                     │  Transformed Graph       │
                     │  primal + tangent + ...  │
                     └────────┬────────────────┘
                              │ compile (toposort + SSA)
                              ▼
                     ┌─────────────────────────┐
                     │  CompiledProgram         │
                     │  flat slot-based IR      │
                     └────────┬────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌────────────────┐      ┌─────────────────────┐
        │  prog.eval()   │      │  lower to StableHLO  │
        │  (CPU)         │      │  → IREE/XLA (GPU)    │
        └────────────────┘      └─────────────────────┘
```

### CompiledProgram (SSA IR)

```rust
struct CompiledProgram<Op: PrimitiveOp> {
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

`compile(target_nodes, input_nodes)` topologically sorts from targets, producing a flat instruction sequence. Compile once, eval many times with different input values.

```rust
let prog = g.compile(&[v9], &[x, a, ct_y]);
let grad1 = prog.eval(&[2.0, 3.0, 1.0]);
let grad2 = prog.eval(&[4.0, 3.0, 1.0]);
```

### Backend dispatch

| Primitive tier | Backend | Use case |
|---------------|---------|----------|
| Tier 1 only | Custom algebra backend | Tropical, p-adic, polynomial rings |
| Tier 1 + 2 | StableHLO → IREE/XLA | Standard float/complex on GPU/TPU |

tidu2 is backend-agnostic. It produces `CompiledProgram<Op>` without knowing the execution target.

---

## VI. Crate Architecture

### chainrules2

**Role**: define the `PrimitiveOp` trait — the contract between tidu2 (engine) and tenferro2 (primitives).

Contains:
- `PrimitiveOp` trait
- `ValId`, `OpId` types
- Error types

Does NOT contain: Graph, differentiation logic, compilation, or any concrete ops.

### tidu2

**Role**: graph engine. Build, transform, compile, evaluate.

Contains:
- `Graph<Op>` data structure
- `differentiate()`, `transpose()` (graph transformations)
- `compile()` → `CompiledProgram`
- `CompiledProgram::eval()` (CPU execution)

Does NOT contain: concrete ops, StableHLO, tensor types.

### tenferro2

**Role**: tensor primitives and hardware backends.

Contains:
- `TensorOp` enum implementing `PrimitiveOp` (Tier 1 + Tier 2)
- `Tensor<T>`, `DynTensor` types
- Linearize rules for each primitive
- StableHLO lowering of `CompiledProgram<TensorOp>`
- IREE/XLA backend integration

### Dependency graph

```
chainrules2          (PrimitiveOp trait, no dependencies)
    ↓
tidu2                (Graph engine, depends on chainrules2)
    ↓
tenferro2            (Tensor prims + backends, depends on chainrules2 + tidu2)
```

---

## VII. Advantages of Graph-Based Design

### Cross-country evaluation

The linearized graph can be **partially transposed**. Forward for the first half, backward for the second half, meeting at an optimal split point.

```
Full forward:   t_x → L1 → L2 → L3 → L4 → t_y
Full backward:  ct_y → L4ᵀ → L3ᵀ → L2ᵀ → L1ᵀ → ct_x
Cross-country:  t_x → L1 → L2 → mid ← L4ᵀ ← L3ᵀ ← ct_y
```

This is a graph transformation (transpose only part of the chain). Evaluation remains always forward. The optimal split point can be determined automatically based on op costs and graph structure.

### Runtime checkpoint optimization

Each op declares cost and output size:

```rust
trait OpInfo {
    fn cost(&self, input_sizes: &[usize]) -> f64;
    fn output_size(&self, input_sizes: &[usize]) -> usize;
}
```

The graph scheduler uses **runtime information** (actual tensor sizes, available memory) to decide which intermediate values to keep vs. recompute. This replaces static `CheckpointStrategy` hints with graph-level optimization:

- Given a memory budget → minimize total computation
- Given a time budget → minimize peak memory
- Automatic, based on the full graph structure — not per-op decisions

### Unified CSE across all levels

Since primal, tangent, cotangent, and higher-order nodes share a single graph, CSE operates across all levels. A value like `exp(x)` is guaranteed to appear once regardless of how many derivative levels reference it.

### Compilation separation

Graph construction and transformation happen once. The compiled SSA IR can be:
- Evaluated repeatedly with different inputs (CPU)
- Lowered to StableHLO for GPU/TPU compilation
- Serialized and cached

---

## Appendix: Design Notes

### A. Custom rules for user-defined functions

Users can extend the primitive set by adding variants to their `PrimitiveOp`
enum. This is analogous to Julia's `ChainRulesCore.frule` / `rrule`.

To avoid exposing low-level graph internals (`ValId`, `Graph`) in user code,
provide a `Traced` wrapper with operator overloading:

```rust
// User writes normal math — Traced records ops into the graph
fn linearize_dyson(
    g0: Traced, sigma: Traced,       // primal values
    t_g0: Traced, t_sigma: Traced,   // tangent values
) -> Traced {
    g0.exp() * t_g0 + sigma * t_sigma
}
```

```rust
// Traced is a thin wrapper around ValId with operator overloading
struct Traced<'g, Op: PrimitiveOp> {
    ctx: &'g GraphContext<Op>,
    val_id: ValId,              // hidden from user
}

impl Mul for Traced<'_, Op> { ... }  // calls ctx.add_op(Mul, ...)
impl Add for Traced<'_, Op> { ... }  // calls ctx.add_op(Add, ...)
```

No `ValId` leaks into user code. Comparable to Julia's ChainRules experience.

### B. Wrapper structs and pytree-style AD

When AD targets are wrapped in domain structs (e.g., `GreenFunction` containing
a `Tensor<f64>` plus metadata), the wrapper declares which fields are
differentiable ("leaves") via a `Differentiable` trait:

```rust
#[derive(Differentiable)]
struct GreenFunction {
    #[leaf]
    data: Tensor<f64>,      // AD target
    mesh: ImagTimeMesh,     // metadata, not differentiated
    beta: f64,              // parameter, not differentiated
}
```

The graph operates on leaves (`Tensor<f64>`). Pack/unpack is automatic:

```rust
trait Differentiable {
    type Leaves;
    fn to_leaves(&self) -> Self::Leaves;
    fn from_leaves(leaves: Self::Leaves, template: &Self) -> Self;
}
```

Custom linearize rules are written at the leaf level using `Traced` values.
Metadata is preserved through `from_leaves(result, template)`.

This is analogous to JAX's pytree + `Functors.jl`'s `@functor`.

---

## VIII. Roadmap

### Phase 1: Scalar graph AD

- `Graph<ScalarOp>` with `Operand = f64`
- Primitives: `Mul`, `Add`, `Scale`, `Exp`, `Dup`, `Neg`
- `differentiate`, `transpose`, `compile`, `eval`
- Tests: forward, backward, FoF, FoR, RoF, RoR for `exp(a*x)`
- Lives in tidu2 on `feat/v2` branch

### Phase 2: Tensor primitives

- `TensorOp` enum with Tier 1 (Semiring Core) + Tier 2 (Standard)
- `Operand = Tensor<f64>` and `DynTensor`
- Linearize rules for all primitives
- Integration with existing tenferro-rs tensor infrastructure

### Phase 3: Backend compilation

- StableHLO lowering of `CompiledProgram<TensorOp>`
- IREE/XLA integration for GPU/TPU execution
- Custom algebra backend for Tier 1 programs

### Phase 4: Optimization

- Graph scheduler for checkpoint optimization
- Cross-country mode (automatic forward/backward split)
- Operator fusion in compiled IR

---

## Superseded Issues

This document unifies and supersedes:

- tidu-rs#12: tape AD design
- tidu-rs#13: graph-based AD design
- chainrules-rs#7: trait unification
- chainrules-rs#8: DifferentiableOp trait
- tenferro-rs#616: Traced Tensor + StableHLO IR
- tenferro-rs#618: tenferro v2 roadmap (AD portions)
