# v2 AD Architecture

**Date:** 2026-04-01 (updated 2026-04-02)
**Status:** Draft
**Repos:** chainrules-rs, tidu-rs, tenferro-rs

---

## I. Vision

Build a differentiable programming stack in Rust where:

- **`differentiate` is the only derivative operation.** It takes a graph and
  produces a NEW independent linear graph representing the JVP. Applying it
  n times gives the n-th derivative. All derivative information comes from
  this single operation.
- **`transpose` is not differentiation — it is an evaluation strategy.** It
  flips a linear graph to reverse the data flow direction (JVP → VJP). It is
  always optional, always at the end, and does not add derivative information.
- **Graphs are independent until merge.** `differentiate` and `transpose`
  produce new graphs. Graphs reference values from other graphs by name
  (content-addressable ID). Only at evaluation time are graphs **merged**,
  unifying same-named nodes (automatic CSE).
- **Evaluation is always forward** (upstream to downstream), after merge.
- The merged graph can be **compiled to CPU execution or lowered to StableHLO**
  for GPU/TPU.

| Goal | Transformations |
|------|----------------|
| JVP (1st derivative, forward) | `differentiate` |
| VJP (1st derivative, backward) | `differentiate` → `transpose` |
| n-th derivative | `differentiate` × n |
| n-th derivative as gradient | `differentiate` × n → `transpose` |

Six operations:

```
build          construct a primal graph
differentiate  graph → NEW linear graph (JVP)
transpose      linear graph → NEW linear graph (flip direction)
merge          unify multiple graphs by content-addressable ID (CSE)
compile        merged graph → CompiledProgram (flat SSA IR)
eval           CompiledProgram + input values → output values
```

Three crates, strictly layered:

```
chainrules2    PrimitiveOp trait (op contract)
    ↓
tidu2          Graph engine (build, differentiate, transpose, merge, compile, eval)
    ↓
tenferro2      Tensor primitives + StableHLO lowering
```

---

## II. Computation Graph

### Data structure

A computation is a directed acyclic graph (DAG) with two node types:

- **Val node** (circles): a value. External input or one output slot of an Op.
  Can be consumed by any number of downstream Ops (fan-out).
- **Op node** (boxes): a primitive operation. Fixed input/output arity per op
  kind. At most 2 inputs is sufficient for most ops.

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

All leaf nodes are `Input`. Whether an input is a "constant" or a
"differentiable variable" is decided at differentiation time (`wrt` argument),
not in the graph structure.

### Content-addressable identity

`OpId` is determined by `(op_kind, input_val_ids...)`.
`ValId` is determined by `(op_id, output_slot)`.

Adding the same op with the same inputs returns the existing node.
**This identity is used across graphs**: when two independent graphs contain
the same `(op_kind, input_val_ids...)`, `merge` unifies them into a single
node. This is how CSE works across primal, tangent, and cotangent graphs.

### Graphs are independent until merge

Each graph is self-contained. `differentiate` and `transpose` produce NEW
graphs that reference values from other graphs by content-addressable ID
(dangling references). These references are resolved at `merge` time.

```
G_primal:     v0=Input(x), v1=Input(a), v2=Mul(v0,v1), v3=Exp(v2)
G_linear:     v4=Input(dx), v5=Mul(v1,v4), v6=Mul(v3,v5)
                                     ^^          ^^
                            references v1, v3 from G_primal (by name)

merge(G_primal, G_linear) → unified graph with shared v1, v3
```

---

## III. Differentiation

### `differentiate`: graph → NEW linear graph

Takes a primal graph and produces a NEW independent graph representing the
linearized computation (JVP). The new graph references primal values by
content-addressable ID.

```rust
fn differentiate(
    primal_graph: &Graph<Op>,
    outputs: &[ValId],
    wrt: &[(ValId, ValId)],    // (Input node, tangent Input node)
) -> (Graph<Op>, Vec<Option<ValId>>)  // NEW graph + tangent outputs
```

Walks Ops in topological order. For each Op, calls `op.linearize()` which
adds tangent nodes to the NEW graph. Primal output vals are referenced (not
recomputed). `wrt` restricted to Input nodes. Zero propagation skips nodes
not depending on `wrt`.

### `transpose`: flip a linear graph

Takes a linear graph and produces a NEW graph with reversed data flow.
Each linear op is replaced by its transpose (adjoint).

**Real case**: transpose = matrix transpose.

```
Add       ↔  Dup          (sum ↔ broadcast)
Scale(c)  ↔  Scale(c)     (self-adjoint)
Mul(a, ·) ↔  Mul(a, ·)    (self-adjoint)
```

**Complex case**: transpose = Hermitian adjoint (conjugate transpose).
The adjoint of a linear map on a complex inner product space `<u,v> = u†v`
requires conjugation. `Conj` nodes are inserted during transpose.

```
Scale(c)  ↔  Scale(Conj(c))     ← conjugate the coefficient
Mul(a, ·) ↔  Mul(Conj(a), ·)    ← conjugate the primal
Add       ↔  Dup                 ← unchanged (real-valued structure)
```

Example: `f(z) = c * z` where `c = 2+3i`, `z ∈ C`.

```
Linearize (same as primal, linear op):
  G_linear:
    dz = Input(tangent)
    dy = Scale(c, dz)          // (2+3i) * dz

Transpose (real):               Transpose (complex):
  ct_z = Scale(c, ct_y)          ct_z = Scale(Conj(c), ct_y)
       = (2+3i) * ct_y                = (2-3i) * ct_y
                                         ← Conj node inserted
```

The graph for the complex transpose:

```
G_transposed:
  ct_y = Input(cotangent)
  c_conj = Conj(c)              // ← NEW: conjugate of primal value c
  ct_z = Mul(c_conj, ct_y)      // (2-3i) * ct_y
```

`differentiate` is identical for real and complex — no conjugation.
`transpose` is the only operation that differs: it inserts `Conj` for
complex-valued primal coefficients. `Conj` must be an IR primitive.

The transposed graph references the same primal values as the original
linear graph (with `Conj` wrappers for complex). With a sufficiently rich
primitive set (including `Dup` and `Conj`), all transposes are expressible
as IR ops.

### `merge`: unify graphs before evaluation

Combines multiple independent graphs. Nodes with the same content-addressable
ID are unified into a single node. This is automatic CSE across all levels.

```
merge(G_primal, G_transposed):
  v0 = Input(x)
  v1 = Input(a)        ← shared by G_primal and G_transposed
  v2 = Mul(v0, v1)
  v3 = Exp(v2)          ← shared: computed once, used by both graphs
  v7 = Input(ct_y)
  v8 = Mul(v3, v7)
  v9 = Mul(v1, v8)
```

### Linearization of binary ops

Most ops have at most 2 inputs. For `y = f(a, b)`, the linearization is
`dy = ∂f/∂a · da + ∂f/∂b · db`. With `Option<ValId>` tangent inputs
(None = zero), there are 4 cases handled uniformly:

| da | db | dy |
|----|----|----|
| 0 | 0 | 0 (skip) |
| da | 0 | ∂f/∂a · da |
| 0 | db | ∂f/∂b · db |
| da | db | Add(∂f/∂a · da, ∂f/∂b · db) |

Zero propagation (case 1) avoids creating unnecessary nodes.

### Higher-order AD = repeat `differentiate`

n-th derivative = apply `differentiate` n times to produce n independent
linear graphs. `transpose` is optional (evaluation choice), applied to the
final linear graph.

```
1st derivative:   differentiate(G_primal) → G_linear
2nd derivative:   differentiate(G_linear) → G_linear2
gradient:         transpose(G_linear) → G_transposed
Hessian-gradient: differentiate(G_transposed) → G_linear2, then transpose
```

### Concrete example: `f(x) = exp(a * x)`

**Step 1 — Build primal graph G_primal:**

```
v0 = Input(x)
v1 = Input(a)
v2 = Mul(v0, v1)    // a*x
v3 = Exp(v2)         // y = exp(a*x)
```

**Step 2 — differentiate(G_primal) → G_linear (independent graph):**

```
v4 = Input(t_x)
v5 = Mul(v1, v4)     // a * t_x         (references v1 from G_primal)
v6 = Mul(v3, v5)     // exp(a*x) * a*t_x (references v3 from G_primal)
```

**Step 3 — transpose(G_linear) → G_transposed (independent graph):**

```
v7 = Input(ct_y)
v8 = Mul(v3, v7)     // exp(a*x) * ct_y  (references v3 from G_primal)
v9 = Mul(v1, v8)     // a*exp(a*x)*ct_y  (references v1 from G_primal)
```

**Step 4 — merge(G_primal, G_transposed) → unified graph:**

```
v0 = Input(x)
v1 = Input(a)        ← shared
v2 = Mul(v0, v1)
v3 = Exp(v2)          ← shared: computed once
v7 = Input(ct_y)
v8 = Mul(v3, v7)
v9 = Mul(v1, v8)     = ct_x
```

**Step 5 — compile → TenferroIR → backend → eval:**

```rust
let tenferro_ir = merged.compile(&[v3, v9], &[v0, v1, v7]);
// Standard algebra: TenferroIR → StableHLO → faer/LAPACK (or XLA)
let [y, ct_x] = backend.eval(&tenferro_ir, &[x_val, a_val, 1.0]);
```

Primal (`v3`) and gradient (`v9`) are compiled into a single TenferroIR
program. The backend executes both in one pass. `v3 = exp(a*x)` computed
once, shared by primal output and backward.
All higher-order methods (FoF, FoR, RoF, RoR) give `d²f/dx² = a² exp(ax)` ✓

---

## IV. Primitive Set and Operand

### Operand: multi-dimensional array with StableHLO-compatible ops

`Operand` is the type of data flowing through the graph. It must be a
**multi-dimensional array** (tensor) of some scalar type. Scalars are
0-dimensional arrays.

Operand must support operations aligned 1:1 with StableHLO ops:

```rust
trait Operand: Clone + Send + Sync + 'static {
    fn reshape(&self, shape: &[usize]) -> Self;                           // stablehlo.reshape
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self;  // stablehlo.broadcast_in_dim
    fn add(&self, other: &Self) -> Self;                                   // stablehlo.add
    fn multiply(&self, other: &Self) -> Self;                              // stablehlo.multiply
    fn dot_general(&self, other: &Self, ...) -> Self;                      // stablehlo.dot_general
}
```

This 1:1 alignment means `CompiledProgram` can be lowered to StableHLO
trivially — each instruction maps directly to a StableHLO op.

Why these operations are needed:

| AD operation | Required Operand ops |
|-------------|---------------------|
| forward eval | multiply, add |
| linearize | multiply, add |
| transpose (Dup) | broadcast_in_dim |
| batched JVP (all tangent dirs at once) | reshape (add axes) + multiply (broadcasting) |
| cross-country composition | reshape + multiply (outer product) or dot_general (contraction) |

### Placeholder evaluation and broadcasting

When computing all tangent directions simultaneously, the tangent input is
a placeholder representing an identity matrix:

```
x: shape [N]
dx: placeholder → shape [N, N] (identity matrix, one direction per column)
```

Broadcasting propagates the batch dimension through the graph. For a
2-input linearization `dy = ∂f/∂a · da + ∂f/∂b · db`:

```
∂f/∂a · da:  shape [N1]    →  reshape to [N1, 1]
∂f/∂b · db:  shape [N2]    →  reshape to [1, N2]
dy = Add:    [N1, 1] + [1, N2] = [N1, N2]  ← broadcast in Add
```

The Jacobian shape `[N_out, N_in]` emerges naturally from broadcasting.
Shape conventions (which axis is which) follow StableHLO / JAX conventions.

### PrimitiveOp trait (defined in chainrules2)

```rust
pub trait PrimitiveOp: Clone + Hash + Eq + Send + Sync + 'static {
    type Operand: Operand;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;

    fn eval(&self, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;

    /// Produce a NEW linear graph for the JVP.
    /// Primal values are referenced by ValId (resolved at merge time).
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

Linearize rules are standalone functions; the trait impl is a `match` dispatch.

### Design principle

A primitive's `linearize` must be **linear in the tangent inputs**. It may:
- Reference primal output vals as coefficients (e.g., `Exp` references `exp(x)`)
- Emit other primitives (e.g., `Exp` emits `Mul`)
- NOT apply nonlinear ops to tangent inputs

### Two tiers (defined in tenferro2)

**Tier 1 — Semiring Core**: sufficient for einsum-based computation. Compatible
with custom algebraic backends (tropical, p-adic, polynomial rings, etc.).

- Elementwise: `Add`, `Mul`, `Scale`, `Neg`, `Conj`
- Tensor: `Einsum`, `Transpose`, `Reshape`, `BroadcastInDim`, `Dup`
- Reduction: `Sum`, `Prod`

**Tier 2 — Standard = Core + JAX prims**: full JAX-compatible set.

- Transcendental: `Exp`, `Log`, `Sin`, `Cos`, ...
- Linalg: `SVD`, `Cholesky`, `QR`, `Eig`, `Solve`
- Indexing: `Gather`, `Scatter`, `Slice`
- Control: `Cond`, `Scan`, `While`
- Misc: `Sort`, `FFT`, `Conv`, ...

Each primitive's `linearize` is expressed in terms of other primitives.

---

## V. Compilation & Execution

### Pipeline

```
                ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
                │  G_primal    │   │  G_linear    │   │  G_transposed│
                │  (build)     │   │  (diff)      │   │  (transpose) │
                └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                       │                  │                   │
                       └─────────┬────────┘───────────────────┘
                                 │ merge (unify by content-addressable ID)
                                 ▼
                       ┌─────────────────────┐
                       │  Merged Graph       │
                       │  (shared nodes, CSE) │
                       └─────────┬───────────┘
                                 │ compile (toposort + SSA)
                                 ▼
                       ┌─────────────────────┐
                       │  TenferroIR         │
                       │  (flat slot-based)  │
                       └─────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  │                   ▼
     ┌────────────────┐         │         ┌──────────────────┐
     │  Standard      │         │         │  Custom algebra  │
     │  → StableHLO   │         │         │  → Custom backend│
     └───────┬────────┘         │         │  (Tier 1 only)   │
             │                  │         └──────────────────┘
        ┌────┴────┐             │
        ▼         ▼             │
   ┌────────┐ ┌───────┐        │
   │  faer  │ │  XLA  │        │
   │(default)│ │(opt.) │        │
   └────────┘ └───────┘        │
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

`compile(target_nodes, input_nodes)` topologically sorts from targets,
producing a flat instruction sequence. Compile once, eval many times.

### Backend dispatch

| Primitive tier | Backend | Use case |
|---------------|---------|----------|
| Tier 1 only | Custom algebra backend | Tropical, p-adic, polynomial rings |
| Tier 1 + 2 | StableHLO → IREE/XLA | Standard float/complex on GPU/TPU |

tidu2 is backend-agnostic. It produces `CompiledProgram<Op>`. StableHLO
lowering is trivial because Operand ops are 1:1 with StableHLO ops.

---

## VI. Crate Architecture

### chainrules2

**Role**: define `PrimitiveOp` and `Operand` traits.

Contains:
- `PrimitiveOp` trait
- `Operand` trait (StableHLO-aligned ops)
- `ValId`, `OpId` types
- Error types

### tidu2

**Role**: graph engine.

Contains:
- `Graph<Op>` data structure
- `differentiate()` → new graph
- `transpose()` → new graph
- `merge()` → unified graph
- `compile()` → `CompiledProgram`
- `CompiledProgram::eval()` (CPU execution)

### tenferro2

**Role**: tensor primitives and hardware backends.

Contains:
- `TensorOp` enum implementing `PrimitiveOp` (Tier 1 + Tier 2)
- `Tensor<T>`, `DynTensor` implementing `Operand`
- Linearize rules for each primitive
- StableHLO lowering of `CompiledProgram<TensorOp>`

### Dependency graph

```
chainrules2          (PrimitiveOp + Operand traits)
    ↓
tidu2                (Graph engine)
    ↓
tenferro2            (Tensor prims + backends)
```

---

## VII. Advantages of Graph-Based Design

### Cross-country evaluation

Partially transpose a linear graph: forward for the first half, backward
for the second half.

```
Linear graph:    dx → [L1] → [L2] → mid → [L3] → [L4] → dy

Split and transpose L3, L4 only:
  G_fwd:  dx  → [L1] → [L2] → mid_fwd     (keep as-is)
  G_bwd:  ct_dy → [L4ᵀ] → [L3ᵀ] → mid_bwd  (transpose L3, L4)
```

Combine via outer product (scalar chain) or einsum (tensor chain):

```
Scalar:   J = mid_bwd ⊗ mid_fwd                          (broadcast + multiply)
Tensor:   J = dot_general(mid_bwd, mid_fwd, contract=k)  (contraction over mid dim)
```

Optimal split point can be determined automatically based on op costs.

### Runtime checkpoint optimization

Each op declares cost and output size:

```rust
trait OpInfo {
    fn cost(&self, input_sizes: &[usize]) -> f64;
    fn output_size(&self, input_sizes: &[usize]) -> usize;
}
```

The graph scheduler uses **runtime information** to decide which intermediate
values to keep vs. recompute. Graph-level optimization, not per-op decisions.

### CSE via merge

Since graphs are merged by content-addressable ID, a value like `exp(x)` is
guaranteed to appear once in the merged graph regardless of how many
independent graphs reference it.

### Compile-once-eval-many: Laplacian example

Tr(H) = Σᵢ ∂²f/∂xᵢ². Key insight: `differentiate` twice with the same
tangent direction v gives vᵀHv directly. No `transpose`, no dot product.

```
t_y = differentiate(y, wrt=x, tangent=v)     // ∇f · v
vHv = differentiate(t_y, wrt=x, tangent=v)   // vᵀHv
```

| Method | Transform | Compile | Eval | Total |
|--------|-----------|---------|------|-------|
| FoF + compile once (exact) | 2 | 1 | n | O(n) eval only |
| FoF + Hutchinson (stochastic) | 2 | 1 | k | O(k), k << n |
| Forward Laplacian (future) | 1 | 1 | 1 | **O(1)** |

Forward Laplacian propagates (value, tangent, Laplacian) simultaneously:

```
u = a * b:   Lap(u) = a·Lap(b) + 2·da·db + b·Lap(a)
u = exp(a):  Lap(u) = exp(a)·(Lap(a) + da²)
```

Implementable as a specialized graph transformation.

---

## Appendix: Design Notes

### A. Custom rules for user-defined functions

Users extend the primitive set by adding variants to their `PrimitiveOp` enum
(analogous to Julia's `ChainRulesCore.frule` / `rrule`).

To avoid exposing low-level graph internals, provide a `Traced` wrapper:

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
struct Traced<'g, Op: PrimitiveOp> {
    ctx: &'g GraphContext<Op>,
    val_id: ValId,              // hidden from user
}

impl Mul for Traced<'_, Op> { ... }  // calls ctx.add_op(Mul, ...)
impl Add for Traced<'_, Op> { ... }  // calls ctx.add_op(Add, ...)
```

### B. Wrapper structs and pytree-style AD

Domain structs wrapping tensors declare differentiable fields via a trait:

```rust
#[derive(Differentiable)]
struct GreenFunction {
    #[leaf]
    data: Tensor<f64>,      // AD target (Operand)
    mesh: ImagTimeMesh,     // metadata, not differentiated
    beta: f64,              // parameter, not differentiated
}
```

The graph operates on leaves (`Tensor<f64>` = Operand). Pack/unpack is
automatic via `to_leaves` / `from_leaves`.

---

## VIII. Roadmap

### Phase 1: Scalar graph AD

- `Graph<ScalarOp>` with `Operand = f64`
- Primitives: `Mul`, `Add`, `Scale`, `Exp`, `Dup`, `Neg`
- `differentiate`, `transpose`, `merge`, `compile`, `eval`
- Tests: forward, backward, FoF, FoR, RoF, RoR for `exp(a*x)`

### Phase 2: Tensor primitives

- `TensorOp` enum with Tier 1 (Semiring Core) + Tier 2 (Standard)
- `Operand = Tensor<f64>` and `DynTensor` implementing StableHLO-aligned ops
- Linearize rules for all primitives
- Batched JVP via placeholder + broadcasting

### Phase 3: Backend compilation

- StableHLO lowering of `CompiledProgram<TensorOp>` (trivial: 1:1 op mapping)
- IREE/XLA integration for GPU/TPU execution
- Custom algebra backend for Tier 1 programs

### Phase 4: Optimization

- Graph scheduler for checkpoint optimization (runtime cost/memory info)
- Cross-country mode (automatic forward/backward split)
- Forward Laplacian
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
