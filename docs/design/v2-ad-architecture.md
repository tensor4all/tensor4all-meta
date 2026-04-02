# v2 AD Architecture

**Date:** 2026-04-01 (updated 2026-04-02)
**Status:** Draft
**Repos:** chainrules-rs, tidu-rs, tenferro-rs

---

## I. Vision

Build a differentiable programming stack in Rust where:

- **`differentiate` is the only derivative operation.** It takes a graph and
  produces a NEW linear graph fragment with external references to primal
  values. Applying it n times (with `merge` in between) gives the n-th
  derivative. All derivative information comes from this single operation.
- **`transpose` is not differentiation — it is a graph transformation that
  reverses tangent data flow** (JVP → VJP). It does not add derivative
  information. It can be applied to all or part of a linear graph
  (partial transpose enables cross-country evaluation).
- **Graph fragments have external references.** `differentiate` and
  `transpose` produce new graph fragments that reference primal values by
  content-addressable ID. `merge` resolves these references and unifies
  same-named nodes (automatic CSE). `merge` must precede the next
  `differentiate` to enable higher-order tracing.
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
differentiate  graph → NEW linear graph (JVP), references primal values
merge          resolve external references + CSE (must precede next differentiate)
transpose      linear graph → NEW linear graph (flip tangent I/O direction)
compile        merged graph → TenferroIR (flat SSA)
eval           TenferroIR + input values → output values
```

**Pipeline for higher-order AD**: `merge` after each `differentiate` to
resolve external references before the next `differentiate` can trace
through primal dependencies.

```
1st order:  build → differentiate → merge → transpose → compile → eval
2nd order:  build → differentiate → merge → differentiate → merge → compile → eval
n-th order: build → (differentiate → merge) × n → [transpose] → compile → eval
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
**This identity is used across graph fragments**: when two fragments contain
the same `(op_kind, input_val_ids...)`, `merge` unifies them into a single
node. This is how CSE works across primal, tangent, and cotangent fragments.

### Graph fragments with external references

`differentiate` and `transpose` produce NEW graph fragments that reference
values from other fragments by content-addressable ID (external references).
These references are resolved at `merge` time. A fragment is NOT
self-contained — it depends on external values.

```
G_primal:     v0=Input(x), v1=Input(a), v2=Mul(v0,v1), v3=Exp(v2)
G_linear:     v4=Input(dx), v5=Scale(v1,v4), v6=Scale(v3,v5)
                                     ^^          ^^
                                  ^^           ^^
                            external references to v1, v3 from G_primal

merge(G_primal, G_linear) → unified graph, external refs resolved
```

---

## III. Differentiation

### `differentiate`: graph → NEW linear graph fragment

Takes a (merged) graph and produces a NEW linear graph fragment representing
the JVP. The fragment contains external references to primal values by
content-addressable ID. These must be resolved via `merge` before the next
`differentiate`.

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

Takes a linear graph and produces a NEW graph with **tangent I/O reversed**.
Fixed operands stay the same; only the active (tangent) data flow direction
flips.

#### Active mask (no dedicated linear primitives)

The linear graph uses **the same ops** as the primal graph (`Mul`, `Add`,
`Exp`, etc.). Each node in the linear graph carries an **active mask** —
a boolean per input indicating which operands are active (tangent) and
which are fixed (coefficient):

```
Mul(a, b) + mask=[fixed, active]   — b is the tangent variable
Add(a, b) + mask=[active, active]  — both are tangent variables
Exp(a)    + mask=[active]          — a is the tangent variable
```

No `Scale` or `AddLin` needed. The mask is determined during `differentiate`:
`Some(tangent)` → active, `None` → fixed.

At higher order, a "fixed" operand is NOT necessarily a primal value. It can
be an earlier tangent or a cotangent-side residual. The mask is recomputed
at each `differentiate` call. What matters: **in this transform, which
operands are treated as fixed, and which are the active linear variables.**

#### Transpose rules with mask

Transpose reads the mask to know what to flip:

```
Op(a, b) mask=[fixed, active]:
  real:    transpose → Op(a, ct)           (fixed operand stays)
  complex: transpose → Op(Conj(a), ct)     (conjugate the fixed operand)

Add(a, b) mask=[active, active]:
  transpose → Dup(ct) → two outputs        (sum ↔ broadcast)

Dup(a) (1 input → 2 outputs):
  transpose → Add(ct_1, ct_2)              (broadcast ↔ sum)
```

Only `Dup` and `Conj` are added to the primitive set. All other ops are
reused from the primal set.

#### Tangent I/O reversal

Transpose swaps active inputs and outputs. Fixed operands are unchanged.

```
Forward linear graph (from differentiate):
  v4 = Input(dx)                           ← tangent INPUT
  v5 = Mul(v1, v4)  mask=[fixed, active]   ← v1 = a (fixed)
  v6 = Mul(v3, v5)  mask=[fixed, active]   ← v3 = exp(a*x) (fixed)
                                              v6 = tangent OUTPUT

Transposed graph (reversed, new variable names):
  v7 = Input(ct_dy)                        ← was OUTPUT, now INPUT
  v8 = Mul(v3, v7)  mask=[fixed, active]   ← reverse order
  v9 = Mul(v1, v8)  mask=[fixed, active]   ← was INPUT, now OUTPUT
                                              (complex: Conj(v3), Conj(v1))
```

**Only tangent I/O flips. Fixed operands (v1, v3) remain as external
references.**

#### Cotangent accumulation

Local transpose rules alone are NOT sufficient. When multiple reverse
contributions land on the **same tangent node**, they must be combined
with `Add`. Example: `f(x) = x + x` produces `Add(dx, dx)` where
both inputs are the same tangent. Transpose gives `Dup(ct_y)` → two
cotangent contributions to `dx`, which must be accumulated.

Algorithm:

```
transpose(linear_fragment):
  1. seed cotangent on linear outputs
  2. traverse linear ops in reverse topological order
  3. read active mask; emit reverse contributions by local transpose rule
  4. bucket contributions by original tangent-node identity
  5. combine each bucket with Add
```

`differentiate` is identical for real and complex — no conjugation.
`transpose` is the only operation that differs for complex: it wraps
fixed operands in `Conj`. Only `Dup` and `Conj` are needed as additional
primitives.

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
`dy = ∂f/∂a · da + ∂f/∂b · db`, using the same ops with active masks:

| da | db | linearized node | mask |
|----|----|----|---|
| 0 | 0 | skip (zero propagation) | — |
| da | 0 | `Mul(∂f/∂a, da)` | `[fixed, active]` |
| 0 | db | `Mul(∂f/∂b, db)` | `[fixed, active]` |
| da | db | `Add(Mul(∂f/∂a, da), Mul(∂f/∂b, db))` | Add: `[active, active]` |

The active mask is metadata on each linear graph node, determined by
which tangent inputs are non-zero. `transpose` reads the mask to know
which operands to flip and which to keep fixed.

### Higher-order AD = repeat `differentiate` with `merge`

n-th derivative = apply `differentiate` n times, with `merge` after each
to resolve external references before the next can trace through primal
dependencies. `transpose` is optional (evaluation choice).

```
1st derivative:
  G_lin1 = differentiate(G_primal)
  G1     = merge(G_primal, G_lin1)          ← resolve external refs

2nd derivative:
  G_lin2 = differentiate(G1, output=tangent_output, wrt=x)
  G2     = merge(G1, G_lin2)                ← resolve again

gradient:
  G_trans = transpose(G_lin1)               ← flip tangent I/O
  G_grad  = merge(G_primal, G_trans)

Hessian-gradient:
  G_lin_of_grad = differentiate(G_grad, output=ct_x, wrt=x)
  G_hess = merge(G_grad, G_lin_of_grad)
```

Without `merge` between differentiations, external references remain
dangling and the second `differentiate` cannot trace through primal nodes.

### Concrete example: `f(x) = exp(a * x)`

**Step 1 — Build primal graph G_primal:**

```
v0 = Input(x)
v1 = Input(a)
v2 = Mul(v0, v1)    // a*x
v3 = Exp(v2)         // y = exp(a*x)
```

**Step 2 — differentiate(G_primal) → G_linear (new graph fragment):**

Same ops as primal, with active masks:

```
v4 = Input(t_x)
v5 = Mul(v1, v4)  mask=[fixed, active]   // a * t_x   (v1 = a, external ref)
v6 = Mul(v3, v5)  mask=[fixed, active]   // exp(a*x) * a*t_x  (v3, external ref)
```

**Step 3 — merge(G_primal, G_linear) → resolve external refs:**

Required before transpose or further differentiate.

```
v0 = Input(x)
v1 = Input(a)
v2 = Mul(v0, v1)
v3 = Exp(v2)          ← v5, v6 can now trace through v3 to x
v4 = Input(t_x)
v5 = Mul(v1, v4)  mask=[fixed, active]
v6 = Mul(v3, v5)  mask=[fixed, active]
```

**Step 4 — transpose(linear part of merged graph) → G_transposed:**

Tangent I/O flips. Fixed operands stay. Reverse order.

```
v7 = Input(ct_y)
v8 = Mul(v3, v7)  mask=[fixed, active]   // reverse of v6
v9 = Mul(v1, v8)  mask=[fixed, active]   // reverse of v5, v9 = ct_dx
```

**Step 5 — merge(G_merged, G_transposed) → final graph:**

```
v0 = Input(x)
v1 = Input(a)         ← shared across primal + transpose
v2 = Mul(v0, v1)
v3 = Exp(v2)          ← shared: computed once, used by v5, v6, v8
v7 = Input(ct_y)
v8 = Mul(v3, v7)  mask=[fixed, active]
v9 = Mul(v1, v8)  mask=[fixed, active]   = ct_x
```

**Step 5 — compile → CompiledProgram → eval:**

```rust
// Graph construction + AD transforms (expensive, do once)
let prog = merged.compile(&[v3, v9], &[v0, v1, v7]);

// Eval (cheap, do many times with different inputs)
let [y, ct_x] = prog.eval(&[2.0, 3.0, 1.0]);
let [y2, ct_x2] = prog.eval(&[4.0, 3.0, 1.0]);
```

**Caching**: graph construction, differentiation, merge, and compile are
all expensive. tidu2 does not cache — it returns `CompiledProgram<Op>`
and the caller is responsible for retaining and reusing it. This is
analogous to JAX's `jit`: trace once, execute many times.

```
Expensive (once):  build → differentiate → merge → compile → CompiledProgram
Cheap (many times): CompiledProgram.eval(inputs)
```

Primal (`v3`) and gradient (`v9`) are compiled into a single program.
`v3 = exp(a*x)` computed once, shared by primal output and backward.
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
straightforwardly for the core subset (elementwise, dot_general,
reductions). Linalg ops use `custom_call`. Complex transcendentals
and control flow require decomposition.

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
- Reference primal output vals as fixed operands (e.g., `Exp` references `exp(x)`)
- Emit the same primal ops with active masks (e.g., `Exp` emits `Mul` with mask)
- NOT apply nonlinear ops to tangent inputs

Nodes emitted by `linearize` carry an **active mask** indicating which
inputs are tangent (active) and which are fixed. `transpose` reads this mask.

### Two tiers (defined in tenferro2)

**Tier 1 — Semiring Core**: sufficient for einsum-based computation. Compatible
with custom algebraic backends (tropical, p-adic, polynomial rings, etc.).

- Elementwise: `Add`, `Mul`, `Neg`, `Conj`, `Dup`
- Tensor: `Einsum`, `Transpose`, `Reshape`, `BroadcastInDim`
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
lowering is conceptually straightforward for the core op subset
(elementwise, dot_general, reductions map 1:1 to StableHLO).

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

## VIII. Golden Tests

Minimal set of tests that validate the graph transform procedure.
Derived from manual experiments (see `docs/design/ad-graph-experiments.md`).

| # | Function | What it checks |
|---|----------|---------------|
| 1 | `x + x` | transpose accumulation (same tangent node) |
| 2 | `x * y` | binary linearization, distinct reverse sinks |
| 3 | `c * z` (complex) | `Conj` placement only in transpose |
| 4 | `x²` | duplication, CSE, all 4 second-order modes (FoF/FoR/RoF/RoR) |
| 5 | `exp(a*x)` | chain rule, external refs, repeated merge, all 4 modes |
| 6 | `x*y` mixed partial | order symmetry: d²f/dxdy = d²f/dydx |
| 7 | `exp(a*x)` 3rd order | repeated higher-order closure |

Expected second-order results for `x²` (all seeds = 1):

| Mode | Output |
|------|--------|
| FoF | 2 |
| FoR | 2 |
| RoF | 2 |
| RoR | 2 |

Expected second-order results for `exp(a*x)` (all seeds = 1):

| Mode | Output |
|------|--------|
| FoF | a² exp(ax) |
| FoR | a² exp(ax) |
| RoF | a² exp(ax) |
| RoR | a² exp(ax) |

---

## IX. Roadmap

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

- StableHLO lowering of `CompiledProgram<TensorOp>` (1:1 for core ops)
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
