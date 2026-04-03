# v2 Tensor API Pseudocode

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`

---

## Design Decisions

- All operations are **lazy** (deferred). No eager mode.
- `TracedTensor` is the user-facing type for **standard algebra**. It holds
  shape/dtype, graph info, and optionally data. All lazy operations return
  `TracedTensor`.
- `Tensor` is the concrete dense runtime value (shape + buffer + placement).
  It may live on CPU or GPU. Users see it at input/output boundaries only.
- `einsum` is a free function that takes `TracedTensor` inputs and returns
  `TracedTensor`. There is no eager `einsum` on `Tensor` — users wrap
  concrete data with `TracedTensor::from` first.
- `eval()` triggers `materialize_merge -> compile (cached) -> execute`,
  filling in the data.
- `Engine` holds backend + compilation cache + einsum cache. It is generic
  over `Backend<StdTensorOp>` (e.g., `FaerBackend`, `XlaBackend`).
- `eval_all` is the primary evaluation API. It resolves all output fragments
  together into one `MaterializedGraph`, so shared intermediate nodes (e.g.,
  primal values needed by both forward output and gradient) are computed once.
  Single-output `eval` is a convenience wrapper around `eval_all`.
- **Custom algebras** (Tropical, etc.) do not use `TracedTensor` or `einsum`.
  They work with `Fragment<SemiringOp<T>>` and the computegraph-rs API
  directly. See "Custom Algebra" section below.

---

## Type Definitions

```rust
struct Placement {
    memory_kind: MemoryKind,
    resident_device: Option<ComputeDevice>,
}

enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

// Typed tensor (internal)
struct TensorData<T: Scalar> {
    buffer: Buffer<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    placement: Placement,
    preferred_compute_device: Option<ComputeDevice>,
}

// Type-erased tensor (user-facing)
enum Tensor {
    F32(TensorData<f32>),
    F64(TensorData<f64>),
    C32(TensorData<Complex<f32>>),
    C64(TensorData<Complex<f64>>),
}

// Graph-aware wrapper — tracks computation for AD and compilation
// Standard algebra only (StdTensorOp)
struct TracedTensor {
    shape: Vec<usize>,
    dtype: DType,
    fragment: Arc<Fragment<StdTensorOp>>,
    val: LocalValId,
    data: Option<Tensor>,  // Some for inputs / eval'd results, None for deferred
}
```

- `TracedTensor::from(Tensor)` creates a Fragment input node with `data = Some(...)`.
- Operations (einsum, exp, ...) create new Fragments and return
  `TracedTensor` with `data = None`.
- `eval()` fills in `data` and returns `&Tensor`. The resulting tensor may live
  on CPU or GPU depending on the `Engine` backend.

```rust
enum Buffer<T> {
    Host(HostBuffer<T>),
    Backend(BufferHandle<T>),
}
```

Host access is explicit:

```rust
impl Tensor {
    fn placement(&self) -> Placement;
    fn memory_kind(&self) -> MemoryKind;
    fn resident_device(&self) -> Option<ComputeDevice>;
    fn preferred_compute_device(&self) -> Option<ComputeDevice>;
    fn to_placement(&self, target: Placement) -> Tensor;
    fn to_cpu(&self) -> Tensor;
    fn to_gpu_on(&self, device_id: usize) -> Tensor;
    fn to_pinned_host(&self) -> Tensor;
}
```

### Graph origin

Every graph starts from `TracedTensor::from`:

```rust
let x = TracedTensor::from(Tensor::new(&[1.0, 2.0], &[2]));

// Internally:
// fragment = Fragment { vals: [Input("x_0")], ops: [], ... }
// val = 0
// data = Some(Tensor([1.0, 2.0]))
```

Operations extend the graph:

```rust
let y = x.exp();

// Internally:
// fragment = Fragment {
//     vals: [Derived { op=Exp, inputs=[Input("x_0")] }],
//     ops: [Exp(External(Input("x_0")))],
//     parents: [x.fragment],
// }
// val = 0
// data = None  (not yet computed)
```

---

## Engine Setup

```rust
// Create engine with faer CPU backend (default)
let mut engine = Engine::new(FaerBackend::new());

// Or with XLA GPU backend
let mut engine = Engine::new(XlaBackend::gpu(device_id));
```

---

## Basic Operations

```rust
let x = TracedTensor::from(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
let w = TracedTensor::from(Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

// All operations build graph, nothing is computed yet
let y = einsum(&mut engine, &[&x, &w], "ij,jk->ik");   // data = None
let z = y.exp();                            // data = None
let loss = z.sum();                         // data = None

// eval triggers compilation + execution
let loss_val: &Tensor = loss.eval(&mut engine);
// loss.data is now Some(...); may be CPU or GPU resident

// Already eval'd tensors return data immediately
let loss_val2: &Tensor = loss.eval(&mut engine);  // no recomputation
```

---

## Einsum (N-ary)

```rust
let a = TracedTensor::from(Tensor::new(&a_raw, &[2, 3]));
let b = TracedTensor::from(Tensor::new(&b_raw, &[3, 4]));
let c = TracedTensor::from(Tensor::new(&c_raw, &[4, 5]));

// N-ary einsum: contraction path optimization happens inside (uses Engine's EinsumCache)
let result = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il");

let result_val: &Tensor = result.eval(&mut engine);
// First call:  path optimization + Fragment build + compile + eval
// Second call (same graph structure): compile cache hit → eval only
```

---

## Reverse-Mode AD (grad / VJP)

```rust
let x = TracedTensor::from(Tensor::new(&x_raw, &[3]));
let a = TracedTensor::from(Tensor::new(&a_raw, &[3]));

// Forward computation (lazy)
let ax = einsum(&mut engine, &[&a, &x], "i,i->");  // dot product
let y = ax.exp();                      // scalar output

// grad: differentiate + transpose (graph transform, still lazy)
let grad_x = y.grad(&x);

// eval_all: single materialized graph, primal intermediates shared
let results = engine.eval_all(&mut [&mut y, &mut grad_x]);
// results[0] = exp(a . x)
// results[1] = a * exp(a . x)
```

**Why `eval_all`, not separate `eval` calls:**

Each `eval` independently runs `materialize_merge → compile → eval`. If `y`
and `grad_x` are evaluated separately, primal intermediates (e.g. `exp(a*x)`)
are recomputed in both programs.

`eval_all` resolves all fragments together into one `MaterializedGraph`.
`GlobalValKey`-based deduplication ensures shared nodes are computed once:

```text
engine.eval_all(&mut [&mut y, &mut grad_x])
    → resolve([primal_fragment, transposed_fragment])
    → materialize_merge(view, [key(y), key(grad_x)])
        // exp(a*x) appears once, shared between y and grad_x
    → compile (single CompiledProgram)
    → eval (single execution, both outputs produced)
```

Separate `eval` calls remain available for convenience when sharing is not
needed:

```rust
let y_val: &Tensor = y.eval(&mut engine);  // OK if grad not needed
```

---

## Forward-Mode AD (JVP)

```rust
let x = TracedTensor::from(Tensor::new(&x_raw, &[3]));
let a = TracedTensor::from(Tensor::new(&a_raw, &[3]));

let y = einsum(&mut engine, &[&a, &x], "i,i->").exp();

// JVP with tangent vector
let t_x = TracedTensor::from(Tensor::new(&tangent_raw, &[3]));
let dy = y.jvp(&x, &t_x);  // differentiate only (no transpose)

// eval_all: primal intermediate exp(a.x) shared between y and dy
let results = engine.eval_all(&mut [&mut y, &mut dy]);
```

---

## Hessian-Vector Product (HVP)

```rust
let x = TracedTensor::from(Tensor::new(&x_raw, &[3]));
let a = TracedTensor::from(Tensor::new(&a_raw, &[3]));

let y = einsum(&mut engine, &[&a, &x], "i,i->").exp();

// Forward-over-reverse: jvp(grad(f))
let grad_x = y.grad(&x);                                      // differentiate + transpose
let t_x = TracedTensor::from(Tensor::new(&tangent_raw, &[3]));
let hvp = grad_x.jvp(&x, &t_x);                              // differentiate again

// eval_all: all primal + first-order + second-order intermediates shared
let results = engine.eval_all(&mut [&mut y, &mut grad_x, &mut hvp]);
```

---

## Reusable Functions

```rust
// Define model as a normal function over TracedTensor
fn my_model(engine: &mut Engine, x: &TracedTensor, w: &TracedTensor) -> TracedTensor {
    einsum(engine, &[x, w], "ij,jk->ik").exp().sum()
}

// Evaluate
let x = TracedTensor::from(Tensor::new(&x_raw, &[2, 2]));
let w = TracedTensor::from(Tensor::new(&w_raw, &[2, 2]));
let loss = my_model(&mut engine, &x, &w);
let loss_val: &Tensor = loss.eval(&mut engine);

// Differentiate the same function
let grad_w = my_model(&mut engine, &x, &w).grad(&w);
let grad_val: &Tensor = grad_w.eval(&mut engine);

// Different data, same graph structure → cache hit
let x2 = TracedTensor::from(Tensor::new(&x_raw2, &[2, 2]));
let w2 = TracedTensor::from(Tensor::new(&w_raw2, &[2, 2]));
let loss2_val = my_model(&mut engine, &x2, &w2).eval(&mut engine);  // cache hit
```

---

## Compilation Cache Behavior

```rust
let x = TracedTensor::from(Tensor::new(&x_raw, &[2, 2]));
let w = TracedTensor::from(Tensor::new(&w_raw, &[2, 2]));
let y = my_model(&mut engine, &x, &w);
let _ = y.eval(&mut engine);  // compile + cache

// Same graph structure, different data → cache hit
let x2 = TracedTensor::from(Tensor::new(&x_raw2, &[2, 2]));
let w2 = TracedTensor::from(Tensor::new(&w_raw2, &[2, 2]));
let _ = my_model(&mut engine, &x2, &w2).eval(&mut engine);  // cache hit

// Same subscripts but different shapes may produce different
// contraction paths → different graph structure → recompile
let x3 = TracedTensor::from(Tensor::new(&x_raw3, &[100, 100]));
let w3 = TracedTensor::from(Tensor::new(&w_raw3, &[100, 100]));
let _ = my_model(&mut engine, &x3, &w3).eval(&mut engine);  // may recompile
```

Cache key is based on **normalized graph topology** (op types, connectivity,
modes), not on concrete `InputKey` values. This ensures that repeated
`differentiate` calls with different `DiffPassId`s produce cache hits when the
graph structure is the same. Shape-dependent ops (e.g. `Reshape(target)`)
encode shape in the op itself, so different shapes produce different topology
automatically.

---

## Custom Algebra (Tropical, etc.) — No TracedTensor

Custom algebras do not use `TracedTensor` or the `einsum` free function.
They work directly with the computegraph-rs Fragment API and
`SemiringOp<T>`:

```rust
use computegraph::{FragmentBuilder, compile, materialize_merge, resolve};
use tenferro_ops::{SemiringOp, SemiringOps, build_einsum_fragment};
use tenferro_einsum::optimize_contraction_path;

// User-defined tensor type with Operand impl
type TropicalOp = SemiringOp<TropicalTensor>;

// 1. Optimize contraction path (algebra-agnostic)
let subscripts = Subscripts::parse("ij,jk,kl->il");
let shapes = [&[3, 4][..], &[4, 5], &[5, 6]];
let path = optimize_contraction_path(&subscripts, &shapes);

// 2. Build Fragment (generic over SemiringOps)
let mut builder = FragmentBuilder::<TropicalOp>::new();
let a = builder.add_input("a".into());
let b = builder.add_input("b".into());
let c = builder.add_input("c".into());
let result = build_einsum_fragment(&mut builder, &path, &[a, b, c]);
builder.set_outputs(vec![result]);
let fragment = Arc::new(builder.build());

// 3. Compile
let view = resolve(vec![fragment]);
let graph = materialize_merge(&view, &outputs);
let prog = compile(&graph);

// 4. Execute with chosen backend
let mut backend = CpuSemiringBackend;
let results = backend.eval_program(&prog, &[tropical_a, tropical_b, tropical_c]);

// No AD available — SemiringOp<T> does not implement PrimitiveOp
```

---

## SVD with AD

```rust
let a = TracedTensor::from(Tensor::new(&a_raw, &[4, 3]));

// SVD is a primitive with linearize + transpose_rule
let (u, s, vt) = svd(&a);

// Use SVD result in further computation (hyper-edge einsum, no scatter needed)
let truncated = einsum(&mut engine, &[&u, &s, &vt], "ij,j,jk->ik");
let loss = truncated.sum();

// AD through SVD
let grad_a = loss.grad(&a);
let results = engine.eval_all(&mut [&mut loss, &mut grad_a]);
```

---

## Summary of Types

```text
Tensor           Concrete data (shape + buffer + strides + dtype)
                 The natural "tensor" — what you think of as a tensor

TracedTensor     Graph-aware wrapper for standard algebra
                 (shape + dtype + Fragment<StdTensorOp> + Option<Tensor>)
                 All lazy operations return this
                 eval() fills in data and returns &Tensor

Engine<B>        Backend + compilation cache + einsum cache
                 Generic over B: Backend<StdTensorOp>
                 Long-lived, reused across all operations

einsum()         Free function: TracedTensor inputs → TracedTensor output
                 Builds Fragment via contraction path optimization
                 No eager einsum on Tensor — use TracedTensor::from first
```

---

## Internal Flow

```text
TracedTensor::from(Tensor::new(raw, shape))
    → creates Fragment with input node
    → TracedTensor { shape, fragment, val, data: Some(Tensor) }

einsum(&mut engine, &[&a, &b], "ij,jk->ik")
    → path optimization (EinsumCache in Engine)
    → builds Fragment: Transpose → Reshape → DotGeneral → Reshape → Transpose
    → TracedTensor { shape=[inferred], fragment, val, data: None }

y.grad(&x)
    → differentiate(graph, wrt=x) → transpose(linear_fragment)
    → TracedTensor { shape=x.shape, fragment, val, data: None }

engine.eval_all(&mut [&mut y, &mut grad_x])
    → resolve (gather all reachable fragments from all outputs)
    → materialize_merge (flatten + CSE, GlobalValKey dedup)
        // shared primal nodes appear once in the materialized graph
    → engine.cache.get_or_compile (cache lookup or compile to SSA)
    → backend.eval_program(prog, inputs)
    → fills y.data = Some(result), grad_x.data = Some(result)
    → returns Vec<&Tensor>

y.eval(&mut engine)
    → if data is Some → return &Tensor immediately
    → same pipeline as above but for single output
    → returns &Tensor
```
