# v2 Tensor API Pseudocode

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `README.md`

---

## Design Decisions

- All operations are **lazy** (deferred). No eager mode.
- `TracedTensor` is the user-facing type that holds shape/dtype, graph info,
  and optionally data. All operations return `TracedTensor`.
- `Tensor` is the concrete data type (shape + buffer). Users see it at
  input/output boundaries.
- `eval()` triggers `materialize_merge -> compile (cached) -> execute`,
  filling in the data.
- `Engine` holds execution context + compilation cache + extension caches.
  `Engine` is a **tenferro-rs concept** that wraps `computegraph-rs`'s
  compilation cache with backend-specific execution context (CPU/GPU) and
  extension caches (e.g., `EinsumCache` for contraction path optimization).
- `eval_all` is the primary evaluation API. It resolves all output fragments
  together into one `MaterializedGraph`, so shared intermediate nodes (e.g.,
  primal values needed by both forward output and gradient) are computed once.
  Single-output `eval` is a convenience wrapper around `eval_all`.

---

## Type Definitions

```rust
// Concrete data — the natural "tensor"
struct Tensor {
    buffer: DataBuffer,
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: DType,
}

// Graph-aware wrapper — tracks computation for AD and compilation
struct TracedTensor {
    shape: Vec<usize>,
    dtype: DType,
    fragment: Arc<Fragment<TensorOp>>,
    val: LocalValId,
    data: Option<Tensor>,  // Some for inputs / eval'd results, None for deferred
}
```

- `TracedTensor::from(Tensor)` creates a Fragment input node with `data = Some(...)`.
- Operations (einsum, exp, ...) create new Fragments and return
  `TracedTensor` with `data = None`.
- `eval()` fills in `data` and returns `&Tensor`.

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
// Create engine with CPU backend
let mut engine = Engine::new(CpuContext::new());

// Or with GPU
let mut engine = Engine::new(CudaContext::new(device_id));
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
// loss.data is now Some(...)

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

Cache key is graph structure (`GlobalValKey`) only. Shape-dependent ops
(e.g. `Reshape(target)`) encode shape in the op itself, so different shapes
produce different `GlobalValKey`s automatically.

---

## Tropical Algebra (No AD)

```rust
let engine = Engine::new(CpuContext::new());

let a = TracedTensor::from(Tensor::new(&tropical_a, &[3, 4]));
let b = TracedTensor::from(Tensor::new(&tropical_b, &[4, 5]));
let c = TracedTensor::from(Tensor::new(&tropical_c, &[5, 6]));

// einsum works generically over different algebras (standard, tropical, etc.)
// The algebra is determined by the tensor's algebra type parameter, not by
// calling a separate function per algebra.
let result = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il");
let result_val: &Tensor = result.eval(&mut engine);

// No AD available for tropical — grad() would return compile-time error
// or runtime error depending on design choice
```

---

## SVD with AD

```rust
let a = TracedTensor::from(Tensor::new(&a_raw, &[4, 3]));

// SVD is a primitive with linearize + transpose_rule
let (u, s, vt) = svd(&a);

// Use SVD result in further computation
let truncated = einsum(&mut engine, &[&u, &diag(&s), &vt], "ij,j,jk->ik");
let loss = truncated.sum();

// AD through SVD
let grad_a = loss.grad(&a);
let results = engine.eval_all(&mut [&mut loss, &mut grad_a]);
```

---

## Summary of Types

```text
Tensor         Concrete data (shape + buffer + strides + dtype)
               The natural "tensor" — what you think of as a tensor

TracedTensor   Graph-aware wrapper (shape + dtype + graph info + Option<Tensor>)
               All lazy operations return this
               eval() fills in data and returns &Tensor

Engine         Execution context + compilation cache + extension caches
               Long-lived, reused across all operations
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
