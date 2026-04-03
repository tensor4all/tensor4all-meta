# v2 Tensor API Pseudocode

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `v2-architecture-overview.md`

---

## Design Decisions

- All operations are **lazy** (deferred). No eager mode.
- `Tensor` is a single type (not an enum). Internally holds a graph node reference.
- `eval()` triggers `materialize_merge -> compile (cached) -> execute`.
- `Engine` holds execution context + compilation cache + extension caches.
- `DynTensor` is the concrete data type. Users interact with it only at
  input/output boundaries.

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
// Input: DynTensor (concrete data) → Tensor (graph node)
let x = Tensor::input("x", DynTensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
let w = Tensor::input("w", DynTensor::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

// All operations build graph, nothing is computed yet
let y = einsum(&[&x, &w], "ij,jk->ik");
let z = y.exp();
let loss = z.sum();

// eval triggers compilation + execution, returns DynTensor
let loss_val: DynTensor = loss.eval(&mut engine);

// Multiple outputs in one eval
let (z_val, loss_val): (DynTensor, DynTensor) = eval_all(&mut engine, &[&z, &loss]);
```

---

## Einsum (N-ary)

```rust
let a = Tensor::input("a", a_data);  // [2, 3]
let b = Tensor::input("b", b_data);  // [3, 4]
let c = Tensor::input("c", c_data);  // [4, 5]

// N-ary einsum: contraction path optimization happens inside
let result = einsum(&[&a, &b, &c], "ij,jk,kl->il");

let result_val: DynTensor = result.eval(&mut engine);
// First call:  path optimization + Fragment build + compile + eval
// Second call (same shapes): all caches hit → eval only
```

---

## Reverse-Mode AD (grad / VJP)

```rust
let x = Tensor::input("x", x_data);  // [3]
let a = Tensor::input("a", a_data);  // [3]

// Forward computation (lazy)
let ax = einsum(&[&a, &x], "i,i->");  // dot product
let y = ax.exp();                      // scalar output

// grad: differentiate + transpose (graph transform, still lazy)
let grad_x = y.grad(&x);

// eval
let y_val: DynTensor = y.eval(&mut engine);
let grad_val: DynTensor = grad_x.eval(&mut engine);
// grad_val = a * exp(a . x)
```

---

## Forward-Mode AD (JVP)

```rust
let x = Tensor::input("x", x_data);
let a = Tensor::input("a", a_data);

let y = einsum(&[&a, &x], "i,i->").exp();

// JVP with tangent vector
let t_x = Tensor::input("t_x", tangent_data);
let dy = y.jvp(&x, &t_x);  // differentiate only (no transpose)

let dy_val: DynTensor = dy.eval(&mut engine);
```

---

## Hessian-Vector Product (HVP)

```rust
let x = Tensor::input("x", x_data);
let a = Tensor::input("a", a_data);

let y = einsum(&[&a, &x], "i,i->").exp();

// Forward-over-reverse: jvp(grad(f))
let grad_x = y.grad(&x);                 // differentiate + transpose
let t_x = Tensor::input("t_x", tangent_data);
let hvp = grad_x.jvp(&x, &t_x);         // differentiate again

let hvp_val: DynTensor = hvp.eval(&mut engine);
```

---

## Reusable Functions

```rust
// Define model as a normal function over Tensor
fn my_model(x: &Tensor, w: &Tensor) -> Tensor {
    einsum(&[x, w], "ij,jk->ik").exp().sum()
}

// Evaluate
let x = Tensor::input("x", x_data);
let w = Tensor::input("w", w_data);
let loss = my_model(&x, &w);
let loss_val = loss.eval(&mut engine);

// Differentiate the same function
let grad_w = my_model(&x, &w).grad(&w);
let grad_val = grad_w.eval(&mut engine);

// Different data, same graph structure → cache hit
let x2 = Tensor::input("x", x_data2);
let w2 = Tensor::input("w", w_data2);
let loss2_val = my_model(&x2, &w2).eval(&mut engine);  // cache hit
```

---

## Compilation Cache Behavior

```rust
let x = Tensor::input("x", x_data_2x2);
let w = Tensor::input("w", w_data_2x2);
let y = my_model(&x, &w);
let _ = y.eval(&mut engine);  // compile + cache

// Same graph structure, different data → cache hit
let x2 = Tensor::input("x", x_data2_2x2);
let w2 = Tensor::input("w", w_data2_2x2);
let _ = my_model(&x2, &w2).eval(&mut engine);  // cache hit

// Same subscripts but different shapes may produce different
// contraction paths → different graph structure → recompile
let x3 = Tensor::input("x", x_data_100x100);
let w3 = Tensor::input("w", w_data_100x100);
let _ = my_model(&x3, &w3).eval(&mut engine);  // may recompile
```

---

## Tropical Algebra (No AD)

```rust
// Tropical semiring: same API, just different algebra
let engine = Engine::new(CpuContext::new());

let a = Tensor::input("a", tropical_data_a);
let b = Tensor::input("b", tropical_data_b);
let c = Tensor::input("c", tropical_data_c);

// N-ary einsum works over tropical semiring
let result = einsum_tropical(&[&a, &b, &c], "ij,jk,kl->il");
let result_val = result.eval(&mut engine);

// No AD available for tropical — grad() would return compile-time error
// or runtime error depending on design choice
```

---

## SVD with AD

```rust
let a = Tensor::input("a", a_data);  // [m, n]

// SVD is a primitive with linearize + transpose_rule
let (u, s, vt) = svd(&a);

// Use SVD result in further computation
let truncated = einsum(&[&u, &diag(&s), &vt], "ij,j,jk->ik");
let loss = truncated.sum();

// AD through SVD
let grad_a = loss.grad(&a);
let grad_val = grad_a.eval(&mut engine);
```

---

## Summary of Types

```text
DynTensor   Concrete data (input/output boundary only)
Tensor      Graph node reference (all operations return this)
Engine      Execution context + compilation cache + extension caches
Trace       (removed — Tensor::input replaces trace.input)
```

---

## Internal Flow

```text
Tensor::input(name, DynTensor)
    → creates Fragment input node, returns Tensor

einsum(&[&a, &b], "ij,jk->ik")
    → path optimization (EinsumCache in Engine)
    → builds Fragment: Transpose → Reshape → DotGeneral → Reshape → Transpose
    → returns Tensor

y.grad(&x)
    → differentiate(graph, wrt=x) → transpose(linear_fragment)
    → returns Tensor (still lazy)

y.eval(&mut engine)
    → resolve (gather reachable fragments)
    → materialize_merge (flatten + CSE)
    → engine.cache.get_or_compile (cache lookup or compile to SSA)
    → compiled.eval(ctx, inputs) → DynTensor
```
