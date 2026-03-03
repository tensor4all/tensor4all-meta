# ad-tensors-rs: Three-Mode AD Tensor Layer Design

## Goal

Define a tensor computation layer (`ad-tensors-rs`) on top of `tenferro-rs` that supports
three AD execution modes with one API surface:

- `Primal` (plain numeric evaluation)
- `Dual` (forward mode / JVP)
- `Tracked` (reverse mode / VJP, plus HVP support)

This layer is conceptually similar to the role of NDTensors in ITensors.jl, but AD-first.

## Scope

This document defines:

- Repository boundary for `ad-tensors-rs`
- API compatibility policy with `tenferro-einsum` / `tenferro-linalg`
- Global-context based `*_auto` API layer
- Core value/scalar model for three AD modes
- Separation between primal kernels and AD rules
- `detach` (`stop_gradient`) semantics
- HVP composition from dual + tracked information
- Policy for non-smooth operations (pivot/rank/truncation)

This document does not define migration phases.

## Design Principles

1. `AnyScalar` remains, and must preserve AD mode metadata.
2. `ad-tensors-rs` is a separate repository and depends on `tenferro-*` crates.
3. `einsum`/`linalg` explicit APIs mirror `tenferro` signatures.
4. Every explicit API has a global-context `*_auto` companion.
5. Primal numeric kernels and AD rules are separate layers.
6. AD behavior is operation-rule based (`rrule`, `frule`, `hvp`).
7. Mode promotion is explicit and centralized.
8. Non-smooth branch decisions are never silently differentiated.

## Repository Strategy

`ad-tensors-rs` is maintained as an independent repository (not inside `tenferro-rs`):

- Depends on published `tenferro-*` crates (tensor/prims/einsum/linalg/algebra/device).
- Owns AD-oriented tensor-network abstractions and orchestration APIs.
- Tracks `tenferro` API compatibility in CI (compile check against target versions).

This keeps `tenferro-rs` focused on core tensor kernels and keeps AD-layer release cadence independent.

## API Compatibility Policy

`ad-tensors-rs` mirrors `tenferro` explicit-context APIs first, then adds convenience auto APIs.

### 1) Explicit API (mirror)

For example (same shape as `tenferro`):

```rust
pub fn einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>;

pub fn svd<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>;
```

### 2) Auto API (global context)

Each explicit function gets a `*_auto` variant that obtains context from thread-local global storage:

```rust
pub fn einsum_auto<Alg, Backend>(
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>;

pub fn svd_auto<T, C>(
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>;
```

Naming rule:

- Explicit: same name/signature family as `tenferro` (`einsum`, `svd`, ...)
- Auto: `<name>_auto` (`einsum_auto`, `svd_auto`, ...)

## Global Context API

`*_auto` APIs use thread-local global contexts keyed by context type:

```rust
pub fn set_global_context<C: 'static>(ctx: C) -> GlobalContextGuard<C>;
pub fn with_global_context<C: 'static, R>(
    f: impl FnOnce(&mut C) -> Result<R>,
) -> Result<R>;
pub fn try_with_global_context<C: 'static, R>(
    f: impl FnOnce(&mut C) -> Result<R>,
) -> Result<Option<R>>;
```

Behavior:

- Missing context in `*_auto` returns `Error::MissingGlobalContext { type_name }`.
- Explicit `ctx` APIs remain the canonical low-level path.
- `*_auto` is convenience sugar only (`with_global_context(|ctx| explicit(ctx, ...))`).
- Thread-local storage avoids cross-thread contention and keeps behavior deterministic.

## Core Value Model

### Base scalar domain

```rust
pub enum BaseScalar {
    F64(f64),
    C64(num_complex::Complex64),
}
```

### Mode-preserving scalar

```rust
pub enum AnyScalar {
    Primal(BaseScalar),
    Dual {
        primal: BaseScalar,
        tangent: BaseScalar,
    },
    Tracked {
        primal: BaseScalar,
        node: NodeId,
        tangent: Option<BaseScalar>,
    },
}
```

`AnyScalar` is required because many tensor-network APIs return runtime-dependent scalar
types, and AD metadata must survive scalar operations.

### Tensor wrappers

```rust
pub struct Primal<T> {
    pub value: T,
}

pub struct Dual<T> {
    pub primal: T,
    pub tangent: T,
}

pub struct Tracked<T> {
    pub primal: T,
    pub node: NodeId,
    pub tape: TapeId,
    pub tangent: Option<T>, // direction cache for HVP workflows
}
```

## Trait Boundaries

### Primal kernel trait

Primal kernels are pure numeric kernels. No tape logic and no AD branching.

```rust
pub trait TensorKernel: Clone {
    type Index: IndexLike;

    fn contract(tensors: &[&Self], allowed: AllowedPairs<'_>) -> Result<Self>;
    fn factorize(
        &self,
        left_inds: &[Self::Index],
        options: &FactorizeOptions,
    ) -> Result<FactorizeResult<Self>>;
    fn axpby(&self, a: BaseScalar, other: &Self, b: BaseScalar) -> Result<Self>;
    fn scale(&self, a: BaseScalar) -> Result<Self>;
    fn inner_product(&self, other: &Self) -> Result<BaseScalar>;
}
```

### AD operation rule trait

```rust
pub trait OpRule<V: Differentiable> {
    fn eval(&self, inputs: &[&V]) -> Result<V>;
    fn rrule(
        &self,
        inputs: &[&V],
        out: &V,
        cotangent: &V::Tangent,
    ) -> AdResult<Vec<V::Tangent>>;
    fn frule(
        &self,
        inputs: &[&V],
        tangents: &[Option<&V::Tangent>],
    ) -> AdResult<V::Tangent>;
    fn hvp(
        &self,
        inputs: &[&V],
        cotangent: &V::Tangent,
        cotangent_tangent: Option<&V::Tangent>,
        input_tangents: &[Option<&V::Tangent>],
    ) -> AdResult<Vec<(V::Tangent, V::Tangent)>>;
}
```

Each operation owns its AD behavior explicitly and can be tested in isolation.

## Execution Semantics by Mode

- `Primal<T>`: `eval` only
- `Dual<T>`: `eval + frule` (JVP propagation)
- `Tracked<T>`: `eval + rrule` on tape (VJP)
- `HVP`: operation-local `hvp` or linearized pullback composition

## Mode and DType Promotion

### DType promotion

- `F64 + F64 -> F64`
- Any combination containing `C64 -> C64`

### AD mode promotion

- `Primal < Dual < Tracked`
- Mixed-mode operations promote to the highest mode

Examples:

- `Primal tensor + Dual scalar -> Dual result`
- `Dual tensor + Tracked scalar -> Tracked result`

## Detach / Stop-Gradient

`detach` is a first-class operator: `stop_gradient(x)`.

Mathematically:

- Primal: identity (`y = x`)
- Reverse derivative: `0`
- Forward tangent: `0`
- HVP contribution: `0`

Mode behavior:

- `Primal`: no-op
- `Dual`: keep primal, set tangent to zero or `None`
- `Tracked`: keep primal, disconnect from tape (`requires_grad = false`)

## Non-Smooth Operations Policy

Pivot selection, rank truncation, and similar branch decisions are generally non-smooth.
Use explicit differentiation policy:

```rust
pub enum DiffPolicy {
    Strict,        // return ModeNotSupported
    StopGradient,  // allow primal eval, block derivatives
}
```

Recommended default for tensor-network workloads: `StopGradient`.

## HVP Composition

Let graph nodes satisfy:

\[
x_k = \phi_k(x_{p_1(k)}, \dots, x_{p_m(k)}), \quad f = x_L \in \mathbb{R}
\]

Forward pass:

\[
\dot{x}_k = \sum_j \frac{\partial \phi_k}{\partial x_{p_j}} \dot{x}_{p_j}
\]

Reverse seeds:

\[
\bar{x}_L = 1,\quad \dot{\bar{x}}_L = 0
\]

Reverse recursion:

\[
\bar{x}_{p_j} \mathrel{+}= J_{k,j}^\top \bar{x}_k
\]

Linearized reverse recursion (for HVP):

\[
\dot{\bar{x}}_{p_j} \mathrel{+}= J_{k,j}^\top \dot{\bar{x}}_k
+ \dot{J}_{k,j}^\top \bar{x}_k
\]

with:

\[
\dot{J}_{k,j} = D J_{k,j}[\dot{x}]
\]

At parameter leaves:

- Gradient: \(\nabla f = \bar{\theta}\)
- HVP: \(H v = \dot{\bar{\theta}}\)

## Complexity

For one direction \(v\), HVP should stay in the same asymptotic class as one model pass:

- Gradient: \(O(C)\)
- One HVP: \(O(C)\)
- \(K\) directions: \(O(KC)\)

where \(C\) is the forward computational cost.

## Testing Requirements

1. Mode invariants:
   - AD metadata is preserved through scalar/tensor operations.
2. Rule consistency:
   - `frule` matches finite-difference directional derivative.
   - `rrule` matches finite-difference VJP checks.
