# Design v2 Overview

**Date:** 2026-04-04
**Status:** Draft

---

## Vision

Build a differentiable programming stack in Rust with clean separation between
graph infrastructure, AD traits, AD transforms, and concrete primitives.

---

## Crate Hierarchy

```text
computegraph-rs    General-purpose computation graph engine
    ↓              (GraphOp, Operand, Fragment, resolve,
    ↓               materialize_merge, compile, eval, cache)
    ↓
chainrules-rs      AD trait definitions
    ↓              (PrimitiveOp: GraphOp + add + linearize + transpose_rule)
    ↓
tidu-rs            AD graph transforms
    ↓              (differentiate, transpose — generic over PrimitiveOp)
    ↓
tenferro-rs        Concrete tensor primitives + backends
                   (Tensor, StableHLO lowering, optimizing compiler,
                    Execution IR, CPU/GPU dispatch)
```

Each layer depends only on the layers above it. No layer references specific
primitives except tenferro-rs.

---

## Document Organization

This directory is organized into four categories:

### spec/ — Normative specifications (source of truth)

Each fact has exactly one owner document. Other documents link here rather
than re-stating.

| Document | Owns |
|----------|------|
| [primitive-catalog.md](spec/primitive-catalog.md) | Tenferro IR op vocabulary, per-op semantics, StableHLO lowering rules, frontend sugar / canonical lowering |
| [backend-contract.md](spec/backend-contract.md) | Backend pipeline, Execution IR dispatch categories, backend trait signatures (`SemiringCore`, `SemiringFastPath`), generic execution engine, backend comparison (XLA vs faer vs custom), buffer lifecycle, memory layout |
| [optimizer-passes.md](spec/optimizer-passes.md) | Optimization pass algorithms (DotDimensionSorter, TransposeFolding, DotDecomposer, ReductionSimplification), pass ordering |
| [tensor-semantics.md](spec/tensor-semantics.md) | Tensor type semantics, stride model, contiguity rules, dense-only principle, structural types in tensor4all-rs |
| [ad-contract.md](spec/ad-contract.md) | `PrimitiveOp` trait signature, `linearize`/`transpose_rule` requirements, closure rule, cotangent accumulation |

### architecture/ — Design rationale

Describes *what* each subsystem does and *why*. Does NOT duplicate normative
tables or trait signatures — links to spec/ instead.

| Document | Covers |
|----------|--------|
| [tenferro-crates.md](architecture/tenferro-crates.md) | Crate structure, `StdTensorOp` / `SemiringOp<T>` design, `SemiringOps` trait, einsum builder, user extension points |
| [computegraph.md](architecture/computegraph.md) | `GraphOp`, `Operand`, `Fragment`, resolve/materialize/compile/eval pipeline |
| [chainrules.md](architecture/chainrules.md) | `PrimitiveOp` trait, AD rule structure |
| [tidu.md](architecture/tidu.md) | `differentiate`, `transpose`, `LinearFragment`, higher-order AD |
| [ad-pipeline.md](architecture/ad-pipeline.md) | End-to-end AD pipeline, scalar/vector examples, golden tests |

### reference/ — Non-normative reference material

| Document | Contents |
|----------|----------|
| [jax-primitives.md](reference/jax-primitives.md) | JAX primitive inventory |
| [stablehlo-primitives.md](reference/stablehlo-primitives.md) | StableHLO op inventory |
| [jax-stablehlo-needed.md](reference/jax-stablehlo-needed.md) | Phase-1 StableHLO target set, JAX↔tenferro correspondence |
| [ad-graph-experiments.md](reference/ad-graph-experiments.md) | Earlier experiments informing the fragment-based AD design |

### examples/ — Non-normative worked examples

| Document | Contents |
|----------|----------|
| [tensor-api-pseudocode.md](examples/tensor-api-pseudocode.md) | `TracedTensor` API usage, lazy evaluation, AD examples |
| [vector_ad_examples_check.py](examples/vector_ad_examples_check.py) | Sanity-check script for vector AD examples |

---

## Key Design Principles

### computegraph-rs is AD-agnostic

The graph engine knows nothing about differentiation. It is equally usable for
multi-tensor einsum (graph of binary contractions) or any DAG-structured
computation.

**Note on `Operand`:** While `Operand` is defined in computegraph-rs, its
methods are tensor-oriented algebraic operations (`dot_general`, `reduce_sum`,
`conj`, etc.). Structural operations (`transpose`, `reshape`,
`broadcast_in_dim`) live in a separate `TensorData` trait. This is a deliberate
design choice -- computegraph-rs is a **tensor computation graph engine**, not a
fully generic DAG engine. The abstraction boundary is AD-agnostic, not
tensor-agnostic.

### tidu-rs is primitive-agnostic

AD transforms (`differentiate`, `transpose`) are fully generic over
`Op: PrimitiveOp`. tidu-rs never references specific primitives such as `Add`
or `Mul`.

The concrete tensor vocabulary that downstream tenferro supplies is documented
in [`primitive-catalog.md`](spec/primitive-catalog.md).

### Closure is downstream responsibility

The only rule for primitives is:

> `linearize` and `transpose_rule` must emit only ops that themselves
> implement `PrimitiveOp`.

tenferro-rs is responsible for satisfying this.

### Physical merge is late

```text
differentiate -> resolve -> differentiate -> resolve -> ...  (logical, cheap)
materialize_merge  (physical, once, before compile)
```

Higher-order AD requires `resolve` (logical), not `materialize_merge`
(physical).

---

## Conventions

- **1 fact, 1 owner**: each normative fact lives in exactly one spec/ document
- **Link, don't re-state**: architecture/ and examples/ reference spec/ tables, never duplicate them
- **Pseudocode is illustrative**: code examples in architecture/ are non-normative unless explicitly marked as spec
- **Normative trait signatures**: spec/primitive-catalog.md (Tenferro IR ops, `Operand`, `GraphOp`), spec/backend-contract.md (backend traits: `SemiringCore`, `SemiringFastPath`), spec/ad-contract.md (AD traits: `PrimitiveOp`)
