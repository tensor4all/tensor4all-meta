# Memory Kind and Placement Design

**Date:** 2026-04-03
**Status:** Approved
**Scope:** `tensor4all-meta/docs/design-v2`

---

## Goal

Replace the old `LogicalMemorySpace` / `memory_space` model in `design-v2`
with a placement model that is closer to JAX/XLA terminology while preserving
tenferro's distinction between data placement and compute preference.

## Decisions

### 1. Public placement model

`LogicalMemorySpace` is removed from the v2 design.

The new source of truth is:

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
```

### 2. Meaning of each field

- `memory_kind` describes the class of memory used for tensor storage.
- `resident_device` identifies the device that owns or primarily addresses that
  memory.
- `preferred_compute_device` remains a separate runtime hint for where an op
  should execute.

This preserves the tenferro rule that data placement and compute placement are
related but not identical concepts.

### 3. Canonical examples

- Ordinary CPU tensor:
  `Placement { memory_kind: UnpinnedHost, resident_device: None }`
- Pinned host tensor:
  `Placement { memory_kind: PinnedHost, resident_device: None }`
- GPU-resident tensor:
  `Placement { memory_kind: Device, resident_device: Some(Cuda { device_id }) }`
- Backend-specific managed memory:
  `Placement { memory_kind: Other("managed".into()), resident_device: Some(...) }`

### 4. Public API terminology

The v2 design should use placement-oriented names:

- `placement() -> Placement`
- `memory_kind() -> MemoryKind`
- `resident_device() -> Option<ComputeDevice>`
- `preferred_compute_device() -> Option<ComputeDevice>`
- `to_placement(target: Placement) -> Tensor`

Sugar APIs remain placement-oriented:

- `to_cpu()`
- `to_gpu_on(device_id)`
- `to_pinned_host()`

The old names are removed from the v2 design:

- `LogicalMemorySpace`
- `memory_space()`
- `to_memory_space(...)`

### 5. Compiler/runtime boundary

`CompiledProgram` stays placement-agnostic.

Placement is carried only by runtime tensors. Cross-device or cross-placement
execution is therefore still a runtime concern rather than an IR concern.

### 6. Compatibility story

The v2 docs do not preserve backward-compatible terminology.

Old names are replaced outright:

- `MainMemory` -> `UnpinnedHost`
- `PinnedMemory` -> `PinnedHost`
- `GpuMemory { device_id }` -> `Placement { memory_kind: Device, resident_device: Some(...) }`
- `ManagedMemory` -> `Other("managed")`

### 7. Backend mapping

Backend docs should describe canonical public memory kinds and their backend
realizations rather than pretending every backend shares one fixed enum.

This allows tenferro to stay compatible with JAX/XLA's `memory_kind` idea
without making the public API fully stringly typed.

## Non-goals

- Modeling kernel-local scratch memories such as shared memory or VMEM as part
  of the public tensor placement API
- Making `to_device(...)` the primary public API
- Preserving v1 or early v2 placement naming in the public docs
