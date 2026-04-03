# Memory Kind Terminology Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `design-v2` placement terminology around `Placement` and `MemoryKind`, removing the old `LogicalMemorySpace` model from the public design.

**Architecture:** The docs will treat placement as runtime tensor metadata made of `memory_kind` plus `resident_device`, with `preferred_compute_device` remaining a separate execution hint. Backend and tensor API documents will be updated together so the design has a single placement vocabulary.

**Tech Stack:** Markdown docs, `rg`, `git diff --check`

---

### Task 1: Update the core placement model docs

**Files:**
- Modify: `docs/design-v2/backend-architecture.md`
- Modify: `docs/design-v2/tenferro-internal-design.md`

**Step 1: Replace old placement terminology in backend architecture**

Edit `docs/design-v2/backend-architecture.md` to:
- replace references to device or memory-space metadata on `Tensor` with `Placement`
- replace `device()` examples with `placement()`, `resident_device()`, and placement-aware transfer examples
- rewrite the compatibility table in terms of canonical `MemoryKind` values

**Step 2: Remove the stale tenferro-device note**

Edit `docs/design-v2/tenferro-internal-design.md` so `tenferro-device` is no longer described as "unchanged from v1", and instead summarize the new `Placement` / `MemoryKind` / `ComputeDevice` split.

**Step 3: Verify the docs are internally consistent**

Run:

```bash
rg -n "LogicalMemorySpace|memory_space|MainMemory|PinnedMemory|GpuMemory|ManagedMemory" docs/design-v2/backend-architecture.md docs/design-v2/tenferro-internal-design.md
```

Expected: no matches, or only intentional historical references that are explicitly marked as replacements.

### Task 2: Update the tensor API docs

**Files:**
- Modify: `docs/design-v2/tensor-design.md`
- Modify: `docs/design-v2/tensor-api-pseudocode.md`

**Step 1: Rewrite the runtime tensor description**

Edit `docs/design-v2/tensor-design.md` to introduce `Placement`, `MemoryKind`, and `resident_device` in the `TensorData<T>` description.

**Step 2: Rewrite the public tensor API snippet**

Edit `docs/design-v2/tensor-api-pseudocode.md` so host/device transfer examples use:

```rust
impl Tensor {
    fn placement(&self) -> Placement;
    fn memory_kind(&self) -> MemoryKind;
    fn resident_device(&self) -> Option<ComputeDevice>;
    fn to_placement(&self, target: Placement) -> Tensor;
    fn to_cpu(&self) -> Tensor;
    fn to_gpu_on(&self, device_id: usize) -> Tensor;
    fn to_pinned_host(&self) -> Tensor;
}
```

**Step 3: Verify old public names are gone**

Run:

```bash
rg -n "device\\(|to_memory_space|memory_space\\(" docs/design-v2/tensor-design.md docs/design-v2/tensor-api-pseudocode.md
```

Expected: no matches.

### Task 3: Final consistency pass

**Files:**
- Modify: `docs/design-v2/backend-architecture.md`
- Modify: `docs/design-v2/tenferro-internal-design.md`
- Modify: `docs/design-v2/tensor-design.md`
- Modify: `docs/design-v2/tensor-api-pseudocode.md`

**Step 1: Run markdown sanity checks**

Run:

```bash
git diff --check -- docs/design-v2/backend-architecture.md docs/design-v2/tenferro-internal-design.md docs/design-v2/tensor-design.md docs/design-v2/tensor-api-pseudocode.md docs/plans/2026-04-03-memory-kind-design.md docs/plans/2026-04-03-memory-kind-implementation.md
```

Expected: no output.

**Step 2: Commit the docs**

Run:

```bash
git add docs/design-v2/backend-architecture.md docs/design-v2/tenferro-internal-design.md docs/design-v2/tensor-design.md docs/design-v2/tensor-api-pseudocode.md docs/plans/2026-04-03-memory-kind-design.md docs/plans/2026-04-03-memory-kind-implementation.md
git commit -m "rewrite v2 placement docs around memory kinds"
```
