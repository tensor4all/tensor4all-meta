# Tier 1 Transpose Rules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated detailed design note for Tier 1 transpose rules and link it from the v2 AD architecture doc.

**Architecture:** Keep the main AD architecture document focused on the high-level model, and move the full Tier 1 rule table plus downstream contract into a separate design note. Update the main doc so it clearly states that downstream primitives must linearize into this transpose-closed Tier 1 set.

**Tech Stack:** Markdown, Rust-like pseudocode, local JAX source for cross-checking

---

### Task 1: Add detailed Tier 1 transpose note

**Files:**
- Create: `docs/design/v2-tier1-transpose-rules.md`

**Step 1: Write the scope and contract**
State that `tidu2::transpose` directly handles only Tier 1 primitives and that downstream complex primitives must linearize into this set.

**Step 2: Write the full Tier 1 rule table**
Cover `Add`, `Mul`, `Neg`, `Conj`, `Dup`, `DotGeneral`, `Transpose`, `Reshape`, `BroadcastInDim`, and `ReduceAdd`.

**Step 3: Add worked notes for `DotGeneral`**
Spell out how contraction, batch axes, and explicit `Transpose` / `Reshape` steps reconstruct the cotangent flow.

### Task 2: Link the main AD architecture doc

**Files:**
- Modify: `docs/design/v2-ad-architecture.md`

**Step 1: Update the transpose section**
Replace the current short local-rule sketch with a summary plus a link to the new detailed note.

**Step 2: Clarify downstream responsibility**
State that Tier 2 primitives such as `SVD` and `Solve` are downstream and must linearize into the transpose-closed Tier 1 set.

### Task 3: Verify docs

**Files:**
- Test: `docs/design/v2-ad-architecture.md`
- Test: `docs/design/v2-tier1-transpose-rules.md`

**Step 1: Search for the new link and contract**
Run: `rg -n "v2-tier1-transpose-rules|transpose-closed|Tier 2 primitives such as" docs/design/v2-ad-architecture.md docs/design/v2-tier1-transpose-rules.md`
Expected: the main doc links to the detailed note and the downstream contract is present.
