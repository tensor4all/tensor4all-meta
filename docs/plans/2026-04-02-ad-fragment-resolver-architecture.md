# AD Fragment Resolver Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the v2 AD architecture spec around `Fragment + Resolver + GlobalValKey`, replacing eager per-stage merge language with `resolve` plus `materialize_merge`.

**Architecture:** Treat AD transforms as fragment-producing operations over a resolver-backed logical DAG. Keep physical flattening, CSE, and slot assignment in a final `materialize_merge` stage before compile, and keep the numeric examples consistent with the existing checker script.

**Tech Stack:** Markdown, Rust-like pseudocode, Python 3 verification script

---

### Task 1: Rewrite the AD core model

**Files:**
- Create: `docs/plans/2026-04-02-ad-fragment-resolver-architecture.md`
- Modify: `docs/design/v2-ad-architecture.md`

**Step 1: Replace the graph data model**

Write the document so the primary runtime object is `Fragment`, not a single eagerly merged `Graph`. Define `ValRef`, `GlobalValKey`, `GlobalOpKey`, `OpMode`, `Resolver`, and `ResolvedView`.

**Step 2: Replace the transform pipeline**

Describe the pipeline as `build -> resolve -> differentiate -> [resolve -> differentiate]* -> [transpose] -> materialize_merge -> compile -> eval`, making clear that `resolve` is logical lookup and `materialize_merge` is physical flattening plus CSE.

**Step 3: Rewrite the scalar and vector examples**

Update the examples so they use external references explicitly and no longer claim that a physical merge is required between transforms. Keep the final formulas unchanged so the existing checker remains valid.

### Task 2: Align the backend entry point

**Files:**
- Modify: `docs/design/v2-backend-architecture.md`

**Step 1: Update the compile input terminology**

Adjust the backend document so the compile pipeline starts from `MaterializedGraph` or equivalent wording, rather than an ambiguously already-merged graph.

**Step 2: Note the delayed materialization model**

Add one short clarification that AD transforms may remain fragmented until `materialize_merge`, and that compile consumes the materialized result.

### Task 3: Verify the rewritten docs

**Files:**
- Test: `docs/design/v2_vector_ad_examples_check.py`

**Step 1: Run the numeric checker**

Run: `python3 docs/design/v2_vector_ad_examples_check.py`

Expected: sample outputs print cleanly and the final line reports verification of both vector example families.

**Step 2: Run doc-level sanity searches**

Run: `rg -n "merge must precede next differentiate|Scale|physical merge" docs/design/v2-ad-architecture.md`

Expected: no stale statements that contradict the new fragment/resolver model remain, except where old terminology is explicitly discussed as superseded.
