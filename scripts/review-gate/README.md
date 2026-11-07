# Review gate: the two rules in their final form

Reference implementation and test suite for the two defects tracked as
[#88](https://github.com/tensor4all/tensor4all-meta/issues/88) and
[#89](https://github.com/tensor4all/tensor4all-meta/issues/89). The code they
describe lives in three consumer repositories, each carrying its own copy of both
pieces:

| repository | `scripts/repository-rules-review.py` | `verify-review-label` |
|---|---|---|
| `strided-rs` | 1491 lines, 52412 bytes | `.github/actions/verify-review-label/action.yml` |
| `tenferro-rs` | 2911 lines, 107123 bytes | inline in `.github/workflows/review_bot.yml` |
| `tensor4all-rs` | 1912 lines, 68919 bytes | `.github/actions/verify-review-label/action.yml` |

The three parser copies are **not** byte-identical (sha256 `eabd489f…`,
`c80caa91…`, `4f6ba646…`), so a fix cannot be replicated by copying one file
over the others. What *can* be shared is the rule and its tests, which is what
this directory holds: each repository replaces its local loops with these
functions and runs the same fixtures.

Measured at `strided-rs@99eefc55`, `tenferro-rs@e2a2e52b`,
`tensor4all-rs@5eee4d95`.

## Rule 1 — a file header cannot be forged from inside a hunk (`diffparse.py`)

A `git diff` is a small state machine, and the old copies of the parser ignored
the state:

```python
for line in diff_text.splitlines():
    if line.startswith("+++ "):        # no hunk state: any line, anywhere
        current_file = line.removeprefix("+++ b/").removeprefix("+++ ")
```

An added line whose *content* begins with `++ ` renders as `+++ b/whatever`, so
the parser adopted a path chosen by the pull-request author and attributed every
following addition to it. Every rule keyed on the path then read the wrong file;
the frozen-tree check is the one worth defeating.

`parse_diff` fixes the grammar: `@@ -a,b +c,d @@` opens a hunk whose body is
exactly `b` old-side and `d` new-side lines, headers are recognised only when no
body is outstanding, and `\ No newline at end of file` is metadata rather than a
body line. `fixtures/forged-header.diff` is a real `git diff` of a change to
`deprecated/strided-old/src/lib.rs` whose block comment carries the line
`++ b/review.txt`.

Drop-in functions, same signatures as the copies they replace (def line at the
measured revisions):

| function | replaces |
|---|---|
| `added_lines_with_text(diff_text)` | `strided-rs:451`, `tenferro-rs:713`, `tensor4all-rs:568` |
| `added_lines_by_file(diff_text)` | `tenferro-rs:613` |
| `files_with_deleted_lines(diff_text)` | `strided-rs:435` |
| `sensitive_diff_location(diff_text)` | `strided-rs:741`, `tenferro-rs:1004`, `tensor4all-rs:858` (`+++` branch at `strided-rs:756`, `tenferro-rs:1019`, `tensor4all-rs:873`) |

Two additions:

- `changed_paths(diff_text)` — every non-`/dev/null` path, both sides.
- `assert_paths_consistent(diff_text, name_status_text)` — cross-checks the
  paths the diff *claims* against `git diff --name-status -z`, which
  NUL-separates them and cannot be influenced by the diff text. A gate should
  fail closed on a non-empty result. This is the second layer from the issue:
  the parser no longer invents paths, and if a future edit reintroduces one the
  cross-check still refuses.

Two helpers have no drop-in here because their bodies are repository-specific
and must be rebuilt on `parse_diff`: `files_with_unanchorable_deletions`
(`tenferro-rs:642`) and `split_large_file_diff` (`strided-rs:657`,
`tenferro-rs:912`, `tensor4all-rs:774`), which splits chunks on
`line.startswith("@@")`. Neither moves a path, so neither is a bypass today, but
both should key on the same hunk state so the three parsers cannot drift apart
again.

## Rule 2 — a waiver names a head SHA (`waiver.js`)

The replaced rule bound the approval to "a maintainer applied the label at some
point", and tried to date the last head update from a field that ordinary
`committed` timeline events do not carry (measured 0/2 on `strided-rs#369`,
0/1 on `tenferro-rs#2501` and `tensor4all-rs#1161`). `committer.date` is not a
substitute — the author controls it.

The rule implemented here:

> A waiver is a comment by someone with write access or better containing
> `waive-review <40-character sha>`. It is valid only while that SHA is the
> pull-request head. The label is a pointer for humans and grants nothing.

Consequences, each pinned by a fixture in `fixtures/waiver/`:

| situation | old decision | new decision |
|---|---|---|
| waiver names the current head | allow (`legacy-label-trusted`) | allow (`waiver-matches-head`) |
| author pushes; waiver names the old head | **allow** | deny (`stale-waiver`) |
| close/reopen, head unchanged | allow | allow (`waiver-matches-head`) |
| waiver posted by `CONTRIBUTOR` | **allow** | deny (`non-maintainer-waiver`) |
| label applied, no waiver comment | **allow** | deny (`no-waiver`) |
| abbreviated sha (`waive-review 2e2a…`) | **allow** | deny (`shorthand-waiver`) |
| newer maintainer waiver names another head | allow | deny (`stale-waiver`) |

Because the binding is to a commit rather than to a timeline event, close and
reopen need no special case: the same head keeps its waiver, a pushed head loses
it. The verifier must not keep the `created_at` comparison as a fallback —
leaving it in place restores the old trust in the label.

`run({github, core, context, labelName})` is the `actions/github-script` entry
point: it paginates the issue comments, resolves a repository role for authors
whose `author_association` is not conclusive, evaluates, and calls
`core.setFailed` with the reason when the waiver does not bind to the head.

Sites to replace: `.github/actions/verify-review-label/action.yml:45-68` in
`strided-rs` and `tensor4all-rs` (the two files are byte-identical), and
`review_bot.yml:357-374` plus `review_bot.yml:190-207` in `tenferro-rs` (waive
and no-LLM verifiers, both inline).

## Applying it to a repository

1. Copy `diffparse.py` next to the review script and call its functions instead
   of the local copies listed above.
2. Fetch `git diff --name-status -z` for the same base/head pair and fail the
   gate when `assert_paths_consistent` is not `ok`.
3. Put `waiver.js` at `.github/actions/verify-review-label/waiver.js` and reduce
   the action step to:

   ```yaml
   - uses: actions/github-script@v7
     env:
       REVIEW_LABEL: ${{ inputs.label }}
     with:
       script: |
         const { run } = require(`${process.env.GITHUB_WORKSPACE}/.github/actions/verify-review-label/waiver.js`);
         await run({ github, core, context, labelName: process.env.REVIEW_LABEL });
   ```

4. Keep `scripts/test-review-gate.py` and `scripts/test-waiver.js` in the
   repository so the next edit to the parser or the waiver has the same
   fixtures to answer to.

## Running the tests

```bash
python3 scripts/test-review-gate.py     # 13 tests, stdlib only, no network
node --test scripts/test-waiver.js      # 14 tests, fixture JSON
```

Both suites are wired into `.github/workflows/review-gate.yml`.
