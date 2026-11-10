# Contribution rules for the Rust repositories

This is the single place where the rules that apply to **every Rust repository in `tensor4all`** are written down.
A repository's own `CONTRIBUTING.md` says how that repository differs — its MSRV, its feature combinations, the
commands its CI runs — and must not restate the rules below in different words. Where the two disagree, this page is
the one that moves, unless the repository records why in its own file.

The list of repositories this applies to is in [`repos.txt`](repos.txt), and
`scripts/check-contribution.py` reads it.

## Why a single page

Fourteen of the twenty-three Rust repositories in the organisation had no `CONTRIBUTING.md` at all, and the two that
did described the same practices in different words — so a contributor who had read one of them could not tell which
parts were the project's rules and which were that repository's local habits. The rules below are the parts that were
already true everywhere but written nowhere.

## 1. Work starts from an issue, and the issue is claimed

- Every change that is not a one-line typo starts from an issue. The issue names the acceptance criteria; if it does
  not, adding them is the first commit's job.
- **Claim the issue before you work on it.** If it is assigned to somebody else, it is theirs. Do not take an
  assignee away, and re-check the assignee immediately before opening the pull request — assignment is a lock, and a
  second writer reading a stale page is what the lock is for.
- If an open pull request already does what the issue asks, the issue does not need a second one. Search the open
  pull requests by title, by `Closes #<n>` in the body, and by branch name before starting.

## 2. A pull request is one decision

- One pull request answers one question. Bundling several unrelated fixes is allowed only when they are the same
  kind of change in one crate and each is too small to review alone; say so in the body when you bundle.
- A pull request that changes behaviour and a pull request that changes performance are separate even when they touch
  the same file, because they are reviewed against different evidence.
- Draft until it can be reviewed. A pull request that still needs its own decisions made should stay a draft rather
  than collect review comments on a moving target.

## 3. Every pull request carries evidence, not claims

The body has a `## Verification` section — that exact heading, no parenthetical after it — containing at least one
fenced code block with a command and its real output. The `body / body` check enforces the presence of that block, and
a reviewer enforces its honesty.

- **Measured, and by what.** Every number says how it was produced: the command, the host, the commit. A number that
  comes from a different revision, a different host, or a patched harness says so in the same sentence.
- **Not measured, and why.** A claim you could not measure is written as `Not measured: <what> — <why>`, not deleted
  and not softened into an unquantified sentence.
- **No general rules from samples.** A sentence of the form "the kernel does X for every layout" needs the count of
  layouts and the command that produced it; three cases are three cases.
- **No claim stronger than the test.** If the test would pass with the feature removed, the test is not evidence.
  State what the test observes, or add the mutation that makes it fail.

## 4. Tests are written red first

- Write the failing test, run it, keep the output. A test written after the fix is a regression guard, which is
  useful, but it is not evidence that the fix addresses a measured defect — say which one you are adding.
- A test that cannot fail is a defect. Where a claim matters (numerical identity, a threshold, a dispatch decision),
  break the implementation on purpose once and check that the new test fails.
- Numerics follow the contracts the crate documents: NaN propagates rather than vanishing behind `f64::max`, an
  operation that cannot represent its result returns an error rather than `Ok` with a wrong number, and where a crate
  promises bit-identical output across layouts or thread counts, a test pins it.

## 5. The gates a Rust repository runs

Every Rust repository in the organisation runs at minimum the gates below. This is the rule the repositories are held
to; it is not a measurement of each repository's CI in this change, and a repository that does not yet run one of them
is a defect to report rather than a licence to keep the others:

| gate | what it means |
|---|---|
| `fmt` | `cargo fmt --check` on the workspace |
| `clippy` | `cargo clippy --all-targets` at `-D warnings` for the crate's own code |
| `test` | the workspace test suite on the platforms the crate claims |
| `msrv` | the manifest's `rust-version` actually builds — a pinned toolchain, not the current stable |
| `no_std` | where the crate advertises `no_std`, a build without `std` |
| `docs` | rustdoc builds without warnings for the public surface |
| `body` | the pull-request body carries `## Verification` with evidence (section 3) |
| `review` | the review gate below |

A repository may add gates; none may drop one of these without recording why in its `CONTRIBUTING.md`.

## 6. Review

- **An adversarial review of the head commit is required before merge, and it must name the commit.**
  The review states the SHA it read; a review of an older head does not count, because a head that moved is a
  different tree. `tensor4all-meta` holds the reference implementation
  ([`.github/workflows/review-gate.yml`](../../.github/workflows/review-gate.yml)). Whether a repository makes it a
  required check is a branch-protection setting that no file records, so this page states the rule and
  [`scripts/check-contribution.py`](../../scripts/check-contribution.py) reports only the part that is in the tree:
  which repositories carry the pointer to this page.
- **Whoever wrote the change does not merge it alone.** The author may merge after the review, but a change merged
  before its review lands is a change nobody reviewed — that has happened, and it is why the gate is a status check
  and not a habit.
- **A review that keeps asking for more is a review to stop.** Past the fourth round, a finding is classified as
  *blocking* (a wrong number, a silent stop, a failure recorded as success) or *polish*. Polish does not block, and a
  change that keeps growing should be shrunk rather than extended — the smaller version is the one that lands.

## 7. Compatibility and versions

- Public APIs are allowed to change while the crates are pre-1.0, and obsolete paths are **removed** rather than kept
  behind compatibility shims. A shim that only exists to avoid a version bump is a cost with no reader.
- A change to a wire format, a C ABI revision, or a serialized layout is a version bump of its own: the number has to
  tell a pin that the contract moved. Bumping a patch version because the code changed is not enough.
- Releases are batched. Several fixes ship as one version, tagged `<crate>-v<version>`; a release every day is a
  release nobody reads.
- Nothing is published from a workspace that has not been restored to `main` and rebuilt.

## 8. Provenance and automation

- **State when a contribution is machine-written.** A pull request, comment or issue produced with an agent names the
  model and says whether a human asked for it. A reader cannot otherwise tell an answer somebody requested from one a
  scheduled job emitted.
- Do not post AI-generated content to repositories that forbid it. Upstream projects that are not ours have their own
  rules; a fork carries the fix and the reproducer, not a bot-authored issue.
- Secrets never enter a commit, a log, or a pull request body. A key that was pasted into a log is a key to rotate.

## 9. What this page does not decide

Per repository, and therefore in that repository's `CONTRIBUTING.md`: the MSRV, the feature combinations CI builds, the
platforms claimed, the benchmark protocol, and any crate-specific numerical contract. Keep those local — the point of
this page is that a contributor can read one file and know what is the same everywhere.
