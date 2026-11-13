# For developers

This page is for contributors to tensor4all, whether human or AI agent. Most code here is written with AI coding
agents under human direction. Some of it comes from resident agents ("superagents") that patrol the repositories.
The workflow below lets all of them work on the same repositories without duplicating work or merging past a
review.

## Where the rules live

| What | Where |
|---|---|
| Rules for every Rust repository (issues, PR evidence, gates, review, versions) | [`rust-repositories.md`](rust-repositories.md) |
| Shared rules for coding agents, linked from every repository's `AGENTS.md` | [tensor4all-agent-rules](https://github.com/tensor4all/tensor4all-agent-rules) · [`rules/index.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md) |
| Rules for agents that dispatch, review, hold or merge | [`rules/common/superagent.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/superagent.md) |
| Who works on an issue (assignment is the lock) | [`rules/common/assignment.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/assignment.md) |
| AI disclosure, copyright and citation | [`rules/common/provenance.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/provenance.md) |
| What differs in one repository | that repository's `CONTRIBUTING.md` and `AGENTS.md` |

Point your agent at the repository's `AGENTS.md`. It links everything else.

## The workflow in one screen

1. **Start from an issue and claim it.** Assign yourself before you start. If you file an issue for a fix you
   already have, use `gh issue create --assignee @me`. An issue assigned to someone else is theirs.
2. **Open a PR that names its model.** The PR body carries a line-initial `Model: <provider>/<model>`, one line per
   model that wrote code, for example `Model: anthropic/claude-opus-5-5`. Use `Model: none/human` if no AI was used.
   A PR without this line is converted to draft automatically.
3. **Get a review from a different provider.** Before merge, a model from a different provider than the author's
   reviews the head commit. Two models from the same provider do not count. The reviewer names its model and the
   SHA it read, then adds `xreview:done`. A later push voids the review. If you have only one provider, add
   `xreview:requested`, and a maintainer's agent will review it.
4. **Merge only after `xreview:done`.** Enable auto-merge only after `xreview:done` is present.
5. **To stop a merge, convert the PR to draft.** Do this with `gh pr ready <n> --undo` and a one-line `hold:` reason.
   On GitHub, converting a PR to draft disables auto-merge, and a change request alone does not. Only whoever set the
   hold, or a maintainer, marks the PR ready again.

Before you merge, close, reopen or hand over a PR, re-read its current state. Do not act on a snapshot from earlier.

## Labels

These labels exist in every tensor4all repository.

| Label | Meaning | Who sets it |
|---|---|---|
| `ai-ready` | Scope is clear. Agents may pick it up and implement it. | Maintainers and their agents |
| `human-approve-required` | Needs an explicit human decision, such as public API, design, release or a new dependency. Agents do not dispatch or merge it, and do not remove the label. | Anyone who sees the need; only a human removes it |
| `xreview:requested` | Asks for a review by a different AI provider | The PR author |
| `xreview:done` | A different provider reviewed the current head | The reviewing agent |

Issues without `ai-ready` are not picked up by agents.

## Automation

[gh-action-bot](https://github.com/tensor4all/gh-action-bot) runs these as the `tensor4all-ai-bot` app:

- **ModelLinePatrol** (every 15 minutes, no LLM) converts new PRs without a `Model:` line to draft and explains why.
- **AgentRulesLinkAudit** (weekly, no LLM) opens an `ai-ready` issue in any Rust repository whose `AGENTS.md` does not
  link the shared rules.
- **AI audits** (bug hunts, closed-issue gap audits, performance and documentation audits) open issues in the main
  Rust repositories.

Every automated post states that it is automated, which workflow produced it, and which model was used, if any.

## Why these rules

Each rule comes from an observed failure, discussed in
[tensor4all-meta#93](https://github.com/tensor4all/tensor4all-meta/issues/93):

- One issue produced two implementations, because the issue was filed without being claimed.
- A PR merged 57 seconds after a change request, because nothing held auto-merge.
- A PR was closed as a duplicate of one that had already been closed, because the agent acted on stale state.

The rules aim to keep the overhead small: a label, a line in the PR body, and a draft toggle.
