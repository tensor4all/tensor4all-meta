#!/usr/bin/env node
/**
 * Tests for scripts/review-gate/waiver.js  —  run with `node --test scripts/`.
 *
 * Fixtures are plain JSON (`fixtures/waiver/*.json`) so the decision can be
 * replayed from a recorded timeline/comment set without touching the network.
 *
 * The pre-fix decision is kept in this file as `legacyEvaluate`: it is the
 * logic that shipped in `.github/actions/verify-review-label/action.yml`
 * (strided-rs and tensor4all-rs) and inline in `review_bot.yml` (tenferro-rs).
 * It binds the approval to "a maintainer applied the label at some point", reads
 * `created_at` from `committed` timeline events (a field those events do not
 * carry), and therefore never observes that the head moved.
 */

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const FIXTURES = path.join(__dirname, "review-gate", "fixtures", "waiver");
const waiver = require("./review-gate/waiver.js");

function load(name) {
  return JSON.parse(fs.readFileSync(path.join(FIXTURES, name + ".json"), "utf8"));
}

/** The pre-fix decision: label presence + a `committed` event timestamp that
 *  ordinary `committed` events do not carry. */
function legacyEvaluate(fixture) {
  const events = fixture.timeline || [];
  const labelEvents = events.filter(
    (e) => e.event === "labeled" && e.label && e.label.name === "rules-review:waive"
  );
  if (labelEvents.length === 0) return { allowed: false, code: "no-label-event" };
  if (fixture.action === "synchronize") {
    return { allowed: false, code: "synchronize" };
  }
  const labelTime = Date.parse(labelEvents[labelEvents.length - 1].created_at);
  const headTimes = events
    .filter(
      (e) =>
        ["committed", "head_ref_force_pushed"].includes(e.event) && e.created_at
    )
    .map((e) => Date.parse(e.created_at))
    .filter((t) => Number.isFinite(t));
  const latest = headTimes.length ? Math.max(...headTimes) : null;
  if (latest !== null && labelTime <= latest) {
    return { allowed: false, code: "label-older-than-head" };
  }
  if (!fixture.labelPresent) return { allowed: false, code: "no-label" };
  return { allowed: true, code: "legacy-label-trusted" };
}

test("a waiver comment naming the current head is accepted", () => {
  const fixture = load("match");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(decision.code, "waiver-matches-head");
  assert.equal(decision.namedSha, fixture.headSha.toLowerCase());
});

test("a waiver naming an older head is rejected after a push", () => {
  const fixture = load("stale");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
  assert.match(decision.reason, /waive-review/);
  assert.equal(decision.namedSha, fixture.comments[0].body.match(/[0-9a-f]{40}/)[0]);
});

test("the pre-fix logic accepted that same stale waiver (pinned)", () => {
  const fixture = load("stale");

  assert.equal(legacyEvaluate(fixture).allowed, true);
  assert.equal(waiver.evaluate(fixture).allowed, false);
});

test("reopening the PR with the same head keeps the waiver valid", () => {
  const fixture = load("reopen-same-sha");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(fixture.action, "reopened");
});

test("a comment by someone below write access is rejected", () => {
  const fixture = load("non-maintainer");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "non-maintainer-waiver");
  assert.match(decision.reason, /CONTRIBUTOR/);
});

test("a write-permission comment is accepted even when the association is odd", () => {
  const fixture = load("permission-write");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, true, decision.reason);
});

test("the label alone grants nothing", () => {
  const fixture = load("label-only");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "no-waiver");
  assert.match(decision.reason, /label/);
});

test("an abbreviated sha is not a waiver", () => {
  const fixture = load("abbreviated");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "shorthand-waiver");
});

test("a missing or malformed head sha fails closed", () => {
  const fixture = load("match");

  for (const headSha of [undefined, null, "", "not-a-sha", fixture.headSha.slice(0, 12)]) {
    const decision = waiver.evaluate({ ...fixture, headSha });
    assert.equal(decision.allowed, false, `head ${headSha}`);
    assert.equal(decision.code, "invalid-head");
  }
});

test("the newest maintainer waiver wins over an older one", () => {
  const fixture = load("newest-wins");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
  assert.equal(decision.namedSha, fixture.comments[1].body.match(/[0-9a-f]{40}/)[0]);
});

test("the bare word does not grant a waiver", () => {
  const fixture = {
    headSha: "a".repeat(40),
    comments: [
      {
        id: 1,
        user: { login: "maint" },
        author_association: "MEMBER",
        created_at: "2026-01-01T00:00:00Z",
        body: "I would waive-review this, but let me look first",
      },
    ],
  };

  assert.equal(waiver.evaluate(fixture).code, "no-waiver");
});

test("a hex run that is not exactly 40 characters is rejected", () => {
  const fixture = {
    headSha: "a".repeat(40),
    comments: [
      {
        id: 1,
        user: { login: "maint" },
        author_association: "MEMBER",
        created_at: "2026-01-01T00:00:00Z",
        body: "waive-review " + "a".repeat(41),
      },
    ],
  };

  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "shorthand-waiver");
});

test("both `waive-review <sha>` and `waive-review: <sha>` are accepted", () => {
  const sha = "b".repeat(40);

  for (const body of [`waive-review ${sha}`, `waive-review: ${sha}`, `LGTM\n\nwaive-review ${sha}.`]) {
    const decision = waiver.evaluate({
      headSha: sha,
      comments: [
        {
          id: 1,
          user: { login: "maint" },
          author_association: "OWNER",
          created_at: "2026-01-01T00:00:00Z",
          body,
        },
      ],
    });
    assert.equal(decision.allowed, true, body);
  }
});

test("the github-script glue fails the job when the waiver is stale", async () => {
  const fixture = load("stale");
  const failures = [];
  const core = {
    setFailed: (message) => failures.push(message),
    warning: () => {},
    info: () => {},
    notice: () => {},
  };
  const github = {
    paginate: async () => fixture.comments,
    rest: {
      issues: { listComments: async () => ({ data: fixture.comments }) },
      pulls: { get: async () => ({ data: { head: { sha: fixture.headSha } } }) },
      repos: {
        getCollaboratorPermissionLevel: async ({ username }) => ({
          data: { role_name: username === "maint" ? "maintain" : "read" },
        }),
      },
    },
  };
  const context = {
    repo: { owner: "tensor4all", repo: "strided-rs" },
    payload: {
      action: fixture.action,
      pull_request: {
        number: 42,
        head: { sha: fixture.headSha },
        labels: [{ name: "rules-review:waive" }],
      },
    },
  };

  const decision = await waiver.run({ github, core, context, labelName: "rules-review:waive" });

  assert.equal(decision.allowed, false);
  assert.equal(failures.length, 1);
  assert.match(failures[0], /waive-review/);
});
