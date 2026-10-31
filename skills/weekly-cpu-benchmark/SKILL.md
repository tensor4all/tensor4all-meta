---
name: weekly-cpu-benchmark
description: Run the weekly full CPU benchmark of the tensor4all stack on the zen5-cpu measurement host, when the monitoring session instructs it. Plain shell procedure (preflight, pins, measure repository by repository, one raw-run pull request per repository, summary) that a pi agent or Claude Code executes identically; every judgement is closed as stop-and-report.
---

# Weekly CPU benchmark (zen5-cpu)

You are the **executor**. The monitoring session started you; it decides go/no-go, merges, version
bumps and what the maintainer is told. You run the commands below in order, read each exit
code, and report. You do not decide, repair, retry or improvise. Where a step says **STOP**,
run nothing further except step 6 (summary), then reply with the one-line reason.

Design and background: tensor4all-meta#76 (Phase 7), `docs/benchmarks/README.md`.

## Ground rules

1. Run on the observer CPUs. `run.sh` moves your process tree there before every measurement
   and verifies it; do not move processes yourself, do not run anything else on the host while a
   measurement runs, do not start other agents or builds.
2. **Wait, do not poll.** Run each long command in the foreground and wait for it in one tool
   call, with the timeout given below. Never read a log or the work directory to see how far it
   got. If your tool cannot hold a call as long as the timeout (Claude Code caps a foreground call at
   10 minutes), use the detached form in "Executors with a tool-call cap" instead.
3. Every command is `bash skills/weekly-cpu-benchmark/run.sh ...`, run from the root of the
   `tensor4all-meta` checkout. Alias it: `RUN="bash skills/weekly-cpu-benchmark/run.sh"`.
4. Exit codes of `run.sh`: 0 ok, 1 a check or a unit failed (cause recorded), 2 refused (usage,
   executor record, lock, bad stage), 3 no checkout. Any other code: STOP.
5. **Must not**: publish a crate; merge a pull request; bump or edit a pin; run anything while
   another measurement or agent is busy on the host (preflight refuses, a guard failure discards
   the run); change `cpufreq/boost` or the governor (you only read them; the operator sets them);
   commit generated artefacts (pages, INDEX, HISTORY, summary files, figures); run
   `cargo build` or a benchmark yourself outside `run.sh`; edit a repository by hand; force-push;
   retry a failed unit; make any decision the monitoring session did not give you.

## Step 0. Declare who you are

The executor record is mandatory; `run.sh` refuses to start without it. Set it once per shell
(every tool call starts a fresh shell on some executors, so put it in front of each command or
`export` it in a file you `source`):

```sh
export WEEKLY_EXECUTOR=pi                      # or claude-code
export WEEKLY_EXECUTOR_VERSION=<agent version> # e.g. output of `pi --version` / `claude --version`
export WEEKLY_EXECUTOR_MODEL=<model name>      # e.g. the model id the session runs on
export WEEKLY_EXECUTOR_PID=<pid of your agent process>   # optional: run.sh finds the nearest ancestor named pi/node/claude
```

If a variable is unknown: STOP, report "executor record incomplete".

The work directory is printed by every command (`~/.local/state/tensor4all-weekly/<UTC date>`);
`WEEKLY_WORK_DIR` overrides it. Its JSON files are the record `summary.py` reads.

## Step 1. Preflight (timeout 300000 ms)

```sh
$RUN preflight
```

Expected: one line per check (`PASS`, `WARN`, `INFO`, `FAIL`) and last line `PREFLIGHT OK`, exit 0.
It checks: `cpufreq/boost` is 0; governor `performance`; no other agent or load on the measured
CPUs and their SMT siblings (the idle gate, with your tree already moved to the observer set); disk
space; `rustc` against the stack's toolchain declaration (the stack declares none yet: that is a
`WARN`, recorded, not a failure); each repository clean, on `main`, equal to `origin/main`, no
measurement holding its lock; `gh auth status`. It writes `hostinfo.json` (copied into every run
directory) and `preflight.json`.

- Exit 1 or any `FAIL` line: **STOP**, reply "preflight failed" plus the `FAIL` lines verbatim.
  Measure nothing. (Exception: `repo[tensor4all]` `FAIL` for a missing checkout stops only that
  repository, see step 3; report it, and continue only if every other check passed.)
- `WARN` lines: continue, and they appear in the summary.

The 14 h wall-clock cap starts now (recorded in `meta.json`). A unit that would not finish inside it
is recorded as `deferred` and left for next week.

## Step 2. Pins (timeout 600000 ms)

```sh
$RUN pins
```

Expected: one line per repository, then the pins-behind report, then `PINS OK ...`, exit 0. For each
repository it reads the toolkit revision from `requirements.txt` and runs `campaign-sync-pins --check`
with the stack file of that revision; it also compares the toolkit installed in the recorder's Python
with that pin, and reports stack pins behind the latest crates.io release (report only).

- `status ok`: continue. `unchecked`: the toolkit revision predates the stack file; continue, it is
  reported. `FAIL` (a pin differs from the stack, or the installed toolkit is not the pinned one):
  exit 1, **do not measure that repository** (`run.sh` refuses it by itself); measure the others;
  the failure goes into the summary. Never edit a pin and never run `campaign-sync-pins` without `--check`.
- Pins behind the latest release are never bumped.

## Step 3. Measure (repository by repository)

Print the plan and the timeouts once: `$RUN plan`.

Run the stages in this order, one stage per tool call, each in the foreground:

| order | command | stage content | timeout (ms) |
|---|---|---|---|
| 1 | `$RUN run narrow` | narrow core (screening of the D4 representative cases) | 1800000 |
| 2 | `$RUN run cpueinsum` | einsum-opt_size, einsum-opt_flops | 1200000 |
| 3 | `$RUN run tcontract` | tcbench, per-shape | 3000000 |
| 4 | `$RUN run tlinalg` | competition (mkl, openblas), scaling | 9000000 |
| 5 | `$RUN run tensor4all` | build, 1T and 4T rows | 4500000 |
| 6 | `$RUN run tenferro` | build, then `run_all.sh 1 4 8` (all CPU suites) | 14400000 |
| 7 | `$RUN run faer` | faer-kernels, mkl arm then openblas arm | 30000000 |

Time budget (estimates, to be replaced by the first real week's measurements): narrow 0.5 h, cpueinsum
0.2 h, tcontract 0.5 h, tlinalg 1.5 h, tensor4all 0.5 h, tenferro 2 h, faer 5.5 h, builds and guard waits
1 h: 11 to 12 h in all. Hard cap 14 h from preflight; what does not fit is deferred to next week by
`run.sh` itself, you do not shorten anything.

For each call:

- Exit 0: continue with the next stage. Exit 1: some unit failed; its log and one-line cause are
  recorded; **continue with the next stage**, do not retry, do not read the log. Exit 3: no checkout
  for this stage; continue. Exit 2 or any other code: STOP.
- A stage prints `[stage/unit] ok|failed|deferred|skipped ...` lines. That is all you read.
- After stage 1: if its line says `failed` for any unit, send the line to the monitoring session
  (`herdr` message or your reply channel) as "narrow-core anomaly" and continue; it may tell you to stop.
  `skipped` (no cases declared yet) is normal today.
- Executors with a tool-call cap: see the next section; the order and the rules do not change.

How each stage records what it measured (so you know what the PR will contain):

- Toolkit repositories (cpueinsum, tcontract, tlinalg, faer): `record_run ... --no-publish`
  (the flag the toolkit has today; `--raw-only` is item 7.1, pending). It builds first, then verifies,
  then measures each thread row; the run directory gets `run.yaml`, the CSVs, guard, build-info and the
  `report.md` that `report_run` writes inside it.
- tenferro-benchmark: its own `scripts/run_all.sh 1 4 8` with its own row pinning; `run.sh` answers the
  runner's observer handshake at every row. The runner also rewrites `result/zen5-cpu/**` (aggregates;
  item 7.4 pending): `run.sh` discards those after the run. The run directories are under
  `data/results/zen5-cpu/` (git-ignored in the repository until item 2.5: the PR step uses `git add -f`).
- tensor4all-benchmark: `scripts/run_all.sh <profile>` once per row (1T, 4T); same discard of
  `report.md` and `INDEX.md` (item 7.4 pending). If there is no checkout, the stage exits 3: report it.

### Executors with a tool-call cap

If your tool cannot wait as long as the timeout in the table, split and detach. A stage can be run
unit by unit (`$RUN plan` lists the units with timeouts), and a unit can be started detached and awaited
with bounded blocking calls that read no output:

```sh
$RUN bg run faer --unit mkl     # prints: started faer-mkl (pid N); returns at once
$RUN await faer-mkl 540         # blocks up to 540 s: prints "RUNNING faer-mkl" (exit 10) or "FINISHED faer-mkl exit=N"
```

Repeat `await` while it prints `RUNNING` (do nothing else in between). When it prints `FINISHED`, the
exit code `N` is the one of `run`; apply the table above. Run the units of a stage in the order `$RUN plan`
prints them, one at a time, and finish stage N before starting stage N+1.

## Step 4. Record (nothing to do by hand)

`run.sh` already wrote, into each repository, the raw run directories only, with two sidecar files
in each: `executor.json` (your executor record) and `hostinfo.json` (the preflight host record).
This is the stand-in for `threads.observer.executor`, which campaign-toolkit gets in item 7.6; the
sidecar becomes that field then. You generate nothing else: no pages, INDEX, HISTORY, summary, figure.

## Step 5. One pull request per repository (timeout 300000 ms each)

After every measurement stage has run (or been refused), for each repository that has run
directories, one call each:

```sh
$RUN pr cpueinsum    # then tcontract, tlinalg, tensor4all, tenferro, faer
```

Expected: `[stage] pull request: <url>`, exit 0. The branch is `weekly/<repo>-<UTC date>`, the
commit holds the run directories only, the body holds the preflight result, the pins, the durations
and the failures. Exit 1 with "no run directories recorded": nothing was measured there; no pull
request; continue. Exit 1 with "the tree has changes besides the run directories": **STOP** for that
repository (nothing was committed), include the message in the summary, continue with the others.
Never merge, never push to `main`, never edit the body by hand. CI checks the pull requests; the monitoring session reads them.

## Step 6. Summary back to the monitoring session

```sh
$RUN summary
```

Expected: a Markdown summary on stdout (executor, wall time, preflight, a table of every unit with
status, duration and one-line cause, failures, pull request URLs and site links, pins and pins behind
the latest release). Send that text, unchanged, to the monitoring session as your final reply. It is
made from the work directory by `summary.py`, not from your memory: do not rewrite it, do not add
conclusions, do not report a unit as successful unless the table says `ok`. If the call itself fails,
reply with the exit code and the last preflight line.

`summary.py --selftest` checks the summariser on a synthetic work directory (no measurement).

## Step 7. Publish (after the pull requests, and it is a separate run)

A merged raw run PR is not a publication. For faer, tlinalg, cpueinsum and tcontract the site draws
**publication runs only**: a run labelled `pub-<rev8>-<date>` at the current pin. Weekly runs are
recorded with `record_run ... --no-publish` (see Step 3), so they are stored, merged and never
rendered. That is the behaviour you will see as `Nothing recorded at the current pin on this
profile` while `pins/<lib>.rev` matches the run's recorded `library.commit` exactly. When you hit
that, the cause is the label and the flag, not the pin, and not the merge.

Never relabel a weekly run to make a page appear. It is the protocol being faked, and the numbers
then carry a claim (a publication with its guard set and timing policy) that nobody measured.

| repository | publication command | what one publication covers | cost |
| --- | --- | --- | --- |
| cpueinsum-benchmark | `scripts/v2/publish.sh` | `einsum-opt_size`, then `einsum-opt_flops` | about 10 min |
| tcontract-benchmark | `scripts/v2/publish.sh` (needs `TBLIS_ROOT`; it refuses a suite that measures the external provider without its identity) | `tcbench` (aa 3), then `per-shape` (aa 2), reps 5 | about 16 min |
| tlinalg-benchmark | `scripts/publish.sh` | competition on both vendor arms, then scaling, aa 4, reps 5 | about 1 h 15 m |
| faer-benchmark | `scripts/publish.sh` | faer-kernels for mkl (3 engines, aa 3) and openblas (2 engines, aa 6), 14 families plus the 1024 tier | **about 11 h both arms** |
| tenferro-benchmark | no publication run. `publish-results` renders tracked run records (`ingest-results` runs on push and on schedule) | the CPU and GPU pages from `data/results/**` | minutes, a workflow |

Costs are the weekly arms measured in `~/.local/state/tensor4all-weekly/<utc-date>/summary*.md`, so
they scale with the suite, not with the machine.

Order matters when the clock is short. The three cheap ones (cpueinsum, tcontract, tlinalg, about
1 h 40 m together) publish as soon as the machine is free. Hold faer until any known performance gap
is fixed, because publishing freezes numbers that are about to change (2026-10-31 decision: faer
waits for the MKL gap, see faer#168 and the MKL comparison in faer#160).

Only `zen5-cpu` exists as a profile here, and only the Zen 5 host can produce its pages. The 64 core
EPYC is `zen2-cpu`: a publication run there fills a different profile and leaves the zen5 pages
untouched.

Verify by reading the site, never by the pipeline: a green Pages workflow and a merged PR are not
evidence.

```bash
curl -sL http://tensor4all.org/faer-benchmark/ | grep -o 'updated [0-9:. \-]*Z[^<]*'
# tenferro spells it differently:
curl -sL http://tensor4all.org/tenferro-benchmark/ | grep -o 'updated 20[0-9:. \-]*Z[^<]*'
```

Quote that line with the run name in the report. If it did not move, say so and give the reason
(machine busy, publication not run, workflow skipped) rather than reporting success.

## Reference

- Order, rows and budget: tensor4all-meta#76, Phase 7. The cap is `WEEKLY_CAP_HOURS` (default 14).
- Pinning: the zen5-cpu rows are 1T on CPU 0, 4T on 0-3 (observer `4-11,16-23`), 8T on 4-11
  (observer `0-3,12-15`). The toolkit repositories' suites measure on CPUs 4, 4-7, 4-11 (one L3 domain),
  so their observer is `0-3,12-15`; `run.sh` derives each observer set from the measured set
  (`idle_cpus.py resolve`), it does not hard-code them.
- `bash skills/weekly-cpu-benchmark/run.sh --dry-run <subcommand>` prints every command (including each
  affinity move) and runs only read-only checks; `selftest-affinity` exercises the affinity functions on a
  throwaway process tree. Neither measures anything.
