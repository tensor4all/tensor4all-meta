# Benchmarks of the tensor4all stack

Where the benchmark campaigns are, what a weekly full CPU run is, who starts it and what it
leaves behind. This page is the human entry point; it links the per-repository procedures and does
not copy them. Design and decisions: [tensor4all-meta#76](https://github.com/tensor4all/tensor4all-meta/issues/76)
(Phase 7 is the weekly run).

## The campaigns

| Campaign | Measures | README | Suite definitions | Latest pages | History | Site | Raw runs |
|---|---|---|---|---|---|---|---|
| [faer-benchmark](https://github.com/tensor4all/faer-benchmark) | t4a-faer (fork) against upstream faer, MKL and OpenBLAS | [README](https://github.com/tensor4all/faer-benchmark#readme) | [`benchmarks/suites/`](https://github.com/tensor4all/faer-benchmark/tree/main/benchmarks/suites) | [`result/INDEX.md`](https://github.com/tensor4all/faer-benchmark/blob/main/result/INDEX.md) | [`result/history.html`](https://github.com/tensor4all/faer-benchmark/blob/main/result/history.html) | <http://tensor4all.org/faer-benchmark/> | `data/results/zen5-cpu/` |
| [tlinalg-benchmark](https://github.com/tensor4all/tlinalg-benchmark) | tlinalg batched kernels against vendor LAPACK | [README](https://github.com/tensor4all/tlinalg-benchmark#readme) | [`benchmarks/suites/`](https://github.com/tensor4all/tlinalg-benchmark/tree/main/benchmarks/suites) | [`result/INDEX.md`](https://github.com/tensor4all/tlinalg-benchmark/blob/main/result/INDEX.md) | [`result/history.html`](https://github.com/tensor4all/tlinalg-benchmark/blob/main/result/history.html) | <http://tensor4all.org/tlinalg-benchmark/> | `data/results/zen5-cpu/` |
| [tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark) | tcontract tensor contraction against TBLIS (tcbench, per-shape) | [README](https://github.com/tensor4all/tcontract-benchmark#readme) | [`benchmarks/suites/`](https://github.com/tensor4all/tcontract-benchmark/tree/main/benchmarks/suites) | [`result-v2/tcbench/INDEX.md`](https://github.com/tensor4all/tcontract-benchmark/blob/main/result-v2/tcbench/INDEX.md), [`per-shape`](https://github.com/tensor4all/tcontract-benchmark/blob/main/result-v2/per-shape/INDEX.md) | [`result-v2/history.html`](https://github.com/tensor4all/tcontract-benchmark/blob/main/result-v2/history.html) | <http://tensor4all.org/tcontract-benchmark/> | `data/v2-results/` |
| [cpueinsum-benchmark](https://github.com/tensor4all/cpueinsum-benchmark) | cpueinsum on the 14-instance einsum corpus (opt_size, opt_flops) | [README](https://github.com/tensor4all/cpueinsum-benchmark#readme) | [`benchmarks/suites/`](https://github.com/tensor4all/cpueinsum-benchmark/tree/main/benchmarks/suites) | [`result-v2/cpueinsum/INDEX.md`](https://github.com/tensor4all/cpueinsum-benchmark/blob/main/result-v2/cpueinsum/INDEX.md) | [`result-v2/cpueinsum/history.html`](https://github.com/tensor4all/cpueinsum-benchmark/blob/main/result-v2/cpueinsum/history.html) | <http://tensor4all.org/cpueinsum-benchmark/> | `data/v2-results/` |
| [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) | tenferro CPU suites (einsum, ops, FFT, public API, small work, ...) and GPU | [README](https://github.com/tensor4all/tenferro-benchmark#readme) | [`benchmarks/cpu/`](https://github.com/tensor4all/tenferro-benchmark/tree/main/benchmarks/cpu) | [`result/zen5-cpu/cpu/`](https://github.com/tensor4all/tenferro-benchmark/tree/main/result/zen5-cpu/cpu) (one page per suite) | per-commit `history.html` on the site | <http://tensor4all.org/tenferro-benchmark/> | `data/results/zen5-cpu/` (git-ignored until plan item 2.5) |
| [tensor4all-benchmark](https://github.com/tensor4all/tensor4all-benchmark) | tensor4all-rs: three maintained cases, 1T and 4T | [README](https://github.com/tensor4all/tensor4all-benchmark#readme) | none; the cases are `src/` binaries driven by [`scripts/run_all.sh`](https://github.com/tensor4all/tensor4all-benchmark/blob/main/scripts/run_all.sh) | [`result/INDEX.md`](https://github.com/tensor4all/tensor4all-benchmark/blob/main/result/INDEX.md) | in `result/INDEX.md` | none yet (plan item 2.6) | `result/<profile>/<rev12>/raw/` |

Shared pieces, all in [campaign-toolkit](https://github.com/tensor4all/campaign-toolkit):

- **Stack pins**: [`stack/stack-pins.yaml`](https://github.com/tensor4all/campaign-toolkit/blob/main/stack/stack-pins.yaml)
  decides which revision of each library the campaigns measure. A repository pins the toolkit by revision in
  `requirements.txt`, so the toolkit revision also fixes the stack. `campaign-sync-pins --check` compares a
  repository's `pins/*.rev` with it; `campaign-stack-status` prints one row per repository (pins, newest run, CI).
- **Result pages**: `result/INDEX.md` is generated; `result/HISTORY.md` (older pages, newest first) is what
  the toolkit's index generator produces from revision `0f1949f` on and appears in a repository when it moves to
  that toolkit (plan Phase 2); until then the older pages are listed in `history.html`.
- **Profiles**: the measurement host is `zen5-cpu` (AMD Ryzen AI 9 HX 470, 12 cores / 24 threads, two L3 domains:
  CPUs 0-3 and 4-11); [`campaign_toolkit/profiles/zen5-cpu.yaml`](https://github.com/tensor4all/campaign-toolkit/blob/main/campaign_toolkit/profiles/zen5-cpu.yaml)
  holds its thread rows and observer sets, and the workspace `AGENTS.md` the pinning rules.

Per-campaign procedures (recording a run, vendor arms, the idle gate, what a run records) are in each
repository's `README.md` / `AGENTS.md` and in the shared `benchmark-campaign` agent skill; they are not repeated here.

## The weekly full CPU run

Every weekend the full set of CPU suites is measured on the `zen5-cpu` host, repository by repository.
The host is quiet and dedicated; the point is a regular, comparable data point per pin and a visible week-to-week
spread of the host itself.

**Who starts it.** The monitoring session owns it (decision D13): its recurring job decides go/no-go and
instructs the benchmark agent on the host. There is no timer on the host. If the monitoring session is not
running, no weekly run happens; its inspection list checks that a summary arrived.

**Go / no-go** (the monitoring session, before instructing; any "no" means skip this week and say why):

1. the host is quiet: no other agent works on it, and the host's Claude Code is idle or stopped unless it is
   the executor (`herdr --machine evo_vpn agent list`; `ps` for a busy `claude`);
2. the previous weekly run has finished (its summary arrived; no `run.lock` in the weekly state directory);
3. `cpufreq/boost` is 0 and the governor is `performance` (read only; the operator sets them);
4. the benchmark repositories are clean on `main` (the skill's preflight checks this again, authoritatively).

**Who runs it.** The executor on the host is a **pi** agent on a cheap model; **Claude Code** is the fallback
(it idles at a visibly higher CPU load, so it is not the default). The monitoring session picks the executor at
instruction time and switches to the fallback if pi or its model provider is down, the agent does not start or
tool errors repeat; the executor never decides this. Both follow the same plain procedure,
[`skills/weekly-cpu-benchmark/SKILL.md`](../../skills/weekly-cpu-benchmark/SKILL.md): preflight, pins, measure in a fixed
order, record raw run directories, one pull request per repository, summary. The shell behind it is
[`run.sh`](../../skills/weekly-cpu-benchmark/run.sh) (a thin layer over each repository's own runner; it moves the
executor's process tree to the row's observer CPUs before each row and verifies it) and
[`summary.py`](../../skills/weekly-cpu-benchmark/summary.py) (the summary is made from recorded JSON, not from the agent's
words). `run.sh --dry-run <subcommand>` prints every command and runs nothing that measures.

**Order and budget.** Narrow core, cpueinsum, tcontract, tlinalg, tensor4all-benchmark, tenferro, faer; about 11-12 h
estimated, cut at 14 h with the rest left for next week (D15). `run.sh plan` prints the table with timeouts. The
estimates are provisional until the first real week.

**Where the output goes.** One pull request per repository, branch `weekly/<repo>-<UTC date>`, containing run
directories only (each with an `executor.json` and the preflight `hostinfo.json` beside `run.yaml`), body with the
preflight result, pins, durations and failures. The executor never merges. The monitoring session reviews, and merges
when CI is green by the usual criteria (D14). CI and the Pages workflow generate the pages, INDEX, HISTORY, figures
and trend from the raw runs. The site shows the latest week per pin; earlier weeks go to the history and the trend
(D16). The maintainer is told by the monitoring session from the summary the executor returns.

**What is not done yet** (so what the first runs will look like):

| Item | State | Effect today |
|---|---|---|
| 7.1 `record_run --raw-only`, `weekly-<date>` label | pending | `run.sh` calls the toolkit recorder with `--no-publish`, which still writes `report.md` inside the run directory; the label `weekly-<UTC date>-<unit>` is passed with `--label` |
| 7.4 raw-only mode of the tenferro and tensor4all runners | pending | the runners also write aggregates; `run.sh` discards them (`git checkout`/`clean` of `result/`), so the pull request has run directories only. tensor4all-benchmark's `report.py --check` CI will flag the stale report until 7.4 |
| 7.6 `threads.observer.executor` in the schema | pending | the executor (`pi` or `claude-code`, version, model) is written as a sidecar `executor.json` in each run directory; it becomes that field when 7.6 lands |
| tenferro raw data tracking (2.5) | pending | `data/results/` is git-ignored; the pull request uses `git add -f` on the new run directories |
| D4 narrow-core cases | not declared in the campaigns | `run narrow` records "skipped" until cases are declared |

## Recreating the monitoring cron

The weekly schedule lives in the monitoring session, so it has to be recreated when that session restarts.

**Schedule.** `0 0 * * 6` in UTC: Saturday 00:00 UTC (09:00 JST). The run takes 11-12 h (cap 14 h), so it ends
on Saturday afternoon JST at the latest 23:00 JST; Sunday stays free as slack to re-run a repository that failed or
was deferred and to review the pull requests; nobody works on the host on a weekend morning. If the scheduler uses
local time (check `date +%Z` where the monitoring session runs), write the same instant in that zone: `0 9 * * 6` in JST.

**What the monitoring session does when it fires.** Go/no-go as above; pick the executor; then send the instruction
below to the host agent; then watch `herdr --machine evo_vpn agent get <name>`; when the summary arrives, review the pull
requests and tell the maintainer; if the executor died or went wrong, do not restart a half-finished run: next preflight
decides (the work directory is per UTC date).

**Agent naming.** Agents are named `<host>-<role>[-<detail>]` (e.g. `evo-tenferro-bench`, `evo-faer-autopilot-dsv4f`).
The weekly agents are `evo-weekly-bench-pi` (primary) and `evo-weekly-bench-claude` (fallback); the machine label
for `herdr --machine` is `evo_vpn` (always the `_vpn` labels). Exactly one of them is started for a given week.

**Instruction text** (substitute `<EXECUTOR>`, `<VERSION>`, `<MODEL>`; send with
`herdr --machine evo_vpn agent prompt evo-weekly-bench-pi "<text>"`):

```text
Weekly CPU benchmark, <UTC date>. Read and follow skills/weekly-cpu-benchmark/SKILL.md in the tensor4all-meta checkout
(git pull --ff-only there first), exactly, step by step: do not improvise, retry, merge, bump pins or change boost or the
governor. Set WEEKLY_EXECUTOR=<EXECUTOR> WEEKLY_EXECUTOR_VERSION=<VERSION> WEEKLY_EXECUTOR_MODEL=<MODEL> as the skill
says. Where the skill says STOP, stop and reply with the one-line reason. Your final reply is the output of
`run.sh summary`, unchanged.
```

**Where it is recorded.** This section is the source; the monitoring session's state file (`state/tensor4all/monitoring.md` in the advisor
repository) carries a link to it.

## Running or checking it by hand

From a `tensor4all-meta` checkout beside the benchmark repositories, with the executor record set (see the skill):

```sh
bash skills/weekly-cpu-benchmark/run.sh --dry-run preflight   # read-only checks, no affinity moves
bash skills/weekly-cpu-benchmark/run.sh --dry-run run faer    # prints every command, builds and measures nothing
bash skills/weekly-cpu-benchmark/run.sh selftest-affinity     # affinity moves on a throwaway process tree
python3 skills/weekly-cpu-benchmark/summary.py --selftest
```

A dry run does not build and does not time anything, so it does not prove that a runner builds; the first real week
(or a separate `--no-publish` smoke on the quiet host) does.
