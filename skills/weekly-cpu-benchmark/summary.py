#!/usr/bin/env python3
"""The weekly summary for the monitoring session, made from the work directory, not from prose.

    summary.py WORK_DIR                 Markdown summary on stdout
    summary.py WORK_DIR --pr-body STAGE the part of a pull request body that comes from the work directory
    summary.py --selftest               build a synthetic work directory and check the output

WORK_DIR is what run.sh wrote: meta.json, executor.json, preflight.json, pins.json,
pins-behind.json, units/<stage>__<unit>.json, pr/<stage>.json. A unit is one tool call's worth
of measurement (a recorder run, a build, a runner invocation). Python 3 standard library only.
"""
import glob
import json
import os
import sys
import tempfile
import time

SITES = {
    "faer": "http://tensor4all.org/faer-benchmark/",
    "tlinalg": "http://tensor4all.org/tlinalg-benchmark/",
    "tcontract": "http://tensor4all.org/tcontract-benchmark/",
    "cpueinsum": "http://tensor4all.org/cpueinsum-benchmark/",
    "tenferro": "http://tensor4all.org/tenferro-benchmark/",
    "tensor4all": None,  # no Pages site yet (plan item 2.6)
}
ORDER = ["narrow", "cpueinsum", "tcontract", "tlinalg", "tensor4all", "tenferro", "faer"]


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def units(work):
    out = [load(p) for p in sorted(glob.glob(os.path.join(work, "units", "*.json")))]
    out = [u for u in out if u]
    out.sort(key=lambda u: (ORDER.index(u["stage"]) if u["stage"] in ORDER else 99, u.get("start_epoch", 0)))
    return out


def one_line(text, limit=160):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize(work):
    meta = load(os.path.join(work, "meta.json"), {})
    exe = load(os.path.join(work, "executor.json"), {})
    pre = load(os.path.join(work, "preflight.json"))
    pins = load(os.path.join(work, "pins.json"), [])
    behind = [r for r in load(os.path.join(work, "pins-behind.json"), []) if r.get("behind")]
    us = units(work)
    prs = [load(p) for p in sorted(glob.glob(os.path.join(work, "pr", "*.json")))]
    prs = [p for p in prs if p]
    counts = {}
    for u in us:
        counts[u["status"]] = counts.get(u["status"], 0) + 1
    dry = bool(meta.get("dry_run")) or any(u.get("dry_run") for u in us)
    L = []
    L.append(f"# Weekly CPU benchmark {meta.get('date', '?')}{' (DRY RUN: nothing was measured)' if dry else ''}")
    L.append("")
    L.append(f"- Executor: {exe.get('executor', '?')} {exe.get('version', '')}, model {exe.get('model', '?')}"
             "  (sidecar `executor.json` in each run directory; `threads.observer.executor` after toolkit item 7.6)")
    if units(work):
        first = min(u.get("start_epoch", 0) for u in us)
        last = max(u.get("end_epoch", 0) for u in us)
        measured = sum(u.get("duration_s", 0) for u in us)
        L.append(f"- Wall time {hms(last - first)} from the first to the last unit (units add up to {hms(measured)}); "
                 f"cap {meta.get('cap_hours', '?')} h")
    L.append("- Units: " + (", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "none run"))
    if pre is None:
        L.append("- Preflight: no preflight.json (not run)")
    else:
        bad = [c for c in pre["checks"] if c["status"] == "FAIL"]
        warn = [c for c in pre["checks"] if c["status"] == "WARN"]
        L.append(f"- Preflight: {'OK' if pre['ok'] else 'FAILED: ' + ', '.join(c['name'] for c in bad)}"
                 f" ({len(pre['checks'])} checks, {len(warn)} warnings)")
        for c in bad + warn:
            L.append(f"  - {c['status']} {c['name']}: {one_line(c['detail'])}")
    L.append("")
    L.append("## Results by unit")
    L.append("")
    L.append("| stage | unit | status | duration | run directories | cause |")
    L.append("|---|---|---|---|---|---|")
    for u in us:
        rd = len(u.get("rundirs") or [])
        L.append(f"| {u['stage']} | {u['unit']} | {u['status']} | {hms(u.get('duration_s', 0))} | {rd or ''} | {one_line(u.get('cause', ''), 120)} |")
    if not us:
        L.append("| - | - | no units recorded | | | |")
    L.append("")
    failed = [u for u in us if u["status"] == "failed"]
    L.append("## Failures")
    L.append("")
    if failed:
        for u in failed:
            L.append(f"- {u['stage']}/{u['unit']}: {one_line(u.get('cause', ''))} (log: `{u.get('log', '')}`)")
    else:
        L.append("None.")
    left = [u for u in us if u["status"] in ("deferred", "skipped")]
    if left:
        L.append("")
        L.append("## Not run (left for next week or not declared)")
        L.append("")
        for u in left:
            L.append(f"- {u['stage']}/{u['unit']}: {u['status']}: {one_line(u.get('cause', ''))}")
    L.append("")
    L.append("## Pull requests")
    L.append("")
    if prs:
        for p in prs:
            site = SITES.get(p["stage"])
            L.append(f"- {p['stage']}: {p['url']}" + (f" ; site {site}" if site else ""))
    else:
        L.append("None opened.")
    L.append("")
    L.append("## Pins")
    L.append("")
    for p in pins:
        L.append(f"- {p['repo']}: {p['status']} (toolkit {p.get('toolkit_rev', '')[:12] or 'none'}) {one_line(p.get('detail', ''), 200)}")
    if not pins:
        L.append("- pins step not run")
    L.append("")
    L.append("Pins behind the latest crates.io release (report only, nothing was bumped):")
    if behind:
        for r in behind:
            L.append(f"- {r['library']}: pin {r['pin_version']}, latest {r['latest_crates']}")
    else:
        L.append("- none, or the report was not made")
    L.append("")
    return "\n".join(L)


def pr_body(work, stage):
    meta = load(os.path.join(work, "meta.json"), {})
    pre = load(os.path.join(work, "preflight.json"))
    pins = [p for p in load(os.path.join(work, "pins.json"), []) if p["repo"] == stage]
    behind = [r for r in load(os.path.join(work, "pins-behind.json"), []) if r.get("behind")]
    us = [u for u in units(work) if u["stage"] == stage]
    L = ["## Preflight", ""]
    if pre:
        L.append(f"{'OK' if pre['ok'] else 'FAILED'}: " + "; ".join(f"{c['name']} {c['status']}" for c in pre["checks"]))
    else:
        L.append("not recorded")
    L += ["", "## Pins", ""]
    for p in pins:
        L.append(f"- toolkit pin {p.get('toolkit_rev', '')[:12] or 'none'}: {p['status']}. {one_line(p.get('detail', ''), 300)}")
    for r in behind:
        L.append(f"- behind latest release: {r['library']} pin {r['pin_version']}, latest {r['latest_crates']}")
    L += ["", "## Duration and units", ""]
    for u in us:
        L.append(f"- {u['unit']}: {u['status']}, {hms(u.get('duration_s', 0))}{', ' + one_line(u['cause']) if u.get('cause') else ''}")
    failed = [u for u in us if u["status"] == "failed"]
    L += ["", "## Failures", ""]
    L += [f"- {u['unit']}: {one_line(u.get('cause', ''))}" for u in failed] or ["None."]
    L += ["", "The pull request carries run directories only. Pages, INDEX, HISTORY, figures and trend are generated by CI and Pages from them; "
          "until campaign-toolkit item 7.1 (`record_run --raw-only`) lands the recorder is called with `--no-publish`, and the "
          "`report.md` that `report_run` writes inside a run directory is part of it.",
          f"Started {meta.get('start', '?')}."]
    return "\n".join(L)


def selftest():
    with tempfile.TemporaryDirectory() as work:
        os.makedirs(os.path.join(work, "units"))
        os.makedirs(os.path.join(work, "pr"))

        def w(name, obj):
            with open(os.path.join(work, name), "w") as f:
                json.dump(obj, f)

        t = int(time.time())
        w("meta.json", {"date": "20261031", "cap_hours": 14, "start_epoch": t, "dry_run": False})
        w("executor.json", {"executor": "pi", "version": "1.2.3", "model": "cheap-model"})
        w("preflight.json", {"ok": True, "failed": [], "checks": [{"name": "boost", "status": "PASS", "detail": "0"},
                                                                  {"name": "toolchain", "status": "WARN", "detail": "stack declares none"}]})
        w("pins.json", [{"repo": "faer", "status": "ok", "toolkit_rev": "a" * 40, "detail": "ok pins/t4a-faer.rev"}])
        w("pins-behind.json", [{"library": "tlinalg", "pin_version": "0.6.0", "latest_crates": "0.7.1", "behind": True},
                               {"library": "faer", "pin_version": "0.25.0", "latest_crates": "0.25.0", "behind": False}])
        w("units/faer__mkl.json", {"stage": "faer", "unit": "mkl", "status": "ok", "cause": "", "start_epoch": t,
                                   "end_epoch": t + 9000, "duration_s": 9000, "rundirs": ["data/results/x/1"], "log": "l1"})
        w("units/tlinalg__scaling.json", {"stage": "tlinalg", "unit": "scaling", "status": "failed",
                                          "cause": "ERROR: guard: cpus 4-11 busy", "start_epoch": t, "end_epoch": t + 60,
                                          "duration_s": 60, "rundirs": [], "log": "l2"})
        w("units/narrow__narrow-core.json", {"stage": "narrow", "unit": "narrow-core", "status": "skipped",
                                             "cause": "not declared", "start_epoch": t, "end_epoch": t, "duration_s": 0,
                                             "rundirs": [], "log": ""})
        w("pr/faer.json", {"stage": "faer", "url": "https://github.com/tensor4all/faer-benchmark/pull/99"})
        text = summarize(work)
        body = pr_body(work, "faer")
    checks = [
        ("# Weekly CPU benchmark 20261031" in text, "title"),
        ("pi 1.2.3" in text, "executor"),
        ("| faer | mkl | ok | 2h30m | 1 |" in text, "ok unit row"),
        ("tlinalg/scaling: ERROR: guard: cpus 4-11 busy" in text, "failure with cause"),
        ("narrow/narrow-core: skipped" in text, "not-run list"),
        ("pull/99" in text and "http://tensor4all.org/faer-benchmark/" in text, "pr and site"),
        ("tlinalg: pin 0.6.0, latest 0.7.1" in text, "pins behind"),
        ("WARN toolchain" in text, "warning surfaced"),
        ("faer: pin 0.25.0" not in text, "up-to-date pin not listed as behind"),
        ("faer: ok" in text, "pins section"),
        ("## Failures" in body and "None." in body and "mkl: ok, 2h30m" in body, "pr body"),
    ]
    bad = [name for ok, name in checks if not ok]
    for ok, name in checks:
        print(("ok   " if ok else "FAIL ") + name)
    print("SELFTEST " + ("OK" if not bad else "FAILED: " + ", ".join(bad)))
    return 1 if bad else 0


def main(argv):
    if argv and argv[0] == "--selftest":
        return selftest()
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    work = argv[0]
    if len(argv) >= 3 and argv[1] == "--pr-body":
        print(pr_body(work, argv[2]))
    else:
        print(summarize(work))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
