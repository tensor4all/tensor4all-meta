#!/usr/bin/env python3
"""Small helpers for run.sh (Python 3 standard library, plus PyYAML for suite files).

run.sh stays a thin shell over each repository's own runner; the few jobs that are awkward
in shell live here, each as one subcommand with a plain exit code:

    helpers.py jwrite FILE key=value key:=<json> ...   write a JSON object (strings, or raw JSON with :=)
    helpers.py norm CPULIST                            canonical form of a CPU list, e.g. "3,0-2" -> "0-3"
    helpers.py cpusets ROOT SUITE...                   union of the cpu_sets a repository's suites declare
    helpers.py tree [--exclude PID] PID...             the PIDs of these processes and all their descendants (own user)
    helpers.py verify CPULIST PID...                   every thread of every PID has Cpus_allowed_list == CPULIST
    helpers.py executor-pid EXECUTOR                   the nearest ancestor of this process that is the agent
    helpers.py agents-busy SECONDS EXCLUDE_FILE        agent processes using CPU in a window (exit 1 if any)
    helpers.py toolchain STACK_FILE                    compare `rustc -vV` with the stack's declared toolchain
    helpers.py pins-behind STACK_FILE OUT.json         stack pins against crates.io and GitHub (report only)
    helpers.py preflight-json TSV OUT.json             the preflight check list as JSON

Nothing here changes affinity or touches a repository; run.sh does that with taskset and git.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PROFILE = os.environ.get("WEEKLY_PROFILE", "zen5-cpu")

AGENT_COMMS = ("claude", "pi", "codex", "gemini", "opencode", "aider", "herdr")


def parse_cpus(text):
    out = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def fmt_cpus(cpus):
    cpus = sorted(cpus)
    runs = []
    for c in cpus:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def cmd_jwrite(argv):
    path, items = argv[0], argv[1:]
    obj = {}
    for item in items:
        key, value = item.split("=", 1)
        if key.endswith(":"):  # key:=<json>
            obj[key[:-1]] = json.loads(value)
        else:
            obj[key] = value
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


def cmd_norm(argv):
    print(fmt_cpus(parse_cpus(argv[0])))
    return 0


def cmd_cpusets(argv):
    import yaml  # PyYAML is a dependency of campaign-toolkit and of every toolkit repository

    root, suites = argv[0], argv[1:]
    cpus = set()
    for suite in suites:
        path = os.path.join(root, "benchmarks", "suites", suite + ".yaml")
        data = yaml.safe_load(open(path))
        for run in data.get("runs", []):
            specs = run.get("cpu_sets", []) or []
            if isinstance(specs, dict):  # per profile since campaign-toolkit#50
                specs = specs.get(PROFILE, []) or []
            for spec in specs:
                cpus |= parse_cpus(str(spec))
    if not cpus:
        print(f"no cpu_sets declared by {suites} in {root}", file=sys.stderr)
        return 2
    print(fmt_cpus(cpus))
    return 0


def read_stat(pid):
    """(ppid, comm, uid) of a process, or None when it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            text = f.read()
        comm = text[text.index("(") + 1:text.rindex(")")]
        rest = text[text.rindex(")") + 2:].split()
        ppid = int(rest[1])
        uid = os.stat(f"/proc/{pid}").st_uid
        return ppid, comm, uid
    except (OSError, ValueError):
        return None


def all_procs():
    table = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = read_stat(int(name))
            if st:
                table[int(name)] = st
    return table


def descendants(table, roots):
    kids = {}
    for pid, (ppid, _, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    seen, queue = set(), [r for r in roots if r in table]
    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(kids.get(pid, []))
    return seen


def cmd_tree(argv):
    exclude = None
    if argv and argv[0] == "--exclude":
        exclude, argv = argv[1], argv[2:]
    table = all_procs()
    me = os.getuid()
    skip = descendants(table, [int(exclude)]) if exclude and exclude.isdigit() else set()
    pids = sorted(p for p in descendants(table, [int(a) for a in argv]) - skip if table[p][2] == me)
    print(" ".join(str(p) for p in pids))
    return 0


def cmd_verify(argv):
    want = parse_cpus(argv[0])
    bad = 0
    checked = 0
    for pid in argv[1:]:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue  # the process ended between listing and checking
        for tid in tids:
            try:
                with open(f"/proc/{pid}/task/{tid}/status") as f:
                    text = f.read()
            except OSError:
                continue
            m = re.search(r"^Cpus_allowed_list:\s*(\S+)", text, re.M)
            if not m:
                continue
            checked += 1
            have = parse_cpus(m.group(1))
            if have != want:
                comm = (read_stat(int(pid)) or (0, "?", 0))[1]
                print(f"MISMATCH pid {pid} thread {tid} ({comm}): Cpus_allowed_list {m.group(1)}, want {fmt_cpus(want)}")
                bad += 1
    print(f"verified {checked} threads of {len(argv) - 1} processes against {fmt_cpus(want)}: {'OK' if not bad else f'{bad} mismatches'}")
    return 1 if bad else 0


def cmd_executor_pid(argv):
    kind = argv[0]
    comms = {"claude-code": {"claude"}, "pi": {"pi", "node", "bun"}}.get(kind, set())
    table = all_procs()
    pid = os.getppid()
    while pid > 1 and pid in table:
        ppid, comm, _ = table[pid]
        if comm in comms:
            print(pid)
            return 0
        pid = ppid
    print(f"no ancestor named {sorted(comms)} found for executor {kind!r}; set WEEKLY_EXECUTOR_PID", file=sys.stderr)
    return 1


def cpu_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            text = f.read()
        rest = text[text.rindex(")") + 2:].split()
        return int(rest[11]) + int(rest[12])
    except (OSError, ValueError, IndexError):
        return None


def cmd_agents_busy(argv):
    seconds = float(argv[0])
    exclude = set()
    if len(argv) > 1 and os.path.exists(argv[1]):
        exclude = {int(x) for x in open(argv[1]).read().split() if x.isdigit()}
    table = all_procs()
    cands = [p for p, (_, comm, _) in table.items() if comm in AGENT_COMMS and p not in exclude]
    before = {p: cpu_ticks(p) for p in cands}
    time.sleep(seconds)
    hz = os.sysconf("SC_CLK_TCK")
    busy = []
    for p in cands:
        a, b = before[p], cpu_ticks(p)
        if a is None or b is None:
            continue
        share = (b - a) / hz / seconds
        if share >= 0.05:
            busy.append((p, table[p][1], share))
    for p, comm, share in busy:
        print(f"pid {p} ({comm}) used {100 * share:.1f}% of a core")
    print(f"{len(cands)} other agent processes sampled for {seconds:g}s, {len(busy)} busy")
    return 1 if busy else 0


def load_stack(path):
    import yaml

    return yaml.safe_load(open(path))


def cmd_toolchain(argv):
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True).stdout
    m = re.search(r"^release:\s*(\S+)", out, re.M)
    have = m.group(1) if m else "unknown"
    try:
        stack = load_stack(argv[0])
    except Exception as exc:  # noqa: BLE001 - report, do not crash the preflight
        print(f"WARN cannot read the stack file ({exc}); rustc is {have}")
        return 0
    declared = stack.get("toolchain")
    if declared is None:
        print(f"WARN the stack declares no toolchain (stack/stack-pins.yaml has no `toolchain` key); rustc is {have}, recorded only")
        return 0
    want = declared.get("rustc") or declared.get("version") or declared.get("channel") if isinstance(declared, dict) else str(declared)
    if want and str(want).lstrip("=") in (have, have.split("-")[0]):
        print(f"PASS rustc {have} matches the stack's declared toolchain {want}")
        return 0
    print(f"FAIL rustc {have} differs from the stack's declared toolchain {want}")
    return 1


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tensor4all-weekly-cpu-benchmark"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def gh_api(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout) if r.returncode == 0 else None


def cmd_pins_behind(argv):
    stack = load_stack(argv[0])
    rows = []
    for name, lib in stack.get("libraries", {}).items():
        row = {"library": name, "pin_version": lib.get("version"), "pin_tag": lib.get("tag"),
               "pin_commit": (lib.get("commit") or "")[:12], "latest_crates": None, "latest_github": None,
               "behind": False, "note": ""}
        crate = lib.get("crate")
        if crate:
            try:
                row["latest_crates"] = http_json(f"https://crates.io/api/v1/crates/{crate}")["crate"]["max_stable_version"]
            except Exception as exc:  # noqa: BLE001
                row["note"] += f"crates.io unreachable ({type(exc).__name__}); "
        m = re.match(r"https://github.com/([^/]+/[^/]+?)(?:\.git)?$", lib.get("repo") or "")
        if m:
            rel = gh_api(f"repos/{m.group(1)}/releases/latest")
            if rel and rel.get("tag_name"):
                row["latest_github"] = rel["tag_name"]
        pin = lib.get("version")
        if pin and row["latest_crates"] and row["latest_crates"] != pin:
            row["behind"] = True
        rows.append(row)
    with open(argv[1], "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    behind = [r for r in rows if r["behind"]]
    print(f"{len(rows)} stack libraries checked, {len(behind)} behind the latest crates.io release (report only)")
    for r in behind:
        print(f"  {r['library']}: pin {r['pin_version']}, latest {r['latest_crates']}")
    return 0


def cmd_preflight_json(argv):
    checks = []
    for line in open(argv[0]):
        line = line.rstrip("\n")
        if not line:
            continue
        name, status, detail = (line.split("\t", 2) + ["", ""])[:3]
        checks.append({"name": name, "status": status, "detail": detail})
    failed = [c["name"] for c in checks if c["status"] == "FAIL"]
    with open(argv[1], "w") as f:
        json.dump({"checks": checks, "ok": not failed, "failed": failed}, f, indent=2)
        f.write("\n")
    return 0


COMMANDS = {
    "jwrite": cmd_jwrite, "norm": cmd_norm, "cpusets": cmd_cpusets, "tree": cmd_tree, "verify": cmd_verify,
    "executor-pid": cmd_executor_pid, "agents-busy": cmd_agents_busy, "toolchain": cmd_toolchain,
    "pins-behind": cmd_pins_behind, "preflight-json": cmd_preflight_json,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2:]))
