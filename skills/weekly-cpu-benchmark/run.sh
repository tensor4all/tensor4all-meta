#!/usr/bin/env bash
# Weekly full CPU benchmark on the zen5-cpu measurement host: a thin shell over each benchmark
# repository's own runner. The procedure an agent follows is SKILL.md; this file is what its
# commands call. It decides nothing: it checks, moves the agent off the measured CPUs, runs the
# repository's own runner, records what happened as JSON in the weekly work directory, and
# stops with a plain exit code.
#
#   run.sh [--dry-run] preflight
#   run.sh [--dry-run] pins
#   run.sh [--dry-run] plan
#   run.sh [--dry-run] run <stage> [--unit NAME]     stage: narrow cpueinsum tcontract tlinalg tensor4all tenferro faer
#   run.sh [--dry-run] bg run <stage> [--unit NAME]  /  run.sh await <name> [seconds]   (executors with a tool-call cap)
#   run.sh [--dry-run] pr <stage>
#   run.sh [--dry-run] summary
#   run.sh selftest-affinity
#
# Exit codes: 0 ok; 1 a check or a unit failed (the cause is in the work directory); 2 refused
# (usage, missing executor record, lock held, deadline); 3 no checkout for the stage.
#
# --dry-run prints every command it would run, including each affinity move, and executes only
# cheap read-only checks (sysfs, /proc, git status, git ls-remote, gh). It starts no build, no
# runner, and moves no process.
#
# Environment (see docs/benchmarks/README.md):
#   WEEKLY_EXECUTOR, WEEKLY_EXECUTOR_VERSION, WEEKLY_EXECUTOR_MODEL   required: who is running this
#   WEEKLY_EXECUTOR_PID   the agent's PID (default: the nearest ancestor that is the agent)
#   WEEKLY_EXTRA_PIDS     more agent PIDs to keep off the measured CPUs (default: a running herdr)
#   WEEKLY_ROOT           directory holding the repository checkouts (default: parent of this repository)
#   WEEKLY_TOOLKIT        campaign-toolkit checkout (default: $WEEKLY_ROOT/campaign-toolkit)
#   WEEKLY_DATE           UTC date stamp YYYYMMDD (default: today)
#   WEEKLY_WORK_DIR       work directory (default: ${XDG_STATE_HOME:-~/.local/state}/tensor4all-weekly/$WEEKLY_DATE)
#   WEEKLY_CAP_HOURS      wall-clock cap from the start of preflight (default 14)
#   WEEKLY_MIN_FREE_GB    disk-space floor (default 100)
#   WEEKLY_PYTHON         python used for the recorders (default python3; use the venv that has the pinned toolkit)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
META=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || (cd "$HERE/../.." && pwd))
ROOT=${WEEKLY_ROOT:-$(dirname "$META")}
TOOLKIT=${WEEKLY_TOOLKIT:-$ROOT/campaign-toolkit}
DATE=${WEEKLY_DATE:-$(date -u +%Y%m%d)}
WORK=${WEEKLY_WORK_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/tensor4all-weekly/$DATE}
WORK_GIVEN=${WEEKLY_WORK_DIR:+1}
PY=${WEEKLY_PYTHON:-python3}
HELP="python3 $HERE/helpers.py"
# python for a repository's recorder: WEEKLY_PYTHON_<STAGE> (e.g. WEEKLY_PYTHON_FAER) overrides WEEKLY_PYTHON
py_for() { local v="WEEKLY_PYTHON_${1^^}"; echo "${!v:-$PY}"; }
PROFILE=zen5-cpu
CAP_HOURS=${WEEKLY_CAP_HOURS:-14}
MIN_FREE_GB=${WEEKLY_MIN_FREE_GB:-100}
IDLE_SCRIPT=$TOOLKIT/benchmarks/scripts/idle_cpus.py
DRY=0

declare -A DIR=([faer]=faer-benchmark [tlinalg]=tlinalg-benchmark [tcontract]=tcontract-benchmark
                [cpueinsum]=cpueinsum-benchmark [tensor4all]=tensor4all-benchmark [tenferro]=tenferro-benchmark)
# The measurement order of the plan (tensor4all-meta#76, Phase 7). `narrow` is the screening pass.
STAGES=(narrow cpueinsum tcontract tlinalg tensor4all tenferro faer)
TOOLKIT_REPOS=(cpueinsum tcontract tlinalg faer)

# RUSTC_WRAPPER (kache) breaks patched and pinned builds; keep it empty for build and run alike.
export RUSTC_WRAPPER=

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit "${2:-2}"; }
quote() { printf '%q ' "$@"; }
# plan: print the command in dry-run mode, run it otherwise. Returns the command's status.
plan() {
    if (( DRY )); then printf '[dry-run] + %s\n' "$(quote "$@")"; return 0; fi
    "$@"
}
now() { date +%s; }
iso() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }
jw() { $HELP jwrite "$@"; }

# ---------------------------------------------------------------------------------------------
# CPU isolation: the executor's process tree goes to the row's observer set before the row.
# ---------------------------------------------------------------------------------------------
ISO_ROOTS=()          # root PIDs of the tree to keep off the measured CPUs
ISO_EXCLUDE_FILE=""   # file holding the PID of the measured subtree (never moved)

iso_pids() {
    local excl=""
    [[ -n $ISO_EXCLUDE_FILE && -s $ISO_EXCLUDE_FILE ]] && excl=$(<"$ISO_EXCLUDE_FILE")
    # shellcheck disable=SC2086
    $HELP tree ${excl:+--exclude "$excl"} "${ISO_ROOTS[@]}"
}

# isolate <cpu-list> [label]: taskset -a -cp the tree, then verify Cpus_allowed_list of every
# thread of a fresh enumeration. A second pass catches a process that was started from a parent
# that had not been moved yet. Returns 1 when any thread still differs: the row must not start.
isolate() {
    local set=$1 label=${2:-row} pids p n
    pids=$(iso_pids)
    n=$(wc -w <<<"$pids")
    if (( DRY )); then
        say "[dry-run] isolate ($label): $n processes (roots: ${ISO_ROOTS[*]}) -> CPUs $set"
        for p in $(tr ' ' '\n' <<<"$pids" | head -3); do
            say "[dry-run] + taskset -a -cp $set $p"
        done
        (( n > 3 )) && say "[dry-run]   ... and $((n - 3)) more; then: verify Cpus_allowed_list of every thread == $set"
        return 0
    fi
    for pass in 1 2; do
        for p in $pids; do
            taskset -a -cp "$set" "$p" >/dev/null 2>&1 || { [[ -d /proc/$p ]] && warn "could not move $p to $set"; }
        done
        pids=$(iso_pids)
        # shellcheck disable=SC2086
        if $HELP verify "$set" $pids; then return 0; fi
    done
    say "isolation ($label) FAILED: the tree is not confined to CPUs $set" >&2
    return 1
}

# Roots of the tree: the executor agent and any extra agent PIDs.
setup_iso_roots() {
    local pid=${WEEKLY_EXECUTOR_PID:-}
    if [[ -z $pid ]]; then
        pid=$($HELP executor-pid "$WEEKLY_EXECUTOR") || die "cannot identify the executor process; set WEEKLY_EXECUTOR_PID"
    fi
    ISO_ROOTS=("$pid")
    local extra=${WEEKLY_EXTRA_PIDS-$(pgrep -u "$(id -u)" -x herdr 2>/dev/null | tr '\n' ' ')}
    for p in $extra; do ISO_ROOTS+=("$p"); done
}

# resolve <measured-cpu-list>: the observer set of a measured set, from the live topology
# (idle_cpus.py resolve); prints it. resolve_row <threads>: the profile's thread row.
resolve_set() { python3 "$IDLE_SCRIPT" resolve "$1" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["observer"])'; }
resolve_row() { python3 "$IDLE_SCRIPT" resolve --profile "$PROFILE" --threads "$1" --json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["measured"], d["observer"])'; }

# ---------------------------------------------------------------------------------------------
# Arguments, the executor record, the work directory
# ---------------------------------------------------------------------------------------------
require_executor() {
    [[ -n ${WEEKLY_EXECUTOR:-} && -n ${WEEKLY_EXECUTOR_VERSION:-} && -n ${WEEKLY_EXECUTOR_MODEL:-} ]] \
        || die "set WEEKLY_EXECUTOR (pi or claude-code), WEEKLY_EXECUTOR_VERSION and WEEKLY_EXECUTOR_MODEL; refusing to start without the executor record"
    case $WEEKLY_EXECUTOR in pi|claude-code) ;; *) die "WEEKLY_EXECUTOR must be pi or claude-code, got '$WEEKLY_EXECUTOR'";; esac
}

init_work() {
    mkdir -p "$WORK/units" "$WORK/logs" "$WORK/pr"
    # The executor record. campaign-toolkit has no `threads.observer.executor` field yet (plan
    # item 7.6), so it travels as a sidecar executor.json next to each run.yaml until it does.
    jw "$WORK/executor.json" executor="$WEEKLY_EXECUTOR" version="$WEEKLY_EXECUTOR_VERSION" \
        model="$WEEKLY_EXECUTOR_MODEL" date="$DATE" recorded_by="skills/weekly-cpu-benchmark/run.sh" \
        note="sidecar; becomes threads.observer.executor in run.yaml once campaign-toolkit item 7.6 lands"
    if [[ ! -f $WORK/meta.json ]]; then
        local t; t=$(now)
        jw "$WORK/meta.json" date="$DATE" executor="$WEEKLY_EXECUTOR" executor_version="$WEEKLY_EXECUTOR_VERSION" \
            model="$WEEKLY_EXECUTOR_MODEL" start_epoch:="$t" start="$(iso "$t")" \
            deadline_epoch:="$((t + CAP_HOURS * 3600))" cap_hours:="$CAP_HOURS" dry_run:="$( ((DRY)) && echo true || echo false)"
    fi
}
meta() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$WORK/meta.json" "$1"; }

repo_path() { echo "$ROOT/${DIR[$1]}"; }
have_checkout() { [[ -d $(repo_path "$1")/.git ]]; }

# ---------------------------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------------------------
CHECKS_TSV=""
chk() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$CHECKS_TSV"; printf '%-5s %-22s %s\n' "$2" "$1" "$3"; }

measured_sets() {
    # every distinct measured CPU set: the profile's rows plus the sets the toolkit suites declare
    local n r
    { for n in 1 4 8; do resolve_row "$n" | cut -d' ' -f1; done
      for r in "${TOOLKIT_REPOS[@]}"; do
          have_checkout "$r" || continue
          # shellcheck disable=SC2046
          $HELP cpusets "$(repo_path "$r")" $(suites_of "$r") 2>/dev/null
      done
    } | sort -u
}

cmd_preflight() {
    CHECKS_TSV=$WORK/preflight.tsv; : >"$CHECKS_TSV"
    local v f

    v=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unreadable)
    if [[ $v == 0 ]]; then chk boost PASS "cpufreq/boost = 0"; else chk boost FAIL "cpufreq/boost = $v (must be 0; the operator sets it, the agent never does)"; fi

    v=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')
    if [[ ${v% } == performance ]]; then chk governor PASS "performance on every CPU"; else chk governor FAIL "governors seen: ${v:-unreadable} (must be performance; the operator sets it)"; fi

    if command -v taskset >/dev/null && [[ $(uname -s) == Linux ]]; then chk taskset PASS "taskset available"; else chk taskset FAIL "taskset or Linux missing"; fi
    if [[ -f $IDLE_SCRIPT ]]; then chk toolkit PASS "idle_cpus.py at $IDLE_SCRIPT"; else chk toolkit FAIL "no $IDLE_SCRIPT (set WEEKLY_TOOLKIT)"; fi

    # executor tree
    local roots=""
    if setup_iso_roots 2>/dev/null; then
        roots=$(iso_pids)
        chk executor PASS "roots ${ISO_ROOTS[*]} (comm $(ps -o comm= -p "${ISO_ROOTS[0]}" 2>/dev/null)), tree of $(wc -w <<<"$roots") processes"
    else
        chk executor FAIL "cannot identify the executor process; set WEEKLY_EXECUTOR_PID"
    fi

    # other agents
    printf '%s\n' "$roots" >"$WORK/executor-tree.txt"
    v=$($HELP agents-busy 2 "$WORK/executor-tree.txt"); f=$?
    if (( f == 0 )); then chk other-agents PASS "$(tail -1 <<<"$v")"; else chk other-agents FAIL "$(tr '\n' ';' <<<"$v")"; fi

    # idle gate on every measured set (with SMT siblings), the executor off it
    local set obs
    if [[ -f $IDLE_SCRIPT ]]; then
        while read -r set; do
            [[ -n $set ]] || continue
            obs=$(resolve_set "$set")
            if (( DRY )); then
                v=$(python3 "$IDLE_SCRIPT" check "$set" --profile "$PROFILE" --dry-run | tr '\n' ';')
                chk "idle-gate[$set]" PASS "dry-run, not sampled: $v"
            else
                if isolate "$obs" "preflight idle gate $set" >/dev/null && v=$(python3 "$IDLE_SCRIPT" check "$set" --profile "$PROFILE" --seconds 3 --max-busy 0.05 2>&1); then
                    chk "idle-gate[$set]" PASS "idle with SMT siblings (executor on $obs)"
                else
                    chk "idle-gate[$set]" FAIL "not idle or not isolated: $(tr '\n' ';' <<<"${v:-}")"
                fi
            fi
        done < <(measured_sets)
    fi

    # disk
    v=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
    if (( v >= MIN_FREE_GB )); then chk disk PASS "${v} GB free under $ROOT (floor $MIN_FREE_GB)"; else chk disk FAIL "${v} GB free under $ROOT, floor $MIN_FREE_GB"; fi

    # toolchain against the stack declaration
    v=$($HELP toolchain "$TOOLKIT/stack/stack-pins.yaml"); f=$?
    chk toolchain "${v%% *}" "${v#* }"

    # repositories: present, clean, on main, equal to origin/main, no measurement holding its lock
    local r d head remote branch
    for r in "${STAGES[@]}"; do
        [[ $r == narrow ]] && continue
        d=$(repo_path "$r")
        if ! have_checkout "$r"; then
            if (( DRY )); then chk "repo[$r]" INFO "no checkout at $d (plan only in dry-run)"; else chk "repo[$r]" FAIL "no checkout at $d"; fi
            continue
        fi
        branch=$(git -C "$d" rev-parse --abbrev-ref HEAD)
        head=$(git -C "$d" rev-parse HEAD)
        remote=$(git -C "$d" ls-remote origin refs/heads/main 2>/dev/null | cut -f1)
        v=$(git -C "$d" status --porcelain --untracked-files=normal | head -3 | tr '\n' ';')
        if [[ $branch == main && -z $v && -n $remote && $head == "$remote" ]]; then
            chk "repo[$r]" PASS "main = origin/main = ${head:0:12}, clean"
        else
            chk "repo[$r]" FAIL "branch=$branch head=${head:0:12} origin/main=${remote:0:12} dirty=[${v}]"
        fi
        if [[ -f $d/target/campaign.lock ]] && ! flock -n "$d/target/campaign.lock" true; then
            chk "lock[$r]" FAIL "a measurement holds $d/target/campaign.lock"
        fi
    done

    if gh auth status >/dev/null 2>&1; then chk gh PASS "gh auth status ok"; else chk gh FAIL "gh auth status failed"; fi

    # the host record every run carries
    if PYTHONPATH=$TOOLKIT python3 -m campaign_toolkit.hostinfo --json >"$WORK/hostinfo.json" 2>"$WORK/hostinfo.err"; then
        chk hostinfo PASS "wrote $WORK/hostinfo.json"
    else
        chk hostinfo FAIL "campaign_toolkit.hostinfo failed: $(head -c 200 "$WORK/hostinfo.err")"
    fi

    $HELP preflight-json "$CHECKS_TSV" "$WORK/preflight.json"
    if grep -q $'\tFAIL\t' "$CHECKS_TSV"; then
        say "PREFLIGHT FAILED: nothing is measured. Report the FAIL lines above."
        return 1
    fi
    say "PREFLIGHT OK"
}

# ---------------------------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------------------------
toolkit_rev_of() {   # the toolkit rev a repository's requirements.txt pins, or empty
    local f; f="$(repo_path "$1")/requirements.txt"
    [[ -f $f ]] && grep -v '^#' "$f" | grep -o 'campaign-toolkit.*@[0-9a-f]\{40\}' | grep -o '[0-9a-f]\{40\}$' | head -1
}

cmd_pins() {
    local r d rev tdir out status lines installed idir irev
    : >"$WORK/pins.tsv"
    for r in faer tlinalg tcontract cpueinsum tenferro tensor4all; do
        d=$(repo_path "$r")
        if ! have_checkout "$r"; then
            say "[$r] no checkout: pins not checked"
            printf '%s\t%s\t%s\t%s\n' "$r" "no-checkout" "" "no checkout at $d" >>"$WORK/pins.tsv"; continue
        fi
        rev=$(toolkit_rev_of "$r")
        tdir=$TOOLKIT; status=""
        if [[ -n $rev ]]; then
            tdir=$WORK/toolkit/$rev
            if [[ ! -d $tdir ]]; then
                mkdir -p "$tdir"
                git -C "$TOOLKIT" archive "$rev" 2>/dev/null | tar -x -C "$tdir" 2>/dev/null || { rm -rf "$tdir"; tdir=""; }
            fi
        fi
        if [[ -z $tdir || ! -f $tdir/stack/stack-pins.yaml || ! -f $tdir/campaign_toolkit/sync_pins.py ]]; then
            status=unchecked
            lines="toolkit rev ${rev:-none} predates stack/stack-pins.yaml and campaign-sync-pins; checked against the toolkit checkout's stack for information only"
            tdir=$TOOLKIT
            out=$(cd "$tdir" && PYTHONPATH=$tdir python3 -m campaign_toolkit.sync_pins "$d" --repo "${DIR[$r]}" --stack "$tdir/stack/stack-pins.yaml" --check 2>&1 | tr '\n' ';')
            lines="$lines; info: ${out:0:600}"
        else
            out=$(cd "$tdir" && PYTHONPATH=$tdir python3 -m campaign_toolkit.sync_pins "$d" --repo "${DIR[$r]}" --stack "$tdir/stack/stack-pins.yaml" --check 2>&1); f=$?
            if (( f == 0 )); then status=ok; else status=FAIL; fi
            lines=$(tr '\n' ';' <<<"$out")
        fi
        # the toolkit the recorder will import must be the pinned one (run.yaml records toolkit.commit)
        if [[ -n $rev ]]; then
            idir=$($(py_for "$r") -c 'import campaign_toolkit,os;print(os.path.dirname(os.path.dirname(campaign_toolkit.__file__)))' 2>/dev/null)
            irev=$(git -C "${idir:-/nonexistent}" rev-parse HEAD 2>/dev/null)
            if [[ -z $irev ]]; then lines="$lines; WARN installed toolkit is not a git checkout; cannot compare with the pin ${rev:0:12}"
            elif [[ $irev != "$rev" ]]; then lines="$lines; WARN installed toolkit ${irev:0:12} != requirements.txt pin ${rev:0:12} (the recorder would record that toolkit, not the pinned one: pip install -r $d/requirements.txt into a venv and set WEEKLY_PYTHON_${r^^})"; status=FAIL
            fi
        fi
        say "[$r] toolkit pin ${rev:0:12}: $status: ${lines:0:400}"
        printf '%s\t%s\t%s\t%s\n' "$r" "$status" "${rev:0:40}" "$lines" >>"$WORK/pins.tsv"
    done
    python3 - "$WORK/pins.tsv" "$WORK/pins.json" <<'PY'
import json, sys
rows = []
for line in open(sys.argv[1]):
    repo, status, rev, detail = (line.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
    rows.append({"repo": repo, "status": status, "toolkit_rev": rev, "detail": detail})
json.dump(rows, open(sys.argv[2], "w"), indent=2)
PY
    # report only: pins behind the latest release
    $HELP pins-behind "$TOOLKIT/stack/stack-pins.yaml" "$WORK/pins-behind.json" || warn "pins-behind report failed (network?)"
    if grep -q $'\tFAIL\t' "$WORK/pins.tsv"; then
        say "PINS FAILED for: $(awk -F'\t' '$2=="FAIL"{printf "%s ",$1}' "$WORK/pins.tsv")- do not measure those repositories; report the lines above."
        return 1
    fi
    say "PINS OK (unchecked = toolkit rev predates the stack file; see the lines above)"
}

# ---------------------------------------------------------------------------------------------
# Units: what one tool call can run. For a toolkit repository, a unit is one record_run.
# ---------------------------------------------------------------------------------------------
suites_of() {
    case $1 in
        faer) echo faer-kernels;;
        tlinalg) echo tlinalg-competition tlinalg-scaling;;
        tcontract) echo tcbench per-shape;;
        cpueinsum) echo einsum-opt_size einsum-opt_flops;;
    esac
}

units_of() {
    case $1 in
        narrow) echo "";;   # entries of narrow-core.tsv, see run_narrow
        cpueinsum) echo opt_size opt_flops;;
        tcontract) echo tcbench per-shape;;
        tlinalg) echo competition-mkl competition-openblas scaling;;
        tensor4all) echo build 1t 4t;;
        tenferro) echo build measure;;
        faer) echo mkl openblas;;
    esac
}

# unit_spec <stage> <unit>: sets U_EST (minutes), U_SUITES, and U_ARGV (the recorder's argv, relative to the repository).
unit_spec() {
    local label="weekly-$DATE-$2"
    U_EST=10; U_SUITES=""; U_ARGV=()
    case $1:$2 in
        cpueinsum:opt_size)  U_EST=5;  U_SUITES=einsum-opt_size;  U_ARGV=(scripts/v2/record_run.py "$PROFILE" einsum-opt_size --label "$label");;
        cpueinsum:opt_flops) U_EST=5;  U_SUITES=einsum-opt_flops; U_ARGV=(scripts/v2/record_run.py "$PROFILE" einsum-opt_flops --label "$label");;
        tcontract:tcbench)   U_EST=20; U_SUITES=tcbench;   U_ARGV=(scripts/v2/record_run.py tcbench "$PROFILE" tcbench --aa 3 --reps 5 --prime-ms 250 --external-provider-json "$WORK/tcontract/tblis-providers.json" --label "$label");;
        tcontract:per-shape) U_EST=10; U_SUITES=per-shape; U_ARGV=(scripts/v2/record_run.py per-shape "$PROFILE" per-shape --aa 2 --reps 5 --prime-ms 250 --external-provider-json "$WORK/tcontract/tblis-providers.json" --label "$label");;
        tlinalg:competition-mkl)      U_EST=40; U_SUITES=tlinalg-competition; U_ARGV=(scripts/record_run.py "$PROFILE" tlinalg-competition --vendor mkl --aa 4 --reps 5 --prime-ms 250 --label "$label");;
        tlinalg:competition-openblas) U_EST=40; U_SUITES=tlinalg-competition; U_ARGV=(scripts/record_run.py "$PROFILE" tlinalg-competition --vendor openblas --aa 4 --reps 5 --prime-ms 250 --label "$label");;
        tlinalg:scaling)     U_EST=20; U_SUITES=tlinalg-scaling; U_ARGV=(scripts/record_run.py "$PROFILE" tlinalg-scaling --aa 4 --reps 5 --prime-ms 250 --label "$label");;
        faer:mkl)      U_EST=165; U_SUITES=faer-kernels; U_ARGV=(scripts/record_run.py "$PROFILE" faer-kernels --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --threads 1,4,8 --dtypes f64,c64 --reps 5 --aa 3 --prime-ms 250 --label "$label");;
        faer:openblas) U_EST=165; U_SUITES=faer-kernels; U_ARGV=(scripts/record_run.py "$PROFILE" faer-kernels --vendor openblas --engines faer-fork,lapack-openblas --threads 1,4,8 --dtypes f64,c64 --reps 5 --aa 6 --prime-ms 250 --label "$label");;
        tensor4all:build) U_EST=15;;
        tensor4all:1t|tensor4all:4t) U_EST=15;;
        tenferro:build) U_EST=40;;
        tenferro:measure) U_EST=120;;
        *) return 1;;
    esac
}

cmd_plan() {
    local s u total=0
    printf '%-10s %-22s %6s  %s\n' stage unit est-min "recommended tool timeout (ms)"
    for s in "${STAGES[@]}"; do
        if [[ $s == narrow ]]; then printf '%-10s %-22s %6s  %s\n' narrow "(narrow-core.tsv)" 30 1800000; total=$((total + 30)); continue; fi
        for u in $(units_of "$s"); do
            unit_spec "$s" "$u"
            printf '%-10s %-22s %6s  %s\n' "$s" "$u" "$U_EST" "$((U_EST * 90000))"
            total=$((total + U_EST))
        done
    done
    say "sum of estimates: ${total} min (+ about 60 min of builds and guard waits not listed per unit); cap ${CAP_HOURS} h"
    say "one run.sh call = one stage; a whole stage needs the timeout of the sum of its units; with a tool-call timeout cap use --unit"
}

# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------
record_unit() {  # stage unit status cause start end exit rundirs...
    local stage=$1 unit=$2 status=$3 cause=$4 start=$5 end=$6 code=$7; shift 7
    local rd; rd=$(printf '%s\n' "$@" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')
    jw "$WORK/units/${stage}__${unit}.json" stage="$stage" unit="$unit" status="$status" cause="$cause" \
        start_epoch:="$start" end_epoch:="$end" duration_s:="$((end - start))" exit_code:="$code" \
        log="$WORK/logs/${stage}__${unit}.log" rundirs:="$rd" dry_run:="$( ((DRY)) && echo true || echo false)"
    # accumulate run paths for the PR step
    local p; for p in "$@"; do [[ -n $p ]] && echo "$p" >>"$WORK/$stage.rundirs"; done
    say "[$stage/$unit] $status${cause:+: $cause} ($((end - start)) s)"
}

# deadline_ok <est-min>: false when starting now would pass the cap.
deadline_ok() { (( $(now) + $1 * 60 <= $(meta deadline_epoch) )); }

failure_cause() {  # last ERROR-looking line of a log, else the last line
    local f=$1 c
    [[ -f $f ]] || { echo "no log"; return; }
    c=$(grep -E 'ERROR|Error|error:|FAIL|refus|Traceback' "$f" | tail -1)
    [[ -n $c ]] || c=$(grep -v '^\s*$' "$f" | tail -1)
    echo "${c:0:300}"
}

add_sidecars() {  # directory...
    local d
    for d in "$@"; do
        [[ -d $d ]] || continue
        cp "$WORK/executor.json" "$d/executor.json"
        [[ -f $WORK/hostinfo.json ]] && cp "$WORK/hostinfo.json" "$d/hostinfo.json"
    done
}

# New run directories since a marker: the directories of the run.yaml files newer than it.
new_run_dirs() {  # repo-path marker
    (cd "$1" && find data -name run.yaml -newer "$2" -printf '%h\n' 2>/dev/null | sort -u)
}

materialise() {  # stage: bring the checkout of the measured library to its pin (no build)
    local d; d=$(repo_path "$1")
    case $1 in
        faer|tlinalg|cpueinsum) plan bash -c "cd $(printf %q "$d") && scripts/setup_extern_deps.sh";;
        tcontract)              plan bash -c "cd $(printf %q "$d") && scripts/setup_v2_extern_deps.sh";;
        tenferro)               plan bash -c "cd $(printf %q "$d") && TENFERRO_UPDATE=1 TENFERRO_REF=$(tenferro_pin) scripts/setup_extern_deps.sh";;
        *) return 0;;
    esac
}

tenferro_pin() {
    local d; d=$(repo_path tenferro)
    if [[ -f $d/pins/tenferro-rs.rev ]]; then grep -v '^#' "$d/pins/tenferro-rs.rev" | grep -E '^[0-9a-f]{40}' | head -1
    else grep -m1 'TENFERRO_REF:' "$d/.github/workflows/ci.yml" 2>/dev/null | grep -oE '[0-9a-f]{40}'; fi
}

run_toolkit_unit() {  # stage unit
    local stage=$1 unit=$2 d log start end code=0 obs sets marker dirs cause
    d=$(repo_path "$stage"); log=$WORK/logs/${stage}__${unit}.log
    unit_spec "$stage" "$unit"
    start=$(now)
    if ! deadline_ok "$U_EST"; then record_unit "$stage" "$unit" deferred "past the ${CAP_HOURS} h cap: left for next week" "$start" "$start" 0; return 0; fi
    # shellcheck disable=SC2086
    sets=$($HELP cpusets "$d" $U_SUITES) || { record_unit "$stage" "$unit" failed "cannot read cpu_sets of $U_SUITES" "$start" "$(now)" 2; return 1; }
    obs=$(resolve_set "$sets")
    say "[$stage/$unit] suite rows measure on CPUs $sets; executor and recorder go to observer CPUs $obs (est ${U_EST} min)"
    if [[ $stage == tcontract ]]; then
        local tb=${TBLIS_ROOT:-$d/target/tblis-v2.0-beta2}
        if (( ! DRY )) && [[ ! -d $tb ]]; then record_unit "$stage" "$unit" failed "TBLIS_ROOT $tb does not exist; set TBLIS_ROOT" "$start" "$(now)" 2; return 1; fi
        export TBLIS_ROOT=$tb
        mkdir -p "$WORK/tcontract"
        if (( DRY )); then
            say "[dry-run] + (cd $d && TBLIS_ROOT=$tb $(py_for "$stage") scripts/tblis_providers_json.py --out $WORK/tcontract/tblis-providers.json)"
        else
            (cd "$d" && "$(py_for "$stage")" scripts/tblis_providers_json.py --out "$WORK/tcontract/tblis-providers.json") >>"$log" 2>&1 \
                || { record_unit "$stage" "$unit" failed "$(failure_cause "$log")" "$start" "$(now)" 1; return 1; }
        fi
    fi
    if ! isolate "$obs" "$stage/$unit"; then record_unit "$stage" "$unit" failed "isolation failed: executor tree not confined to $obs" "$start" "$(now)" 1; return 1; fi
    marker=$WORK/$stage.$unit.marker
    (( DRY )) || touch "$marker"
    if (( DRY )); then
        say "[dry-run] + (cd $d && taskset -c $obs $(py_for "$stage") $(quote "${U_ARGV[@]}") --no-publish)   # builds first inside the recorder, then verify, then each thread row"
        record_unit "$stage" "$unit" dry-run "" "$start" "$start" 0
        return 0
    fi
    (cd "$d" && taskset -c "$obs" "$(py_for "$stage")" "${U_ARGV[@]}" --no-publish) >"$log" 2>&1; code=$?
    end=$(now)
    if (( code != 0 )); then record_unit "$stage" "$unit" failed "$(failure_cause "$log")" "$start" "$end" "$code"; return 1; fi
    dirs=$(new_run_dirs "$d" "$marker")
    if [[ -z $dirs ]]; then record_unit "$stage" "$unit" failed "recorder exited 0 but wrote no run.yaml" "$start" "$end" 0; return 1; fi
    # shellcheck disable=SC2046
    (cd "$d" && add_sidecars $dirs)
    # the recorder must have left nothing else changed in the tree
    cause=$(git -C "$d" status --porcelain --untracked-files=all | grep -v -F -f <(printf '%s\n' $dirs) | head -3 | tr '\n' ';')
    # shellcheck disable=SC2086
    record_unit "$stage" "$unit" ok "${cause:+other changes in the tree: $cause}" "$start" "$end" 0 $dirs
}

# Build for the repositories whose runner builds inside the pinned row (a build there would run on the measured cores).
run_build_unit() {  # stage
    local stage=$1 d start log=$WORK/logs/${stage}__build.log code=0 obs f
    d=$(repo_path "$stage"); start=$(now); unit_spec "$stage" build
    obs=$(resolve_row 1 | cut -d' ' -f2)
    isolate "$obs" "$stage build" || { record_unit "$stage" build failed "isolation failed" "$start" "$(now)" 1; return 1; }
    if (( DRY )); then
        case $stage in
            tensor4all) say "[dry-run] + (cd $d && taskset -c $obs cargo build --release --locked --bins)";;
            tenferro)
                f=blas-mkl
                say "[dry-run] + (cd $d && MKLROOT=\${MKLROOT:-/opt/intel/oneapi/mkl/latest} taskset -c $obs cargo build --release --no-default-features --features $f --bins)"
                say "[dry-run] + (cd $d && taskset -c $obs cargo build --release --no-default-features --features $f,strided-rs,hptt --bin benchmark_permutation)"
                say "[dry-run] + (cd $d && taskset -c $obs uv sync --frozen)";;
        esac
        record_unit "$stage" build dry-run "" "$start" "$start" 0; return 0
    fi
    case $stage in
        tensor4all) (cd "$d" && taskset -c "$obs" cargo build --release --locked --bins) >"$log" 2>&1; code=$?;;
        tenferro)
            f=${TENFERRO_CPU_FEATURES:-blas-mkl}
            export MKLROOT=${MKLROOT:-/opt/intel/oneapi/mkl/latest}
            (cd "$d" && taskset -c "$obs" cargo build --release --no-default-features --features "$f" --bins \
                && taskset -c "$obs" cargo build --release --no-default-features --features "$f,strided-rs,hptt" --bin benchmark_permutation \
                && { taskset -c "$obs" uv sync --frozen || taskset -c "$obs" uv sync; }) >"$log" 2>&1; code=$?;;
    esac
    if (( code != 0 )); then record_unit "$stage" build failed "$(failure_cause "$log")" "$start" "$(now)" "$code"; return 1; fi
    record_unit "$stage" build ok "" "$start" "$(now)" 0
}

# tensor4all-benchmark: scripts/run_all.sh <profile> takes the row from BENCH_THREADS / BENCH_CPU_CORE and pins each binary itself.
# Its profile names are still the machine names (plan item 2.6 renames them to zen5-cpu).
t4a_profile() { local base=${WEEKLY_T4A_PROFILE_BASE:-linux-ryzen-ai-9-hx-470}; [[ $1 == 1 ]] && echo "$base" || echo "$base-${1}t"; }

run_t4a_row() {  # threads
    local n=$1 stage=tensor4all unit=${1}t d log start end code=0 prof rev row measured obs marker
    d=$(repo_path tensor4all); log=$WORK/logs/${stage}__${unit}.log; start=$(now); prof=$(t4a_profile "$n")
    if ! deadline_ok 15; then record_unit "$stage" "$unit" deferred "past the ${CAP_HOURS} h cap: left for next week" "$start" "$start" 0; return 0; fi
    row=$(resolve_row "$n"); measured=${row% *}; obs=${row#* }
    say "[$stage/$unit] row $n: measured CPUs $measured (BENCH_CPU_CORE=0 BENCH_THREADS=$n), observer $obs, profile dir $prof"
    isolate "$obs" "$stage $unit" || { record_unit "$stage" "$unit" failed "isolation failed: executor tree not confined to $obs" "$start" "$(now)" 1; return 1; }
    rev=$(grep -m1 'tensor4all-itensorlike = ' "$d/Cargo.toml" 2>/dev/null | sed -n 's/.*rev = "\([0-9a-f]*\)".*/\1/p')
    (( DRY )) && [[ -z $rev ]] && rev="<tensor4all-rs-pin-from-Cargo.toml>"
    if (( DRY )); then
        say "[dry-run] + (cd $d && BENCH_CPU_CORE=0 BENCH_THREADS=$n scripts/run_all.sh $prof)   # measures, then runs report.py and gen_index.py (aggregates; item 7.4 pending)"
        say "[dry-run] + (cd $d && git checkout -- result/INDEX.md result/$prof/${rev:0:12}/report.md)   # discard the aggregates the runner wrote (a new report.md is deleted); the PR carries run.yaml and raw/ only"
        record_unit "$stage" "$unit" dry-run "" "$start" "$start" 0; return 0
    fi
    [[ -n $rev ]] || { record_unit "$stage" "$unit" failed "cannot read the tensor4all-rs pin from Cargo.toml" "$start" "$(now)" 2; return 1; }
    (cd "$d" && BENCH_CPU_CORE=0 BENCH_THREADS=$n scripts/run_all.sh "$prof") >"$log" 2>&1; code=$?
    end=$(now)
    if (( code != 0 )); then record_unit "$stage" "$unit" failed "$(failure_cause "$log")" "$start" "$end" "$code"; return 1; fi
    local out="result/$prof/${rev:0:12}"
    # No raw-only mode yet (plan item 7.4): discard what the runner's aggregation step wrote.
    (cd "$d" && git checkout -- result/INDEX.md 2>/dev/null; git checkout -- "$out/report.md" 2>/dev/null || rm -f "$out/report.md")
    add_sidecars "$d/$out"
    record_unit "$stage" "$unit" ok "aggregates (report.md, INDEX.md) discarded; 7.4 pending" "$start" "$end" 0 "$out/run.yaml" "$out/raw" "$out/executor.json" "$out/hostinfo.json"
}

# tenferro-benchmark: one scripts/run_all.sh 1 4 8 invocation (the per-row reports are merged from one
# invocation). The runner pins every row itself; at every row it asks, through
# BENCHMARK_OBSERVER_HANDSHAKE, for the observer to move. This shell answers: it moves the executor
# tree (all but the runner's own subtree) to the row's observer set, verifies, and only then acks.
handshake_follow() {  # dir
    local dir=$1 last="" request mask
    while :; do
        request=$(cat "$dir/request" 2>/dev/null || true)
        if [[ -n $request && $request != "$last" ]]; then
            mask=${request##* }
            if isolate "$mask" "tenferro row" >"$dir/follow.log" 2>&1; then
                printf '%s\n' "$request" >"$dir/ack.tmp" && mv "$dir/ack.tmp" "$dir/ack"
            else
                printf '%s\n' "$mask" >"$dir/failed"
                cat "$dir/follow.log" >>"$dir/failed"
            fi
            last=$request
        fi
        sleep 0.2
    done
}

run_tenferro_measure() {
    local stage=tenferro unit=measure d log start end code=0 hs marker pid fpid dirs n row
    d=$(repo_path tenferro); log=$WORK/logs/${stage}__${unit}.log; start=$(now); unit_spec tenferro measure
    if ! deadline_ok "$U_EST"; then record_unit "$stage" "$unit" deferred "past the ${CAP_HOURS} h cap: left for next week" "$start" "$start" 0; return 0; fi
    hs=$WORK/tenferro/handshake; marker=$WORK/tenferro.measure.marker
    export BENCHMARK_TARGET_PROFILE=$PROFILE TENFERRO_CPU_FEATURES=${TENFERRO_CPU_FEATURES:-blas-mkl} \
           PUBLICATION_GATE_FEATURES=${TENFERRO_CPU_FEATURES:-blas-mkl} PUBLICATION_GATE_PROFILE=full \
           PERMUTATION_EXTRA_FEATURES=hptt MKLROOT=${MKLROOT:-/opt/intel/oneapi/mkl/latest} \
           BENCHMARK_OBSERVER_HANDSHAKE=$hs
    for n in 1 4 8; do row=$(resolve_row $n); say "[$stage/$unit] row $n: measured ${row% *}, observer ${row#* }"; done
    if (( DRY )); then
        say "[dry-run] + mkdir -p $hs   # handshake directory"
        say "[dry-run] + handshake_follow $hs &   # answers each row's request: isolate (executor tree minus the runner subtree) to the row's observer set, verify, then ack"
        say "[dry-run] + (cd $d && ./scripts/run_all.sh 1 4 8)   # env: BENCHMARK_TARGET_PROFILE=$PROFILE TENFERRO_CPU_FEATURES=$TENFERRO_CPU_FEATURES PUBLICATION_GATE_PROFILE=full BENCHMARK_OBSERVER_HANDSHAKE=$hs"
        for n in 1 4 8; do
            row=$(resolve_row $n); ISO_EXCLUDE_FILE=""; isolate "${row#* }" "tenferro row $n (planned handshake answer)"
        done
        say "[dry-run] + discard aggregates: git checkout -- result; git clean -fdq -- result   # then the PR carries data/results/$PROFILE run directories only (git add -f: data/results is ignored until plan item 2.5)"
        record_unit "$stage" "$unit" dry-run "" "$start" "$start" 0; return 0
    fi
    mkdir -p "$hs"; rm -f "$hs"/request "$hs"/ack "$hs"/failed
    ISO_EXCLUDE_FILE=$WORK/tenferro/runner.pid
    : >"$ISO_EXCLUDE_FILE"
    touch "$marker"
    handshake_follow "$hs" & fpid=$!
    (cd "$d" && exec ./scripts/run_all.sh 1 4 8) >"$log" 2>&1 &
    pid=$!; echo "$pid" >"$ISO_EXCLUDE_FILE"
    wait "$pid"; code=$?
    kill "$fpid" 2>/dev/null; wait "$fpid" 2>/dev/null
    ISO_EXCLUDE_FILE=""
    end=$(now)
    (cd "$d" && git checkout -- result 2>/dev/null; git clean -fdq -- result 2>/dev/null)   # discard the aggregates (item 7.4 pending)
    if [[ -s $hs/failed ]]; then record_unit "$stage" "$unit" failed "isolation failed at a row: $(head -1 "$hs/failed")" "$start" "$end" 1; return 1; fi
    if (( code != 0 )); then record_unit "$stage" "$unit" failed "$(failure_cause "$log")" "$start" "$end" "$code"; return 1; fi
    dirs=$(cd "$d" && find "data/results/$PROFILE" -mindepth 3 -maxdepth 3 -type d -newer "$marker" 2>/dev/null | sort)
    if [[ -z $dirs ]]; then record_unit "$stage" "$unit" failed "runner exited 0 but wrote no run directory under data/results/$PROFILE" "$start" "$end" 0; return 1; fi
    # shellcheck disable=SC2046
    (cd "$d" && add_sidecars $dirs)
    # shellcheck disable=SC2086
    record_unit "$stage" "$unit" ok "aggregates discarded; 7.4 pending" "$start" "$end" 0 $dirs
}

# narrow core: entries of narrow-core.tsv, one record_run per line: <repo><TAB><record_run script and arguments>.
# Their run directories are moved out of the repository (a partial run is not a weekly result).
run_narrow() {
    local file=${WEEKLY_NARROW_FILE:-$HERE/narrow-core.tsv} n=0 repo args start
    start=$(now)
    [[ -f $file ]] || file=/dev/null
    while IFS=$'\t' read -r repo args; do
        [[ -z $repo || $repo == \#* ]] && continue
        n=$((n + 1))
        # shellcheck disable=SC2086
        say "[narrow/$n] $repo: $args"
        say "narrow-core entries are not executed by this version: translate them into units first"
    done <"$file"
    if (( n == 0 )); then
        record_unit narrow narrow-core skipped "no narrow-core cases declared yet (D4 representative cases are pending; see narrow-core.tsv); continue with the next stage" "$start" "$start" 0
    else
        record_unit narrow narrow-core skipped "$n narrow-core entries present but not executable by this version" "$start" "$start" 0
    fi
    return 0
}

cmd_run() {
    local stage=${1:-} unit="" u failed=0
    shift || true
    [[ $stage == narrow || -n ${DIR[$stage]:-} ]] || die "usage: run <stage> [--unit NAME]; stages: ${STAGES[*]}"
    while (( $# )); do case $1 in --unit) unit=${2:?--unit needs a name}; shift 2;; *) die "unknown option $1";; esac; done
    init_work
    if [[ $stage == narrow ]]; then run_narrow; return 0; fi
    if ! have_checkout "$stage"; then
        say "[$stage] no checkout at $(repo_path "$stage")"
        if (( DRY )); then
            say "[dry-run] plan only; the commands below are what a real run would execute"
        else
            record_unit "$stage" "(stage)" failed "no checkout at $(repo_path "$stage")" "$(now)" "$(now)" 3
            return 3
        fi
    fi
    # Pin gate: a repository whose pin check FAILed is not measured.
    if [[ -f $WORK/pins.tsv ]] && awk -F'\t' -v r="$stage" '$1==r && $2=="FAIL"{f=1} END{exit !f}' "$WORK/pins.tsv"; then
        record_unit "$stage" "(stage)" failed "pins check FAILed (see $WORK/pins.json); not measured" "$(now)" "$(now)" 1; return 1
    fi
    setup_iso_roots || die "cannot identify the executor process; set WEEKLY_EXECUTOR_PID"
    (( DRY )) || { exec 9>"$(dirname "$WORK")/run.lock"; flock -n 9 || die "another weekly run holds $(dirname "$WORK")/run.lock (one measurement at a time)"; }
    : >"$WORK/$stage.rundirs"
    if (( DRY )); then materialise "$stage"
    else materialise "$stage" >>"$WORK/logs/${stage}__setup.log" 2>&1 \
        || { record_unit "$stage" setup failed "$(failure_cause "$WORK/logs/${stage}__setup.log")" "$(now)" "$(now)" 1; return 1; }
    fi
    local units; units=${unit:-$(units_of "$stage")}
    for u in $units; do
        case $stage:$u in
            tensor4all:build|tenferro:build) run_build_unit "$stage" || failed=1;;
            tensor4all:1t) run_t4a_row 1 || failed=1;;
            tensor4all:4t) run_t4a_row 4 || failed=1;;
            tenferro:measure) run_tenferro_measure || failed=1;;
            *) unit_spec "$stage" "$u" || die "unknown unit '$u' of $stage (units: $(units_of "$stage"))"
               run_toolkit_unit "$stage" "$u" || failed=1;;
        esac
    done
    return $failed
}

# ---------------------------------------------------------------------------------------------
# pr: one pull request per repository, run directories only
# ---------------------------------------------------------------------------------------------
cmd_pr() {
    local stage=${1:-} d branch body title paths url p
    [[ -n ${DIR[$stage]:-} ]] || die "usage: pr <stage>; stages: ${STAGES[*]:1}"
    init_work
    d=$(repo_path "$stage"); branch="weekly/${DIR[$stage]}-$DATE"; body=$WORK/pr/$stage.body.md
    have_checkout "$stage" || { say "[$stage] no checkout: no pull request"; return 3; }
    [[ -s $WORK/$stage.rundirs ]] || { say "[$stage] no run directories recorded (nothing measured, or every unit failed): no pull request"; return 1; }
    mapfile -t paths < <(sort -u "$WORK/$stage.rundirs")
    {
        echo "Weekly full CPU benchmark on the zen5-cpu measurement host, $DATE (UTC). Raw run directories only."
        echo; echo "Executor: $WEEKLY_EXECUTOR $WEEKLY_EXECUTOR_VERSION, model $WEEKLY_EXECUTOR_MODEL. Recorded as executor.json next to each run.yaml until campaign-toolkit item 7.6 adds threads.observer.executor."
        echo; echo "$(python3 "$HERE/summary.py" "$WORK" --pr-body "$stage")"
    } >"$body"
    title="Weekly zen5-cpu run $DATE: raw run directories"
    if (( DRY )); then
        say "[dry-run] + git -C $d switch -c $branch"
        say "[dry-run] + git -C $d add -f -- ${paths[*]}"
        say "[dry-run] + check: nothing but those paths is staged or modified"
        say "[dry-run] + git -C $d commit -m '$title' (trailer: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com> when the executor is Claude Code)"
        say "[dry-run] + git -C $d push -u origin $branch"
        say "[dry-run] + gh pr create -R tensor4all/${DIR[$stage]} --base main --head $branch --title '$title' --body-file $body"
        say "[dry-run] (never merges; the monitoring session reviews)"
        return 0
    fi
    git -C "$d" switch -c "$branch" || die "cannot create $branch" 1
    git -C "$d" add -f -- "${paths[@]}" || die "git add failed" 1
    p=$(git -C "$d" status --porcelain --untracked-files=all | grep -v '^A  ' | head -5 | tr '\n' ';')
    if [[ -n $p ]]; then
        git -C "$d" reset -q
        die "the tree has changes besides the run directories: $p. Not committed; report this." 1
    fi
    local trailer=""
    [[ $WEEKLY_EXECUTOR == claude-code ]] && trailer=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
    git -C "$d" commit -q -m "$title${trailer}" || die "git commit failed" 1
    git -C "$d" push -u origin "$branch" || die "git push failed" 1
    url=$(gh pr create -R "tensor4all/${DIR[$stage]}" --base main --head "$branch" --title "$title" --body-file "$body") || die "gh pr create failed" 1
    jw "$WORK/pr/$stage.json" stage="$stage" repo="${DIR[$stage]}" branch="$branch" url="$url"
    say "[$stage] pull request: $url"
    git -C "$d" switch -q main
}

# bg / await: for an executor whose tool call cannot be held as long as a unit takes. `bg run ...`
# starts the same command detached and returns at once; `await <name> [seconds]` blocks up to that
# long (default 540) and prints RUNNING (exit 10) or FINISHED exit=<code> (the code of the run).
# This is a bounded wait, not a poll of a log file: it reads no output.
cmd_bg() {
    local name pid_exec
    [[ ${1:-} == run && -n ${2:-} ]] || die "usage: bg run <stage> [--unit NAME]"
    name=$2${4:+-$4}
    mkdir -p "$WORK/bg"; rm -f "$WORK/bg/$name.exit"
    pid_exec=${WEEKLY_EXECUTOR_PID:-$($HELP executor-pid "$WEEKLY_EXECUTOR")} || die "cannot identify the executor process; set WEEKLY_EXECUTOR_PID"
    if (( DRY )); then say "[dry-run] + setsid nohup bash $0 $* > $WORK/logs/bg-$name.out   # then: await $name"; return 0; fi
    WEEKLY_EXECUTOR_PID=$pid_exec WEEKLY_WORK_DIR=$WORK setsid nohup bash -c 'bash "$0" "$@"; echo $? >"'"$WORK/bg/$name.exit"'"' "$0" "$@" \
        >"$WORK/logs/bg-$name.out" 2>&1 &
    echo $! >"$WORK/bg/$name.pid"
    say "started $name (pid $!); wait with: run.sh await $name"
}
cmd_await() {
    local name=${1:?usage: await <name> [seconds]} limit=${2:-540} t=0
    while [[ ! -f $WORK/bg/$name.exit ]]; do
        (( t >= limit )) && { say "RUNNING $name"; return 10; }
        sleep 5; t=$((t + 5))
    done
    local code; code=$(<"$WORK/bg/$name.exit")
    say "FINISHED $name exit=$code"; return "$code"
}

cmd_summary() {
    init_work
    python3 "$HERE/summary.py" "$WORK"
}

# ---------------------------------------------------------------------------------------------
# selftest-affinity: the isolation functions on a throwaway process tree, nothing else is touched
# ---------------------------------------------------------------------------------------------
cmd_selftest() {
    local allowed a b root keep rc=0 pids tmp
    allowed=$(awk '/^Cpus_allowed_list/{print $2}' /proc/self/status)
    a=${SELFTEST_SET_A:-0-3}; b=${SELFTEST_SET_B:-1-2}
    say "selftest: this shell may run on $allowed; a throwaway tree moves to $a, then $b"
    tmp=$(mktemp -d)
    cat >"$tmp/threads.py" <<'PY'
import threading, time
for _ in range(3):
    threading.Thread(target=time.sleep, args=(600,), daemon=True).start()
time.sleep(600)
PY
    cat >"$tmp/tree.sh" <<TREE
sleep 600 &
python3 $tmp/threads.py &
bash -c 'sleep 600 & wait' &
wait
TREE
    bash "$tmp/tree.sh" & root=$!
    bash -c 'sleep 600' & keep=$!
    sleep 1
    ISO_ROOTS=("$root" "$keep"); ISO_EXCLUDE_FILE=$tmp/exclude; echo "$keep" >"$ISO_EXCLUDE_FILE"
    say "tree: $(iso_pids | wc -w) processes (root $root), excluded subtree root $keep"
    if isolate "$a" "selftest A"; then say "OK: tree confined to $a"; else say "FAIL: could not confine to $a"; rc=1; fi
    if isolate "$b" "selftest B"; then say "OK: tree confined to $b"; else say "FAIL: could not confine to $b"; rc=1; fi
    if [[ $(awk '/^Cpus_allowed_list/{print $2}' "/proc/$keep/status") == "$allowed" ]]; then
        say "OK: the excluded subtree kept its own affinity ($allowed)"
    else say "FAIL: the excluded subtree was moved"; rc=1; fi
    # the verification must notice a straggler: put one process back and expect a mismatch
    pids=$(iso_pids)
    taskset -a -cp "$allowed" "${pids##* }" >/dev/null 2>&1
    # shellcheck disable=SC2086
    if $HELP verify "$b" $pids >/dev/null; then say "FAIL: verify missed a straggler"; rc=1; else say "OK: verify detects a straggler (negative test)"; fi
    # shellcheck disable=SC2046
    kill $($HELP tree "$root" "$keep") 2>/dev/null
    wait 2>/dev/null
    rm -rf "$tmp"
    if (( rc == 0 )); then say "SELFTEST OK"; else say "SELFTEST FAILED"; fi
    return $rc
}

# ---------------------------------------------------------------------------------------------
args=()
for a in "$@"; do case $a in --dry-run) DRY=1;; *) args+=("$a");; esac; done
set -- "${args[@]:-}"
sub=${1:-}; shift || true
case $sub in
    selftest-affinity) cmd_selftest; exit $?;;
    ""|-h|--help|help) sed -n '2,26p' "$0"; exit 0;;
    plan) cmd_plan; exit 0;;
    preflight|pins|run|pr|summary|bg|await) ;;
    *) die "unknown subcommand '$sub' (preflight pins plan run pr summary selftest-affinity)";;
esac
require_executor
# a dry run never shares a work directory (or a cap clock) with a real week
if (( DRY )) && [[ -z $WORK_GIVEN ]]; then WORK=$WORK-dryrun; fi
# a real preflight starts the week: the cap clock restarts
if [[ $sub == preflight ]] && (( ! DRY )); then rm -f "$WORK/meta.json"; fi
init_work
(( DRY )) && say "[dry-run] work directory: $WORK"
case $sub in
    preflight) cmd_preflight; exit $?;;
    pins) cmd_pins; exit $?;;
    run) cmd_run "$@"; exit $?;;
    pr) cmd_pr "$@"; exit $?;;
    summary) cmd_summary; exit $?;;
    bg) cmd_bg "$@"; exit $?;;
    await) cmd_await "$@"; exit $?;;
esac
