#!/usr/bin/env python3
"""Generate the Matrix room mock-ups for agent-swarm.md.

Usage (from the repository root):

    python3 docs/design/figures/make_agent_swarm_figures.py

Writes swarm_room_*.svg next to this script. Standard library only.
The conversations are invented examples: names, numbers, commits and
measurements are made up to show what using the swarm feels like.
"""

from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
W = 760
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"

# role -> (avatar fill, name colour)
ROLE = {
    "operator": ("#5b6676", "#3b4552"),
    "chief": ("#4a4fd1", "#4a4fd1"),
    "superagent": ("#6f73e0", "#4a4fd1"),
    "worker": ("#0f8a7e", "#0b6e64"),
    "broker": ("#a86410", "#8a520c"),
}
INK, MUTED, LINE, BG, HEAD = "#18202b", "#5b6676", "#d5dae2", "#ffffff", "#f3f5f8"
MENTION_BG, MENTION_FG = "#e8e9fb", "#3f43b8"
TAG_FG = "#0b6e64"


def wrap(text, width):
    """Greedy word wrap; width in characters."""
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            if line and len(line) + 1 + len(word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        out.append(line)
    return out


def styled_line(line, x, y):
    """One text line; @mentions get a pill colour, [tags] a teal colour."""
    spans = []
    for tok in line.split(" "):
        t = escape(tok)
        if tok.startswith("@"):
            spans.append(f'<tspan fill="{MENTION_FG}" font-weight="600">{t}</tspan>')
        elif tok.startswith("[") or tok.endswith("]"):
            spans.append(f'<tspan fill="{TAG_FG}" font-family="{MONO}" font-size="12">{t}</tspan>')
        else:
            spans.append(t)
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="13.5" '
            f'fill="{INK}" xml:space="preserve">{" ".join(spans)}</text>')


class Room:
    def __init__(self, name, members, note):
        self.name, self.members, self.note = name, members, note
        self.items = []  # (kind, payload)

    def msg(self, who, role, time, text, replies=None):
        self.items.append(("msg", (who, role, time, text, replies)))
        return self

    def divider(self, label):
        self.items.append(("div", label))
        return self

    def render(self, x0, width, y0, chars):
        parts, y = [], y0
        for kind, p in self.items:
            if kind == "div":
                parts.append(f'<line x1="{x0+16}" y1="{y+10}" x2="{x0+width-16}" y2="{y+10}" stroke="{LINE}"/>')
                tw = len(p) * 6.6 + 16
                cx = x0 + width / 2
                parts.append(f'<rect x="{cx-tw/2}" y="{y+1}" width="{tw}" height="18" rx="9" fill="{BG}" stroke="{LINE}"/>')
                parts.append(f'<text x="{cx}" y="{y+14}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{MUTED}">{escape(p)}</text>')
                y += 30
                continue
            who, role, time, text, replies = p
            fill, namec = ROLE[role]
            parts.append(f'<circle cx="{x0+30}" cy="{y+14}" r="13" fill="{fill}"/>')
            parts.append(f'<text x="{x0+30}" y="{y+18.5}" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#fff">{escape(who.lstrip("@")[0].upper())}</text>')
            parts.append(f'<text x="{x0+54}" y="{y+12}" font-family="{FONT}" font-size="13" font-weight="700" fill="{namec}">{escape(who)}</text>')
            nx = x0 + 54 + len(who) * 7.4 + 8
            parts.append(f'<text x="{nx}" y="{y+12}" font-family="{FONT}" font-size="11" fill="{MUTED}">{escape(role)} · {time}</text>')
            ty = y + 31
            for line in wrap(text, chars):
                parts.append(styled_line(line, x0 + 54, ty))
                ty += 19
            if replies:
                parts.append(f'<rect x="{x0+54}" y="{ty-8}" width="{len(replies)*6.8+28}" height="22" rx="11" fill="{HEAD}" stroke="{LINE}"/>')
                parts.append(f'<text x="{x0+68}" y="{ty+7}" font-family="{FONT}" font-size="12" fill="{MENTION_FG}">↳ {escape(replies)}</text>')
                ty += 22
            y = ty + 10
        return parts, y


def frame(name, members, note, body_parts, height, extra_head=""):
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}">',
        f'<rect x="0.5" y="0.5" width="{W-1}" height="{height-1}" rx="12" fill="{BG}" stroke="{LINE}"/>',
        f'<path d="M0.5 12.5 a12 12 0 0 1 12 -12 h{W-25} a12 12 0 0 1 12 12 v40 h-{W-1} z" fill="{HEAD}" stroke="{LINE}"/>',
        f'<text x="18" y="24" font-family="{FONT}" font-size="15" font-weight="700" fill="{INK}">{escape(name)}</text>',
        f'<text x="18" y="43" font-family="{FONT}" font-size="12" fill="{MUTED}">{escape(members)}</text>',
        f'<text x="{W-18}" y="24" text-anchor="end" font-family="{FONT}" font-size="11.5" fill="{MUTED}">{escape(note)}</text>',
        extra_head,
    ]
    return "\n".join(head + body_parts + ["</svg>\n"])


def single(room, path):
    parts, y = room.render(0, W, 66, 92)
    path.write_text(frame(room.name, room.members, room.note, parts, y + 8))


def chief():
    r = Room("#t4a-chief", "alice (operator), bob (operator), chief",
             "operators give instructions only here")
    r.msg("alice", "operator", "09:02",
          "tenferro-rs: CPU einsum got about 15% slower since last week. Find out why and fix it. "
          "Check on a quiet machine before merging.")
    r.msg("chief", "chief", "09:02",
          "On it. Handed to @tenferro-sa. Recorded: instruction from alice, 2026-11-28 09:02.")
    r.msg("bob", "operator", "09:40", "What is everyone doing right now? Anything waiting for us?")
    r.msg("chief", "chief", "09:41",
          "tenferro-rs: bisecting the einsum slowdown on minipc-1 (reserved until 10:30); 2 other jobs queued.\n"
          "tensor4all-rs: PR #1400 waiting for a second-provider review.\n"
          "colmat-rs: idle.\n"
          "Waiting for a human: tenferro-rs#2690 changes a public API and is labelled human-approve-required. "
          "A maintainer has to decide it on GitHub.")
    r.divider("2 hours later")
    r.msg("chief", "chief", "11:26",
          "Done: the slowdown came from #2681 (new GEMM blocking). Fix PR #2710 restores the speed "
          "(+16% on minipc-1, cores 0-7), reviewed by another provider and merged.")
    r.msg("alice", "operator", "11:30", "Thanks. Stop all benchmark jobs for today, I need minipc-1.")
    r.msg("chief", "chief", "11:30",
          "Stopped 1 queued bench job (tensor4all-rs). minipc-1 has no leases until you say so.")
    single(r, HERE / "swarm_room_chief.svg")


def status():
    r = Room("#t4a-status", "chief, superagents · operators read",
             "superagents report here; wakes only the chief")
    r.msg("tenferro-sa", "superagent", "09:05",
          "[tenferro-rs] einsum slowdown: started bisect job [tenferro-rs#2709] on minipc-1 (exclusive). "
          "2 jobs wait behind it.")
    r.msg("t4a-rs-sa", "superagent", "09:31",
          "[tensor4all-rs] PR #1400 ready, waiting for a second-provider review.")
    r.msg("tenferro-sa", "superagent", "10:04",
          "[tenferro-rs] cause found: #2681. Fix job [tenferro-rs#2710] started on gpu-host-w1.")
    r.msg("tenferro-sa", "superagent", "11:24",
          "[tenferro-rs] #2710 merged through the merge broker. Bench: +16% vs. main, back to the old speed.")
    single(r, HERE / "swarm_room_status.svg")


def repo():
    r = Room("#t4a-tenferro-rs", "chief, tenferro-sa · operators read",
             "the chief gives this superagent its work here")
    r.msg("chief", "chief", "09:02",
          "@tenferro-sa From alice: CPU einsum is about 15% slower than last week. Find the cause, fix it, "
          "and benchmark on a quiet host before merging.")
    r.msg("tenferro-sa", "superagent", "09:04",
          "Plan: 1. bisect on a quiet host (bench profile, exclusive lease). 2. fix job. "
          "3. review by another provider. 4. benchmark the fix. 5. merge through the broker. "
          "Progress goes to #t4a-status.")
    single(r, HERE / "swarm_room_repo.svg")


def host():
    left = Room("", "", "")
    left.msg("tenferro-sa", "superagent", "09:05",
             "[tenferro-rs#2709] bisect the CPU einsum slowdown between v0.9.3 and main. "
             "@minipc-1-w1", replies="4 replies")
    left.msg("broker", "broker", "09:06",
             "[tenferro-rs#2709] minipc-1 reserved, exclusive, cores 0-7, until 10:30.")
    left.msg("t4a-rs-sa", "superagent", "09:12",
             "[tensor4all-rs#1402] run the TCI test suite. @minipc-1-w2",
             replies="queued: minipc-1 is reserved")
    right = Room("", "", "")
    right.msg("tenferro-sa", "superagent", "09:05",
              "[tenferro-rs#2709] bisect the CPU einsum slowdown between v0.9.3 and main. @minipc-1-w1")
    right.msg("minipc-1-w1", "worker", "09:07",
              "[tenferro-rs#2709] quiet checks passed (boost off). Bisecting 14 commits.")
    right.msg("minipc-1-w1", "worker", "09:58",
              "[tenferro-rs#2709] first bad commit: #2681 (new GEMM blocking), -15.8% on einsum_cpu_f64.")
    right.msg("minipc-1-w1", "worker", "10:01",
              "[tenferro-rs#2709] @tenferro-sa done. Report attached to issue #2700.")
    right.msg("broker", "broker", "10:02", "[tenferro-rs#2709] minipc-1 released.")

    lw, rw = 400, W - 400
    lp, ly = left.render(0, lw, 66, 46)
    rp, ry = right.render(lw, rw, 96, 41)
    h = max(ly, ry) + 8
    extra = (
        f'<rect x="{lw}" y="53" width="{rw-0.5}" height="{h-53.5}" fill="#fafbfc"/>'
        f'<line x1="{lw}" y1="53" x2="{lw}" y2="{h-1}" stroke="{LINE}"/>'
        f'<text x="{lw+16}" y="80" font-family="{FONT}" font-size="12.5" font-weight="700" fill="{INK}">Thread</text>'
        f'<text x="{lw+70}" y="80" font-family="{MONO}" font-size="12" fill="{TAG_FG}">[tenferro-rs#2709]</text>'
    )
    path = HERE / "swarm_room_host.svg"
    path.write_text(frame("#t4a-host-minipc-1", "minipc-1 slots, superagents · operators read",
                          "one thread per job; every message carries the job tag",
                          lp + rp, h, extra))


if __name__ == "__main__":
    chief()
    status()
    repo()
    host()
