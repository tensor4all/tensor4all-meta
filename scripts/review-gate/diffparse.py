#!/usr/bin/env python3
"""Canonical, hunk-aware parser for ``git diff`` output.

The three review-gate repositories (``strided-rs``, ``tenferro-rs``,
``tensor4all-rs``) each carried a copy of a parser that decided "this line is a
file header" with ``line.startswith("+++ ")`` and no notion of hunk state.  An
added line whose *content* begins with ``++ `` renders as ``+++ b/whatever``, so
the parser adopted that string as the current file and every following added
line was attributed to a path the pull-request author chose.  Every rule keyed
on the path (the frozen-tree check above all) then looked at the wrong file.

This module is the reference implementation of the corrected rule:

* a header line is recognised **only outside a hunk**;
* inside a hunk, the body budget from ``@@ -a,b +c,d @@`` decides how many
  ``-``/``+``/space lines follow, and those lines are content;
* paths can additionally be cross-checked against ``git diff --name-status -z``,
  which cannot be influenced by the diff text.

No network access, no third-party dependencies; Python 3.9+.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = [
    "FileDiff",
    "Hunk",
    "HunkLine",
    "PathConsistency",
    "added_lines_by_file",
    "added_lines_with_text",
    "assert_paths_consistent",
    "changed_paths",
    "files_with_deleted_lines",
    "parse_diff",
    "parse_name_status",
]

HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def _split_diff_git_header(header: str) -> "tuple[Optional[str], Optional[str]]":
    """Split the ``a/x b/y`` tail of a ``diff --git`` line.

    Plain paths are separated by " b/"; paths that git had to quote (spaces,
    non-ASCII) arrive as two quoted strings.
    """

    header = header.strip()
    if header.startswith('"'):
        parts = re.findall(r'"(?:[^"\\]|\\.)*"', header)
        if len(parts) == 2:
            return _strip_side(parts[0], "a"), _strip_side(parts[1], "b")
    marker = header.rfind(" b/")
    if marker == -1:
        return None, None
    return _strip_side(header[:marker], "a"), _strip_side(header[marker + 1 :], "b")

#: Lines git emits that are never file content and never file headers.
_META_PREFIXES = (
    "index ",
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "similarity index ",
    "dissimilarity index ",
    "copy from ",
    "copy to ",
    "rename from ",
    "rename to ",
    "Binary files ",
)

DEV_NULL = "/dev/null"


@dataclass(frozen=True)
class HunkLine:
    """One body line of a hunk.  ``kind`` is ``"+"``, ``"-"`` or ``" "``."""

    kind: str
    text: str
    old_lineno: Optional[int]
    new_lineno: Optional[int]


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple

    def added(self) -> Iterable:
        return (line for line in self.lines if line.kind == "+")


@dataclass(frozen=True)
class FileDiff:
    old_path: Optional[str]
    new_path: Optional[str]
    hunks: tuple

    @property
    def path(self) -> Optional[str]:
        """Path rules should key on: the new side, or the old one for a delete."""

        return self.new_path or self.old_path


@dataclass(frozen=True)
class PathConsistency:
    """Difference between the paths a diff *claims* and the paths git lists."""

    unexpected: tuple
    missing: tuple

    @property
    def ok(self) -> bool:
        return not self.unexpected and not self.missing


def _unquote(path: str) -> str:
    """Undo git's C-style quoting of paths that contain spaces or escapes."""

    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        try:
            return json.loads(path)
        except ValueError:  # pragma: no cover - defensive
            return path[1:-1]
    return path


def _strip_side(path: str, side: str) -> Optional[str]:
    """``b/foo`` -> ``foo``; ``/dev/null`` -> ``None``."""

    path = _unquote(path.strip())
    if path == DEV_NULL:
        return None
    prefix = side + "/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def parse_diff(diff_text: str) -> tuple:
    """Parse ``git diff`` output into :class:`FileDiff` records.

    Header lines are recognised only when no hunk body is outstanding.  Inside a
    hunk the ``@@`` budget decides, so a ``+++``-looking line is content.
    """

    files = []
    old_path = None
    new_path = None
    hunks = []
    hunk = None  # (old_left, new_left, old_no, new_no, lines)
    in_body = False

    def flush_file() -> None:
        nonlocal old_path, new_path, hunks
        if old_path is None and new_path is None and not hunks:
            return
        files.append(FileDiff(old_path=old_path, new_path=new_path, hunks=tuple(hunks)))
        old_path = None
        new_path = None
        hunks = []

    for raw in diff_text.splitlines():
        if in_body:
            marker = raw[:1]
            if raw.startswith("\\"):
                # "\ No newline at end of file" is metadata, not a body line.
                continue
            old_left, new_left, old_no, new_no, lines = hunk
            if marker == "+":
                lines.append(HunkLine("+", raw[1:], None, new_no))
                new_no += 1
                new_left -= 1
            elif marker == "-":
                lines.append(HunkLine("-", raw[1:], old_no, None))
                old_no += 1
                old_left -= 1
            elif marker == " " or raw == "":
                lines.append(
                    HunkLine(" ", raw[1:] if raw else "", old_no, new_no)
                )
                old_no += 1
                new_no += 1
                old_left -= 1
                new_left -= 1
            else:
                # Truncated or malformed hunk: fall through and re-read the line
                # as a header candidate rather than swallowing it as content.
                in_body = False
                hunk = None
                old_left = new_left = 0
                if raw.startswith("diff --git ") or raw.startswith("@@") or raw.startswith(
                    "--- "
                ):
                    pass
                else:
                    continue
            if in_body:
                hunk = (old_left, new_left, old_no, new_no, lines)
                if old_left <= 0 and new_left <= 0:
                    hunks[-1] = Hunk(
                        hunks[-1].old_start,
                        hunks[-1].old_count,
                        hunks[-1].new_start,
                        hunks[-1].new_count,
                        tuple(lines),
                    )
                    in_body = False
                    hunk = None
                continue

        if raw.startswith("diff --git "):
            flush_file()
            # The header names both sides; `--- `/`+++ ` refine it below and a
            # rename/copy pair overrides both.  Keeping this default makes a
            # mode-only or binary file block report its paths at all.
            header = raw[len("diff --git ") :]
            old_from_header, new_from_header = _split_diff_git_header(header)
            old_path = old_from_header
            new_path = new_from_header
            continue
        if raw.startswith("rename from ") or raw.startswith("copy from "):
            old_path = _unquote(raw.split(" ", 2)[2].strip())
            continue
        if raw.startswith("rename to ") or raw.startswith("copy to "):
            new_path = _unquote(raw.split(" ", 2)[2].strip())
            continue
        if raw.startswith("--- "):
            old_path = _strip_side(raw[4:], "a")
            continue
        if raw.startswith("+++ "):
            new_path = _strip_side(raw[4:], "b")
            continue
        match = HUNK_HEADER.match(raw)
        if match:
            old_start = int(match.group("old_start"))
            old_count = int(match.group("old_count") or 1)
            new_start = int(match.group("new_start"))
            new_count = int(match.group("new_count") or 1)
            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=(),
                )
            )
            hunk = (old_count, new_count, old_start, new_start, [])
            in_body = True
            if old_count == 0 and new_count == 0:  # pragma: no cover - degenerate
                in_body = False
                hunk = None
            continue
        if raw.startswith(_META_PREFIXES):
            continue
        # Anything else outside a hunk is either a section heading of a hunk
        # (`@@ ... @@ heading` was matched above) or noise; never a header.

    flush_file()
    return tuple(files)


def added_lines_with_text(diff_text: str) -> "dict[str, list[tuple[int, str]]]":
    """Map each file to its added ``(new_line_number, text)`` pairs."""

    result = {}  # type: dict[str, list[tuple[int, str]]]
    for file_diff in parse_diff(diff_text):
        path = file_diff.path
        if path is None:
            continue
        bucket = result.setdefault(path, [])
        for hunk in file_diff.hunks:
            for line in hunk.added():
                bucket.append((line.new_lineno, line.text))
    return result


def added_lines_by_file(diff_text: str) -> "dict[str, set[int]]":
    """Map each file to the set of new-side line numbers it adds."""

    return {
        path: {lineno for lineno, _ in lines}
        for path, lines in added_lines_with_text(diff_text).items()
    }


def files_with_deleted_lines(diff_text: str) -> "set[str]":
    """Files this diff removes lines from, keyed by their new-side path."""

    result = set()
    for file_diff in parse_diff(diff_text):
        path = file_diff.path
        if path is None:
            continue
        if any(line.kind == "-" for hunk in file_diff.hunks for line in hunk.lines):
            result.add(path)
    return result


def changed_paths(diff_text: str) -> "set[str]":
    """Every non-``/dev/null`` path the diff names, old or new side."""

    paths = set()
    for file_diff in parse_diff(diff_text):
        if file_diff.old_path:
            paths.add(file_diff.old_path)
        if file_diff.new_path:
            paths.add(file_diff.new_path)
    return paths


def parse_name_status(text: str) -> "dict[str, tuple[str, Optional[str]]]":
    """Parse ``git diff --name-status -z`` output.

    Returns ``{status: {new_or_only_path: old_path_or_None}}``-flattened as
    ``{path: (status, other_path)}`` where ``other_path`` is the old path of a
    rename/copy and ``None`` otherwise.
    """

    fields = [chunk for chunk in text.split("\0") if chunk != ""]
    result = {}
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        letter = status[:1]
        if letter in ("R", "C"):
            if index + 1 >= len(fields):
                break
            old = fields[index]
            new = fields[index + 1]
            index += 2
            result[new] = (status, old)
            continue
        if index >= len(fields):
            break
        path = fields[index]
        index += 1
        result[path] = (status, None)
    return result


def assert_paths_consistent(diff_text: str, name_status_text: str) -> PathConsistency:
    """Cross-check the diff's claimed paths against git's authoritative list.

    ``git diff --name-status -z`` reports the paths git itself used to build the
    diff, NUL-separated and therefore unambiguous.  A diff that names a path git
    does not list has been tampered with (a forged header); a gate should fail
    closed on a non-empty result instead of trusting either side.
    """

    listed = set(parse_name_status(name_status_text))
    for status, other in parse_name_status(name_status_text).values():
        if other:
            listed.add(other)
    claimed = changed_paths(diff_text)
    return PathConsistency(
        unexpected=tuple(sorted(claimed - listed)),
        missing=tuple(sorted(listed - claimed)),
    )
