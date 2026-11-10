#!/usr/bin/env python3
"""Report which Rust repositories carry the unified contribution policy, and which do not.

Reads `docs/contributing/repos.txt` and, for each repository, fetches `CONTRIBUTING.md` and looks for the pointer
marker the template writes:

    <!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->

Why a checker and not a paragraph: the policy is only unified if a repository cannot quietly not have it. Fourteen of
the twenty-three Rust repositories had no `CONTRIBUTING.md` at all before this, and nothing noticed.

Usage:
    python3 scripts/check-contribution.py [--repo-root <dir with checkouts>] [--json]

Exit status: 0 when every listed repository carries the pointer, 1 otherwise, 2 when the list itself is missing.
With `--repo-root`, a local checkout is read instead of the API (so this runs without a token, and in CI of a
repository that has the others checked out beside it).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

MARKER = "unified contribution policy"
POLICY = "docs/contributing/rust-repositories.md"
ORG = "tensor4all"


def read_list(root: pathlib.Path) -> list[str]:
    path = root / "docs" / "contributing" / "repos.txt"
    if not path.exists():
        print(f"check-contribution: missing {path}", file=sys.stderr)
        raise SystemExit(2)
    out = []
    for line in path.read_text().splitlines():
        body, _, comment = line.partition("#")
        name = body.strip()
        if name:
            out.append((name, "fork" in comment))
    return out


def from_api(repo: str) -> tuple[str | None, str]:
    """(text, state): the repository's CONTRIBUTING.md, and how retrieval went.

    A 404 means the file is absent; any other failure means we do not know — a rate limit or a 503 reported as
    "no CONTRIBUTING.md" would be a diagnostic that lies about the repository rather than about the network.
    """
    res = subprocess.run(
        ["gh", "api", f"repos/{ORG}/{repo}/contents/CONTRIBUTING.md", "--jq", ".content"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        err = (res.stderr or "") + (res.stdout or "")
        if "Not Found" in err or "404" in err:
            return None, "absent"
        return None, "unretrieved"
    import base64
    try:
        return base64.b64decode(res.stdout).decode("utf-8", "replace"), "read"
    except Exception:  # noqa: BLE001
        return None, "unretrieved"


def from_disk(repo_root: pathlib.Path, repo: str) -> str | None:  # noqa: D401
    path = repo_root / repo / "CONTRIBUTING.md"
    return path.read_text(errors="replace") if path.exists() else None


def main(argv: list[str]) -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    repo_root = None
    as_json = "--json" in argv
    if "--repo-root" in argv:
        repo_root = pathlib.Path(argv[argv.index("--repo-root") + 1])

    repos = read_list(root)
    rows = []
    for repo, is_fork in repos:
        if is_fork:
            rows.append((repo, "fork", "upstream project: its own contribution rules apply"))
            continue
        if repo_root:
            text, how = from_disk(repo_root, repo), "read"
        else:
            text, how = from_api(repo)
        if how == "unretrieved":
            rows.append((repo, "error", "could not read the repository (rate limit or network), not proof of absence"))
        elif text is None:
            rows.append((repo, "missing", "no CONTRIBUTING.md"))
        elif MARKER not in text:
            rows.append((repo, "unpointed", "CONTRIBUTING.md does not point at the policy"))
        else:
            rows.append((repo, "ok", ""))

    ok = [r for r in rows if r[1] == "ok"]
    exempt = [r for r in rows if r[1] == "fork"]
    errored = [r for r in rows if r[1] == "error"]
    if as_json:
        print(json.dumps([{"repo": r[0], "state": r[1], "detail": r[2]} for r in rows], indent=1))
    else:
        width = max(len(r[0]) for r in rows) if rows else 10
        for repo, state, detail in rows:
            print(f"{repo:{width}}  {state:9s}  {detail}")
        print(f"\n{len(ok)}/{len(rows) - len(exempt)} of our repositories point at {POLICY}"
              f" ({len(exempt)} forks exempt)")
        if errored:
            print(f"{len(errored)} could not be read at all — fix that before trusting the count: "
                  + ", ".join(r[0] for r in errored))

    return 0 if len(ok) == len(rows) - len(exempt) and not errored else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
