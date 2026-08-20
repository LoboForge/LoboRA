#!/usr/bin/env python3
"""procscan -- the one and only process counter in this tree.

WHY THIS FILE EXISTS
  Both watchers used to count processes on the box with a remote shell pipeline:

      grep -l "model_tr""aining/train.py" /proc/[0-9]*/cmdline 2>/dev/null | wc -l

  That construct can never return 0. In a pipeline the shell forks the child
  first and the CHILD expands the glob, so the child's own /proc/<pid>/cmdline is
  in the file list it hands to grep. By the time grep opens that file it has
  already execve'd, and the file now holds grep's own argv -- which contains the
  search pattern as argv[2]. Grep matches itself, every single time.

  The "model_tr""aining" string-splitting trick does not help: shell concatenation
  happens before execve, so grep's argv carries the joined pattern. The trick
  defeats matching the *shell's* cmdline and nothing else.

  Measured consequence: the count is permanently `real + 1`, so a `trainers == 0`
  fire condition is unsatisfiable and every stop/handoff decision built on it is
  dead on arrival while looking like it works.

WHAT THIS DOES INSTEAD
  Read /proc/<pid>/cmdline directly, in Python, with no shell and no glob, and
  exclude the pids that cannot honestly be counted:

    * the scanning process itself
    * every ANCESTOR of it -- when this runs over SSH the invoking shell's argv
      contains the pattern (it is in the command we sent), and when a bash port
      of these rules runs inside tmux the tmux server is an ancestor too
    * every DESCENDANT of it -- anything it spawns inherits the same argv problem
    * processes in Z (zombie) or X/x (dead) state -- a zombie holds no GPU memory
      and cannot run; counting one as a live trainer is simply wrong

  A pid whose state cannot be read is counted as LIVE. Every unknown here must
  push callers towards "training may still be running", never towards a stop.

ONE IMPLEMENTATION, TWO PLACES IT RUNS
  The local watchers import this module. For the box, `remote_scan_command()`
  ships this exact source over SSH inside a quoted heredoc and runs it there, so
  the remote count and the local count are the same code. Second implementations
  are how this bug class survives; there is deliberately not one.

  scripts/vast/stop_at_step.sh is the one unavoidable exception: it runs on the
  box with no checkout of this repo, so it re-states these rules in pure bash.
  Its self-test cross-checks that bash matcher against this module and fails if
  the two ever disagree.

OUTPUT FORMAT (what the remote side prints, what parse_section reads)
      ###TRAINERS
      1234 R
      1235 Z
      OK 1 1
  One "<pid> <state>" line per match, then "OK <live> <undead>". No cmdlines are
  ever printed: they contain dataset paths, and nothing here needs them.

  The trailing OK line is load-bearing. If python is missing on the box, or the
  scan dies, the section has no OK line and parse_section reports live = -1
  (unknown) rather than 0, so no caller can mistake a broken probe for an idle box.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

# Z: zombie, awaiting a wait() from its parent. X/x: dead (transient, kernel-side).
# None of these can execute an instruction or hold a CUDA context.
UNDEAD_STATES = frozenset({"Z", "X", "x"})

DEFAULT_PROC_ROOT = "/proc"

# The source is shipped to the box between two lines of this delimiter, so it must
# never occur ALONE on a line of this file. Assigning it here does not: a heredoc
# ends only on a line equal to the delimiter and nothing else.
_HEREDOC_DELIM = "LOBORA_PROCSCAN_SOURCE_EOF"

# Tried in order on the box. The last entry is this project's venv, which exists
# even on an image whose PATH has been mangled by a broken activate.
REMOTE_PYTHON_CANDIDATES = ("python3", "python", "/workspace/venv/bin/python")


# --------------------------------------------------------------------------- #
# /proc readers -- every one of them returns a benign default rather than raising
# --------------------------------------------------------------------------- #

def list_pids(proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> list[int]:
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    out = []
    for name in names:
        if name.isdigit():
            out.append(int(name))
    out.sort()
    return out


def read_cmdline(pid: int, proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> str:
    """argv with NULs turned into spaces. '' for kernel threads and zombies."""
    try:
        raw = Path(proc_root, str(pid), "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def read_stat(pid: int,
              proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> tuple[str, int]:
    """(state, ppid). ('', 0) when the process is gone or /proc is not readable.

    Everything after the LAST ')' is `state ppid ...`, which is the only parse
    that survives a comm containing spaces or parentheses.
    """
    try:
        text = Path(proc_root, str(pid), "stat").read_text()
    except OSError:
        return "", 0
    tail = text.rpartition(")")[2].split()
    if len(tail) < 2:
        return "", 0
    ppid = int(tail[1]) if tail[1].lstrip("-").isdigit() else 0
    return tail[0], ppid


def read_comm(pid: int, proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> str:
    """The kernel's short name for the process, e.g. 'bash' or 'tmux: server'."""
    try:
        text = Path(proc_root, str(pid), "stat").read_text()
    except OSError:
        return ""
    inner = text.partition("(")[2]
    return inner.rpartition(")")[0]


def is_undead(state: str) -> bool:
    return state in UNDEAD_STATES


# --------------------------------------------------------------------------- #
# Exclusion set: self, ancestors, descendants
# --------------------------------------------------------------------------- #

def ancestors_of(pid: int,
                 proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> list[int]:
    """Walk ppid upwards. Bounded, so a corrupt /proc cannot spin forever."""
    out: list[int] = []
    seen = {pid}
    cur = pid
    for _ in range(64):
        _, ppid = read_stat(cur, proc_root)
        if ppid <= 0 or ppid in seen:
            break
        out.append(ppid)
        seen.add(ppid)
        cur = ppid
    return out


def descendants_of(pid: int,
                   proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> list[int]:
    children: dict[int, list[int]] = {}
    for other in list_pids(proc_root):
        _, ppid = read_stat(other, proc_root)
        children.setdefault(ppid, []).append(other)
    out: list[int] = []
    queue = list(children.get(pid, ()))
    seen = set(queue)
    while queue:
        cur = queue.pop()
        out.append(cur)
        for kid in children.get(cur, ()):
            if kid not in seen:
                seen.add(kid)
                queue.append(kid)
    return out


def self_excluded_pids(self_pid: int | None = None,
                       proc_root: str | os.PathLike = DEFAULT_PROC_ROOT) -> set[int]:
    """Every pid whose argv is contaminated by our own invocation."""
    pid = os.getpid() if self_pid is None else self_pid
    out = {pid}
    out.update(ancestors_of(pid, proc_root))
    out.update(descendants_of(pid, proc_root))
    return out


# --------------------------------------------------------------------------- #
# The scan
# --------------------------------------------------------------------------- #

class Match:
    __slots__ = ("pid", "state", "cmdline")

    def __init__(self, pid: int, state: str, cmdline: str) -> None:
        self.pid = pid
        self.state = state
        self.cmdline = cmdline

    @property
    def undead(self) -> bool:
        return is_undead(self.state)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Match(pid={self.pid}, state={self.state!r})"


def scan(pattern: str, *,
         proc_root: str | os.PathLike = DEFAULT_PROC_ROOT,
         self_pid: int | None = None,
         exclude: set[int] | None = None,
         include_undead: bool = True) -> list[Match]:
    """Every process whose argv contains `pattern`, minus the ones we must not count.

    `include_undead` keeps Z/X matches in the returned list so callers can log
    them; use `count()` (or Match.undead) to leave them out of a live tally.
    """
    skip = self_excluded_pids(self_pid, proc_root)
    if exclude:
        skip |= set(exclude)
    out: list[Match] = []
    for pid in list_pids(proc_root):
        if pid in skip:
            continue
        cmd = read_cmdline(pid, proc_root)
        if not cmd or pattern not in cmd:
            continue
        state, _ = read_stat(pid, proc_root)
        if is_undead(state) and not include_undead:
            continue
        out.append(Match(pid, state, cmd))
    return out


def count(pattern: str, **kw) -> int:
    """How many LIVE processes match. Zombies and dead entries do not count."""
    return sum(1 for m in scan(pattern, **kw) if not m.undead)


def alive(pid: int, *, proc_root: str | os.PathLike = DEFAULT_PROC_ROOT,
          hint: str = "") -> bool:
    """Is `pid` a live process, and -- with a hint -- still the one we mean?

    The hint defends against pid reuse: a recycled pid must not be able to
    impersonate a watcher forever.
    """
    if pid <= 0:
        return False
    state, _ = read_stat(pid, proc_root)
    if not Path(proc_root, str(pid)).exists():
        return False
    if is_undead(state):
        return False
    return hint in read_cmdline(pid, proc_root) if hint else True


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #

def format_section(name: str, matches: list[Match]) -> list[str]:
    lines = [f"###{name}"]
    live = undead = 0
    for m in matches:
        lines.append(f"{m.pid} {m.state or '?'}")
        if m.undead:
            undead += 1
        else:
            live += 1
    lines.append(f"OK {live} {undead}")
    return lines


def parse_section(lines: list[str]) -> dict:
    """Read one section back. live == -1 means 'the probe did not complete'.

    A section with no OK line is a broken probe, not an empty box. Callers that
    treat -1 as 0 would stop an instance because python was missing.
    """
    pids: list[tuple[int, str]] = []
    live = -1
    undead = 0
    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        if parts[0] == "OK" and len(parts) >= 3 \
                and parts[1].isdigit() and parts[2].isdigit():
            live, undead = int(parts[1]), int(parts[2])
            continue
        if parts[0].isdigit() and len(parts) >= 2:
            pids.append((int(parts[0]), parts[1]))
    return {"live": live, "undead": undead, "pids": pids}


def module_source() -> str:
    try:
        return Path(__file__).read_text()
    except OSError as exc:  # pragma: no cover - only if the checkout is broken
        raise RuntimeError(f"cannot read {__file__} to ship it to the box: {exc}")


def remote_scan_command(patterns: dict[str, str],
                        python_candidates: tuple[str, ...] | None = None) -> str:
    """A POSIX-sh snippet that runs THIS module on the box and prints its sections.

    The pattern appears in the argv of the remote shell that runs this, which is
    exactly why the scanner excludes its own ancestors.
    """
    cands = python_candidates or REMOTE_PYTHON_CANDIDATES
    args = " ".join(shlex.quote(f"{name}={pat}") for name, pat in patterns.items())
    lookup = " || ".join(f"command -v {shlex.quote(c)} 2>/dev/null" for c in cands[:-1])
    return "\n".join([
        f"LOBORA_PY=$({lookup} || printf %s {shlex.quote(cands[-1])})",
        f'"$LOBORA_PY" - {args} <<\'{_HEREDOC_DELIM}\'',
        module_source(),
        _HEREDOC_DELIM,
    ])


# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    """`procscan NAME=pattern [NAME=pattern ...]` -> one ###NAME section each."""
    printed = 0
    for arg in argv:
        name, sep, pattern = arg.partition("=")
        if not sep or not name:
            continue
        for line in format_section(name, scan(pattern)):
            print(line)
        printed += 1
    if not printed:
        print("usage: procscan NAME=pattern [NAME=pattern ...]", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
