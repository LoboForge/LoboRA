"""Tests for lobora/procscan.py -- the one process counter in this tree.

The bug these exist to prevent: counting processes with

    grep -l PAT /proc/[0-9]*/cmdline | wc -l

which can never return 0, because in a pipeline the child expands the glob and so
hands grep grep's own /proc entry -- which by then holds grep's argv, pattern
included. A watcher whose fire condition is `count == 0` is then dead code that
looks alive.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lobora import procscan


def fake_proc(root: Path, pid: int, cmdline: str, state: str = "S",
              ppid: int = 1, comm: str = "python3") -> None:
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode() + b"\0")
    (d / "stat").write_text(f"{pid} ({comm}) {state} {ppid} 0 0 0 -1 0 0 0\n")


# --------------------------------------------------------------------------- #
# Reading /proc
# --------------------------------------------------------------------------- #

def test_reads_state_and_ppid_past_a_comm_containing_spaces(tmp_path):
    fake_proc(tmp_path, 42, "tmux new-session -d", state="S", ppid=7,
              comm="tmux: server")
    assert procscan.read_stat(42, tmp_path) == ("S", 7)
    assert procscan.read_comm(42, tmp_path) == "tmux: server"


def test_a_missing_process_reads_as_gone_rather_than_raising(tmp_path):
    assert procscan.read_stat(999, tmp_path) == ("", 0)
    assert procscan.read_cmdline(999, tmp_path) == ""


# --------------------------------------------------------------------------- #
# The counting rules
# --------------------------------------------------------------------------- #

def test_counts_live_matches(tmp_path):
    fake_proc(tmp_path, 10, "python model_training/train.py --lora")
    fake_proc(tmp_path, 11, "python model_training/train.py --lora")
    fake_proc(tmp_path, 12, "bash supervise_stage2.sh")
    assert procscan.count("model_training/train.py",
                          proc_root=tmp_path, self_pid=999) == 2


def test_zombies_and_dead_are_reported_but_never_counted_as_live(tmp_path):
    fake_proc(tmp_path, 10, "python model_training/train.py", state="R")
    fake_proc(tmp_path, 11, "python model_training/train.py", state="Z")
    fake_proc(tmp_path, 12, "python model_training/train.py", state="X")
    matches = procscan.scan("model_training/train.py", proc_root=tmp_path,
                            self_pid=999)
    assert {m.pid for m in matches} == {10, 11, 12}
    assert {m.pid for m in matches if m.undead} == {11, 12}
    assert procscan.count("model_training/train.py",
                          proc_root=tmp_path, self_pid=999) == 1


def test_an_unreadable_state_counts_as_live(tmp_path):
    """Every unknown must push callers towards 'still running', never to a stop."""
    d = tmp_path / "10"
    d.mkdir()
    (d / "cmdline").write_bytes(b"python\0model_training/train.py\0")
    assert procscan.count("model_training/train.py",
                          proc_root=tmp_path, self_pid=999) == 1


def test_excludes_itself_its_ancestors_and_its_descendants(tmp_path):
    # The shape of the real remote probe: sshd -> shell whose argv carries the
    # pattern (because we sent it) -> the python scanner -> a child of it.
    fake_proc(tmp_path, 1, "/sbin/init", ppid=0)
    fake_proc(tmp_path, 100, "sshd: root@notty", ppid=1)
    fake_proc(tmp_path, 200, "sh -c echo model_training/train.py", ppid=100)
    fake_proc(tmp_path, 300, "python - TRAINERS=model_training/train.py", ppid=200)
    fake_proc(tmp_path, 400, "sh -c stat model_training/train.py", ppid=300)
    # ...and one genuine trainer, in a completely different subtree.
    fake_proc(tmp_path, 500, "python model_training/train.py --lora", ppid=1)

    matches = procscan.scan("model_training/train.py", proc_root=tmp_path,
                            self_pid=300)
    assert [m.pid for m in matches] == [500]


def test_the_ancestor_walk_survives_a_ppid_cycle(tmp_path):
    fake_proc(tmp_path, 10, "a", ppid=11)
    fake_proc(tmp_path, 11, "b", ppid=10)
    assert procscan.ancestors_of(10, tmp_path) == [11]


# --------------------------------------------------------------------------- #
# A real zombie: a child that exits while its parent never calls wait().
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                    reason="needs a Linux procfs")
def test_a_real_zombie_is_not_alive():
    parent = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import os, sys, time
            pid = os.fork()
            if pid == 0:
                os._exit(0)          # exits; the parent never wait()s for it
            sys.stdout.write(str(pid) + "\\n")
            sys.stdout.flush()
            time.sleep(30)
        """)],
        stdout=subprocess.PIPE, text=True)
    try:
        zpid = int(parent.stdout.readline().strip())
        state = ""
        for _ in range(100):
            state, _ppid = procscan.read_stat(zpid)
            if state == "Z":
                break
            time.sleep(0.05)
        assert state == "Z", f"the child never became a zombie (state {state!r})"
        # It still has a /proc entry and `kill -0` still succeeds on it, which is
        # exactly why "the pid is still there" is the wrong liveness test.
        assert Path(f"/proc/{zpid}").exists()
        os.kill(zpid, 0)
        assert procscan.alive(zpid) is False
        # And it can never appear in a scan: a zombie has no argv left to match.
        assert procscan.read_cmdline(zpid) == ""
    finally:
        parent.kill()
        parent.wait(timeout=10)


# --------------------------------------------------------------------------- #
# The wire format and the shipped-to-the-box path
# --------------------------------------------------------------------------- #

def test_parse_section_reports_unknown_when_the_scan_did_not_finish():
    assert procscan.parse_section(["/bin/sh: python3: not found"])["live"] == -1
    assert procscan.parse_section([])["live"] == -1
    good = procscan.parse_section(["10 S", "11 Z", "OK 1 1"])
    assert (good["live"], good["undead"], good["pids"]) == (1, 1, [(10, "S"), (11, "Z")])


def test_format_and_parse_round_trip(tmp_path):
    fake_proc(tmp_path, 10, "python model_training/train.py", state="R")
    fake_proc(tmp_path, 11, "python model_training/train.py", state="Z")
    lines = procscan.format_section(
        "TRAINERS", procscan.scan("model_training/train.py", proc_root=tmp_path,
                                  self_pid=999))
    assert lines[0] == "###TRAINERS"
    assert procscan.parse_section(lines[1:])["live"] == 1


@pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                    reason="needs a Linux procfs")
def test_the_shipped_payload_counts_zero_for_a_pattern_that_matches_nothing():
    """The regression test for the self-matching grep.

    Run the payload the way ssh runs it -- as one argv string -- so the pattern is
    sitting in the invoking shell's own /proc/<pid>/cmdline while the scan runs.
    """
    nothing = "zzzz-no-process-anywhere-has-this-pattern-9876"
    payload = procscan.remote_scan_command({"TRAINERS": nothing})
    out = subprocess.run(["sh", "-c", payload], capture_output=True, text=True,
                         timeout=120).stdout
    assert procscan.parse_section(out.splitlines()[1:])["live"] == 0, out

    # The construct this replaced, for contrast: it reports a process that is not
    # there, because grep matches its own post-exec argv.
    old = subprocess.run(
        ["sh", "-c", f'grep -l "{nothing[:8]}""{nothing[8:]}" /proc/[0-9]*/cmdline '
                     f'2>/dev/null | wc -l'],
        capture_output=True, text=True, timeout=120).stdout.strip()
    assert old == "1", f"expected the old pipeline to phantom-count, got {old!r}"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                    reason="needs a Linux procfs")
def test_the_shipped_payload_still_finds_a_process_that_is_really_there():
    marker = "lobora-procscan-pytest-marker"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                              marker],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.4)
        payload = procscan.remote_scan_command({"TRAINERS": marker})
        out = subprocess.run(["sh", "-c", payload], capture_output=True, text=True,
                             timeout=120).stdout
        sec = procscan.parse_section(out.splitlines()[1:])
        assert sec["live"] == 1, out
        assert sec["pids"][0][0] == child.pid, out
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_the_shipped_source_is_this_module():
    payload = procscan.remote_scan_command({"TRAINERS": "x"})
    assert procscan.module_source() in payload
    assert "TRAINERS=x" in payload
    # A heredoc ends on a line equal to the delimiter and nothing else; if the
    # delimiter ever appeared alone in the source, the payload would be truncated.
    body = [ln.strip() for ln in procscan.module_source().splitlines()]
    assert procscan._HEREDOC_DELIM not in body
