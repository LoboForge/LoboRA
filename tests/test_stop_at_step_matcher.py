"""The bash matcher in scripts/vast/stop_at_step.sh, checked off the box.

scripts/vast/stop_at_step_selftest.sh is the end-to-end test, and it can only run
ON the box: it starts a replica supervisor and stub trainers and then asserts the
live run is still up. That is the right test for the stop sequence, and the wrong
one for the matching rules, which is where both real bugs lived and which nobody
can exercise while a paid run is in progress.

So the script takes STOP_PROC_ROOT and a `--list-matches` mode that prints what it
would touch and signals nothing. These tests build a synthetic /proc containing
the exact shapes that broke it -- a tmux server whose argv still names the
supervisor script, an accelerate launcher above a leaf trainer, a zombie -- and
assert the bash rules agree with lobora/procscan.py on the same tree.

The agreement check is the point. The bash rules exist as a second implementation
only because the box has no checkout of this repo, and an unchecked second
implementation is precisely how the self-matching grep survived in two files at
once.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lobora import procscan

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vast" / "stop_at_step.sh"

TRAINER_PAT = "examples/minimax_h3/model_training/train.py"
SUP_PAT = "supervise_stage2.sh"


def mkproc(root: Path, pid: int, *, argv: str, state: str = "S", ppid: int = 1,
           comm: str | None = None) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(argv.replace(" ", "\0").encode() + b"\0")
    if comm is None:
        first = argv.split()[0] if argv.split() else ""
        comm = Path(first).name[:15]
    # Field order after comm is: state ppid pgrp ...
    (d / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pid} 0 0 -1 0\n")
    (d / "comm").write_text(comm + "\n")


@pytest.fixture()
def box(tmp_path: Path) -> Path:
    """The live box's shape, as of the run this was written during."""
    root = tmp_path / "proc"
    root.mkdir()
    mkproc(root, 1, argv="/sbin/init", comm="init")
    # The process that broke it: tmux keeps the argv of whatever forked the
    # server, so the server's cmdline names the supervisor script it launched.
    mkproc(root, 200, comm="tmux: server", ppid=1,
           argv="tmux new-session -d -s anatomy_train bash /workspace/supervise_stage2.sh")
    mkproc(root, 300, comm="bash", ppid=200,
           argv="bash /workspace/supervise_stage2.sh")
    mkproc(root, 310, comm="python", ppid=300,
           argv=f"/workspace/venv/bin/python /workspace/venv/bin/accelerate launch {TRAINER_PAT} --config x")
    mkproc(root, 320, comm="pt_main_thread", ppid=310,
           argv=f"/workspace/venv/bin/python {TRAINER_PAT} --config x")
    # Unrelated, and one dead trainer from the previous attempt.
    mkproc(root, 400, comm="sshd", ppid=1, argv="sshd: root@notty")
    mkproc(root, 330, comm="python", ppid=300, state="Z",
           argv=f"/workspace/venv/bin/python {TRAINER_PAT} --config x")
    return root


def list_matches(proc_root: Path, *, trainer: str = TRAINER_PAT,
                 sup: str = SUP_PAT) -> dict[str, list[str]]:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--list-matches"],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "STOP_PROC_ROOT": str(proc_root),
             "STOP_TRAINER_PATTERN": trainer, "STOP_SUP_PATTERN": sup},
    )
    assert proc.returncode == 0, proc.stderr
    out: dict[str, list[str]] = {}
    cur = ""
    for line in proc.stdout.splitlines():
        if line.startswith("###"):
            cur = line[3:]
            out[cur] = []
        elif cur:
            out[cur].append(line)
    out["_stderr"] = proc.stderr.splitlines()
    return out


def live_pids(section: list[str]) -> set[int]:
    return {p for p, state in procscan.parse_section(section)["pids"]
            if not procscan.is_undead(state)}


def undead_pids(section: list[str]) -> set[int]:
    return {p for p, state in procscan.parse_section(section)["pids"]
            if procscan.is_undead(state)}


def test_the_tmux_server_is_never_a_supervisor(box: Path):
    """The bug that ended the last stop attempt: pid 200 was killed as a
    supervisor, taking every session on that socket with it."""
    m = list_matches(box)
    assert live_pids(m["SUPERVISORS"]) == {300}
    assert 200 not in live_pids(m["SUPERVISORS"]) | undead_pids(m["SUPERVISORS"])
    assert m["SUPERVISORS"][-1] == "OK 1 0"


def test_the_whole_trainer_tree_is_matched_not_just_the_leaf(box: Path):
    """Signalling only the leaf leaves the accelerate parent holding the GPU."""
    assert live_pids(list_matches(box)["TRAINERS"]) == {310, 320}


def test_a_dead_trainer_is_reported_but_not_counted_live(box: Path):
    m = list_matches(box)
    assert m["TRAINERS"][-1] == "OK 2 1", m["TRAINERS"]
    assert undead_pids(m["TRAINERS"]) == {330}


def test_the_matcher_cannot_match_itself(box: Path):
    """The whole point. `grep -l PAT /proc/*/cmdline | wc -l` returns 1 here
    because grep's own argv contains the pattern; this must return 0."""
    m = list_matches(box, trainer="a-pattern-no-process-has")
    assert m["TRAINERS"] == ["OK 0 0"]


def test_bash_and_python_agree_on_the_same_tree(box: Path):
    """Two implementations, one answer -- or this fails."""
    bash = list_matches(box)
    py = procscan.scan(TRAINER_PAT, proc_root=box, self_pid=999999)
    assert live_pids(bash["TRAINERS"]) == {m.pid for m in py if not m.undead}
    assert undead_pids(bash["TRAINERS"]) == {m.pid for m in py if m.undead}
    assert bash["TRAINERS"] == procscan.format_section("TRAINERS", py)[1:]


def test_agreement_holds_when_there_is_nothing_to_match(box: Path):
    pat = "nothing/matches/this.py"
    bash = list_matches(box, trainer=pat)
    py = procscan.scan(pat, proc_root=box, self_pid=999999)
    assert bash["TRAINERS"] == ["OK 0 0"]
    assert py == []


def test_a_supervisor_that_is_not_a_shell_is_refused(tmp_path: Path):
    """Belt and braces: even a non-tmux impostor has to look like a shell."""
    root = tmp_path / "proc"
    root.mkdir()
    mkproc(root, 1, argv="/sbin/init", comm="init")
    mkproc(root, 500, comm="python", ppid=1,
           argv="python /workspace/tail_the_log.py /workspace/supervise_stage2.sh")
    mkproc(root, 501, comm="bash", ppid=1, argv="bash /workspace/supervise_stage2.sh")
    m = list_matches(root)
    assert live_pids(m["SUPERVISORS"]) == {501}
    assert any("ignoring pid=500" in ln for ln in m["_stderr"]), m["_stderr"]


def test_an_empty_self_tag_does_not_swallow_the_whole_process_table(box: Path):
    """`case "$cmd" in *""*)` matches everything, so an unset tag would skip every
    process and report an idle box. Caught in production, on the box, by a count
    that came back 0 with two trainers plainly running."""
    shared = SCRIPT.parent / "procscan.sh"
    script = (f'SELF_TAG=""\nsource "{shared}"\n'
              f'count_pids "{TRAINER_PAT}"\n')
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=120,
                          env={"PATH": "/usr/bin:/bin", "PROCFS": str(box)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2", proc.stdout + proc.stderr


def test_the_proc_root_variable_cannot_collide_with_a_callers_own(tmp_path: Path):
    """Found in production: the box's stage-1 entrypoint sets PROC to a model
    path, and a sourced file shares the caller's namespace. The matcher happily
    scanned that directory, found nothing, and reported an idle GPU with two
    trainers running."""
    shared = SCRIPT.parent / "procscan.sh"
    body = shared.read_text()
    assert "PROCFS=" in body and "$PROC/" not in body and '"$PROC"' not in body

    root = tmp_path / "proc"
    root.mkdir()
    mkproc(root, 1, argv="/sbin/init", comm="init")
    mkproc(root, 320, comm="python", ppid=1, argv=f"python {TRAINER_PAT}")
    script = (f'PROC=/some/model/path\nsource "{shared}"\n'
              f'count_pids "{TRAINER_PAT}"\n')
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=120,
                          env={"PATH": "/usr/bin:/bin", "PROCFS": str(root)})
    assert proc.stdout.strip() == "1", proc.stdout + proc.stderr


def test_init_is_protected(box: Path):
    m = list_matches(box)
    assert "1" in m["PROTECTED"][0].split()
