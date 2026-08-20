"""Tests for scripts/vast/stage1_guard.sh.

The guard protects an eight-hour cache from a one-word mistake, so the thing that
matters is that it refuses by DEFAULT and that getting past it takes deliberate
effort. A guard that can be silenced with STAGE1_FORCE=1 is a speed bump.

Nothing here touches a real cache or a real box: the cache is a tmp_path of empty
files, and the "live trainer" is a sleep this test starts and kills itself.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parent.parent / "scripts" / "vast" / "stage1_guard.sh"

NO_SUCH_TRAINER = "a-trainer-pattern-nothing-matches"


def run_guard(cache: Path, *, run: str = "h3_ref2va", force: str | None = None,
              pattern: str = NO_SUCH_TRAINER,
              entrypoint: str | None = None) -> subprocess.CompletedProcess:
    script = (f'source "{GUARD}"\n'
              f'stage1_guard "{cache}" "{run}" "480x832x73" "{pattern}"\n')
    env = {"PATH": "/usr/bin:/bin"}
    if entrypoint:
        env["STAGE1_ENTRYPOINT"] = entrypoint
    if force is not None:
        env["STAGE1_FORCE"] = force
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=120, env=env)


def populate(cache: Path, n: int) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (cache / f"clip_{i:04d}.pth").write_bytes(b"x")
    return cache


def test_an_empty_cache_is_allowed_through(tmp_path: Path):
    r = run_guard(tmp_path / "split-cache")
    assert r.returncode == 0, r.stderr
    assert "stage 1 has work to do" in r.stdout


def test_a_populated_cache_is_refused(tmp_path: Path):
    cache = populate(tmp_path / "split-cache", 5)
    r = run_guard(cache)
    assert r.returncode == 3
    assert "REFUSING TO START STAGE 1" in r.stderr
    assert "5 encoded clips" in r.stderr
    assert "8 hours" in r.stderr


def test_the_refusal_points_at_the_entrypoint_the_operator_probably_wanted(tmp_path):
    r = run_guard(populate(tmp_path / "split-cache", 3))
    assert "train_stage2.sh" in r.stderr and "supervise_stage2.sh" in r.stderr


def test_a_truthy_force_is_not_enough(tmp_path: Path):
    """The whole point: STAGE1_FORCE=1 is what someone types when they are not
    reading, so it must not work."""
    cache = populate(tmp_path / "split-cache", 5)
    for value in ("1", "true", "yes", "force", "-f"):
        r = run_guard(cache, force=value)
        assert r.returncode == 3, f"STAGE1_FORCE={value} got through"
        assert "which is not the token above" in r.stderr


def test_the_exact_token_gets_through(tmp_path: Path):
    cache = populate(tmp_path / "split-cache", 5)
    token = "i-know-this-recomputes-5-cached-clips-for-h3_ref2va"
    r = run_guard(cache, force=token)
    assert r.returncode == 0, r.stderr
    assert "override accepted" in r.stdout


def test_the_token_is_specific_to_this_cache_and_this_run(tmp_path: Path):
    """So one pasted out of an older session cannot unlock a different job."""
    cache = populate(tmp_path / "split-cache", 5)
    stale = "i-know-this-recomputes-963-cached-clips-for-h3_ref2va"
    assert run_guard(cache, force=stale).returncode == 3
    other_run = "i-know-this-recomputes-5-cached-clips-for-something_else"
    assert run_guard(cache, force=other_run).returncode == 3


def test_the_refusal_prints_the_token_to_use(tmp_path: Path):
    cache = populate(tmp_path / "split-cache", 7)
    r = run_guard(cache)
    token = next((w for line in r.stderr.splitlines() for w in line.split()
                  if w.startswith("STAGE1_FORCE=")), None)
    assert token, r.stderr
    assert run_guard(cache, force=token.split("=", 1)[1]).returncode == 0


def test_the_refusal_names_the_entrypoint_you_actually_typed(tmp_path: Path):
    """Sourced, $0 is the shell -- often literally "bash" -- so the printed
    override line has to be told what to call itself or it is unusable."""
    cache = populate(tmp_path / "split-cache", 2)
    r = run_guard(cache, entrypoint="./run_anatomy_train.sh")
    assert "./run_anatomy_train.sh" in r.stderr


def test_a_live_trainer_refuses_and_no_token_overrides_it(tmp_path: Path):
    """Starting stage 1 next to a live stage 2 puts two jobs on one GPU."""
    marker = "pretend-train.py-42424"
    # `exec -a` puts the marker in argv, which is where the guard looks.
    proc = subprocess.Popen(["bash", "-c", f"exec -a '{marker}' sleep 60"])
    try:
        time.sleep(0.5)
        empty = tmp_path / "split-cache"
        r = run_guard(empty, pattern=marker)
        assert r.returncode == 4, r.stdout + r.stderr
        assert "training is running right now" in r.stderr
        assert "No token overrides this" in r.stderr

        cache = populate(tmp_path / "full-cache", 5)
        token = "i-know-this-recomputes-5-cached-clips-for-h3_ref2va"
        r = run_guard(cache, force=token, pattern=marker)
        assert r.returncode == 4, "a cache token must not unlock a live GPU"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_the_guard_never_writes_to_the_cache(tmp_path: Path):
    cache = populate(tmp_path / "split-cache", 4)
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in sorted(cache.iterdir())}
    run_guard(cache)
    run_guard(cache, force="i-know-this-recomputes-4-cached-clips-for-h3_ref2va")
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
             for p in sorted(cache.iterdir())}
    assert before == after


def test_the_entrypoint_actually_calls_the_guard():
    """A guard nothing calls is a file."""
    entry = (GUARD.parent / "run_two_stage_train.sh").read_text()
    assert "stage1_guard.sh" in entry
    assert "stage1_guard " in entry
    body = entry.split("stage1_guard ", 1)[1].splitlines()[0]
    assert "|| exit" in body, "the guard's refusal must stop the script"
    # ...and before anything expensive or destructive happens.
    assert entry.index("stage1_guard ") < entry.index('"sft:data_process"')


@pytest.mark.parametrize("suffix", [".pth", ".pt", ".safetensors", ".npy"])
def test_every_shape_of_cache_entry_counts(tmp_path: Path, suffix: str):
    cache = tmp_path / "split-cache"
    cache.mkdir()
    (cache / f"clip_0000{suffix}").write_bytes(b"x")
    assert run_guard(cache).returncode == 3
