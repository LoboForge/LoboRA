#!/usr/bin/env python3
"""lora_pull_watcher.py -- pull checkpoints off the box every 30 min WHILE training runs.

WHAT THIS IS FOR
  post_stop_watcher.py handles the END of the run: it waits for the stop condition,
  backfills everything, verifies it, and stops the instance. That leaves a gap of
  hours during which fresh checkpoints sit only on rented hardware. This process
  closes that gap: every INTERVAL seconds it asks the box what exists, and hands any
  step that is not local yet to scripts/pull_latest_lora.py --all (idempotent, so a
  cycle with nothing new costs one SSH round trip and a few stats).

  It is read-only on the box apart from the pull itself. It never signals anything,
  never writes to the box, and -- deliberately, structurally -- has no vastai stop
  or teardown path in this file at all. Billing control belongs to post_stop_watcher.

WHEN IT DECIDES THE RUN IS OVER (rewritten; the first version got this wrong)
  It used to exit as soon as it saw a stop sentinel OR a step-TARGET checkpoint on
  the box. Both are FILES, and files outlive the run that made them. Training was
  relaunched from step-2000 towards a higher cap while step-2000 and the previous
  run's sentinel were still sitting on the box, so on its next start the old logic
  would have declared the run finished and exited -- pulling nothing at all for the
  entire 41-hour run it exists to cover.

  Termination is therefore evidence about the PRESENT, not artefacts of the past:

    * no live trainer process, confirmed on consecutive probes, AND
    * a heartbeat that has not been written for LPW_STATIC_SECS

  Both, together. Either alone is a supervisor restart or a slow step. A sentinel
  is corroboration when it is FRESH -- it records a step at/after the configured
  cap and the heartbeat has not been written since -- and is ignored otherwise.

  The cap itself comes from scripts/vast/h3_env.sh via lobora/runcap.py, so this
  file cannot hold a number the run has already passed.

WHAT IT DOES AT THE END
  One final `--all` backfill, itself, and then a FINAL: line saying exactly what is
  local and what is not. It does not hand the job to another process and assume it
  is done: the post-stop watcher's own fire condition was unsatisfiable for months
  because of the self-matching grep, so "someone else will pull it" was not true.
  Before pulling it still yields if the post-stop watcher's log shows an
  unterminated pull, so the two never fight over the same file.

OTHER THINGS IT REFUSES TO DO
  - fill the disk: free space is checked against the backfill size before every pull
  - loop forever against a dead host: once SSH fails repeatedly it asks vastai (read
    only) whether the instance is stopped, and exits with a final verdict line
  - die on a hiccup: a failed SSH or a failed pull is logged and retried next cycle
  - deadlock on a stale lock: a lock whose owner pid is gone (or has been recycled
    into some unrelated process) is reported and taken over

NO PIPELINE-EXIT-STATUS FOOTGUNS
  Child output goes straight to the log file handle and status is read from
  subprocess returncode. Nothing is piped through tee and inspected with $?, which is
  the RUNBOOK section 7 bug class.

PATHS
  Log, state and lock live in LPW_ARTIFACT_DIR (default ~/.lobora/lora_pull_watcher),
  never inside this checkout. Every external dependency is env-overridable so the
  self-test drives this exact code against stubs -- see lora_pull_watcher_selftest.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Process counting lives in the package, in ONE implementation, shared with
# post_stop_watcher.py. A second copy of those rules is how the self-matching
# grep survived unnoticed in both watchers at once.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lobora import procscan, runcap  # noqa: E402

# --------------------------------------------------------------------------- #
# Config (env-overridable; defaults are the real run)
# --------------------------------------------------------------------------- #

def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


HERE = Path(__file__).resolve().parent

ARTIFACT_DIR = Path(env("LPW_ARTIFACT_DIR",
                        str(Path.home() / ".lobora" / "lora_pull_watcher")))
LOG_PATH = Path(env("LPW_LOG", str(ARTIFACT_DIR / "lora_pull_watcher.log")))
STATE_PATH = Path(env("LPW_STATE", str(ARTIFACT_DIR / "lora_pull_watcher.state.json")))
LOCK_PATH = Path(env("LPW_LOCK", str(ARTIFACT_DIR / "lora_pull_watcher.lock")))

VAST_INSTANCE = env("LPW_INSTANCE", "48056192")
VAST_LABEL = env("LPW_LABEL", "lobora-h3-a800")
VASTAI_BIN = env("LPW_VASTAI_BIN", "vastai")

SSH_BIN = env("LPW_SSH_BIN", "ssh")
SSH_HOST = env("LPW_SSH_HOST", "ssh9.vast.ai")
SSH_PORT = env("LPW_SSH_PORT", "16192")
SSH_USER = env("LPW_SSH_USER", "root")
SSH_KEY = os.path.expanduser(env("LPW_SSH_KEY", "~/.ssh/vast_tmp"))

REMOTE_LORA_DIR = env("LPW_REMOTE_LORA_DIR",
                      "/workspace/output/anatomy_ref2va_a800/lora")
REMOTE_HEARTBEAT = env("LPW_REMOTE_HEARTBEAT", "/workspace/logs/anatomy_heartbeat.txt")
REMOTE_SENTINEL = env("LPW_REMOTE_SENTINEL", "/workspace/STOP_AT_STEP.sentinel")

DEST_DIR = Path(env("LPW_DEST_DIR", "/media/wrath/AI/ComfyUI/models/loras/minimax-h3"))
DEST_PREFIX = "genpt-step-"
STEP_PAD = 4

PULL_SCRIPT = env("LPW_PULL_SCRIPT", str(HERE / "pull_latest_lora.py"))
# LPW_PULL_ARGV lets the self-test substitute a stub for the real pull script.
PULL_ARGV = json.loads(env("LPW_PULL_ARGV", json.dumps([sys.executable, PULL_SCRIPT])))

# The cap, from scripts/vast/h3_env.sh unless LPW_TARGET_STEP overrides it.
TARGET_STEP = runcap.target_step("LPW_TARGET_STEP")
INTERVAL_SECS = env_int("LPW_INTERVAL_SECS", 1800)      # 30 minutes
RETRY_SECS = env_int("LPW_RETRY_SECS", 300)             # after a failed cycle
CONFIRM_SECS = env_int("LPW_CONFIRM_SECS", 120)         # re-probe gap for "trainer gone"
CONFIRM_POLLS = env_int("LPW_CONFIRM_POLLS", 2)
# A heartbeat older than this, with no trainer process, means the run is over and
# not merely between attempts. At ~43 s/it and a save every 50 steps, 15 minutes
# of silence is far outside anything a healthy run produces.
STATIC_SECS = env_int("LPW_STATIC_SECS", 900)
SSH_TIMEOUT = env_int("LPW_SSH_TIMEOUT", 60)
PULL_TIMEOUT = env_int("LPW_PULL_TIMEOUT", 5400)
MAX_CYCLES = env_int("LPW_MAX_CYCLES", 0)               # 0 = until a terminal state
# Comfortably longer than the run, because a watcher that retires early leaves the
# LAST checkpoints -- the ones worth having -- sitting on rented hardware, which is
# the exact failure it exists to prevent. The run to 6622 at the measured 43.5 s/it
# is about 55 h, and it only ends once the trainer has also been quiet for
# STATIC_SECS. 96 h leaves room for the ETA slipping, an OOM retry or two, and a
# restart of this watcher partway through, at no cost: it exits as soon as the run
# genuinely ends, and it has no stop path, so an over-long ceiling cannot bill
# anything.
MAX_RUNTIME_SECS = env_int("LPW_MAX_RUNTIME_SECS", 96 * 3600)
SSH_FAILS_BEFORE_VASTAI = env_int("LPW_SSH_FAILS_BEFORE_VASTAI", 2)
SSH_FAILS_BEFORE_GIVEUP = env_int("LPW_SSH_FAILS_BEFORE_GIVEUP", 10)

# A checkpoint is ~125.2 MiB; the pull stages a raw copy plus a remapped .part.
CKPT_BYTES = env_int("LPW_CKPT_BYTES", 131227832)
DISK_BUFFER_BYTES = env_int("LPW_DISK_BUFFER_BYTES", 2 * 1024**3)

# post_stop_watcher coordination
PSW_LOCK = Path(env("LPW_PSW_LOCK",
                    str(Path.home() / ".lobora" / "post_stop_watcher"
                        / "post_stop_watcher.lock")))
PSW_LOG = Path(env("LPW_PSW_LOG",
                   str(Path.home() / ".lobora" / "post_stop_watcher"
                       / "post_stop_watcher.log")))
PSW_MARK_BEGIN = "output begins"
PSW_MARK_END_RE = re.compile(r"pull (?:output ends rc=|TIMED OUT after)")
PSW_LOG_TAIL_BYTES = env_int("LPW_PSW_LOG_TAIL_BYTES", 512 * 1024)
PSW_PROC_HINT = env("LPW_PSW_PROC_HINT", "post_stop_watcher")

TRAINER_PAT = env("LPW_TRAINER_PAT", "model_training/train.py")
SUPERVISOR_PAT = env("LPW_SUPERVISOR_PAT", "supervise_stage2.sh")

PROC_ROOT = Path(env("LPW_PROC_ROOT", "/proc"))

STOPPED_STATES = {"stopped", "exited", "offline", "inactive"}


class CycleFailure(RuntimeError):
    """This cycle could not complete. Log it, keep the watcher alive, retry later."""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str = "") -> None:
    line = f"[{now_iso()}] {msg}"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
            fh.flush()
    except OSError as exc:  # logging must never take the watcher down
        print(f"(log write failed: {exc})", flush=True)
    print(line, flush=True)


def alert(msg: str) -> None:
    log("!" * 72)
    log(f"ALERT: {msg}")
    log("!" * 72)


def final(msg: str) -> None:
    """The one line the human greps for when he gets home."""
    log("=" * 72)
    log(f"FINAL: {msg}")
    log("=" * 72)


# --------------------------------------------------------------------------- #
# State + lock
# --------------------------------------------------------------------------- #

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(**kw) -> None:
    state = load_state()
    state.update(kw)
    state["pid"] = os.getpid()
    state["updated"] = now_iso()
    try:
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, STATE_PATH)
    except OSError as exc:
        log(f"(state write failed: {exc})")


def proc_alive(pid: int, hint: str = "") -> bool:
    """Alive, and -- if a hint is given -- still the process we think it is.

    Delegated to the shared scanner so the pid-reuse check and the zombie check
    are the same ones the remote count uses. That matters here: this box has
    already lost power once, and a recycled pid must not be able to impersonate a
    watcher forever.
    """
    return procscan.alive(pid, proc_root=PROC_ROOT, hint=hint)


def take_lock() -> bool:
    """One watcher at a time. A lock whose owner is gone is taken over, not obeyed."""
    try:
        prior = int(LOCK_PATH.read_text().strip())
    except (OSError, ValueError):
        prior = 0
    if prior and prior != os.getpid():
        if proc_alive(prior, Path(__file__).name):
            log(f"another lora_pull_watcher is already running (pid {prior}); "
                f"exiting without doing anything")
            return False
        why = ("its process is gone" if not (PROC_ROOT / str(prior)).exists()
               else f"pid {prior} is now an unrelated process")
        log(f"stale lock from pid {prior} ({why}); taking it over")
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(f"{os.getpid()}\n")
    except OSError as exc:
        log(f"cannot write lock {LOCK_PATH}: {exc}; refusing to run unlocked")
        return False
    return True


def release_lock() -> None:
    try:
        if int(LOCK_PATH.read_text().strip()) == os.getpid():
            LOCK_PATH.unlink()
    except (OSError, ValueError):
        pass


# --------------------------------------------------------------------------- #
# post_stop_watcher coordination (read-only inspection of its pid and its log)
# --------------------------------------------------------------------------- #

def psw_pid() -> int:
    try:
        return int(PSW_LOCK.read_text().strip())
    except (OSError, ValueError):
        return 0


def psw_running() -> tuple[bool, int]:
    pid = psw_pid()
    return (proc_alive(pid, PSW_PROC_HINT), pid)


def psw_mid_pull() -> bool:
    """True if post_stop_watcher's log shows a pull that started and has not ended."""
    try:
        size = PSW_LOG.stat().st_size
        with PSW_LOG.open("rb") as fh:
            if size > PSW_LOG_TAIL_BYTES:
                fh.seek(size - PSW_LOG_TAIL_BYTES)
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return False
    begin = tail.rfind(PSW_MARK_BEGIN)
    if begin < 0:
        return False
    ends = [m.start() for m in PSW_MARK_END_RE.finditer(tail)]
    return not ends or ends[-1] < begin


# --------------------------------------------------------------------------- #
# Remote probe -- strictly read-only, one SSH round trip per cycle
# --------------------------------------------------------------------------- #

# The process sections come from lobora/procscan.py, shipped to the box and run
# there. They used to come from `grep -l PAT /proc/[0-9]*/cmdline | wc -l`, which
# cannot return 0: the pipeline's child expands the glob, so grep is handed its
# own /proc entry, which by then holds grep's argv -- pattern included. Splitting
# the pattern with "" does not help, because the shell joins it before execve.
def _probe_payload() -> str:
    return "\n".join([
        'echo "###NOW"; date -u +%s',
        f'echo "###SENTINEL"; cat {REMOTE_SENTINEL} 2>/dev/null',
        f'echo "###SENTMTIME"; stat -c %Y {REMOTE_SENTINEL} 2>/dev/null',
        f'echo "###HB"; tail -n 1 {REMOTE_HEARTBEAT} 2>/dev/null',
        f'echo "###HBMTIME"; stat -c %Y {REMOTE_HEARTBEAT} 2>/dev/null',
        f'echo "###CKPT"; ls -l --time-style=+%s {REMOTE_LORA_DIR} 2>/dev/null',
        procscan.remote_scan_command({"TRAINERS": TRAINER_PAT,
                                      "SUPERVISORS": SUPERVISOR_PAT}),
        'echo "###END"',
    ])


def ssh_run(payload: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    cmd = [
        SSH_BIN,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=20",
        "-i", SSH_KEY,
        "-p", SSH_PORT,
        f"{SSH_USER}@{SSH_HOST}",
        payload,
    ]
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout or SSH_TIMEOUT)


def parse_sections(out: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    cur = "_pre"
    for line in out.splitlines():
        if line.startswith("###"):
            cur = line[3:].strip()
            sections[cur] = []
            continue
        sections.setdefault(cur, []).append(line)
    return sections


# Same anchoring rule as pull_latest_lora.py: whitespace before "step-", so the
# hand-parked attempt1_step-*.safetensors copies cannot match.
CKPT_RE = re.compile(r"\s(\d+)\s+\d+\s+step-(\d+)\.safetensors$")


def remote_probe() -> dict:
    """One SSH call -> a snapshot of the box. Raises CycleFailure on SSH trouble."""
    try:
        proc = ssh_run(_probe_payload())
    except subprocess.TimeoutExpired:
        raise CycleFailure(f"SSH probe timed out after {SSH_TIMEOUT}s")
    except OSError as exc:
        raise CycleFailure(f"SSH could not be run: {exc}")
    if proc.returncode != 0:
        raise CycleFailure(f"SSH probe failed (exit {proc.returncode}): "
                           f"{(proc.stderr or '').strip()[:200]}")
    sec = parse_sections(proc.stdout)
    if "END" not in sec:
        raise CycleFailure("SSH probe output truncated (no ###END marker)")

    ckpts: dict[int, int] = {}
    for line in sec.get("CKPT", []):
        m = CKPT_RE.search(line)
        if m:
            ckpts[int(m.group(2))] = int(m.group(1))

    hb_line = next((ln for ln in sec.get("HB", []) if "step=" in ln), "")
    hb = dict(re.findall(r"([A-Za-z_]+)=(\S+)", hb_line))

    def one_int(name: str, default: int = -1) -> int:
        for ln in sec.get(name, []):
            ln = ln.strip()
            if ln.isdigit():
                return int(ln)
        return default

    sentinel = "\n".join(ln for ln in sec.get("SENTINEL", []) if ln.strip())
    sent_step = ""
    for ln in sentinel.splitlines():
        if ln.startswith("stopped_at_cumulative_step="):
            sent_step = ln.split("=", 1)[1].strip()

    # No trailing OK line means the scan did not run (no python on the box, a
    # truncated payload). That is UNKNOWN (-1), never 0: a broken probe must not
    # look like an idle box.
    trainers = procscan.parse_section(sec.get("TRAINERS", []))
    supervisors = procscan.parse_section(sec.get("SUPERVISORS", []))

    box_epoch = one_int("NOW")
    hb_mtime = one_int("HBMTIME")
    return {
        "sentinel": sentinel,
        "sentinel_present": bool(sentinel),
        "sentinel_step": int(sent_step) if sent_step.isdigit() else -1,
        "sentinel_mtime": one_int("SENTMTIME"),
        "box_epoch": box_epoch,
        "hb_step": int(hb["step"]) if hb.get("step", "").isdigit() else -1,
        "hb_loss": hb.get("loss", "?"),
        "hb_mtime": hb_mtime,
        # Measured on the BOX's clock, so it survives a restart of this watcher.
        "hb_static_secs": (box_epoch - hb_mtime
                           if box_epoch > 0 and hb_mtime > 0 else -1),
        "ckpts": ckpts,
        "newest": max(ckpts) if ckpts else -1,
        "trainers": trainers["live"],
        "trainers_undead": trainers["undead"],
        "supervisors": supervisors["live"],
    }


# --------------------------------------------------------------------------- #
# vastai -- read-only status only. There is no stop or teardown path in this file.
# --------------------------------------------------------------------------- #

def instance_status() -> str:
    """'stopped'/'running'/... or '' when vastai cannot answer."""
    cmd = [VASTAI_BIN, "show", "instance", VAST_INSTANCE, "--raw"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f"  vastai status unavailable: {exc}")
        return ""
    if proc.returncode != 0:
        log(f"  vastai status exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:200]}")
        return ""
    text = proc.stdout
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        return ""
    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        return ""
    if isinstance(data, list):
        data = next((x for x in data if str(x.get("id")) == VAST_INSTANCE), {})
    return str(data.get("actual_status") or data.get("cur_state") or "")


# --------------------------------------------------------------------------- #
# Local side
# --------------------------------------------------------------------------- #

def dest_name(step: int) -> str:
    return f"{DEST_PREFIX}{step:0{STEP_PAD}d}.safetensors"


def save_gap(ckpts: dict) -> int:
    """The observed save cadence, from the two newest CUMULATIVE step numbers.

    Read rather than assumed: this run saves every 50 micro-batches where the
    previous one saved every 25, and a hardcoded 25 turns the log line that tells
    the operator when to expect the next file into a lie.
    """
    steps = sorted(ckpts)
    return (steps[-1] - steps[-2]) if len(steps) >= 2 else 50


def local_steps() -> set[int]:
    found = set()
    try:
        entries = list(DEST_DIR.iterdir())
    except OSError:
        return found
    pat = re.compile(rf"^{re.escape(DEST_PREFIX)}(\d+)\.safetensors$")
    for p in entries:
        m = pat.match(p.name)
        if m:
            found.add(int(m.group(1)))
    return found


def check_disk(n_missing: int) -> None:
    needed = n_missing * CKPT_BYTES + 2 * CKPT_BYTES + DISK_BUFFER_BYTES
    try:
        free = shutil.disk_usage(DEST_DIR).free
    except OSError as exc:
        raise CycleFailure(f"cannot stat {DEST_DIR} for free space: {exc}")
    gib = 1024**3
    log(f"  disk: {n_missing} file(s) to fetch, need ~{needed / gib:.1f} GiB "
        f"(incl. staging + {DISK_BUFFER_BYTES / gib:.0f} GiB buffer), "
        f"{free / gib:.1f} GiB free on {DEST_DIR}")
    if free < needed:
        raise CycleFailure(
            f"NOT ENOUGH DISK: need ~{needed / gib:.1f} GiB, have {free / gib:.1f} GiB "
            f"on {DEST_DIR}. Skipping this cycle rather than filling the disk; free "
            f"space and the next cycle will pick the checkpoints up.")


def run_pull() -> None:
    """Invoke pull_latest_lora.py --all. Child output goes straight to the log file
    handle: no pipe, so no ${PIPESTATUS} ambiguity."""
    cmd = [*PULL_ARGV, "--all", "--dest-dir", str(DEST_DIR)]
    log(f"  running: {' '.join(cmd)}")
    env_out = dict(os.environ)
    env_out.setdefault("LORA_SSH_HOST", SSH_HOST)
    env_out.setdefault("LORA_SSH_PORT", SSH_PORT)
    env_out.setdefault("LORA_SSH_KEY", SSH_KEY)
    env_out.setdefault("LORA_SSH_USER", SSH_USER)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(f"----- pull_latest_lora.py --all output begins {now_iso()} -----\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=PULL_TIMEOUT, env=env_out)
        except subprocess.TimeoutExpired:
            fh.write(f"----- pull TIMED OUT after {PULL_TIMEOUT}s -----\n")
            raise CycleFailure(f"pull exceeded {PULL_TIMEOUT}s; retrying next cycle")
        except OSError as exc:
            fh.write(f"----- pull could not be started: {exc} -----\n")
            raise CycleFailure(f"pull could not be started: {exc}")
        fh.write(f"----- pull output ends rc={proc.returncode} {now_iso()} -----\n")
    if proc.returncode != 0:
        raise CycleFailure(f"pull exited {proc.returncode} -- see its output above in "
                           f"this log; retrying next cycle")


def pull_missing(missing: list[int]) -> None:
    """Disk guard -> pull -> report what actually landed."""
    check_disk(len(missing))
    before = local_steps()
    run_pull()
    gained = sorted(local_steps() - before)
    if gained:
        log(f"  PULLED {len(gained)}: "
            f"{', '.join(dest_name(s) for s in gained)}")
    else:
        log("  pull reported success but no new file appeared locally "
            "(already present, or the box has nothing new to give)")
    still = [s for s in missing if s not in local_steps()]
    if still:
        log(f"  still missing after the pull: "
            f"{', '.join(f'step-{s}' for s in still)} -- will retry next cycle")
    save_state(last_pull_at=now_iso(), last_pull_gained=gained)


# --------------------------------------------------------------------------- #
# Handoff
# --------------------------------------------------------------------------- #

def finish(reason: str, missing: list[int]) -> int:
    """Training is over: do the last backfill HERE, then report what is local.

    This used to hand the final pull to post_stop_watcher and exit. That was a
    promise this file could not keep -- the post-stop watcher's fire condition was
    unsatisfiable, so "it will pull the rest" simply was not true. The last pull
    is cheap and idempotent, so it happens here and the log states the outcome
    instead of forwarding it.
    """
    log(f"training is over ({reason})")
    if missing:
        log(f"  {len(missing)} checkpoint(s) still only on the box; fetching them now")
        try:
            pull_missing(missing)
        except CycleFailure as exc:
            alert(f"final backfill failed: {exc}")
    else:
        log("  every checkpoint on the box is already local")

    still = sorted(s for s in missing if s not in local_steps())
    have = sorted(local_steps())
    running, pid = psw_running()
    if still:
        alert(f"{len(still)} checkpoint(s) could NOT be fetched: "
              f"{', '.join(f'step-{s}' for s in still)}. They exist only on the box. "
              f"Do not let the instance go away before they are pulled: "
              f"scripts/pull_latest_lora.py --all")
    final(f"training is over ({reason}). Local: {len(have)} checkpoint(s), newest "
          f"step-{have[-1] if have else 'none'}, in {DEST_DIR}. Still remote-only: "
          f"{len(still)}. This watcher pulls only; it has never had a stop path, so "
          f"instance {VAST_INSTANCE} ({VAST_LABEL}) is still running and billing "
          f"until a human decides otherwise"
          + (f". post_stop_watcher is alive (pid {pid}) but do not assume it will act "
             f"-- verify its log." if running else ". Nothing else is watching."))
    save_state(exit_reason=f"finished:{reason}", exited_at=now_iso(),
               remote_only_at_exit=still)
    return 0


# --------------------------------------------------------------------------- #
# One cycle
# --------------------------------------------------------------------------- #

def sentinel_is_fresh(probe: dict) -> tuple[bool, str]:
    """Does the sentinel describe THIS run, or one that already ended?

    The previous run's sentinel and its step-2000 are both still on the box while
    a new run climbs past them. Acting on either would abandon a live run.
    """
    if not probe["sentinel_present"]:
        return False, "no sentinel"
    if probe["sentinel_step"] < TARGET_STEP:
        return False, (f"it records step {probe['sentinel_step']}, below the cap "
                       f"{TARGET_STEP} -- from an earlier run")
    hb_m, sent_m = probe["hb_mtime"], probe["sentinel_mtime"]
    if hb_m > 0 and sent_m > 0 and hb_m - sent_m > STATIC_SECS:
        return False, (f"the heartbeat was written {hb_m - sent_m}s after it -- "
                       f"training ran again since that stop")
    return True, f"it records step {probe['sentinel_step']} and nothing ran since"


def terminal_reason(probe: dict, trainer_gone: int) -> str:
    """Live evidence that the run has ended, or '' to keep pulling.

    Deliberately NOT "a file called step-<cap> exists" and NOT "a sentinel exists".
    Files outlive the run that wrote them; this run began by resuming from exactly
    such a file.
    """
    if probe["trainers"] != 0 or trainer_gone < CONFIRM_POLLS:
        return ""
    static = probe["hb_static_secs"]
    if static < 0:
        return (f"no live trainer process (confirmed {trainer_gone}x) and no "
                f"readable heartbeat on the box")
    if static < STATIC_SECS:
        return ""
    at_cap = probe["newest"] >= TARGET_STEP
    fresh, why = sentinel_is_fresh(probe)
    corroboration = (f"; the stop sentinel agrees ({why})" if fresh else "")
    return (f"no live trainer process (confirmed {trainer_gone}x) and the heartbeat "
            f"has been static for {static}s; newest checkpoint step-{probe['newest']}"
            f"{' which is at/past the cap ' + str(TARGET_STEP) if at_cap else ''}"
            f"{corroboration}")


def main() -> int:
    log("=" * 72)
    log("lora_pull_watcher start -- periodic mid-training checkpoint puller")
    log(f"  pid           : {os.getpid()}")
    log(f"  box           : {SSH_USER}@{SSH_HOST}:{SSH_PORT} (read-only probe)")
    log(f"  remote dir    : {REMOTE_LORA_DIR}")
    log(f"  dest dir      : {DEST_DIR}")
    log(f"  pull          : {' '.join(PULL_ARGV)} --all")
    log(f"  interval      : {INTERVAL_SECS}s (retry after a failed cycle: {RETRY_SECS}s)")
    log(f"  cap           : step-{TARGET_STEP} (cumulative; read from "
        f"{runcap.source_of('LPW_TARGET_STEP')})")
    log(f"  ends when     : no live trainer (confirmed {CONFIRM_POLLS}x) AND the "
        f"heartbeat has been static for {STATIC_SECS}s. NOT when a step-{TARGET_STEP} "
        f"file or a sentinel merely exists -- both outlive the run that wrote them")
    log(f"  proc count    : lobora/procscan.py shipped to the box (reads "
        f"/proc/<pid>/cmdline directly; no grep pipeline, no self-match)")
    # Printed as a wall-clock instant, not just a duration: "60h" is impossible to
    # compare against an ETA at a glance, and the question that matters is whether
    # this watcher outlives the run.
    expiry = datetime.now(timezone.utc) + timedelta(seconds=MAX_RUNTIME_SECS)
    log(f"  max runtime   : {MAX_RUNTIME_SECS}s ({MAX_RUNTIME_SECS / 3600:.0f}h) -- "
        f"expires {expiry.strftime('%a %Y-%m-%d %H:%M')} UTC. The run must finish "
        f"before then or the last checkpoints stay on the box.")
    log(f"  log/state/lock: {LOG_PATH.parent}")
    alive, pid = psw_running()
    log(f"  post_stop_watcher: {'RUNNING pid ' + str(pid) if alive else 'not running'} "
        f"-- this watcher yields to it mid-pull but never delegates its own final "
        f"backfill to it; this file has no stop path at all")

    if not take_lock():
        return 0
    save_state(started_at=now_iso(), exit_reason=None)

    started = time.monotonic()
    cycle = 0
    ssh_fails = 0
    trainer_gone = 0

    try:
        while True:
            cycle += 1
            if MAX_CYCLES and cycle > MAX_CYCLES:
                final(f"stopping after {MAX_CYCLES} cycle(s) (LPW_MAX_CYCLES)")
                save_state(exit_reason="max_cycles", exited_at=now_iso())
                return 0
            elapsed = time.monotonic() - started
            if elapsed > MAX_RUNTIME_SECS:
                final(f"stopping after {elapsed / 3600:.1f}h without ever seeing the "
                      f"run end (max runtime). Nothing is broken by this; re-launch "
                      f"with scripts/lora_pull_watcher_start.sh if training continues.")
                save_state(exit_reason="max_runtime", exited_at=now_iso())
                return 0

            try:
                probe = remote_probe()
            except CycleFailure as exc:
                ssh_fails += 1
                log(f"cycle {cycle}: {exc} (consecutive SSH failures: {ssh_fails})")
                if ssh_fails >= SSH_FAILS_BEFORE_VASTAI:
                    status = instance_status()
                    log(f"  vastai says instance {VAST_INSTANCE} is "
                        f"'{status or 'unknown'}'")
                    if status.lower() in STOPPED_STATES:
                        final(f"instance {VAST_INSTANCE} ({VAST_LABEL}) is "
                              f"'{status}': the box is gone, SSH can never succeed "
                              f"again, and there is nothing left to pull. Exiting "
                              f"cleanly. Everything that was fetched is in {DEST_DIR}.")
                        save_state(exit_reason=f"instance_{status}",
                                   exited_at=now_iso())
                        return 0
                if ssh_fails >= SSH_FAILS_BEFORE_GIVEUP:
                    final(f"giving up after {ssh_fails} consecutive SSH failures to "
                          f"{SSH_HOST}:{SSH_PORT} without vastai confirming a stopped "
                          f"instance. The box may be wedged or the SSH port may have "
                          f"been reassigned (Vast does that on every restart). Nothing "
                          f"was pulled in those cycles; re-launch after re-resolving "
                          f"the port with `vastai show instances`.")
                    save_state(exit_reason="ssh_giveup", exited_at=now_iso())
                    return 1
                time.sleep(RETRY_SECS)
                continue

            ssh_fails = 0
            have = local_steps()
            missing = sorted(s for s in probe["ckpts"] if s not in have)
            fresh, fresh_why = sentinel_is_fresh(probe)
            undead = probe["trainers_undead"]
            trainers_txt = ("UNKNOWN" if probe["trainers"] < 0
                            else str(probe["trainers"]))
            if undead:
                trainers_txt += f" (+{undead} zombie/dead, not counted)"
            log(f"cycle {cycle}: remote newest=step-{probe['newest']} "
                f"({len(probe['ckpts'])} on box) local={len(have)} "
                f"missing={len(missing)} hb_step={probe['hb_step']} "
                f"loss={probe['hb_loss']} hb_static={probe['hb_static_secs']}s "
                f"trainers={trainers_txt} supervisors={probe['supervisors']} "
                f"sentinel={'FRESH' if fresh else ('stale' if probe['sentinel_present'] else 'no')}")
            if probe["sentinel_present"] and not fresh:
                log(f"  sentinel ignored: {fresh_why}")
            if probe["trainers"] < 0:
                alert("the box-side process count did not complete (no OK line from "
                      "procscan -- is python missing on the box?). Treating it as "
                      "UNKNOWN and continuing to pull, which is the safe direction.")
            save_state(last_cycle_at=now_iso(), last_newest=probe["newest"],
                       last_local_count=len(have))

            if probe["trainers"] == 0:
                trainer_gone += 1
                log(f"  no live trainer process on the box "
                    f"({trainer_gone}/{CONFIRM_POLLS} confirmations). A supervisor "
                    f"restart looks exactly like this for a few seconds, so this is "
                    f"re-checked in {CONFIRM_SECS}s and cross-checked against the "
                    f"heartbeat before anything is concluded.")
            else:
                trainer_gone = 0

            reason = terminal_reason(probe, trainer_gone)
            if reason:
                return finish(reason, missing)

            if probe["trainers"] == 0:
                # Gone but not yet corroborated: come back sooner than a full cycle.
                time.sleep(CONFIRM_SECS)
                continue

            if not missing:
                log(f"  nothing to do: every one of the {len(probe['ckpts'])} "
                    f"checkpoints on the box is already local. Next one should be "
                    f"step-{probe['newest'] + save_gap(probe['ckpts'])}, and the cap "
                    f"is step-{TARGET_STEP}.")
                time.sleep(INTERVAL_SECS)
                continue

            if psw_mid_pull():
                log("  post_stop_watcher is mid-backfill (its log shows a pull that "
                    "has not finished); yielding this cycle so we never race it over "
                    "the same files")
                time.sleep(RETRY_SECS)
                continue

            log(f"  to fetch: {', '.join(f'step-{s}' for s in missing)}")
            try:
                pull_missing(missing)
            except CycleFailure as exc:
                alert(f"cycle {cycle} pull skipped/failed: {exc}")
                time.sleep(RETRY_SECS)
                continue

            time.sleep(INTERVAL_SECS)
    finally:
        release_lock()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        final("interrupted. Nothing on the box was touched; post_stop_watcher is "
              "unaffected and still owns the end of the run.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - last resort; say so in the log
        alert(f"unexpected {type(exc).__name__}: {exc}. This watcher is exiting; "
              f"post_stop_watcher is unaffected. Nothing on the box was changed.")
        raise SystemExit(1)
