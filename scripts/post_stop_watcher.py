#!/usr/bin/env python3
"""post_stop_watcher.py -- LOCAL companion to the box-side stop_at_step.sh watcher.

WHAT THIS DOES (and, importantly, what it does not)
  The box-side watcher (/workspace/stop_at_step.sh, tmux session stop_at_step) halts
  training at cumulative step 2000. It cannot do the rest: scripts/pull_latest_lora.py
  runs locally and SSHes into the box, and the vastai CLI is authenticated locally
  only. So this process, on the local machine, drives the follow-on:

    1. poll the box until training has GENUINELY stopped and step-2000 is complete
    2. pull EVERY remote checkpoint that is not already local (--all backfill)
    3. verify every pulled file independently of the pull script
    4. only if all of that passes: `vastai stop instance 48056192`
    5. confirm the instance actually reached a stopped state, and log what it saw

  This process never stops training, never signals anything on the box, and never
  writes to the box. It is a reader. The box-side watcher owns the shutdown.

FAIL-OPEN IS THE WHOLE POINT
  Once the instance is stopped, SSH is dead and nothing more can be retrieved.
  Every uncertain outcome therefore leaves the instance RUNNING and logs loudly:
  SSH failure, pull failure, a verification failure, a missing checkpoint, a
  stalled or dead run, a step-2000 that never appeared. $1.10/hr for a few hours
  is enormously cheaper than losing the final LoRA.

  `vastai destroy` is never invoked. It is not in this file. Destroying would take
  the 800 GB volume and the 8-hour split-cache with it.

NO PIPELINE-EXIT-STATUS FOOTGUNS
  Every subprocess status is read from subprocess returncode; child output goes
  straight to the log file handle. Nothing is piped through tee and then inspected
  with $?, which is the bug class that already bit this project.

TESTABILITY
  Every external dependency is env-overridable so the self-test drives this exact
  code against stubs: PSW_SSH_BIN, PSW_VASTAI_BIN, PSW_PULL_ARGV, PSW_DEST_DIR,
  PSW_PRINT_ONLY, PSW_MAX_POLLS, ... See post_stop_watcher_selftest.py.

PATHS
  The log, state and lock live in PSW_ARTIFACT_DIR (default ~/.lobora/post_stop_watcher),
  never inside this repo. The pull script is resolved next to this file. Host, port
  and instance id are the run this was written for; override them per run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config (env-overridable; defaults are the real run)
# --------------------------------------------------------------------------- #

def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


HERE = Path(__file__).resolve().parent

ARTIFACT_DIR = Path(env("PSW_ARTIFACT_DIR",
                        str(Path.home() / ".lobora" / "post_stop_watcher")))
LOG_PATH = Path(env("PSW_LOG", str(ARTIFACT_DIR / "post_stop_watcher.log")))
STATE_PATH = Path(env("PSW_STATE", str(ARTIFACT_DIR / "post_stop_watcher.state.json")))
LOCK_PATH = Path(env("PSW_LOCK", str(ARTIFACT_DIR / "post_stop_watcher.lock")))

VAST_INSTANCE = env("PSW_INSTANCE", "48056192")
VAST_LABEL = env("PSW_LABEL", "lobora-h3-a800")
VASTAI_BIN = env("PSW_VASTAI_BIN", "vastai")

SSH_BIN = env("PSW_SSH_BIN", "ssh")
SSH_HOST = env("PSW_SSH_HOST", "ssh9.vast.ai")
SSH_PORT = env("PSW_SSH_PORT", "16192")
SSH_USER = env("PSW_SSH_USER", "root")
SSH_KEY = os.path.expanduser(env("PSW_SSH_KEY", "~/.ssh/vast_tmp"))

REMOTE_LORA_DIR = env("PSW_REMOTE_LORA_DIR",
                      "/workspace/output/anatomy_ref2va_a800/lora")
REMOTE_HEARTBEAT = env("PSW_REMOTE_HEARTBEAT", "/workspace/logs/anatomy_heartbeat.txt")
REMOTE_SENTINEL = env("PSW_REMOTE_SENTINEL", "/workspace/STOP_AT_STEP.sentinel")
REMOTE_STOP_LOG = env("PSW_REMOTE_STOP_LOG", "/workspace/logs/stop_at_step.log")

DEST_DIR = Path(env("PSW_DEST_DIR", "/media/wrath/AI/ComfyUI/models/loras/minimax-h3"))
DEST_PREFIX = "genpt-step-"
STEP_PAD = 4

PULL_SCRIPT = env("PSW_PULL_SCRIPT", str(HERE / "pull_latest_lora.py"))
# PSW_PULL_ARGV lets the self-test substitute a stub for the real pull script.
PULL_ARGV = json.loads(env("PSW_PULL_ARGV", json.dumps([sys.executable, PULL_SCRIPT])))

TARGET_STEP = env_int("PSW_TARGET_STEP", 2000)
POLL_SECS = env_int("PSW_POLL_SECS", 90)
MAX_POLLS = env_int("PSW_MAX_POLLS", 0)            # 0 = forever
STALL_SECS = env_int("PSW_STALL_SECS", 3600)       # heartbeat frozen this long = ALERT
QUIESCE_SECS = env_int("PSW_QUIESCE_SECS", 120)    # ckpt dir untouched before we trust it
CONFIRM_POLLS = env_int("PSW_CONFIRM_POLLS", 2)    # consecutive "stopped" reads required
SSH_TIMEOUT = env_int("PSW_SSH_TIMEOUT", 60)
PULL_TIMEOUT = env_int("PSW_PULL_TIMEOUT", 10800)  # 3 h for a full 55-file backfill
STOP_CONFIRM_SECS = env_int("PSW_STOP_CONFIRM_SECS", 420)
STOP_POLL_SECS = env_int("PSW_STOP_POLL_SECS", 20)
FAIL_SLEEP_SECS = env_int("PSW_FAIL_SLEEP_SECS", 300)  # backoff after a fail-open
ALERT_EVERY = env_int("PSW_ALERT_EVERY", 600)      # rate-limit repeated ALERT lines

# A checkpoint is ~125.2 MiB; keep real headroom on a 99%-full disk.
CKPT_BYTES = env_int("PSW_CKPT_BYTES", 131227832)
DISK_BUFFER_BYTES = env_int("PSW_DISK_BUFFER_BYTES", 2 * 1024**3)

EXPECTED_TENSORS = env_int("PSW_EXPECTED_TENSORS", 208)
KEY_PREFIX = "diffusion_model."

PRINT_ONLY = env("PSW_PRINT_ONLY", "0") == "1"     # print the stop cmd, do not run it

TRAINER_PAT = env("PSW_TRAINER_PAT", "model_training/train.py")
SUPERVISOR_PAT = env("PSW_SUPERVISOR_PAT", "supervise_stage2.sh")


class FailOpen(RuntimeError):
    """Something is not right. Leave the instance running; a human decides."""


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


_last_alert: dict[str, float] = {}


def alert(key: str, msg: str, force: bool = False) -> None:
    """Loud, rate-limited. Used for every condition a human must look at."""
    now = time.time()
    if not force and now - _last_alert.get(key, 0.0) < ALERT_EVERY:
        return
    _last_alert[key] = now
    log("!" * 72)
    log(f"ALERT [{key}]: {msg}")
    log("!" * 72)


# --------------------------------------------------------------------------- #
# State (idempotency across re-runs)
# --------------------------------------------------------------------------- #

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(**kw) -> None:
    state = load_state()
    state.update(kw)
    state["instance"] = VAST_INSTANCE
    state["updated"] = now_iso()
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, STATE_PATH)


def take_lock() -> bool:
    """Refuse to run two watchers at once (a second one could double-stop)."""
    try:
        prior = int(LOCK_PATH.read_text().strip())
    except (OSError, ValueError):
        prior = 0
    if prior and prior != os.getpid() and Path(f"/proc/{prior}").exists():
        log(f"another post_stop_watcher is already running (pid {prior}); exiting")
        return False
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(f"{os.getpid()}\n")
    return True


# --------------------------------------------------------------------------- #
# Remote probe -- strictly read-only, one SSH round trip per poll
# --------------------------------------------------------------------------- #

# The trainer/supervisor patterns are split with an empty-string concatenation so
# that this payload's own argv (which contains the split form) can never match
# itself. grep's own /proc entry does not exist when the glob is expanded.
def _probe_payload() -> str:
    t_a, t_b = TRAINER_PAT[:8], TRAINER_PAT[8:]
    s_a, s_b = SUPERVISOR_PAT[:6], SUPERVISOR_PAT[6:]
    return "\n".join([
        'echo "###NOW"; date -u +%s',
        f'echo "###SENTINEL"; cat {REMOTE_SENTINEL} 2>/dev/null',
        f'echo "###HB"; tail -n 1 {REMOTE_HEARTBEAT} 2>/dev/null',
        f'echo "###HBMTIME"; stat -c %Y {REMOTE_HEARTBEAT} 2>/dev/null',
        f'echo "###CKPT"; ls -l --time-style=+%s {REMOTE_LORA_DIR} 2>/dev/null',
        f'echo "###RECENT"; find {REMOTE_LORA_DIR} -maxdepth 1 -type f '
        f'-newermt "-{QUIESCE_SECS} seconds" 2>/dev/null | wc -l',
        f'echo "###TRAINERS"; grep -l "{t_a}""{t_b}" /proc/[0-9]*/cmdline '
        f'2>/dev/null | wc -l',
        f'echo "###SUPERVISORS"; grep -l "{s_a}""{s_b}" /proc/[0-9]*/cmdline '
        f'2>/dev/null | wc -l',
        f'echo "###STOPLOG"; tail -n 25 {REMOTE_STOP_LOG} 2>/dev/null',
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


# Same anchoring rule as pull_latest_lora.py: whitespace before "step-" so the
# hand-parked attempt1_step-*.safetensors copies cannot match.
CKPT_RE = re.compile(r"\s(\d+)\s+\d+\s+step-(\d+)\.safetensors$")


def remote_probe() -> dict:
    """One SSH call -> a snapshot of the box. Raises FailOpen on any SSH trouble."""
    try:
        proc = ssh_run(_probe_payload())
    except subprocess.TimeoutExpired:
        raise FailOpen("SSH probe timed out")
    if proc.returncode != 0:
        raise FailOpen(f"SSH probe failed (exit {proc.returncode}): "
                       f"{(proc.stderr or '').strip()[:300]}")
    sec = parse_sections(proc.stdout)
    if "END" not in sec:
        raise FailOpen("SSH probe output truncated (no ###END marker)")

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
    return {
        "box_epoch": one_int("NOW"),
        "sentinel": sentinel,
        "sentinel_present": bool(sentinel),
        "hb_step": int(hb["step"]) if hb.get("step", "").isdigit() else -1,
        "hb_line": hb_line,
        "hb_mtime": one_int("HBMTIME"),
        "ckpts": ckpts,
        "newest": max(ckpts) if ckpts else -1,
        "recent_writes": one_int("RECENT", 1),
        "trainers": one_int("TRAINERS", -1),
        "supervisors": one_int("SUPERVISORS", -1),
        "stoplog": [ln for ln in sec.get("STOPLOG", []) if ln.strip()],
    }


def remote_sha256(step: int) -> str:
    """sha256 of one checkpoint on the box (read-only). '' if unavailable."""
    payload = f"sha256sum {REMOTE_LORA_DIR}/step-{step}.safetensors 2>/dev/null"
    try:
        proc = ssh_run(payload, timeout=300)
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    parts = proc.stdout.split()
    return parts[0] if parts else ""


# --------------------------------------------------------------------------- #
# Local verification -- deliberately independent of pull_latest_lora.py
# --------------------------------------------------------------------------- #

def dest_name(step: int) -> str:
    return f"{DEST_PREFIX}{step:0{STEP_PAD}d}.safetensors"


def verify_local(step: int) -> dict:
    """Header parses, 208 tensors, contiguous payload, size adds up, keys remapped.

    Raises FailOpen with a specific reason on any problem.
    """
    path = DEST_DIR / dest_name(step)
    if not path.exists():
        raise FailOpen(f"{path.name} is missing locally")
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise FailOpen(f"{path.name}: no safetensors header length (truncated)")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or 8 + header_len > size:
            raise FailOpen(f"{path.name}: declares {header_len}-byte header but file "
                           f"is {size} bytes -- truncated or corrupt")
        try:
            header = json.loads(fh.read(header_len))
        except json.JSONDecodeError as exc:
            raise FailOpen(f"{path.name}: unparseable safetensors header: {exc}")

    meta = header.get("__metadata__") or {}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(tensors) != EXPECTED_TENSORS:
        raise FailOpen(f"{path.name}: {len(tensors)} tensors, expected {EXPECTED_TENSORS}")

    end = 0
    for name, info in tensors.items():
        lo, hi = info["data_offsets"]
        if lo != end:
            raise FailOpen(f"{path.name}: non-contiguous tensor data at {name}")
        end = hi
        if not name.startswith(KEY_PREFIX):
            raise FailOpen(f"{path.name}: key not remapped (no {KEY_PREFIX}): {name}")
        if ".default." in name or name.endswith(".default"):
            raise FailOpen(f"{path.name}: key still carries .default: {name}")

    if size != 8 + header_len + end:
        raise FailOpen(f"{path.name}: {size} bytes but header+payload require "
                       f"{8 + header_len + end} -- truncated transfer")
    return {"path": path, "size": size, "tensors": len(tensors),
            "source_sha256": meta.get("source_sha256", ""),
            "step_meta": meta.get("step", "")}


def verify_all(steps: list[int], target: int) -> dict:
    """Verify every expected step. Returns a summary; raises FailOpen on the first
    problem, which is what keeps the instance alive."""
    log(f"verifying {len(steps)} local checkpoint(s) in {DEST_DIR}")
    results = {}
    for step in steps:
        info = verify_local(step)
        results[step] = info
        log(f"  ok step-{step:<5} {info['path'].name}  {info['size']} bytes  "
            f"{info['tensors']} tensors  src_sha={info['source_sha256'][:12] or 'n/a'}")

    if target not in results:
        raise FailOpen(f"step-{target} is not among the verified local files")

    # Highest-value extra guarantee: cross-check the target checkpoint against the
    # box while the box is still reachable.
    recorded = results[target]["source_sha256"]
    box_sha = remote_sha256(target)
    if not box_sha:
        log(f"  note: could not read box sha256 for step-{target}; relying on the "
            f"structural checks and the pull script's own transfer hash")
    elif not recorded:
        log(f"  note: local step-{target} carries no source_sha256 metadata; "
            f"structural checks only")
    elif recorded != box_sha:
        raise FailOpen(
            f"step-{target} sha256 mismatch -- local metadata {recorded} vs box "
            f"{box_sha}. The local file is NOT the checkpoint on the box.")
    else:
        log(f"  sha256 cross-check vs box PASSED for step-{target} ({box_sha[:16]}...)")
    return results


# --------------------------------------------------------------------------- #
# Pull
# --------------------------------------------------------------------------- #

def check_disk(n_missing: int) -> None:
    if n_missing <= 0:
        return
    # The pull stages a raw copy plus a remapped .part in a tmpdir on the same
    # filesystem, so peak usage is the backfill plus two extra files.
    needed = n_missing * CKPT_BYTES + 2 * CKPT_BYTES + DISK_BUFFER_BYTES
    free = shutil.disk_usage(DEST_DIR).free
    gib = 1024**3
    log(f"disk check: {n_missing} file(s) to fetch, need ~{needed / gib:.1f} GiB "
        f"(incl. staging + {DISK_BUFFER_BYTES / gib:.0f} GiB buffer), "
        f"{free / gib:.1f} GiB free on {DEST_DIR}")
    if free < needed:
        raise FailOpen(
            f"not enough free space for the backfill: need ~{needed / gib:.1f} GiB, "
            f"have {free / gib:.1f} GiB on {DEST_DIR}. Free space and re-run; the "
            f"instance is being left RUNNING so nothing is lost.")


def run_pull() -> None:
    """Invoke pull_latest_lora.py --all. Child output goes straight to the log file
    handle: no pipe, so no ${PIPESTATUS} ambiguity."""
    cmd = [*PULL_ARGV, "--all", "--dest-dir", str(DEST_DIR)]
    log(f"running backfill pull: {' '.join(cmd)}")
    env_out = dict(os.environ)
    env_out.setdefault("LORA_SSH_HOST", SSH_HOST)
    env_out.setdefault("LORA_SSH_PORT", SSH_PORT)
    env_out.setdefault("LORA_SSH_KEY", SSH_KEY)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(f"----- pull_latest_lora.py --all output begins {now_iso()} -----\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=PULL_TIMEOUT, env=env_out)
        except subprocess.TimeoutExpired:
            fh.write(f"----- pull TIMED OUT after {PULL_TIMEOUT}s -----\n")
            raise FailOpen(f"pull_latest_lora.py exceeded {PULL_TIMEOUT}s")
        fh.write(f"----- pull output ends rc={proc.returncode} {now_iso()} -----\n")
    if proc.returncode != 0:
        raise FailOpen(f"pull_latest_lora.py exited {proc.returncode} -- see the pull "
                       f"output above in this log")
    log("pull reported success (rc=0)")


# --------------------------------------------------------------------------- #
# vastai -- stop only. There is deliberately no destroy path in this file.
# --------------------------------------------------------------------------- #

def vastai_json(args: list[str]) -> dict:
    cmd = [VASTAI_BIN, *args, "--raw"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise FailOpen(f"vastai {' '.join(args)} unusable: {exc}")
    if proc.returncode != 0:
        raise FailOpen(f"vastai {' '.join(args)} exited {proc.returncode}: "
                       f"{(proc.stderr or proc.stdout or '').strip()[:300]}")
    text = proc.stdout
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise FailOpen(f"vastai {' '.join(args)} produced no JSON: {text.strip()[:200]}")
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise FailOpen(f"vastai {' '.join(args)} JSON unparseable: {exc}")


def instance_status() -> str:
    d = vastai_json(["show", "instance", VAST_INSTANCE])
    if isinstance(d, list):
        d = next((x for x in d if str(x.get("id")) == VAST_INSTANCE), {})
    return str(d.get("actual_status") or d.get("cur_state") or "unknown")


STOPPED_STATES = {"stopped", "exited", "offline", "inactive"}


def stop_instance() -> None:
    status = instance_status()
    log(f"instance {VAST_INSTANCE} ({VAST_LABEL}) current status: {status}")
    if status in STOPPED_STATES:
        log("instance is already stopped -- not issuing a second stop (idempotent)")
        save_state(stopped_at=load_state().get("stopped_at") or now_iso(),
                   observed_status=status)
        return

    cmd = [VASTAI_BIN, "stop", "instance", VAST_INSTANCE]
    if PRINT_ONLY:
        log(f"PRINT-ONLY MODE: would run: {' '.join(cmd)}")
        save_state(print_only_stop_at=now_iso(), stop_cmd=" ".join(cmd))
        return

    log(f"issuing STOP (never destroy): {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise FailOpen(f"vastai stop could not be run: {exc}")
    log(f"vastai stop rc={proc.returncode} out={(proc.stdout or '').strip()[:300]} "
        f"err={(proc.stderr or '').strip()[:300]}")
    if proc.returncode != 0:
        raise FailOpen(f"vastai stop exited {proc.returncode}; instance may still be "
                       f"running -- check `vastai show instances`")

    save_state(stop_issued_at=now_iso())
    deadline = time.time() + STOP_CONFIRM_SECS
    observed = "unknown"
    while time.time() < deadline:
        time.sleep(STOP_POLL_SECS)
        try:
            observed = instance_status()
        except FailOpen as exc:
            log(f"  status query hiccup: {exc}")
            continue
        log(f"  observed status: {observed}")
        if observed in STOPPED_STATES:
            log(f"CONFIRMED: instance {VAST_INSTANCE} reached state '{observed}'. "
                f"Billing drops from ~$1.1022/hr to storage-only (~$0.22/hr).")
            save_state(stopped_at=now_iso(), observed_status=observed)
            return
    alert("stop_unconfirmed",
          f"stop was issued but the instance still reports '{observed}' after "
          f"{STOP_CONFIRM_SECS}s. Checkpoints are pulled and verified, so data is "
          f"safe, but VERIFY BILLING MANUALLY: vastai show instances", force=True)
    save_state(stop_issued_at=load_state().get("stop_issued_at"),
               observed_status=observed)


# --------------------------------------------------------------------------- #
# The sequence that runs once training has stopped
# --------------------------------------------------------------------------- #

def do_pull_verify_stop(probe: dict) -> bool:
    """Pull -> verify -> stop. Returns True when the instance stop path completed.
    Any FailOpen propagates and leaves the instance running."""
    remote_steps = sorted(probe["ckpts"])
    log(f"remote checkpoints: {len(remote_steps)} "
        f"(step-{remote_steps[0]} .. step-{remote_steps[-1]})")

    missing = [s for s in remote_steps if not (DEST_DIR / dest_name(s)).exists()]
    log(f"not yet local: {', '.join(f'step-{s}' for s in missing) or 'none'}")
    check_disk(len(missing))

    run_pull()
    verify_all(remote_steps, TARGET_STEP)
    log(f"ALL {len(remote_steps)} checkpoints pulled and verified, including "
        f"step-{TARGET_STEP}. Nothing left to retrieve from the box.")
    save_state(pull_verified_at=now_iso(), verified_steps=remote_steps)

    # Preserve the box-side audit trail locally before SSH goes away forever.
    for line in probe.get("stoplog", []):
        log(f"  box stop_at_step.log | {line}")
    if probe.get("sentinel"):
        for line in probe["sentinel"].splitlines():
            log(f"  box sentinel | {line}")

    stop_instance()
    return True


# --------------------------------------------------------------------------- #
# Classification of each poll
# --------------------------------------------------------------------------- #

class StallTracker:
    def __init__(self) -> None:
        self.last_step = -1
        self.last_mtime = -1
        self.since = time.time()

    def update(self, step: int, mtime: int) -> float:
        """Returns seconds since the heartbeat last advanced."""
        if step != self.last_step or mtime != self.last_mtime:
            self.last_step, self.last_mtime = step, mtime
            self.since = time.time()
        return time.time() - self.since


def main() -> int:
    log("=" * 72)
    log("post_stop_watcher start (LOCAL companion to the box-side stop_at_step.sh)")
    log(f"  instance      : {VAST_INSTANCE} ({VAST_LABEL}) via {VASTAI_BIN}")
    log(f"  box           : {SSH_USER}@{SSH_HOST}:{SSH_PORT}  (read-only polling)")
    log(f"  target step   : {TARGET_STEP} (cumulative, from step-N.safetensors names)")
    log(f"  dest dir      : {DEST_DIR}")
    log(f"  pull          : {' '.join(PULL_ARGV)} --all")
    log(f"  poll/stall    : {POLL_SECS}s / stall alert after {STALL_SECS}s")
    log(f"  print_only    : {PRINT_ONLY}  (destroy is NOT implemented in this file)")
    log(f"  fail-open     : ANY pull/verify/SSH/state problem leaves the instance "
        f"RUNNING and logs an ALERT")

    if not take_lock():
        return 0

    state = load_state()
    if state.get("stopped_at"):
        log(f"state file says the instance was already stopped at "
            f"{state['stopped_at']}; confirming with vastai")
        try:
            status = instance_status()
        except FailOpen as exc:
            log(f"could not confirm ({exc}); exiting without action")
            return 0
        if status in STOPPED_STATES:
            log(f"confirmed status '{status}' -- work already complete, nothing to do")
            return 0
        log(f"status is '{status}', not stopped: the run may have been restarted. "
            f"Clearing the stop marker and resuming the watch.")
        save_state(stopped_at=None, note="stop marker cleared; instance was running")

    stall = StallTracker()
    ssh_fails = 0
    confirmed = 0
    dead_early_pull_done = bool(load_state().get("dead_early_pull_at"))
    polls = 0

    while True:
        polls += 1
        if MAX_POLLS and polls > MAX_POLLS:
            log(f"MAX_POLLS={MAX_POLLS} reached; exiting (test mode)")
            return 0

        try:
            probe = remote_probe()
            ssh_fails = 0
        except FailOpen as exc:
            ssh_fails += 1
            log(f"poll {polls}: {exc} (consecutive SSH failures: {ssh_fails})")
            if ssh_fails >= 10:
                alert("ssh_down",
                      f"{ssh_fails} consecutive SSH failures to {SSH_HOST}:{SSH_PORT}. "
                      f"Not stopping the instance -- an unreachable box cannot be "
                      f"verified, and a stop would end all chance of retrieval.")
            time.sleep(POLL_SECS)
            continue

        newest = probe["newest"]
        trainers = probe["trainers"]
        target_present = newest >= TARGET_STEP and TARGET_STEP in probe["ckpts"]
        stalled_for = stall.update(probe["hb_step"], probe["hb_mtime"])

        log(f"poll {polls}: newest=step-{newest} hb_step={probe['hb_step']} "
            f"trainers={trainers} supervisors={probe['supervisors']} "
            f"sentinel={'YES' if probe['sentinel_present'] else 'no'} "
            f"recent_ckpt_writes={probe['recent_writes']} "
            f"hb_static_for={int(stalled_for)}s")

        training_stopped = trainers == 0
        fired = probe["sentinel_present"] or target_present

        if training_stopped and fired:
            confirmed += 1
            log(f"  training appears STOPPED and the stop condition fired "
                f"({confirmed}/{CONFIRM_POLLS} confirmations)")
            if confirmed < CONFIRM_POLLS:
                time.sleep(min(POLL_SECS, 60))
                continue
            if probe["recent_writes"] > 0:
                log("  checkpoint dir was written to very recently; waiting for a "
                    "quiet window before touching anything")
                confirmed = max(0, confirmed - 1)
                time.sleep(POLL_SECS)
                continue
            if not target_present:
                alert("fired_without_target",
                      f"training is stopped and the sentinel is present, but there is "
                      f"no step-{TARGET_STEP} on the box (newest is step-{newest}). "
                      f"Pulling everything to preserve it, then leaving the instance "
                      f"RUNNING for a human.", force=True)
                try:
                    remote_steps = sorted(probe["ckpts"])
                    check_disk(len([s for s in remote_steps
                                    if not (DEST_DIR / dest_name(s)).exists()]))
                    run_pull()
                    log("preservation pull done; instance intentionally left RUNNING")
                except FailOpen as exc:
                    alert("preserve_failed", f"preservation pull failed: {exc}",
                          force=True)
                save_state(anomaly="fired_without_target", anomaly_at=now_iso())
                time.sleep(max(POLL_SECS, FAIL_SLEEP_SECS))
                continue

            log("-" * 72)
            log("STOP CONDITION CONFIRMED -- beginning pull / verify / stop sequence")
            log("-" * 72)
            try:
                do_pull_verify_stop(probe)
            except FailOpen as exc:
                alert("sequence_failed",
                      f"{exc}\n  The instance has NOT been stopped and is still "
                      f"RUNNING at ~$1.1022/hr on purpose: losing the final LoRA is "
                      f"far worse than a few hours of GPU rent. Fix the problem and "
                      f"re-run this watcher (it is idempotent), or pull by hand.",
                      force=True)
                save_state(last_failure=str(exc), last_failure_at=now_iso())
                time.sleep(max(POLL_SECS, FAIL_SLEEP_SECS))
                continue
            log("SEQUENCE COMPLETE -- watcher exiting")
            return 0

        confirmed = 0

        if training_stopped and not fired:
            # The box watcher does not alert on a run that dies early. This does.
            alert("dead_early",
                  f"NO trainer process on the box, no stop sentinel, and the newest "
                  f"checkpoint is only step-{newest} (< target {TARGET_STEP}). The run "
                  f"looks DEAD or was stopped by something else. The instance is being "
                  f"left RUNNING and will NOT be stopped -- a human should look.")
            if not dead_early_pull_done:
                log("  doing a one-off preservation pull of everything on the box")
                try:
                    remote_steps = sorted(probe["ckpts"])
                    check_disk(len([s for s in remote_steps
                                    if not (DEST_DIR / dest_name(s)).exists()]))
                    run_pull()
                    dead_early_pull_done = True
                    save_state(dead_early_pull_at=now_iso())
                    log("  preservation pull complete; still not stopping the instance")
                except FailOpen as exc:
                    alert("preserve_failed", f"preservation pull failed: {exc}",
                          force=True)
            time.sleep(POLL_SECS)
            continue

        if stalled_for > STALL_SECS:
            alert("stalled",
                  f"the heartbeat has not advanced for {int(stalled_for)}s "
                  f"(> {STALL_SECS}s) while {trainers} trainer process(es) are still "
                  f"alive at step {probe['hb_step']}, newest checkpoint step-{newest}. "
                  f"The run may be wedged. NOT stopping the instance; a human should "
                  f"look. Last heartbeat line: {probe['hb_line'][:200]}")

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted -- instance left in whatever state it was in (never stopped "
            "by this path)")
        raise SystemExit(130)
    except FailOpen as exc:
        alert("fatal", f"unhandled fail-open condition: {exc}. Instance left RUNNING.",
              force=True)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 - last resort; must not stop the instance
        alert("crash", f"unexpected {type(exc).__name__}: {exc}. Instance left "
                       f"RUNNING; nothing was stopped.", force=True)
        raise SystemExit(1)
