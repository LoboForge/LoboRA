#!/usr/bin/env python3
"""Pull the latest MiniMax-H3 LoRA checkpoint off the Vast training box into ComfyUI.

Read-only on the box. Remaps adapter keys into the form ComfyUI's generic
`model_lora_keys_unet` path expects (prefix `diffusion_model.`, drop `.default`),
verifies the weight payload byte-for-byte, and installs atomically.

Usage:
    scripts/pull_latest_lora.py            # fetch the newest checkpoint (no-op if current)
    scripts/pull_latest_lora.py --all      # backfill every checkpoint missing locally
    scripts/pull_latest_lora.py 700        # fetch one specific step
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Vast box -- PORT CHANGES ON EVERY INSTANCE RESTART. Re-resolve with:
#   vastai show instances        (look for label lobora-h3-a800)
# Edit these, or override for one run with LORA_SSH_HOST / LORA_SSH_PORT.
# --------------------------------------------------------------------------- #
SSH_HOST = os.environ.get("LORA_SSH_HOST", "ssh9.vast.ai")
SSH_PORT = os.environ.get("LORA_SSH_PORT", "16192")
SSH_USER = os.environ.get("LORA_SSH_USER", "root")
SSH_KEY = os.path.expanduser(os.environ.get("LORA_SSH_KEY", "~/.ssh/vast_tmp"))
VAST_INSTANCE = "48056192"
VAST_LABEL = "lobora-h3-a800"

SOURCE_RUN = "anatomy_ref2va_a800"
REMOTE_LORA_DIR = f"/workspace/output/{SOURCE_RUN}/lora"
REMOTE_HEARTBEAT = "/workspace/logs/anatomy_heartbeat.txt"
REMOTE_TRAIN_STATE = f"{REMOTE_LORA_DIR}/train_state.json"
CKPT_EVERY = 25  # a new checkpoint lands every N micro-steps

# Step numbers on the box are CUMULATIVE across restarts, so step-N means "N
# micro-steps of training total": the numbering is a lineage and genpt-step-NNNN
# sorts chronologically in ComfyUI. A given step-N is written exactly once, so a
# local copy can never be silently superseded by different weights under the same
# name.
#
# Runs launched through scripts/train_h3_resumable.py also leave a train_state.json
# manifest and a step-N.optim.pt sidecar (Adam moments + LR-scheduler position) beside
# each adapter. This script only ever fetches the .safetensors -- the sidecars are for
# the trainer, not for ComfyUI -- but it reads the manifest when present because that
# is the authoritative cumulative step. Runs from before that patch have neither file
# and warm-restart their optimizer state; everything here still works, it just falls
# back to the heartbeat.
#
# Files parked aside by hand (attempt1_step-N.safetensors) are deliberately not
# matched by remote_checkpoints(): its regex requires whitespace before "step-".

# --------------------------------------------------------------------------- #
# Local destination
# --------------------------------------------------------------------------- #
DEST_DIR = Path("/media/wrath/AI/ComfyUI/models/loras/minimax-h3")
DEST_PREFIX = "genpt-step-"
STEP_PAD = 4

# Key remap, matched to ComfyUI comfy/lora.py::model_lora_keys_unet (generic
# branch) + comfy/weight_adapter/lora.py::LoRAAdapter.load.
KEY_PREFIX = "diffusion_model."
REMAP_NOTE = "strip .default + prefix diffusion_model."

EXPECTED_TENSORS = 208
MIN_PLAUSIBLE_BYTES = 100 * 1024 * 1024  # a mid-write partial is far smaller

# Interpreters known to carry `safetensors` (used only for an extra
# open-with-the-real-library check; the remap itself is pure stdlib).
CANDIDATE_PYTHONS = (
    "/media/wrath/AI/ComfyUI/venv/bin/python",
    "/media/wrath/SSD2TB/Development/LoboRA/.venv/bin/python",
)
REEXEC_GUARD = "_PULL_LATEST_LORA_REEXEC"

SSH_BASE = [
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=20",
    "-i", SSH_KEY,
    "-p", SSH_PORT,
]


class Failure(RuntimeError):
    """Anything that should abort the run with a human-readable message."""


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# safetensors availability (optional, for a belt-and-braces validation)
# --------------------------------------------------------------------------- #

def ensure_safetensors_interpreter() -> None:
    """Re-exec under an interpreter that has safetensors, if this one lacks it."""
    import importlib.util

    if importlib.util.find_spec("safetensors") is not None:
        return
    if os.environ.get(REEXEC_GUARD):
        return
    for py in CANDIDATE_PYTHONS:
        if not os.access(py, os.X_OK):
            continue
        probe = subprocess.run(
            [py, "-c", "import safetensors"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            os.environ[REEXEC_GUARD] = "1"
            os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])
    log("note: no interpreter with `safetensors` found; using stdlib validation only")


def safetensors_open_check(path: Path) -> str:
    """Open the file with the real safetensors library (header only, no weights)."""
    try:
        from safetensors import safe_open
    except Exception:
        return "skipped (safetensors unavailable)"
    last: Exception | None = None
    for framework in ("numpy", "pt"):
        try:
            with safe_open(str(path), framework=framework) as f:
                keys = list(f.keys())
                meta = f.metadata() or {}
            return f"ok ({len(keys)} keys, {len(meta)} metadata fields)"
        except Exception as exc:  # noqa: BLE001 - try the next framework
            last = exc
    raise Failure(f"safetensors refused to open {path.name}: {last}")


# --------------------------------------------------------------------------- #
# Remote side (strictly read-only: ls / stat / sha256sum / cat)
# --------------------------------------------------------------------------- #

def ssh_capture(remote_cmd: str) -> str:
    proc = subprocess.run(
        [*SSH_BASE, f"{SSH_USER}@{SSH_HOST}", remote_cmd],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise Failure(
            f"SSH to {SSH_USER}@{SSH_HOST}:{SSH_PORT} failed (exit {proc.returncode}).\n"
            f"  {proc.stderr.strip() or '(no stderr)'}\n"
            f"Vast reassigns the SSH port whenever the instance restarts. Re-resolve it with\n"
            f"  vastai show instances    # instance {VAST_INSTANCE}, label {VAST_LABEL}\n"
            f"then edit SSH_HOST / SSH_PORT at the top of {Path(__file__).name}, or retry with\n"
            f"  LORA_SSH_PORT=<newport> {Path(__file__).name}"
        )
    return proc.stdout


def remote_checkpoints() -> dict[int, int]:
    """step -> size in bytes, for every step-N.safetensors on the box."""
    out = ssh_capture(f"ls -l --time-style=+%s {REMOTE_LORA_DIR} 2>/dev/null || true")
    found: dict[int, int] = {}
    for line in out.splitlines():
        m = re.search(r"\s(\d+)\s+\d+\s+step-(\d+)\.safetensors$", line)
        if m:
            found[int(m.group(2))] = int(m.group(1))
    if not found:
        raise Failure(f"no step-*.safetensors found in {REMOTE_LORA_DIR} on the box")
    return found


def remote_heartbeat() -> dict[str, str]:
    """Last heartbeat line as key=value pairs (step, loss, attempt, ...)."""
    out = ssh_capture(f"tail -n 5 {REMOTE_HEARTBEAT} 2>/dev/null || true")
    lines = [ln for ln in out.splitlines() if "step=" in ln]
    if not lines:
        return {}
    return dict(re.findall(r"([A-Za-z_]+)=(\S+)", lines[-1]))


def remote_train_state() -> dict:
    """train_state.json from the box, or {} for a run that predates resume state."""
    out = ssh_capture(f"cat {REMOTE_TRAIN_STATE} 2>/dev/null || true").strip()
    if not out:
        return {}
    try:
        state = json.loads(out)
    except json.JSONDecodeError as exc:
        log(f"note: train_state.json on the box is unparseable ({exc}); ignoring it")
        return {}
    return state if isinstance(state, dict) else {}


def remote_size(step: int) -> int:
    out = ssh_capture(f"stat -c %s {REMOTE_LORA_DIR}/step-{step}.safetensors").strip()
    return int(out)


def remote_sha256(step: int) -> str:
    out = ssh_capture(f"sha256sum {REMOTE_LORA_DIR}/step-{step}.safetensors").split()
    return out[0]


def wait_for_stable_size(step: int, first_size: int, settle: float) -> int:
    """Refuse to copy a checkpoint that is still being written."""
    if first_size < MIN_PLAUSIBLE_BYTES:
        raise Failure(
            f"step-{step} is only {human(first_size)} on the box -- it is still being "
            f"written. Wait a minute and re-run."
        )
    log(f"  checking step-{step} is fully written ({settle:.0f}s settle)...")
    time.sleep(settle)
    again = remote_size(step)
    if again != first_size:
        raise Failure(
            f"step-{step} grew from {first_size} to {again} bytes during the settle "
            f"window -- it is mid-write. Wait a minute and re-run."
        )
    return again


# --------------------------------------------------------------------------- #
# safetensors header handling (pure stdlib; payload is copied verbatim)
# --------------------------------------------------------------------------- #

def read_header(path: Path) -> tuple[int, dict]:
    size = path.stat().st_size
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise Failure(f"{path.name} is truncated (no safetensors header length)")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or 8 + header_len > size:
            raise Failure(
                f"{path.name} declares a {header_len}-byte header but the file is only "
                f"{size} bytes -- truncated or corrupt"
            )
        try:
            header = json.loads(f.read(header_len))
        except json.JSONDecodeError as exc:
            raise Failure(f"{path.name} has an unparseable safetensors header: {exc}")
    return header_len, header


def validate(path: Path, expect_remapped: bool) -> dict:
    """Structural check: header parses, offsets are contiguous, size adds up."""
    header_len, header = read_header(path)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(tensors) != EXPECTED_TENSORS:
        raise Failure(
            f"{path.name} has {len(tensors)} tensors, expected {EXPECTED_TENSORS}"
        )
    end = 0
    for name in tensors:
        info = tensors[name]
        lo, hi = info["data_offsets"]
        if lo != end:
            raise Failure(f"{path.name}: non-contiguous tensor data at {name}")
        end = hi
        if expect_remapped:
            if not name.startswith(KEY_PREFIX):
                raise Failure(f"{path.name}: key not remapped: {name}")
            if ".default." in name:
                raise Failure(f"{path.name}: key still carries .default: {name}")
    actual = path.stat().st_size
    if actual != 8 + header_len + end:
        raise Failure(
            f"{path.name} is {actual} bytes but header+payload require "
            f"{8 + header_len + end} -- truncated transfer"
        )
    return {"header_len": header_len, "payload_bytes": end, "tensors": tensors}


def remap_key(key: str) -> str:
    key = re.sub(r"\.default(?=\.|$)", "", key)
    if not key.startswith(KEY_PREFIX):
        key = KEY_PREFIX + key
    return key


def sha256_range(path: Path, start: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_remapped(raw: Path, out: Path, step: int, source_sha: str) -> str:
    """Rewrite the header with ComfyUI-shaped keys; copy the payload byte for byte."""
    header_len, header = read_header(raw)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    new_header: dict = {
        "__metadata__": {
            "format": "pt",
            "step": str(step),
            "source_run": SOURCE_RUN,
            "source_file": f"step-{step}.safetensors",
            "source_sha256": source_sha,
            "remap": REMAP_NOTE,
        }
    }
    for name, info in tensors.items():
        new_name = remap_key(name)
        if new_name in new_header:
            raise Failure(f"remap collision on {new_name} (from {name})")
        new_header[new_name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": info["data_offsets"],
        }

    blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)  # safetensors 8-byte data alignment

    payload = hashlib.sha256()
    with raw.open("rb") as src, out.open("wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        src.seek(8 + header_len)
        while chunk := src.read(1 << 20):
            payload.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    return payload.hexdigest()


# --------------------------------------------------------------------------- #
# Fetch + install
# --------------------------------------------------------------------------- #

def dest_name(step: int) -> str:
    return f"{DEST_PREFIX}{step:0{STEP_PAD}d}.safetensors"


def human(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def rsync_one(step: int, target: Path) -> None:
    remote = f"{SSH_USER}@{SSH_HOST}:{REMOTE_LORA_DIR}/step-{step}.safetensors"
    ssh_cmd = " ".join(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=20", "-i", SSH_KEY, "-p", SSH_PORT]
    )
    cmd = ["rsync", "-rtvP", "--timeout=180", "-e", ssh_cmd, remote, str(target)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise Failure(
            f"rsync of step-{step} failed (exit {proc.returncode}). If this is a "
            f"connection error the Vast SSH port may have changed -- see "
            f"`vastai show instances` for instance {VAST_INSTANCE}."
        )


def fetch_step(step: int, remote_bytes: int, dest_dir: Path, settle: float) -> None:
    final = dest_dir / dest_name(step)
    remote_bytes = wait_for_stable_size(step, remote_bytes, settle)

    log("  hashing the source on the box...")
    source_sha = remote_sha256(step)

    tmpdir = dest_dir / f".pull_latest_lora.{os.getpid()}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    raw = tmpdir / f"step-{step}.raw"
    staged = tmpdir / f"{final.name}.part"
    try:
        rsync_one(step, raw)

        got = raw.stat().st_size
        if got != remote_bytes:
            raise Failure(
                f"transferred {got} bytes but the box reports {remote_bytes} -- "
                f"incomplete transfer, nothing installed"
            )
        local_sha = sha256_range(raw, 0)
        if local_sha != source_sha:
            raise Failure(
                f"sha256 mismatch after transfer\n  box:   {source_sha}\n"
                f"  local: {local_sha}\nnothing installed"
            )
        raw_info = validate(raw, expect_remapped=False)

        payload_sha = write_remapped(raw, staged, step, source_sha)
        out_info = validate(staged, expect_remapped=True)
        if out_info["payload_bytes"] != raw_info["payload_bytes"]:
            raise Failure("payload size changed during remap -- refusing to install")
        installed_payload = sha256_range(staged, 8 + out_info["header_len"])
        if installed_payload != payload_sha:
            raise Failure("payload hash mismatch after write -- refusing to install")
        st_check = safetensors_open_check(staged)

        os.chmod(staged, 0o644)
        os.replace(staged, final)

        log(f"  installed  : {final}")
        log(f"  size       : {final.stat().st_size} bytes ({human(final.stat().st_size)})")
        log(f"  source     : step-{step}.safetensors ({SOURCE_RUN})")
        log(f"  source sha : {source_sha}")
        log(f"  payload sha: {payload_sha}  (identical bytes to source)")
        log(f"  verified   : YES -- transfer sha256 match, "
            f"{len(out_info['tensors'])} tensors remapped, safetensors open {st_check}")
        log(f"  remap      : {REMAP_NOTE}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def local_ok(dest_dir: Path, step: int, deep: bool) -> bool:
    """True if a valid local copy of this step already exists.

    `deep` also confirms the local file came from the checkpoint currently on the
    box: the run can restart and rewrite step-N with different weights.
    """
    path = dest_dir / dest_name(step)
    if not path.exists():
        return False
    try:
        validate(path, expect_remapped=True)
    except Failure as exc:
        log(f"note: existing {path.name} is invalid ({exc}); refetching")
        return False
    if not deep:
        return True
    _, header = read_header(path)
    recorded = (header.get("__metadata__") or {}).get("source_sha256")
    current = remote_sha256(step)
    if recorded != current:
        log(f"note: {path.name} was built from a different step-{step} on the box; "
            f"refetching. Under cumulative numbering a step-N is written once, so this "
            f"means either a pre-cumulative run rewrote it or the file changed on disk")
        return False
    return True


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull the latest MiniMax-H3 LoRA checkpoint from the Vast box "
                    "into ComfyUI, remapped so ComfyUI actually loads it."
    )
    parser.add_argument("steps", nargs="*", type=int,
                        help="explicit step number(s) to fetch, e.g. 700")
    parser.add_argument("--all", action="store_true",
                        help="backfill every remote checkpoint missing locally")
    parser.add_argument("--force", action="store_true",
                        help="refetch even if a valid local copy exists")
    parser.add_argument("--dest-dir", type=Path, default=DEST_DIR,
                        help=f"install directory (default {DEST_DIR})")
    parser.add_argument("--settle", type=float, default=6.0,
                        help="seconds to confirm the remote file size is stable")
    args = parser.parse_args()

    dest_dir = args.dest_dir
    if not dest_dir.is_dir():
        log(f"error: destination {dest_dir} does not exist")
        return 2

    log(f"box   : {SSH_USER}@{SSH_HOST}:{SSH_PORT}  ({VAST_LABEL}, instance {VAST_INSTANCE})")
    log(f"remote: {REMOTE_LORA_DIR}")
    log(f"dest  : {dest_dir}")

    remote = remote_checkpoints()
    latest = max(remote)
    log(f"remote checkpoints: {len(remote)} (step-{min(remote)} .. step-{latest})")

    state = remote_train_state()
    if state:
        optim = state.get("optimizer_state") or "none"
        log(f"resume state: cumulative step {state.get('cumulative_step', '?')}"
            f"/{state.get('total_steps', '?')} from {state.get('checkpoint', '?')} "
            f"(attempt {state.get('attempt', '?')}, optimizer sidecar {optim})")
    else:
        log("resume state: none on the box (run predates train_state.json; a restart "
            "warm-starts its optimizer moments)")

    hb = remote_heartbeat()
    if "step" in hb:
        live = int(hb["step"])
        attempt = hb.get("attempt", "?")
        nxt = ((live // CKPT_EVERY) + 1) * CKPT_EVERY
        log(f"training live at micro-step {live} (attempt {attempt}, "
            f"loss {hb.get('loss', '?')}); next checkpoint at step-{nxt}")
        if live >= latest:
            log(f"  -> {live - latest} steps past step-{latest}, "
                f"{nxt - live} to go until the next one")
        elif state:
            log(f"  -> heartbeat counter ({live}) is behind step-{latest}; with cumulative "
                f"numbering that means the heartbeat is stale, not that step-{latest} "
                f"will be rewritten")
        else:
            log(f"  -> counter is BEHIND step-{latest}: this pre-cumulative run restarted, "
                f"so the box may rewrite step-{latest} with different weights")
    else:
        log("training heartbeat unreadable (log missing or rotated)")

    if args.steps:
        wanted = sorted(set(args.steps))
        missing = [s for s in wanted if s not in remote]
        if missing:
            log(f"error: step(s) {missing} are not on the box; available: {sorted(remote)}")
            return 2
    elif args.all:
        wanted = sorted(remote)
    else:
        wanted = [latest]

    if args.force:
        todo = wanted
    else:
        # Confirm-against-the-box for the newest / explicitly requested steps; for a
        # bulk backfill trust a structurally valid local file for the older ones.
        deep_all = not args.all
        todo = [s for s in wanted
                if not local_ok(dest_dir, s, deep=deep_all or s == latest)]

    if not todo:
        if len(wanted) == 1:
            log(f"already up to date (step {wanted[0]} -> {dest_name(wanted[0])})")
        else:
            log(f"already up to date (all {len(wanted)} checkpoints present and valid)")
        return 0

    log(f"to fetch: {', '.join(f'step-{s}' for s in todo)}")
    for step in todo:
        log("")
        log(f"step-{step} -> {dest_name(step)}")
        fetch_step(step, remote[step], dest_dir, args.settle)

    log("")
    log(f"done: {len(todo)} checkpoint(s) installed in {dest_dir}")
    return 0


if __name__ == "__main__":
    ensure_safetensors_interpreter()
    try:
        raise SystemExit(main())
    except Failure as exc:
        log(f"\nERROR: {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        log("\ninterrupted -- nothing installed")
        raise SystemExit(130)
