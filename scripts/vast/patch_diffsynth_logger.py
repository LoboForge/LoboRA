#!/usr/bin/env python3
"""Patch the installed DiffSynth `ModelLogger` with two ops hooks. Idempotent.

DiffSynth's trainer gives an unattended run nothing to hold on to, so this adds:

  DIFFSYNTH_STEP_OFFSET   base for the step counter. A warm start (--lora_checkpoint)
                          restarts counting at 0, so the next save is `step-100` again
                          and it OVERWRITES the previous attempt's step-100 with
                          different weights. Setting the offset to the resumed
                          checkpoint's number continues the lineage instead.
                          Numbering only: this does not restore optimizer state.

  DIFFSYNTH_HEARTBEAT_FILE  plain-text progress beacon (step, loss, VRAM peak and
                          headroom, attempt) rewritten every micro-step, plus a
                          .jsonl append log. Lets you check a run with one `cat`
                          instead of trawling a multi-GB training log.

Both are no-ops unless the environment variable is set, so a patched install behaves
exactly like a stock one for anybody else.

Idempotency is keyed on `_write_heartbeat`, which only this script adds. It must NOT be
keyed on `DIFFSYNTH_STEP_OFFSET`: the step-offset hunk is also carried by
`patches/diffsynth/site-packages/diffsynth_diffusion.diff`, so on a tree that already has
that patch the script would report "already patched" and skip the heartbeat in silence --
and `scripts/pull_latest_lora.py` plus both watchers depend on the heartbeat. The
step-offset hunk is skipped individually when it is already present.

This patches the file that `import diffsynth` resolves to, i.e. the venv's site-packages,
which is the copy the trainer imports -- not the git checkout. Run it AFTER installing
diffsynth, and again after every reinstall or upgrade: pip overwrites site-packages and
silently takes the hooks with it.

    python scripts/vast/patch_diffsynth_logger.py            # patch
    python scripts/vast/patch_diffsynth_logger.py --check    # exit 1 if unpatched
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

# Idempotency marker. It must be unique to THIS script's contribution, so
# `_write_heartbeat` and not `DIFFSYNTH_STEP_OFFSET`: the step-offset hunk is also carried
# by patches/diffsynth/site-packages/diffsynth_diffusion.diff. Keyed off the env var name,
# a tree that already had that patch applied read as "already patched" and the heartbeat
# was skipped in silence -- and pull_latest_lora.py and both watchers depend on it.
MARKER = "_write_heartbeat"
OFFSET_MARKER = 'os.environ.get("DIFFSYNTH_STEP_OFFSET")'

OFFSET_ANCHOR = "        self.num_steps = 0\n"
OFFSET_PATCH = """        # Cumulative-step base so a warm-started run continues the checkpoint
        # lineage (step-700, step-800, ...) instead of restarting at step-100 and
        # clobbering the previous attempt's files. Adapter weights carry over on a
        # warm start; optimizer/scheduler state does not (DiffSynth saves neither).
        self.num_steps = int(os.environ.get("DIFFSYNTH_STEP_OFFSET") or 0)
"""

STEP_END_ANCHOR = "        self.num_steps += 1\n"
STEP_END_PATCH = (
    "        self.num_steps += 1\n"
    "        self._write_heartbeat(accelerator, kwargs.get(\"loss\"))\n"
)

HEARTBEAT_ANCHOR = "    def on_epoch_end("
HEARTBEAT_PATCH = '''    def _write_heartbeat(self, accelerator: Accelerator, loss=None):
        # Plain-text progress beacon so an unattended run can be checked with one `cat`.
        # Never raises: a broken beacon must not take the training down with it.
        path = os.environ.get("DIFFSYNTH_HEARTBEAT_FILE")
        if not path or not accelerator.is_main_process:
            return
        try:
            import json
            import time
            try:
                loss_value = float(loss.detach().float().item())
            except Exception:
                loss_value = float("nan")
            peak = torch.cuda.max_memory_allocated() / 2**30
            current = torch.cuda.memory_allocated() / 2**30
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            attempt = os.environ.get("ANATOMY_ATTEMPT", "?")
            line = (
                "ts=%s step=%d loss=%.5f vram_current_gib=%.2f vram_peak_gib=%.2f "
                "vram_total_gib=%.2f headroom_gib=%.2f attempt=%s\\n"
            ) % (
                time.strftime("%Y-%m-%dT%H:%M:%S%z"), self.num_steps, loss_value,
                current, peak, total, total - peak, attempt,
            )
            with open(path, "w") as f:
                f.write(line)
            with open(path + ".jsonl", "a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "step": self.num_steps, "loss": loss_value,
                    "vram_peak_gib": round(peak, 3), "vram_current_gib": round(current, 3),
                    "attempt": attempt,
                }) + "\\n")
        except Exception:
            pass

'''


def find_logger() -> Path:
    try:
        import diffsynth.diffusion.logger as mod
    except Exception as exc:  # noqa: BLE001 - any import failure is the same problem
        raise SystemExit(f"error: cannot import diffsynth.diffusion.logger ({exc})")
    return Path(mod.__file__)


def patch(text: str) -> str:
    # The heartbeat hunks are this script's own contribution and are always required.
    for anchor, name in ((STEP_END_ANCHOR, "num_steps += 1"),
                         (HEARTBEAT_ANCHOR, "def on_epoch_end(")):
        if anchor not in text:
            raise SystemExit(
                f"error: anchor {name!r} not found -- this DiffSynth version differs from "
                f"the one these hooks were written against; patch it by hand"
            )
    # The step-offset hunk is shared with patches/diffsynth/site-packages/, so it may
    # already be in place. Skip it then rather than refusing to install the heartbeat.
    if OFFSET_MARKER in text:
        print("step offset already present (from patches/diffsynth/); adding heartbeat only")
    elif OFFSET_ANCHOR in text:
        text = text.replace(OFFSET_ANCHOR, OFFSET_PATCH, 1)
    else:
        raise SystemExit(
            "error: anchor 'num_steps = 0' not found and DIFFSYNTH_STEP_OFFSET is absent -- "
            "this DiffSynth version differs from the one these hooks were written against; "
            "patch it by hand"
        )
    text = text.replace(STEP_END_ANCHOR, STEP_END_PATCH, 1)
    return text.replace(HEARTBEAT_ANCHOR, HEARTBEAT_PATCH + HEARTBEAT_ANCHOR, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logger-path", type=Path, help="override the file to patch")
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    args = ap.parse_args()

    target = args.logger_path or find_logger()
    text = target.read_text(encoding="utf-8")
    print(f"target: {target}")

    if MARKER in text:
        print("already patched (step offset + heartbeat present)")
        return 0
    if args.check:
        print("NOT patched")
        return 1

    patched = patch(text)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(patched)
        staged = Path(fh.name)
    try:
        # Refuse to install a file that does not compile; a syntax error here would
        # only surface hours later, inside the trainer.
        py_compile.compile(str(staged), doraise=True)
        backup = target.with_suffix(target.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"backup: {backup}")
        shutil.copy(staged, target)
    finally:
        staged.unlink(missing_ok=True)

    if MARKER not in target.read_text(encoding="utf-8"):
        raise SystemExit("error: patch did not take")
    print("patched: DIFFSYNTH_STEP_OFFSET + DIFFSYNTH_HEARTBEAT_FILE hooks installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
