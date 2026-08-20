#!/usr/bin/env python3
"""Assert the DiffSynth source edits reached the modules the trainer actually imports.

Run this after every install, reinstall or upgrade, and after applying
`patches/diffsynth/`. It is the only thing that catches a mis-targeted patch before the
GPU bill.

Why it exists: `bootstrap_diffsynth.sh` creates TWO independent DiffSynth trees -- a
non-editable `pip install` into the venv's site-packages, and a separate `git clone` for
the training example. Patching the wrong one is easy and, for one of the four edits,
completely silent:

  training_module.py   fp8/offload silently do not apply to a sharded model, because
                       upstream tests `path in fp8_models` while `path` is a LIST of
                       shard paths. No warning, no traceback -- the frozen 13-shard DiT
                       just loads bf16 (~62 GiB) instead of fp8_e4m3fn (~31 GiB) and the
                       run dies of an inexplicable OOM hours in. This is the check that
                       matters; the others merely fail in ways you would notice.

  runner.py            stage-1 preprocessing is not resumable and one bad sample aborts
                       the whole cache pass.
  logger.py            checkpoints restart at step-100 and overwrite each other.
  train.py (checkout)  the run refuses to start at all.

Each check reads the source of the *imported* module, so it reports on the file that will
really be loaded rather than on a path someone believes is in use.

    python scripts/vast/verify_diffsynth_patches.py
    python scripts/vast/verify_diffsynth_patches.py --checkout /workspace/DiffSynth-Studio
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

# (module, needle, what breaks without it, silent?)
IMPORTED_CHECKS = (
    (
        "diffsynth.diffusion.training_module",
        "any(shard in fp8_models",
        "fp8 and offload are SILENTLY ignored for sharded models -> VRAM blowup with no error",
        True,
    ),
    (
        "diffsynth.diffusion.runner",
        "skipped_existing=",
        "stage-1 preprocessing is not resumable and one bad sample aborts the whole pass",
        False,
    ),
    (
        "diffsynth.diffusion.logger",
        'os.environ.get("DIFFSYNTH_STEP_OFFSET")',
        "checkpoints restart at step-100 on a warm start and overwrite each other",
        False,
    ),
    (
        "diffsynth.diffusion.logger",
        "_write_heartbeat",
        "no heartbeat beacon, so pull_latest_lora.py and the watchers cannot see progress",
        False,
    ),
)

CHECKOUT_CHECK = (
    Path("examples/minimax_h3/model_training/train.py"),
    "processor_config.skip_download = True",
    "the trainer refuses to start: ValueError: No valid model files",
)


def source_of(module_name: str) -> tuple[str, Path]:
    __import__(module_name)
    module = sys.modules[module_name]
    path = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if path is None:
        raise SystemExit(f"FAIL: cannot locate source of {module_name}")
    return Path(path).read_text(encoding="utf-8"), Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", type=Path,
                    default=Path(os.environ.get("DIFFSYNTH") or "/workspace/DiffSynth-Studio"),
                    help="DiffSynth-Studio git checkout holding the training example")
    args = ap.parse_args()

    try:
        import diffsynth
    except Exception as exc:  # noqa: BLE001 - any import failure is the same problem
        raise SystemExit(f"FAIL: cannot import diffsynth ({type(exc).__name__}: {exc})")

    site = Path(diffsynth.__file__).parent.parent
    print(f"imported package : {Path(diffsynth.__file__).parent}")
    print(f"site-packages    : {site}")
    print(f"checkout         : {args.checkout}")
    print()

    failures: list[str] = []
    for module_name, needle, consequence, silent in IMPORTED_CHECKS:
        try:
            text, path = source_of(module_name)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_name}: import failed ({type(exc).__name__}: {exc})")
            print(f"  FAIL  {module_name}  (import failed)")
            continue
        if needle in text:
            print(f"  ok    {module_name}  <- {path}")
        else:
            tag = "SILENT FAILURE" if silent else "consequence"
            failures.append(f"{module_name} is missing {needle!r} in {path}\n"
                            f"          {tag}: {consequence}")
            print(f"  FAIL  {module_name}  <- {path}")

    rel, needle, consequence = CHECKOUT_CHECK
    target = args.checkout / rel
    if not target.exists():
        failures.append(f"{target} does not exist; --checkout points at the wrong tree")
        print(f"  FAIL  {rel}  (not found under {args.checkout})")
    elif needle in target.read_text(encoding="utf-8"):
        print(f"  ok    {rel}  <- {target}")
    else:
        failures.append(f"{target} is missing {needle!r}\n          consequence: {consequence}")
        print(f"  FAIL  {rel}  <- {target}")

    print()
    if not failures:
        print("ALL PATCHES PRESENT in the trees that will actually be used.")
        return 0

    # Loud on purpose. The fp8 failure mode emits nothing at all on its own, so this
    # banner is the only signal that exists before the OOM.
    bar = "!" * 78
    print(bar, file=sys.stderr)
    print(f"!! DIFFSYNTH PATCHES MISSING -- DO NOT START TRAINING ({len(failures)} problem(s))",
          file=sys.stderr)
    print(bar, file=sys.stderr)
    for problem in failures:
        print(f"  * {problem}", file=sys.stderr)
    print(bar, file=sys.stderr)
    print("Fix: apply patches/diffsynth/checkout/*.diff to the CHECKOUT and "
          "patches/diffsynth/site-packages/*.diff to SITE-PACKAGES, then re-run this.",
          file=sys.stderr)
    print(bar, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
