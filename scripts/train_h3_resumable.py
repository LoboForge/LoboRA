#!/usr/bin/env python3
"""Run DiffSynth's MiniMax-H3 trainer with real resume state.

Drop-in for ``examples/minimax_h3/model_training/train.py``: same arguments, same
``accelerate launch`` invocation, same checkpoint files. The difference is that
``lobora.diffsynth_resume.install()`` runs first, so the step counter is cumulative
across supervisor attempts and each ``step-N.safetensors`` gets a ``step-N.optim.pt``
sidecar plus a ``train_state.json`` manifest.

The upstream example is executed as-is rather than forked. It does
``from diffsynth.diffusion import *`` at import time, so patching the package
attributes before that import is what makes the swap take effect — and it means this
wrapper does not have to be re-synced every time DiffSynth updates the example.

Usage (replaces the bare example path in train_stage2_fp8.sh):

    accelerate launch --num_processes 1 --mixed_precision bf16 \\
      /workspace/LoboRA/scripts/train_h3_resumable.py \\
      --dataset_base_path ... [all the usual flags]

Exit codes:
    0   training finished
    3   resume state present but unusable -- FATAL, do not retry (see EXIT_RESUME_UNUSABLE)
    130 stopped on signal after an emergency checkpoint was written
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lobora.console import error, info  # noqa: E402
from lobora.resume_state import EXIT_RESUME_UNUSABLE, ResumeStateError  # noqa: E402

DEFAULT_EXAMPLE = "examples/minimax_h3/model_training/train.py"

#: Dropped in --output_path when a resume is refused. `accelerate launch` does not
#: always pass a child's exit code through verbatim, so the supervisor has a file to
#: check as well as a status to read.
BLOCKED_MARKER = "RESUME_BLOCKED.txt"


def argv_output_path() -> Path | None:
    argv = sys.argv
    for index, item in enumerate(argv):
        if item == "--output_path" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if item.startswith("--output_path="):
            return Path(item.split("=", 1)[1])
    return None


def locate_example() -> Path:
    """Find DiffSynth's MiniMax-H3 example script.

    ``LOBORA_H3_TRAIN_SCRIPT`` wins; otherwise look next to the installed ``diffsynth``
    package and in the cwd, which is where ``train_stage2_fp8.sh`` already cd's to.
    """
    override = os.environ.get("LOBORA_H3_TRAIN_SCRIPT")
    if override:
        path = Path(override)
        if not path.is_file():
            raise SystemExit(f"LOBORA_H3_TRAIN_SCRIPT={override} is not a file")
        return path

    candidates = [Path.cwd() / DEFAULT_EXAMPLE]
    try:
        import diffsynth

        candidates.append(Path(diffsynth.__file__).resolve().parents[1] / DEFAULT_EXAMPLE)
    except ImportError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "could not find DiffSynth's " + DEFAULT_EXAMPLE + ". Run from the DiffSynth-Studio "
        "checkout or set LOBORA_H3_TRAIN_SCRIPT."
    )


def main() -> int:
    from lobora.diffsynth_resume import install

    example = locate_example()
    info(f"wrapping {example}")
    install()

    output_path = argv_output_path()
    if output_path is not None:
        # Clear a marker from a previous refusal so it can only ever describe this run.
        (output_path / BLOCKED_MARKER).unlink(missing_ok=True)

    # runpy re-parses sys.argv, so the example sees exactly the flags we were given.
    try:
        runpy.run_path(str(example), run_name="__main__")
    except ResumeStateError as exc:
        error(str(exc))
        error(
            "refusing to restart from scratch. This is fatal on purpose: a silent restart "
            "would burn a supervisor attempt and throw away hours of training."
        )
        if output_path is not None and output_path.is_dir():
            (output_path / BLOCKED_MARKER).write_text(str(exc) + "\n", encoding="utf-8")
        return EXIT_RESUME_UNUSABLE
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
