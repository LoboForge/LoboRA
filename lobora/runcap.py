#!/usr/bin/env python3
"""runcap -- read the run's cumulative step cap from the one place it is declared.

WHY NOT JUST A CONSTANT IN EACH FILE
  The cap has moved twice in one week (2000 -> 6000 -> 5500) and it is read by
  three programs in two languages: scripts/vast/stop_at_step.sh stops training at
  it, scripts/post_stop_watcher.py decides the run is over at it, and
  scripts/lora_pull_watcher.py keeps pulling until it. A copy of the number in
  each file is a copy that will eventually be wrong, and the failure is silent
  rather than loud: a watcher whose target is a step the run passed hours ago
  concludes that a live run has already finished, and stops pulling.

  So the number lives in scripts/vast/h3_env.sh, which the box-side scripts
  already source, and this module parses that same line for the Python side.

PRECEDENCE
  explicit argument  >  the caller's own environment variable  >  h3_env.sh
  >  built-in fallback. The environment variable is what an operator overrides
  for a one-off, and how the self-tests pin a scenario to a chosen step.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Matches the declaration in h3_env.sh:  STOP_TARGET_STEP=${STOP_TARGET_STEP:-5500}
_DECL_RE = re.compile(r"^\s*STOP_TARGET_STEP=\$\{STOP_TARGET_STEP:-(\d+)\}", re.M)

DEFAULT_ENV_FILE = (Path(__file__).resolve().parent.parent
                    / "scripts" / "vast" / "h3_env.sh")

# Only reached if h3_env.sh is unreadable, e.g. this file copied somewhere alone.
FALLBACK_TARGET_STEP = 5500


def from_env_file(path: str | os.PathLike | None = None) -> int:
    """The declared cap, or -1 when the declaration cannot be found."""
    p = Path(path) if path else DEFAULT_ENV_FILE
    try:
        text = p.read_text()
    except OSError:
        return -1
    m = _DECL_RE.search(text)
    return int(m.group(1)) if m else -1


def target_step(env_var: str, *, override: int | None = None,
                env_file: str | os.PathLike | None = None,
                fallback: int = FALLBACK_TARGET_STEP) -> int:
    """Resolve the cap for one caller. `env_var` is that caller's own knob."""
    if override is not None:
        return override
    raw = os.environ.get(env_var, "")
    if raw.strip().isdigit():
        return int(raw)
    declared = from_env_file(env_file)
    return declared if declared > 0 else fallback


def source_of(env_var: str, env_file: str | os.PathLike | None = None) -> str:
    """Where the number came from, so the startup banner can say so."""
    if os.environ.get(env_var, "").strip().isdigit():
        return f"${env_var}"
    if from_env_file(env_file) > 0:
        p = Path(env_file) if env_file else DEFAULT_ENV_FILE
        return f"{p.name}:STOP_TARGET_STEP"
    return "built-in fallback"
