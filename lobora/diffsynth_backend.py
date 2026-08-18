"""Optional DiffSynth-Studio hooks.

Install extras: ``pip install -e '.[gpu]'``

Stage-1 (``sft:data_process``) should load Qwen3-VL + video/audio VAEs only.
Stage-2 (``sft:train``) loads the Ref2VA DiT, attaches LoRA, and reads ``.pt`` caches.

This module is imported only when DiffSynth is installed. The CPU dry-run path
never touches it.
"""

from __future__ import annotations

from typing import Any


def diffsynth_available() -> bool:
    try:
        import diffsynth  # noqa: F401

        return True
    except ImportError:
        return False


def encode_sample(_sample: Any, _row: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for MiniMaxH3Unit_* encode. Wire in a follow-up GPU PR."""
    raise RuntimeError(
        "DiffSynth encode_sample is not implemented in this checkout. "
        "Use DiffSynth examples/minimax_h3/model_training/train.py --task sft:data_process "
        "or extend this function to call InputVideoEmbedder / PromptEmbedder / ReferenceEncoder."
    )
