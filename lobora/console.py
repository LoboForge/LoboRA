"""Colored console + tagged tqdm."""

from __future__ import annotations

import sys
from typing import Any, Iterable

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"


def _use_color() -> bool:
    return sys.stderr.isatty()


def _paint(color: str, text: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{RESET}"


def info(message: str) -> None:
    print(_paint(CYAN, f"[lobora] {message}"), file=sys.stderr)


def ok(message: str) -> None:
    print(_paint(GREEN, f"[lobora] {message}"), file=sys.stderr)


def warn(message: str) -> None:
    print(_paint(YELLOW, f"[lobora] warn: {message}"), file=sys.stderr)


def error(message: str) -> None:
    print(_paint(RED, f"[lobora] error: {message}"), file=sys.stderr)


def banner(job_name: str) -> None:
    print(_paint(BOLD, f"LoboRA  ·  {job_name}"), file=sys.stderr)


def tagged_tqdm(iterable: Iterable[Any], *, desc: str, total: int | None = None):
    from tqdm import tqdm

    return tqdm(
        iterable,
        desc=_paint(DIM, desc),
        total=total,
        file=sys.stderr,
        leave=True,
    )
