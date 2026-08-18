#!/usr/bin/env python3
"""Thin entrypoint so ``python train.py configs/....yaml`` works like LensTrainer."""

from lobora.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
