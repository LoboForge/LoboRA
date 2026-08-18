"""Baseline-vs-LoRA A/B sample cadence (LensTrainer UX)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SampleJob:
    name: str
    prompt: str
    seed: int
    tag: str  # control | lora
    source_id: str = ""  # dataset sample_id when drawn from data; never logged by default


def expand_prompt(prompt: str, trigger_word: str) -> str:
    return prompt.replace("[trigger]", trigger_word)


def should_sample_at_step(
    step: int,
    *,
    sample_every: int,
    sample_every_early: int,
    sample_early_until: int,
    skip_step_zero: bool = True,
) -> bool:
    if step < 0:
        return False
    if step == 0 and skip_step_zero:
        return False
    if sample_every_early > 0 and step <= sample_early_until:
        return step % sample_every_early == 0
    if sample_every <= 0:
        return False
    return step % sample_every == 0


def pick_prompts_from_dataset(
    samples: Sequence[Any],
    *,
    n: int,
    seed: int,
) -> list[dict[str, str]]:
    """Randomly select caption rows for eval.

    Callers must not print ``prompt`` to logs if privacy matters — only ``name`` / ``source_id``.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not samples:
        raise ValueError("dataset is empty; cannot pick sample prompts")
    rng = random.Random(seed)
    videos = [s for s in samples if getattr(s, "kind", "") == "video"]
    pool = videos if len(videos) >= n else list(samples)
    if len(pool) <= n:
        chosen = list(pool)
        rng.shuffle(chosen)
    else:
        chosen = rng.sample(pool, n)
    out: list[dict[str, str]] = []
    for i, sample in enumerate(chosen):
        sid = str(getattr(sample, "sample_id", f"ds_{i:02d}"))
        stem = Path(sid).stem or f"ds_{i:02d}"
        caption = str(getattr(sample, "caption", "") or "").strip()
        if not caption:
            continue
        out.append({"name": f"ds_{i:02d}_{stem}"[:80], "prompt": caption, "source_id": sid})
    if not out:
        raise ValueError("no non-empty captions available for dataset sample prompts")
    return out


def build_sample_jobs(
    prompts: list,
    *,
    trigger_word: str,
    seed: int,
    walk_seed: bool,
    tag: str,
) -> list[SampleJob]:
    jobs: list[SampleJob] = []
    for i, item in enumerate(prompts):
        if isinstance(item, str):
            name, prompt, source_id = f"p{i:02d}", item, ""
        else:
            name = str(item.get("name") or f"p{i:02d}")
            prompt = str(item.get("prompt") or "")
            source_id = str(item.get("source_id") or "")
        jobs.append(
            SampleJob(
                name=name,
                prompt=expand_prompt(prompt, trigger_word),
                seed=seed + i if walk_seed else seed,
                tag=tag,
                source_id=source_id,
            )
        )
    return jobs


def sample_filename(step: int, job: SampleJob) -> str:
    return f"step_{step:06d}_{job.tag}_{job.name}.mp4"


def sample_dir(output_dir: Path) -> Path:
    path = Path(output_dir) / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path
