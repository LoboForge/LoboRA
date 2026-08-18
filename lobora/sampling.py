"""Baseline-vs-LoRA A/B sample cadence (LensTrainer UX)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleJob:
    name: str
    prompt: str
    seed: int
    tag: str  # control | lora


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


def build_sample_jobs(
    prompts: list[dict],
    *,
    trigger_word: str,
    seed: int,
    walk_seed: bool,
    tag: str,
) -> list[SampleJob]:
    jobs: list[SampleJob] = []
    for i, item in enumerate(prompts):
        if isinstance(item, str):
            name, prompt = f"p{i:02d}", item
        else:
            name = str(item.get("name") or f"p{i:02d}")
            prompt = str(item.get("prompt") or "")
        jobs.append(
            SampleJob(
                name=name,
                prompt=expand_prompt(prompt, trigger_word),
                seed=seed + i if walk_seed else seed,
                tag=tag,
            )
        )
    return jobs


def sample_filename(step: int, job: SampleJob) -> str:
    return f"step_{step:06d}_{job.tag}_{job.name}.mp4"


def sample_dir(output_dir: Path) -> Path:
    path = Path(output_dir) / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path
