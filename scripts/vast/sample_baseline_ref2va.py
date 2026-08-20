#!/usr/bin/env python3
"""Generate one Ref2VA sample from the BASE weights, no LoRA.

Run this before trusting any adapter sample. It answers "can this box generate at
all?" separately from "did the LoRA train?", which is the difference between
debugging a broken environment and debugging a broken run. It also gives you the
control clip for an A/B: if a LoRA sample looks identical to this, the adapter is
not being applied (see the ComfyUI key-remap trap in RUNBOOK.md).

Reference image and prompt are the neutral DiffSynth example ones on purpose -- no
project dataset media is read or written here.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", WORKSPACE / "models/MiniMax-H3"))
OUT = Path(os.environ.get("BASELINE_OUT", WORKSPACE / "output/samples_baseline"))
EXAMPLE_DIR = WORKSPACE / "data/diffsynth_example_dataset"
# Headroom left for activations; the pipeline sizes its offload plan from this.
VRAM_HEADROOM_GIB = float(os.environ.get("VRAM_HEADROOM_GIB", 5))

HEIGHT = int(os.environ.get("HEIGHT", 480))
WIDTH = int(os.environ.get("WIDTH", 832))
# Shortest legal H3 clip (n % 17 == 5, min 22) -- this is a smoke test, not a render.
NUM_FRAMES = int(os.environ.get("BASELINE_FRAMES", 22))
STEPS = int(os.environ.get("BASELINE_STEPS", 12))
SEED = int(os.environ.get("BASELINE_SEED", 42))


def local(pattern: str) -> ModelConfig:
    """A ModelConfig pointing at the local snapshot instead of a hub download."""
    return ModelConfig(
        model_id=str(MODELS_ROOT),
        origin_file_pattern=pattern,
        offload_dtype=torch.bfloat16,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device="cpu",
        preparing_dtype=torch.bfloat16,
        preparing_device="cuda",
        computation_dtype=torch.bfloat16,
        computation_device="cuda",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading Ref2VA pipeline (base only, no LoRA)...")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            local("Ref2VA/text_encoder/model*.safetensors"),
            local("Ref2VA/transformer/model*.safetensors"),
            local("Ref2VA/video_vae/source/model.safetensors"),
            local("Ref2VA/audio_vae/model.safetensors"),
        ],
        processor_config=ModelConfig(
            model_id=str(MODELS_ROOT), origin_file_pattern="Ref2VA/processor/"
        ),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - VRAM_HEADROOM_GIB,
    )

    dataset_snapshot_download(
        dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
        local_dir=str(EXAMPLE_DIR),
        allow_file_pattern="minimax_h3/MiniMax-H3-Ref2VA/*",
    )
    ref_path = EXAMPLE_DIR / "minimax_h3/MiniMax-H3-Ref2VA/0.png"
    ref = Image.open(ref_path).convert("RGB")
    prompt = (
        "a person standing facing camera, full body, neutral pose, even studio light, "
        "photorealistic, medium wide shot, slow subtle breathing motion"
    )

    print(f"generating {NUM_FRAMES} frames @ {WIDTH}x{HEIGHT}, {STEPS} steps...")
    video, audio = pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        seed=SEED,
        references=[{"type": "image", "image": ref}],
    )
    out = OUT / "baseline_step000000_control.mp4"
    write_video_audio(
        video=video, audio=audio, output_path=str(out), fps=24, audio_sample_rate=32000
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
