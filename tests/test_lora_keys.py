from pathlib import Path

import torch
from safetensors.torch import load_file

from lobora.lora import (
    comfy_key_to_peft,
    find_resume_checkpoint,
    lora_state_dict_for_comfy_from_raw,
    save_lora_from_raw,
)


def test_comfy_round_trip_keys():
    raw = {
        "base_model.model.blocks.0.qkv_proj.lora_A.weight": torch.ones(8, 4),
        "base_model.model.blocks.0.qkv_proj.lora_B.weight": torch.ones(4, 8),
        "ignored.weight": torch.ones(1),
    }
    comfy = lora_state_dict_for_comfy_from_raw(raw)
    assert "diffusion_model.blocks.0.qkv_proj.lora_A.weight" in comfy
    assert "ignored.weight" not in comfy
    back = comfy_key_to_peft("diffusion_model.blocks.0.qkv_proj.lora_A.weight")
    assert back == "base_model.model.blocks.0.qkv_proj.lora_A.weight"


def test_resume_prefers_highest_numbered_step(tmp_path: Path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    sd = {"diffusion_model.x.lora_A.weight": torch.zeros(2, 2)}
    save_lora_from_raw(sd, ckpt / "lora_step_000010.safetensors", {"step": 10})
    save_lora_from_raw(sd, ckpt / "lora_step_000250.safetensors", {"step": 250})
    save_lora_from_raw(sd, tmp_path / "lora_latest.safetensors", {"step": 10})
    chosen = find_resume_checkpoint(output_dir=tmp_path, checkpoints_dir=ckpt)
    assert chosen is not None
    assert chosen.name == "lora_step_000250.safetensors"
    loaded = load_file(str(chosen))
    assert "diffusion_model.x.lora_A.weight" in loaded
