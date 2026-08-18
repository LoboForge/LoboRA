"""PEFT attach + Comfy-key export/import + Lens-style resume sidecars."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file

from lobora.console import warn

SIDECARS = ("lora_emergency.safetensors", "lora_final.safetensors", "lora_latest.safetensors")


def default_target_modules() -> list[str]:
    return ["qkv_proj", "out_proj"]


def normalize_target_modules(target_modules: Iterable[str]) -> list[str]:
    aliases = {
        "qkv": ["qkv_proj"],
        "out": ["out_proj"],
        "ffn": ["linear_1", "linear_2"],
    }
    expanded: list[str] = []
    for name in target_modules:
        expanded.extend(aliases.get(name, [name]))
    return list(dict.fromkeys(expanded))


def attach_lora(transformer, rank: int, alpha: int, dropout: float, target_modules: Iterable[str]):
    from peft import LoraConfig, get_peft_model

    targets = normalize_target_modules(target_modules)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(transformer, config)
    model.print_trainable_parameters()
    return model


def _normalize_peft_key(key: str) -> str:
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    return key


def lora_state_dict_for_comfy(model) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if "lora_" not in key:
            continue
        key = _normalize_peft_key(key)
        if key.startswith("transformer."):
            key = key.replace("transformer.", "diffusion_model.", 1)
        if not key.startswith("diffusion_model."):
            key = f"diffusion_model.{key}"
        sd[key] = value.detach().cpu()
    return sd


def lora_state_dict_for_comfy_from_raw(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap a PEFT-style state dict without a live model (used by tests)."""
    sd: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if "lora_" not in key:
            continue
        key = _normalize_peft_key(key)
        if key.startswith("transformer."):
            key = key.replace("transformer.", "diffusion_model.", 1)
        if not key.startswith("diffusion_model."):
            key = f"diffusion_model.{key}"
        sd[key] = value
    return sd


def comfy_key_to_peft(key: str) -> str:
    if key.startswith("diffusion_model."):
        key = "base_model.model." + key[len("diffusion_model.") :]
    elif not key.startswith("base_model.model."):
        key = f"base_model.model.{key}"
    return key


def save_lora(model, output_path: Path, metadata: dict | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sd = lora_state_dict_for_comfy(model)
    if not sd:
        raise RuntimeError("No LoRA weights found to save")
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    save_file(sd, str(output_path), metadata=meta)


def save_lora_from_raw(raw: dict[str, torch.Tensor], output_path: Path, metadata: dict | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sd = lora_state_dict_for_comfy_from_raw(raw)
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    save_file(sd, str(output_path), metadata=meta)


def read_lora_checkpoint_metadata(checkpoint_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def checkpoint_step(path: Path) -> int:
    if path.name in SIDECARS:
        meta = read_lora_checkpoint_metadata(path)
        return int(meta.get("step", 0) or 0)
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def find_resume_checkpoint(*, output_dir: Path, checkpoints_dir: Path) -> Path | None:
    numbered = list(checkpoints_dir.glob("lora_step_*.safetensors"))
    if numbered:
        return max(numbered, key=checkpoint_step)
    sidecars = []
    for name in SIDECARS:
        path = output_dir / name
        if path.is_file():
            sidecars.append(path)
    if not sidecars:
        return None
    # emergency preferred over stale latest when steps match via max(step)
    return max(sidecars, key=lambda p: (checkpoint_step(p), 0 if "emergency" in p.name else -1))


def resolve_resume_checkpoint(
    resume_from: str,
    *,
    output_dir: Path,
    checkpoints_dir: Path,
) -> Path | None:
    token = (resume_from or "").strip()
    if not token:
        return None
    if token.lower() in {"latest", "auto"}:
        return find_resume_checkpoint(output_dir=output_dir, checkpoints_dir=checkpoints_dir)
    path = Path(token)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def write_checkpoint_sidecars(
    src: Path,
    *,
    output_dir: Path,
    checkpoints_dir: Path,
    step: int,
    also_latest: bool = True,
) -> Path:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    numbered = checkpoints_dir / f"lora_step_{step:06d}.safetensors"
    if src.resolve() != numbered.resolve():
        shutil.copy2(src, numbered)
    if also_latest:
        shutil.copy2(numbered, output_dir / "lora_latest.safetensors")
    return numbered


def optimizer_sidecar_path(lora_path: Path) -> Path:
    return lora_path.with_suffix(".optim.pt")


def save_optimizer(optimizer, path: Path, *, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, path)


def load_optimizer(optimizer, path: Path) -> int:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    return int(blob.get("step", 0))


def load_lora_weights(model, checkpoint_path: Path) -> dict[str, object]:
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    metadata = read_lora_checkpoint_metadata(checkpoint_path)
    raw = load_file(str(checkpoint_path))
    if not raw:
        raise RuntimeError(f"No tensors found in checkpoint: {checkpoint_path}")

    model_keys = set(model.state_dict().keys())
    mapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if "lora_" not in key:
            continue
        peft_key = comfy_key_to_peft(key)
        if peft_key in model_keys:
            mapped[peft_key] = value
            continue
        suffix = peft_key.split("base_model.model.", 1)[-1]
        alt_keys = [k for k in model_keys if k.endswith(suffix)]
        if len(alt_keys) == 1:
            mapped[alt_keys[0]] = value

    if not mapped:
        raise RuntimeError(
            f"Could not map any LoRA keys from {checkpoint_path.name} into the current model"
        )

    incompatible = model.load_state_dict(mapped, strict=False)
    missing_lora = [k for k in incompatible.missing_keys if "lora_" in k]
    if missing_lora:
        warn(
            f"{len(missing_lora)} LoRA key(s) in model not present in checkpoint "
            f"(loaded {len(mapped)} tensors from {checkpoint_path.name})"
        )

    step = int(metadata.get("step", 0) or 0)
    step = max(step, checkpoint_step(checkpoint_path))
    return {
        "step": step,
        "rank": metadata.get("rank"),
        "alpha": metadata.get("alpha"),
        "path": str(checkpoint_path),
        "loaded_keys": len(mapped),
    }
