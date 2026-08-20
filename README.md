# LoboRA

LoRA trainer for **MiniMax-H3** (Ref2VA). Ops and UX from LensTrainer — resume, two-stage cache, A/B samples — plus real aspect/duration **buckets** and mixed **image + video** datasets.

LoboRA does **not** invent a new H3 loss. It follows DiffSynth-Studio’s flow-match conventions so adapters stay compatible with that stack and ComfyUI.

## Why this exists

MiniMax shipped weights and no official trainer. Copying a Wan/Lens velocity sign will silently rot the run. DiffSynth’s convention (the one we match):

- `x_t = (1 − σ) · x₀ + σ · ε`
- training target **`ε − x₀`**
- DiT timestep **`t = 1 − σ`**
- train shift **2.22 / 2.22** (inference uses 12 / 3 — that mismatch is intentional)
- frames `n % 17 == 5`, minimum 22; H/W divisible by 32; audio 32 kHz stereo

## Install

```bash
pip install -e ".[dev]"          # CPU tests / dry-run
pip install -e ".[dev,gpu]"      # + DiffSynth + bitsandbytes
```

## Dataset

Drop media next to `.txt` captions (Lens convention):

```
dataset/
  shot_001.mp4
  shot_001.txt
  still_a.png
  still_a.txt
```

Images and videos can live in the **same folder**. They are bucketed separately so a batch is never mixed.

Optional DiffSynth-compatible metadata:

- Ref2VA: `metadata.json` (video + prompt + input_audio + references)
- FL2VA: `metadata.csv` (video, prompt, input_audio)

Captions: 30–80 words, trigger token first, then subject → action → camera → audio.

**Volume (concept LoRA):** 60–150 clips + 100–300 stills, 73 frames @ 24 fps primary.
**Motion LoRA:** 150–300 clips, 73–124 frames, camera grammar labeled.

## Quickstart (CPU dry-run)

```bash
python train.py configs/concept_minimal.yaml \
  --dataset-path ./dataset \
  --output-dir ./output/smoke \
  --steps 8 --dry-run --skip-numerics-gate
```

## GPU recipes

| Preset | Box | Notes |
|---|---|---|
| `configs/ref2va_bf16_80gb.yaml` | 2× H100 80GB ~$5/hr | bf16 weights, FP8-frozen DiT, rank 32, 1500–2500 steps |
| `configs/ref2va_nf4_48gb.yaml` | 1× 48GB | NF4 inference quant as train base — experimental |

```bash
python scripts/download_weights.py --dest ./models/MiniMax-H3
lobora configs/ref2va_bf16_80gb.yaml \
  --dataset-path ./dataset \
  --output-dir ./output/run0
```

## Running on a rented box

[`RUNBOOK.md`](RUNBOOK.md) is the 2am document: the DiffSynth two-stage recipe, the
VRAM levers that actually work on an 80 GB card, crash recovery, and the traps that
silently waste a day — above all the **ComfyUI key remap**, without which a trained
adapter loads zero patches and renders as the base model. Scripts in
[`scripts/vast/`](scripts/vast).

Resume after Ctrl+C:

```bash
lobora configs/ref2va_bf16_80gb.yaml --output-dir ./output/run0 --resume latest
```

Sidecars: `checkpoints/lora_step_NNNNNN.safetensors`, `lora_latest.safetensors`, `lora_emergency.safetensors`, `lora_final.safetensors`, plus `.optim.pt` optimizer state.

## Resume state

Every checkpoint is keyed on the **cumulative** step across all attempts, and carries
enough state to continue rather than warm-restart:

| File | Holds |
|---|---|
| `lora_step_NNNNNN.safetensors` / `step-N.safetensors` | adapter tensors |
| `<same stem>.optim.pt` | Adam moments, LR-scheduler position, RNG |
| `train_state.json` | authoritative cumulative step, epoch position, shuffle seed, run fingerprint |

Because the step number is cumulative, a restarted run continues the lineage
(`step-700`, `step-800`, …) and can never write different weights under a name an
earlier attempt already used.

Resume state that exists but cannot be used is **fatal, never silent**: a missing
checkpoint, a manifest older than the directory, a missing `.optim.pt`, or a run
config that drifted since the checkpoint was written all stop the run with an
explanation. Pass `--allow-weights-only-resume` (or `LOBORA_ALLOW_WEIGHTS_ONLY_RESUME=1`
on the DiffSynth path) to deliberately accept a warm restart with fresh Adam moments.

### On the production DiffSynth path

Live H3 runs go through DiffSynth's `examples/minimax_h3/model_training/train.py`,
which persists none of this. `scripts/train_h3_resumable.py` is a drop-in wrapper —
same flags, same output files — that patches the loop before running it:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  scripts/train_h3_resumable.py  [all the usual DiffSynth flags]
```

`scripts/supervise_h3_resumable.sh` drives it and treats exit code 3
(“resume state unusable”) as fatal instead of burning a retry on it.

## Numerics gate

A 50-step tiny run must keep mean loss roughly in `[0.15, 1.5]` and not rise. Long runs refuse to start without a passed `numerics_gate.json` unless you pass `--skip-numerics-gate`.

## Config precedence

YAML preset < `--env-file` < `--set key=value` < explicit CLI flags.

Unknown YAML keys are **warned**, not silently dropped.

## Tests

```bash
pytest -q
```

CPU-only: grid math, buckets, cache keys (includes `model_rev`), Comfy key remap, scheduler signs, dry-run train, resume round-trip (cumulative step, Adam moments, scheduler position) and checkpoint-name non-collision.

## License

Apache-2.0. DiffSynth-Studio is Apache-2.0 (Zhongjie Duan, 2023). MiniMax-H3 **weights** are under the MiniMax Community License and are not shipped here.
