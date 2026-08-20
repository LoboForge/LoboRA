#!/bin/bash
# Stage-2-only LoRA train for MiniMax-H3 Ref2VA off the prebuilt split-cache.
#
# Byte-for-byte the same training recipe as /workspace/train_stage2_fp8.sh -- same
# height/width/num_frames/dataset_repeat, so the 8-hour split-cache stays valid. The
# only change is the script `accelerate launch` runs: LoboRA's wrapper instead of
# DiffSynth's example, which adds cumulative-step + optimizer/scheduler persistence.
#
# Stage 1 (data_process) is deliberately NOT run: the cache is complete, and stage 1
# would reload the text encoder + both VAEs for nothing.
set -euo pipefail
export PATH="/workspace/venv/bin:/usr/bin:/bin"
export VIRTUAL_ENV=/workspace/venv
unset PYTHONHOME
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
# 3.06 GiB sat reserved-but-unallocated at the last OOM; expandable segments removes
# that fragmentation as a failure mode.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hash -r

ACC=/workspace/venv/bin/accelerate
LOBORA=${LOBORA_REPO:-/workspace/LoboRA}
cd /workspace/DiffSynth-Studio

OUT=/workspace/output/anatomy_ref2va_a800
CACHE=$OUT/split-cache
LORA=$OUT/lora
PROC=/workspace/models/MiniMax-H3/Ref2VA/processor
mkdir -p "$LORA" /workspace/logs

STAGE2_PATHS=$(tr -d '\n' < /workspace/output/stage2_model_paths.json)
# Naming one shard selects the whole sharded DiT group (shard-aware matching in
# training_module.parse_model_configs). This is what drops the frozen base from
# bf16 (~62 GiB) to fp8_e4m3fn (~31 GiB) while computing in bf16.
FP8_TARGET=/workspace/models/MiniMax-H3/Ref2VA/transformer/model-00001-of-00013.safetensors

RESUME_ARGS=()
if [ "${1:-}" != "" ]; then
  RESUME_ARGS=(--lora_checkpoint "$1")
  echo "[$(date -Is)] resuming adapter weights from $1"
fi

exec "$ACC" launch --num_processes 1 --mixed_precision bf16 \
  "$LOBORA/scripts/train_h3_resumable.py" \
  --dataset_base_path "$CACHE" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 73 \
  --dataset_repeat 7 \
  --model_paths "$STAGE2_PATHS" \
  --fp8_models "$FP8_TARGET" \
  --processor_path "$PROC" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$LORA" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --save_steps 100 \
  --use_gradient_checkpointing \
  --silent_on_missing_audio \
  --task "sft:train" \
  "${RESUME_ARGS[@]}"
