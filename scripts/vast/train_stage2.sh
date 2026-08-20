#!/usr/bin/env bash
# Stage-2-only LoRA train for MiniMax-H3 Ref2VA off a prebuilt split-cache.
#
# Stage 1 (data_process) is deliberately not run here: once the cache is complete
# stage 1 only reloads the text encoder and both VAEs for nothing. Stage 2 loads
# ONLY the DiT and still peaks near 78 GiB on an 80 GiB card -- see RUNBOOK.md.
#
# Usage: train_stage2.sh [resume_checkpoint.safetensors]
#   --lora_checkpoint restores ADAPTER WEIGHTS ONLY. Step counter, Adam moments and
#   LR schedule all restart, so the progress bar goes back to 0 on a warm start.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/vast/h3_env.sh
source "$HERE/h3_env.sh"

# Necessary but not sufficient: this removes reserved-but-unallocated fragmentation
# as a failure mode, it does not create headroom. The offload flag below does.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$LORA" "$LOGDIR"
[ -d "$CACHE" ] || { echo "error: no stage-1 cache at $CACHE"; exit 2; }

if [ ! -f "$OUT/stage2_model_paths.json" ]; then
  "$PYTHON" "$HERE/make_model_paths.py" --models-root "$MODELS_ROOT" --out-dir "$OUT"
fi
STAGE2_PATHS=$(tr -d '\n' < "$OUT/stage2_model_paths.json")

# Naming ONE shard selects the whole sharded DiT group (shard-aware matching in
# training_module.parse_model_configs). This is what drops the frozen base from
# bf16 (~62 GiB) to fp8_e4m3fn (~31 GiB) while still computing in bf16.
FP8_TARGET=${FP8_TARGET:-$MODELS_ROOT/Ref2VA/transformer/model-00001-of-00013.safetensors}

RESUME_ARGS=()
if [ "${1:-}" != "" ]; then
  RESUME_ARGS=(--lora_checkpoint "$1")
  echo "[$(date -Is)] resuming adapter weights (weights only) from $1"
fi

cd "$DIFFSYNTH"
exec "$ACC" launch --num_processes 1 --mixed_precision bf16 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$CACHE" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --dataset_repeat "$DATASET_REPEAT" \
  --model_paths "$STAGE2_PATHS" \
  --fp8_models "$FP8_TARGET" \
  --processor_path "$PROCESSOR" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$LORA" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGETS" \
  --lora_rank "$LORA_RANK" \
  --save_steps "$SAVE_STEPS" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --silent_on_missing_audio \
  --task "sft:train" \
  "${RESUME_ARGS[@]}"
