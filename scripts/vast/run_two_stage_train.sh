#!/usr/bin/env bash
# Full two-stage MiniMax-H3 Ref2VA LoRA run on a rented single-GPU box.
#
#   stage 1  sft:data_process  encodes the dataset into $CACHE (slow, one time).
#            Re-running it only VERIFIES an existing cache, it does not re-encode,
#            so it is safe to run again after a crash. Changing HEIGHT / WIDTH /
#            NUM_FRAMES does invalidate the cache.
#   stage 2  sft:train         trains the adapter off that cache. Delegated to
#            train_stage2.sh so there is exactly one canonical stage-2 flag set.
#
# For an unattended run use supervise_stage2.sh instead of calling stage 2 here.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/vast/h3_env.sh
source "$HERE/h3_env.sh"

# Stage 1 is eight hours of GPU time and the one artefact on this box that cannot
# be re-fetched. Refuse to start it when there is a populated cache or a live
# trainer; see stage1_guard.sh for the override, which names the cache on purpose.
# shellcheck source=scripts/vast/stage1_guard.sh
source "$HERE/stage1_guard.sh"
STAGE1_ENTRYPOINT="bash $HERE/run_two_stage_train.sh" \
  stage1_guard "$CACHE" "$RUN" "${HEIGHT}x${WIDTH}x${NUM_FRAMES}" || exit $?

mkdir -p "$OUT" "$CACHE" "$LORA" "$LOGDIR"
LOG=${TRAIN_LOG:-$LOGDIR/h3_train.log}

"$PYTHON" "$HERE/make_model_paths.py" --models-root "$MODELS_ROOT" --out-dir "$OUT" | tee -a "$LOG"
STAGE1_PATHS=$(tr -d '\n' < "$OUT/stage1_model_paths.json")

cd "$DIFFSYNTH"
{
  echo "[$(date -Is)] python=$("$PYTHON" -c 'import sys; print(sys.executable)')"
  echo "[$(date -Is)] diffsynth=$("$PYTHON" -c 'import diffsynth; print(diffsynth.__file__)')"
  echo "[$(date -Is)] STAGE1 data_process start -> $CACHE"
} | tee -a "$LOG"

"$ACC" launch --num_processes 1 --mixed_precision bf16 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$DATASET" \
  --dataset_metadata_path "$DATASET/metadata.json" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --dataset_repeat 1 \
  --model_paths "$STAGE1_PATHS" \
  --processor_path "$PROCESSOR" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$CACHE" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGETS" \
  --lora_rank "$LORA_RANK" \
  --use_gradient_checkpointing \
  --silent_on_missing_audio \
  --initialize_model_on_cpu \
  --task "sft:data_process" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "[$(date -Is)] STAGE1 FAILED rc=$rc" | tee -a "$LOG"; exit "$rc"; }

echo "[$(date -Is)] STAGE2 train start" | tee -a "$LOG"
bash "$HERE/train_stage2.sh" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "[$(date -Is)] STAGE2 FAILED rc=$rc" | tee -a "$LOG"; exit "$rc"; }

echo "[$(date -Is)] TRAIN_DONE" | tee -a "$LOG"
