#!/usr/bin/env bash
# Shared environment for the DiffSynth MiniMax-H3 training scripts in this folder.
# Source it, do not execute it. Every path is overridable from the caller.
#
#   WORKSPACE   root of the rented box               (/workspace)
#   VENV        python venv holding diffsynth        ($WORKSPACE/venv)
#   DIFFSYNTH   DiffSynth-Studio checkout            ($WORKSPACE/DiffSynth-Studio)
#   MODELS_ROOT MiniMax-H3 snapshot                  ($WORKSPACE/models/MiniMax-H3)
#   DATASET     media + sidecar captions             ($WORKSPACE/dataset)
#   RUN         run name, used for output subdirs     (h3_ref2va)
#   OUT         run output dir                       ($WORKSPACE/output/$RUN)

WORKSPACE=${WORKSPACE:-/workspace}
VENV=${VENV:-$WORKSPACE/venv}
DIFFSYNTH=${DIFFSYNTH:-$WORKSPACE/DiffSynth-Studio}
MODELS_ROOT=${MODELS_ROOT:-$WORKSPACE/models/MiniMax-H3}
DATASET=${DATASET:-$WORKSPACE/dataset}
RUN=${RUN:-h3_ref2va}
OUT=${OUT:-$WORKSPACE/output/$RUN}

CACHE=${CACHE:-$OUT/split-cache}
LORA=${LORA:-$OUT/lora}
LOGDIR=${LOGDIR:-$WORKSPACE/logs}
PROCESSOR=${PROCESSOR:-$MODELS_ROOT/Ref2VA/processor}

PYTHON=${PYTHON:-$VENV/bin/python}
ACC=${ACC:-$VENV/bin/accelerate}

# The venv is entered by PATH rather than `activate` so these scripts work the
# same under tmux, cron and `ssh box bash script.sh`, where no shell rc is read.
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export VIRTUAL_ENV="$VENV"
unset PYTHONHOME
export HF_HOME=${HF_HOME:-$WORKSPACE/hf_cache}
export TOKENIZERS_PARALLELISM=false
# Point DiffSynth at the local snapshot and stop it phoning home mid-run.
export DIFFSYNTH_MODEL_BASE_PATH="$MODELS_ROOT"
export DIFFSYNTH_SKIP_DOWNLOAD=true
hash -r

# Shot geometry. H3 wants num_frames % 17 == 5 (min 22) and H/W divisible by 32.
# Changing any of these three invalidates the stage-1 cache -- see RUNBOOK.md.
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-73}

LORA_RANK=${LORA_RANK:-32}
LORA_TARGETS=${LORA_TARGETS:-qkv_proj,out_proj}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATASET_REPEAT=${DATASET_REPEAT:-7}
GRAD_ACCUM=${GRAD_ACCUM:-4}
# Micro-batches, NOT optimizer steps: with GRAD_ACCUM=4 this saves every 25
# optimizer steps. Keep it small enough that a crash cannot wipe a whole attempt.
SAVE_STEPS=${SAVE_STEPS:-25}
