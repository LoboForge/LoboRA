#!/usr/bin/env bash
# Stop a running stage-2 attempt by EXACT PID, wait for the GPU to drain, then
# relaunch the supervisor so it warm-starts from the newest valid checkpoint.
#
# Never use pkill/killall on a training box: the pattern that matches the trainer
# also matches sibling jobs, and on a shared box it will take out someone else's run.
# Re-capture PIDs every time; a stale PID from a previous shell can be reused by an
# unrelated process.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/vast/h3_env.sh
source "$HERE/h3_env.sh"

SESSION=${H3_TMUX_SESSION:-h3_train}
SUPERVISOR=${H3_SUPERVISOR:-$HERE/supervise_stage2.sh}
SUP_LOG=${H3_SUP:-$LOGDIR/h3_supervisor.log}

echo "=== PIDs before ==="
ps -eo pid,ppid,etime,cmd | grep -E "supervise_stage2|train\.py|accelerate launch" | grep -v grep

SUP_PID=$(ps -eo pid,cmd | awk -v s="$(basename "$SUPERVISOR")" '$0 ~ ("bash .*" s) && !/awk/ {print $1; exit}')
LAUNCH_PID=$(ps -eo pid,cmd | awk '/accelerate launch/ && !/awk/ {print $1; exit}')
TRAIN_PID=$(ps -eo pid,cmd | awk '/[p]ython examples\/minimax_h3\/model_training\/train\.py/ {print $1; exit}')

echo "captured: SUP=$SUP_PID LAUNCH=$LAUNCH_PID TRAIN=$TRAIN_PID"

# 1. Supervisor first, so it cannot read the trainer's death as a crash and relaunch.
if [ -n "${SUP_PID:-}" ]; then echo "TERM supervisor $SUP_PID"; kill "$SUP_PID"; fi
sleep 3

# 2. Then the trainer itself.
for p in "$TRAIN_PID" "$LAUNCH_PID"; do
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then echo "TERM $p"; kill "$p"; fi
done

# 3. Give them up to 60s to exit gracefully so the last checkpoint is not truncated.
for _ in $(seq 1 20); do
  alive=""
  for p in "$SUP_PID" "$LAUNCH_PID" "$TRAIN_PID"; do
    [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && alive="$alive $p"
  done
  [ -z "$alive" ] && break
  echo "waiting on:$alive"
  sleep 3
done

# 4. Only then SIGKILL, and only those exact PIDs.
for p in "$TRAIN_PID" "$LAUNCH_PID" "$SUP_PID"; do
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then echo "SIGKILL $p"; kill -9 "$p"; fi
done
sleep 3

echo "=== PIDs after ==="
ps -eo pid,cmd | grep -E "supervise_stage2|train\.py|accelerate launch" | grep -v grep || echo "(none - all stopped)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "killing stale tmux session $SESSION"
  tmux kill-session -t "$SESSION"
fi

# A relaunch before the driver has released VRAM just OOMs immediately.
echo "=== waiting for GPU memory to be released ==="
for _ in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "gpu_used_mib=$used"
  [ "${used:-0}" -lt 4000 ] && break
  sleep 5
done

echo "=== checkpoints present (selector takes the highest valid step-N) ==="
ls -la --time-style=long-iso "$LORA" | grep safetensors || echo "(none yet)"

echo "=== RELAUNCH in tmux $SESSION ==="
tmux new-session -d -s "$SESSION" "bash $SUPERVISOR"
sleep 5
tmux ls
tail -5 "$SUP_LOG" 2>/dev/null || true
