#!/bin/bash
# Supervisor for the resumable MiniMax-H3 trainer.
#
# Replaces /workspace/supervise_stage2.sh. Two behavioural changes over that script:
#
#   1. The resume target comes from train_state.json (lobora.resume_state), not from
#      `ls step-*.safetensors`. The manifest is the authority on the cumulative step,
#      and it refuses to hand back a checkpoint whose optimizer sidecar is missing.
#   2. Exit code 3 (EXIT_RESUME_UNUSABLE) is FATAL. The old script retried every
#      non-zero exit, so a broken resume quietly burned all 8 attempts restarting from
#      scratch. A resume that cannot be trusted must stop and wait for a human.
#
# Trainer output is piped through `tee`, so the trainer's status must be read from
# ${PIPESTATUS[0]} -- plain $? would be tee's (always 0) and every crash would look
# like success.
set -uo pipefail

OUT=${ANATOMY_OUT:-/workspace/output/anatomy_ref2va_a800}
LORA=$OUT/lora
LOBORA=${LOBORA_REPO:-/workspace/LoboRA}
PY=${ANATOMY_PYTHON:-/workspace/venv/bin/python}
INNER=${ANATOMY_INNER:-/workspace/train_stage2_resumable.sh}
LOG=${ANATOMY_LOG:-/workspace/logs/anatomy_train_fp8.log}
SUP=${ANATOMY_SUP:-/workspace/logs/anatomy_supervisor.log}
HB=${ANATOMY_HB:-/workspace/logs/anatomy_heartbeat.txt}
MAX_ATTEMPTS=${ANATOMY_MAX_ATTEMPTS:-8}
RETRY_SLEEP=${ANATOMY_RETRY_SLEEP:-60}
export DIFFSYNTH_HEARTBEAT_FILE=$HB
export PYTHONPATH="$LOBORA${PYTHONPATH:+:$PYTHONPATH}"

# The wrapper reads train_state.json; a stale env offset must not seed the counter.
unset DIFFSYNTH_STEP_OFFSET

EXIT_RESUME_UNUSABLE=3

say() { echo "[$(date -Is)] $*" | tee -a "$SUP"; }

# stdout is the return value; diagnostics go to $SUP only.
resume_checkpoint() {
  "$PY" - "$LORA" <<'PY' 2>>"$SUP"
import sys
from pathlib import Path
from lobora.resume_state import ResumeStateError, find_resume_target

try:
    target = find_resume_target(Path(sys.argv[1]), require_optimizer=False)
except ResumeStateError as exc:
    print(f"resume state unusable: {exc}", file=sys.stderr)
    raise SystemExit(3)
if target is not None:
    print(target.checkpoint)
PY
}

say "supervisor start (max_attempts=$MAX_ATTEMPTS, inner=$INNER)"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  export ANATOMY_ATTEMPT=$attempt

  CKPT=$(resume_checkpoint); rc=$?
  if [ "$rc" -eq "$EXIT_RESUME_UNUSABLE" ]; then
    say "FATAL: resume state in $LORA is unusable (see above). Not retrying."
    exit "$EXIT_RESUME_UNUSABLE"
  fi

  say "===== ATTEMPT $attempt/$MAX_ATTEMPTS resume_from='${CKPT:-<scratch>}' ====="
  echo "[$(date -Is)] ===== ATTEMPT $attempt (resume='${CKPT:-scratch}') =====" >> "$LOG"

  bash "$INNER" $CKPT 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}

  # `accelerate launch` does not always pass a child's exit code through verbatim, so
  # the trainer also drops a marker file when it refuses to resume.
  if [ -f "$LORA/RESUME_BLOCKED.txt" ]; then
    say "FATAL: trainer refused to resume -- $(head -1 "$LORA/RESUME_BLOCKED.txt")"
    exit "$EXIT_RESUME_UNUSABLE"
  fi

  if [ "$rc" -eq 0 ]; then
    say "TRAIN_DONE attempt=$attempt rc=0"
    echo "[$(date -Is)] TRAIN_DONE" >> "$LOG"
    exit 0
  fi

  if [ "$rc" -eq "$EXIT_RESUME_UNUSABLE" ]; then
    say "FATAL: trainer refused to resume (rc=$rc). Fix the state by hand; not retrying."
    exit "$EXIT_RESUME_UNUSABLE"
  fi
  if [ "$rc" -eq 130 ]; then
    say "stopped on signal after an emergency checkpoint (rc=130); not retrying"
    exit 130
  fi

  say "attempt $attempt FAILED rc=$rc"
  if tail -40 "$LOG" | grep -q "OutOfMemoryError"; then
    say "  -> detected CUDA OutOfMemoryError in tail of log"
  fi
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    say "retrying in ${RETRY_SLEEP}s"
    sleep "$RETRY_SLEEP"
  fi
done
say "GAVE UP after $MAX_ATTEMPTS attempts"
exit 1
