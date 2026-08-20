#!/usr/bin/env bash
# Crash-recovery supervisor for stage 2. Restarts the trainer on any non-zero exit,
# resuming from the newest checkpoint that actually validates.
#
# Two traps this exists to avoid:
#
#  1. Trainer output is piped through `tee` so it is visible live in the tmux pane
#     AND appended to the log. With that pipe, the trainer's status must be read
#     from ${PIPESTATUS[0]}: plain $? is tee's status, which is always 0, so every
#     crash would look like success and recovery would silently never fire.
#  2. The trainer restarts its step counter on a warm start, so checkpoint names
#     would collide and overwrite the previous attempt's files. DIFFSYNTH_STEP_OFFSET
#     continues the lineage (step-700, step-800, ...) instead of restarting at 100.
#     That is numbering only -- adapter weights carry over, optimizer and scheduler
#     state do not, because --lora_checkpoint restores weights alone.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/vast/h3_env.sh
source "$HERE/h3_env.sh"

# INNER is overridable so the rc plumbing can be exercised with a stub that exits
# non-zero, without touching real logs. Test that; do not assume it works.
INNER=${H3_INNER:-$HERE/train_stage2.sh}
LOG=${H3_LOG:-$LOGDIR/h3_stage2.log}
SUP=${H3_SUP:-$LOGDIR/h3_supervisor.log}
HB=${H3_HB:-$LOGDIR/h3_heartbeat.txt}
MAX_ATTEMPTS=${H3_MAX_ATTEMPTS:-8}
RETRY_SLEEP=${H3_RETRY_SLEEP:-60}
export DIFFSYNTH_HEARTBEAT_FILE=$HB

mkdir -p "$LOGDIR" "$LORA"

# Status lines go to the pane (stdout) and to the supervisor log.
say() { echo "[$(date -Is)] $*" | tee -a "$SUP"; }

newest_valid_ckpt() {
  # NOTE: stdout of this function IS the return value. Diagnostics go to $SUP only.
  # A checkpoint can be mid-write when a crash lands, so validate the header and
  # the declared payload length before trusting it.
  local f
  for f in $(ls -1 "$LORA"/step-*.safetensors 2>/dev/null \
             | sed 's#.*/step-\([0-9]\+\)\.safetensors#\1 &#' \
             | sort -rn -k1,1 | cut -d' ' -f2-); do
    if "$PYTHON" - "$f" <<'PY' >/dev/null 2>&1
import sys, json, struct
p = sys.argv[1]
with open(p, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    head = json.loads(fh.read(n))
    end = max((v["data_offsets"][1] for k, v in head.items() if k != "__metadata__"), default=0)
    fh.seek(0, 2)
    assert fh.tell() == 8 + n + end, "truncated"
    assert any("lora" in k for k in head), "no lora keys"
PY
    then echo "$f"; return; fi
    echo "[$(date -Is)] rejecting unreadable/truncated checkpoint $f" >> "$SUP"
  done
}

say "supervisor start (max_attempts=$MAX_ATTEMPTS, inner=$INNER)"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  export H3_ATTEMPT=$attempt
  CKPT=$(newest_valid_ckpt)
  STEP_OFFSET=0
  if [ -n "$CKPT" ]; then
    STEP_OFFSET=$(basename "$CKPT" | sed -n 's#^step-\([0-9]\{1,\}\)\.safetensors$#\1#p')
    [ -n "$STEP_OFFSET" ] || STEP_OFFSET=0
  fi
  export DIFFSYNTH_STEP_OFFSET="$STEP_OFFSET"
  say "===== ATTEMPT $attempt/$MAX_ATTEMPTS resume_from='${CKPT:-<scratch>}' step_offset=$STEP_OFFSET ====="
  echo "[$(date -Is)] ===== ATTEMPT $attempt (resume='${CKPT:-scratch}') =====" >> "$LOG"

  # Live to pane + appended to log. Capture PIPESTATUS[0] on the VERY NEXT LINE:
  # any command in between clobbers it.
  bash "$INNER" $CKPT 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    say "TRAIN_DONE attempt=$attempt rc=0"
    echo "[$(date -Is)] TRAIN_DONE" >> "$LOG"
    exit 0
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
