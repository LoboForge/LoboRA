#!/bin/bash
# stop_at_step.sh -- external hard stop for the MiniMax-H3 Ref2VA LoRA run.
#
# WHY THIS EXISTS
#   The DiffSynth example trainer ignores `train.steps` from the project config and
#   runs a full pass of len(dataloader) = 963 samples x dataset_repeat 7 = 6741
#   micro-batches. Warm-started on top of a 600-step offset that is cumulative
#   step 7341 (~77-90 h, ~$85-99). This watcher enforces the intended cap.
#
# WHICH COUNTER (this is the whole correctness argument -- do not "simplify" it)
#   Two counters exist and differ by a constant offset:
#     * tqdm progress bar  -> IN-PASS steps, restarts at 0 on every attempt.
#     * step-N.safetensors -> CUMULATIVE steps. diffsynth/diffusion/logger.py seeds
#       ModelLogger.num_steps from $DIFFSYNTH_STEP_OFFSET (exported by the
#       supervisor from the checkpoint it resumed from) and saves when
#       num_steps % save_steps == 0.
#   This script fires on the CUMULATIVE counter, read from checkpoint filenames on
#   disk, because those are cumulative by construction and a checkpoint has to be
#   verified on disk anyway. One tick = one dataloader micro-batch; with
#   gradient_accumulation_steps=4 the optimizer updates every 4 ticks.
#
# ORDER OF OPERATIONS AT FIRE TIME
#   1. verify the target checkpoint is complete and size-stable
#   2. SIGSTOP the supervisor -- freezes its restart loop WITHOUT closing the tmux
#      pane. Killing it instead would make tmux tear the pane down and SIGHUP the
#      whole process group (the trainer shares the supervisor's pgid), killing the
#      trainer uncontrolled. Freezing keeps the shutdown ours to sequence.
#   3. SIGINT the leaf trainer, escalate SIGTERM then SIGKILL on timeouts
#   4. only then SIGKILL the frozen supervisor
#   5. keep watching, so a resurrection is caught and logged rather than silent
#
# Signals go to specific PIDs discovered and logged at fire time. pkill/killall are
# banned in this project and are never used here. There are deliberately no pipes
# in any exit-status path: a previous bug in this project was `tee` swallowing a
# status via $? instead of ${PIPESTATUS[0]}, so log() writes with two printfs.
#
# Everything is env-overridable so the self-test drives this exact code against
# stub processes. See stop_at_step_selftest.sh.
set -uo pipefail

TARGET=${STOP_TARGET_STEP:-2000}
LORA_DIR=${STOP_LORA_DIR:-/workspace/output/anatomy_ref2va_a800/lora}
LOG=${STOP_LOG:-/workspace/logs/stop_at_step.log}
SENTINEL=${STOP_SENTINEL:-/workspace/STOP_AT_STEP.sentinel}
POLL=${STOP_POLL_SECS:-90}
STABLE_WAIT=${STOP_STABLE_WAIT:-20}
QUIESCE=${STOP_QUIESCE_SECS:-25}
SUP_PATTERN=${STOP_SUP_PATTERN:-supervise_stage2.sh}
TRAINER_PATTERN=${STOP_TRAINER_PATTERN:-examples/minimax_h3/model_training/train.py}
PY=${STOP_PY:-/workspace/venv/bin/python}
INT_GRACE=${STOP_INT_GRACE:-240}
TERM_GRACE=${STOP_TERM_GRACE:-120}
KILL_WAIT=${STOP_KILL_WAIT:-30}
WATCH_AFTER=${STOP_WATCH_AFTER:-900}
DRY_RUN=${STOP_DRY_RUN:-0}

SELF_TAG=$(basename "$0")

log() {
  local line
  line="[$(date -Is)] $*"
  printf '%s\n' "$line" >>"$LOG"
  printf '%s\n' "$line"
}

# ---------------------------------------------------------------- checkpoints --

# Lowest cumulative step >= TARGET, as "N<TAB>path". The glob is anchored, so the
# parked attempt1_step-*.safetensors copies cannot match; the digit check rejects
# anything else that is not exactly step-<digits>.safetensors.
candidate_ckpt() {
  local f base n
  for f in "$LORA_DIR"/step-*.safetensors; do
    [ -f "$f" ] || continue
    base=${f##*/}
    n=${base#step-}
    n=${n%.safetensors}
    case "$n" in '' | *[!0-9]*) continue ;; esac
    if [ "$n" -ge "$TARGET" ]; then
      printf '%s\t%s\n' "$n" "$f"
    fi
  done | sort -n | head -1
}

newest_ckpt_step() {
  local f base n best=0
  for f in "$LORA_DIR"/step-*.safetensors; do
    [ -f "$f" ] || continue
    base=${f##*/}
    n=${base#step-}
    n=${n%.safetensors}
    case "$n" in '' | *[!0-9]*) continue ;; esac
    [ "$n" -gt "$best" ] && best=$n
  done
  printf '%s\n' "$best"
}

# Complete == safetensors header parses, file size equals 8 + header + payload,
# at least one lora key present, and size identical across two stats STABLE_WAIT
# apart. Python is invoked without a pipe so its status is read directly.
verify_ckpt() {
  local f=$1 s1 s2 rc
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  sleep "$STABLE_WAIT"
  s2=$(stat -c %s "$f" 2>/dev/null) || return 1
  if [ "$s1" != "$s2" ]; then
    log "  size not yet stable ($s1 -> $s2), will re-check"
    return 1
  fi
  "$PY" - "$f" <<'PY'
import json
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    head = json.loads(fh.read(n))
    end = max(
        (v["data_offsets"][1] for k, v in head.items() if k != "__metadata__"),
        default=0,
    )
    fh.seek(0, 2)
    total = fh.tell()
if total != 8 + n + end:
    raise SystemExit("truncated: size %d != %d" % (total, 8 + n + end))
if not any("lora" in k for k in head if k != "__metadata__"):
    raise SystemExit("no lora keys in header")
PY
  rc=$?
  [ "$rc" -eq 0 ] || log "  header/payload check failed rc=$rc"
  return "$rc"
}

# Nothing in the checkpoint dir touched in the last QUIESCE seconds, so we cannot
# be signalling in the middle of a save.
quiesced() {
  local recent
  recent=$(find "$LORA_DIR" -maxdepth 1 -type f -newermt "-${QUIESCE} seconds" 2>/dev/null | head -1)
  [ -z "$recent" ]
}

# -------------------------------------------------------------------- process --

cmd_of() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null; }

ppid_of() {
  # Everything after the last ')' is: state ppid ... -- avoids comm containing
  # spaces or parens throwing the field count off.
  sed 's/.*)//' "/proc/$1/stat" 2>/dev/null | awk '{print $2}'
}

state_of() { sed 's/.*)//' "/proc/$1/stat" 2>/dev/null | awk '{print $1}'; }

# PIDs whose cmdline contains $1. Reads /proc directly rather than shelling out to
# ps|grep, so the matcher can never match its own pipeline. Skips this script and
# any of its subshells.
find_pids() {
  local pat=$1 pid cmd
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    [ "$pid" = "$$" ] && continue
    cmd=$(cmd_of "$pid")
    [ -n "$cmd" ] || continue
    case "$cmd" in *"$SELF_TAG"*) continue ;; esac
    case "$cmd" in *"$pat"*) printf '%s\n' "$pid" ;; esac
  done
}

# Of the matching PIDs, those that are not the parent of another match. accelerate
# launch and the python it spawns both carry train.py in argv; the leaf is the real
# trainer and the one that should get the first SIGINT.
leaf_pids() {
  local all=("$@") p q is_parent
  for p in "${all[@]}"; do
    is_parent=0
    for q in "${all[@]}"; do
      [ "$p" = "$q" ] && continue
      [ "$(ppid_of "$q")" = "$p" ] && is_parent=1
    done
    [ "$is_parent" -eq 0 ] && printf '%s\n' "$p"
  done
}

alive() { kill -0 "$1" 2>/dev/null; }

signal_pid() {
  local sig=$1 pid=$2
  if [ "$DRY_RUN" = "1" ]; then
    log "  [dry-run] would send SIG$sig to $pid"
    return 0
  fi
  if kill -"$sig" "$pid" 2>/dev/null; then
    log "  sent SIG$sig to $pid"
    return 0
  fi
  log "  SIG$sig to $pid FAILED (already gone?)"
  return 1
}

wait_exit() {
  local pid=$1 secs=$2 waited=0
  while [ "$waited" -lt "$secs" ]; do
    alive "$pid" || return 0
    sleep 5
    waited=$((waited + 5))
  done
  alive "$pid" && return 1
  return 0
}

# SIGINT -> SIGTERM -> SIGKILL, each with its own budget, verifying between.
stop_pid_gracefully() {
  local pid=$1 label=$2
  log "stopping $label pid=$pid gracefully"
  signal_pid INT "$pid"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  if wait_exit "$pid" "$INT_GRACE"; then
    log "  $label pid=$pid exited after SIGINT"
    return 0
  fi
  log "  $label pid=$pid still alive after ${INT_GRACE}s, escalating to SIGTERM"
  signal_pid TERM "$pid"
  if wait_exit "$pid" "$TERM_GRACE"; then
    log "  $label pid=$pid exited after SIGTERM"
    return 0
  fi
  log "  $label pid=$pid still alive after ${TERM_GRACE}s, escalating to SIGKILL"
  signal_pid KILL "$pid"
  if wait_exit "$pid" "$KILL_WAIT"; then
    log "  $label pid=$pid exited after SIGKILL"
    return 0
  fi
  log "  ERROR: $label pid=$pid survived SIGKILL"
  return 1
}

# ----------------------------------------------------------------- stop logic --

freeze_supervisor() {
  local pids p st n=0
  mapfile -t pids < <(find_pids "$SUP_PATTERN")
  if [ "${#pids[@]}" -eq 0 ]; then
    log "no supervisor process matching '$SUP_PATTERN' -- nothing to freeze"
    return 0
  fi
  for p in "${pids[@]}"; do
    log "supervisor candidate pid=$p cmd=$(cmd_of "$p" | cut -c1-160)"
  done
  for p in "${pids[@]}"; do
    signal_pid STOP "$p"
    [ "$DRY_RUN" = "1" ] && continue
    sleep 1
    st=$(state_of "$p")
    if [ "$st" = "T" ]; then
      log "  supervisor pid=$p is STOPPED (state=T) -- restart loop frozen"
      n=$((n + 1))
    else
      log "  WARNING: supervisor pid=$p state=$st (expected T)"
    fi
  done
  printf '%s\n' "$n" >/dev/null
  return 0
}

kill_frozen_supervisor() {
  local pids p
  mapfile -t pids < <(find_pids "$SUP_PATTERN")
  if [ "${#pids[@]}" -eq 0 ]; then
    log "supervisor already gone"
    return 0
  fi
  for p in "${pids[@]}"; do
    # It is frozen, so SIGTERM would only be delivered on a later SIGCONT and the
    # restart loop could run one more iteration. SIGKILL acts on a stopped process
    # immediately. It is a supervisor shell with nothing to flush.
    log "killing frozen supervisor pid=$p (state=$(state_of "$p"))"
    signal_pid KILL "$p"
    [ "$DRY_RUN" = "1" ] && continue
    wait_exit "$p" 20 || log "  ERROR: supervisor pid=$p survived SIGKILL"
  done
  return 0
}

stop_trainer() {
  local pids leaves p rc=0
  mapfile -t pids < <(find_pids "$TRAINER_PATTERN")
  if [ "${#pids[@]}" -eq 0 ]; then
    log "no trainer process matching '$TRAINER_PATTERN' -- already stopped?"
    return 0
  fi
  for p in "${pids[@]}"; do
    log "trainer match pid=$p ppid=$(ppid_of "$p") cmd=$(cmd_of "$p" | cut -c1-120)"
  done
  mapfile -t leaves < <(leaf_pids "${pids[@]}")
  log "leaf trainer pid(s): ${leaves[*]:-none}"
  for p in "${leaves[@]}"; do
    stop_pid_gracefully "$p" "trainer(leaf)" || rc=1
  done
  # Whatever is left of the match set (the accelerate launcher) gets the same
  # ladder, so no orphan holds the GPU.
  mapfile -t pids < <(find_pids "$TRAINER_PATTERN")
  for p in "${pids[@]}"; do
    alive "$p" || continue
    stop_pid_gracefully "$p" "trainer(launcher)" || rc=1
  done
  return "$rc"
}

write_sentinel() {
  local step=$1 path=$2
  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] would write sentinel $SENTINEL"
    return 0
  fi
  {
    printf 'stopped_at_cumulative_step=%s\n' "$step"
    printf 'checkpoint=%s\n' "$path"
    printf 'stopped_by=%s\n' "$SELF_TAG"
    printf 'stopped_at=%s\n' "$(date -Is)"
    printf 'reason=intentional cap at cumulative step %s (train.steps intent)\n' "$TARGET"
    printf 'do_not_restart=1\n'
  } >"$SENTINEL"
  log "wrote sentinel $SENTINEL"
}

# After the intentional stop, keep looking: if a trainer reappears the restart
# prevention failed and that must be visible in the audit trail, not silent.
watch_for_resurrection() {
  local waited=0 pids p
  log "watching ${WATCH_AFTER}s for any resurrected trainer"
  while [ "$waited" -lt "$WATCH_AFTER" ]; do
    sleep 30
    waited=$((waited + 30))
    mapfile -t pids < <(find_pids "$TRAINER_PATTERN")
    if [ "${#pids[@]}" -gt 0 ]; then
      log "ALERT: trainer reappeared after intentional stop: ${pids[*]}"
      for p in "${pids[@]}"; do
        stop_pid_gracefully "$p" "resurrected-trainer" || true
      done
      mapfile -t pids < <(find_pids "$SUP_PATTERN")
      for p in "${pids[@]}"; do
        log "ALERT: supervisor also back pid=$p -- killing"
        signal_pid KILL "$p"
      done
    fi
  done
  log "no resurrection during ${WATCH_AFTER}s window"
}

# ---------------------------------------------------------------------- main --

log "=========================================================="
log "stop_at_step start: target_cumulative_step=$TARGET"
log "  lora_dir=$LORA_DIR"
log "  poll=${POLL}s stable_wait=${STABLE_WAIT}s quiesce=${QUIESCE}s dry_run=$DRY_RUN"
log "  sup_pattern='$SUP_PATTERN' trainer_pattern='$TRAINER_PATTERN'"
log "  counter=CUMULATIVE (step-N.safetensors filenames), NOT the tqdm in-pass bar"
log "  newest cumulative checkpoint right now: step-$(newest_ckpt_step)"

while :; do
  line=$(candidate_ckpt)
  if [ -n "$line" ]; then
    step=${line%%$'\t'*}
    path=${line#*$'\t'}
    log "candidate at/beyond target: step=$step path=$path"
    if ! quiesced; then
      log "  checkpoint dir still active (<${QUIESCE}s), waiting for a quiet window"
    elif verify_ckpt "$path"; then
      log "VERIFIED complete: $path ($(stat -c %s "$path") bytes)"
      log "TARGET REACHED -- beginning graceful stop"
      write_sentinel "$step" "$path"
      freeze_supervisor
      stop_trainer || log "WARNING: trainer stop reported a problem"
      kill_frozen_supervisor
      log "final on-disk cumulative checkpoint: step-$(newest_ckpt_step)"
      watch_for_resurrection
      log "STOP_COMPLETE target=$TARGET stopped_at_step=$step"
      log "next: from a local LoboRA checkout, pull the final checkpoint with"
      log "  python3 scripts/pull_latest_lora.py $step"
      exit 0
    fi
  else
    log "waiting: newest cumulative step-$(newest_ckpt_step) < target $TARGET"
  fi
  sleep "$POLL"
done
