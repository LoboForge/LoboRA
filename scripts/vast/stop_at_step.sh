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
# WHAT IT MUST NEVER SIGNAL (learned the hard way, see RUNBOOK section 10)
#   The first version matched two "supervisor candidates": the real
#   `bash supervise_stage2.sh`, and the TMUX SERVER, whose argv is still that of
#   whatever forked it (`tmux new-session -d -s anatomy_train ...`) and which is
#   not a supervisor at all. It SIGSTOPped both, stopped the trainer, then logged
#   `killing frozen supervisor pid=<tmux server>` and SIGKILLed it -- destroying
#   every session on that server including the one it was running in. Its log ends
#   mid-sequence on that line, with no verification and no completion.
#
#   So: a supervisor must match the pattern AND be a shell AND not be tmux, and
#   signal_pid refuses outright to signal this script, any of its ancestors, pid 1,
#   or anything that is a tmux server or client. A misclassification now costs a
#   log line instead of the session.
#
# ORDER OF OPERATIONS AT FIRE TIME
#   1. verify the target checkpoint is complete and size-stable
#   2. SIGSTOP the supervisor -- freezes its restart loop WITHOUT closing the tmux
#      pane. Killing it instead would make tmux tear the pane down and SIGHUP the
#      whole process group (the trainer shares the supervisor's pgid), killing the
#      trainer uncontrolled. Freezing keeps the shutdown ours to sequence.
#   3. stop the WHOLE trainer tree, leaves first: the leaf trainer, then the
#      `accelerate launch` parent that also carries train.py in its argv. Stopping
#      only the leaf leaves the launcher holding the GPU, which is what happened.
#      Each gets SIGINT, escalating to SIGTERM then SIGKILL on timeouts.
#   4. CONT the frozen supervisor for a few seconds so it can wait() its dead
#      children, then re-freeze and SIGKILL it. A SIGSTOPped parent can never
#      reap, so anything killed under a frozen parent stays a zombie; the window
#      is bounded and far shorter than the supervisor's retry sleep, and any
#      trainer that appears during it is killed and the supervisor re-frozen, so
#      the no-relaunch guarantee is unchanged.
#   5. verify its own work -- recount both patterns and record a verdict in the log
#      and in the sentinel. The previous version died before finishing and left no
#      record of the outcome at all.
#   6. keep watching, so a resurrection is caught and logged rather than silent
#
# Signals go to specific PIDs discovered and logged at fire time. pkill/killall are
# banned in this project and are never used here. There are deliberately no pipes
# in any exit-status path: a previous bug in this project was `tee` swallowing a
# status via $? instead of ${PIPESTATUS[0]}, so log() writes with two printfs.
#
# Liveness is read from /proc/<pid>/stat, not from `kill -0`: a zombie answers
# `kill -0` forever, so a ladder that waits for `kill -0` to fail never finishes.
# These are the same rules lobora/procscan.py applies on the python side; the
# self-test cross-checks the two matchers against each other.
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
# Seconds the supervisor is allowed to run so it can wait() its dead children.
# Must stay well under the supervisor's H3_RETRY_SLEEP (60s) so it cannot reach
# the point of launching another attempt.
REAP_WINDOW=${STOP_REAP_WINDOW:-10}
# Only ever anything else under test: tests/test_stop_at_step_matcher.py points it
# at a synthetic tree and checks these rules agree with lobora/procscan.py.
PROC=${STOP_PROC_ROOT:-/proc}

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

cmd_of() { tr '\0' ' ' <"$PROC/$1/cmdline" 2>/dev/null; }

# From stat, not /proc/<pid>/comm, so it is the same field lobora/procscan.py
# reads. Everything between the FIRST '(' and the LAST ')' -- a comm may contain
# either character ("tmux: server" does not, but the parse should not depend on
# that).
comm_of() {
  local s
  s=$(cat "$PROC/$1/stat" 2>/dev/null) || return 0
  s=${s#*(}
  printf '%s' "${s%)*}"
}

ppid_of() {
  # Everything after the last ')' is: state ppid ... -- avoids comm containing
  # spaces or parens throwing the field count off.
  sed 's/.*)//' "$PROC/$1/stat" 2>/dev/null | awk '{print $2}'
}

state_of() { sed 's/.*)//' "$PROC/$1/stat" 2>/dev/null | awk '{print $1}'; }

# Z (zombie) and X/x (dead) are not alive: they hold no GPU memory and cannot run
# an instruction, but they answer `kill -0` and appear in `ps` indefinitely. An
# empty state means the process is gone.
undead() {
  case "$(state_of "$1")" in Z | X | x) return 0 ;; *) return 1 ;; esac
}

alive() {
  local st
  st=$(state_of "$1")
  case "$st" in
    '' | Z | X | x) return 1 ;;
    *) return 0 ;;
  esac
}

# tmux keeps the argv of whatever forked the server, so `tmux new-session -d -s
# anatomy_train bash supervise_stage2.sh` leaves a server process whose cmdline
# contains the supervisor pattern. It is not a supervisor, and killing it takes
# down every session on that socket -- including this script's own.
is_tmux() {
  case "$(comm_of "$1")" in tmux*) return 0 ;; esac
  case "$(cmd_of "$1")" in
    tmux\ * | */tmux\ * | tmux) return 0 ;;
  esac
  return 1
}

is_shell() {
  case "$(comm_of "$1")" in
    bash | sh | dash | zsh | ksh) return 0 ;;
    *) return 1 ;;
  esac
}

# This script, everything that spawned it, and init. Computed once at start, and
# consulted before every single signal.
protected_pids() {
  local p=$$ pp n=0
  printf '%s\n1\n' "$$"
  while [ "$n" -lt 64 ]; do
    pp=$(ppid_of "$p")
    case "$pp" in '' | 0) break ;; esac
    printf '%s\n' "$pp"
    p=$pp
    n=$((n + 1))
  done
}
PROTECTED=$(protected_pids)

is_protected() {
  case "
$PROTECTED
" in *"
$1
"*) return 0 ;; esac
  return 1
}

# PIDs whose cmdline contains $1. Reads /proc directly rather than shelling out to
# `ps | grep`: in a pipeline the child expands the glob and hands grep grep's own
# /proc entry, which holds grep's argv -- pattern included -- so a grep matcher
# always finds one process that is not there. Skips this script, its subshells
# (a forked subshell keeps the script's argv), its ancestors, and tmux.
find_pids() {
  local pat=$1 pid cmd
  for pid in "$PROC"/[0-9]*; do
    pid=${pid##*/}
    is_protected "$pid" && continue
    cmd=$(cmd_of "$pid")
    [ -n "$cmd" ] || continue
    case "$cmd" in *"$SELF_TAG"*) continue ;; esac
    case "$cmd" in *"$pat"*) ;; *) continue ;; esac
    is_tmux "$pid" && continue
    undead "$pid" && continue
    printf '%s\n' "$pid"
  done
}

# Matches that are in Z/X state: reported so they are visible in the audit trail,
# never signalled and never waited for.
find_undead_pids() {
  local pat=$1 pid cmd
  for pid in "$PROC"/[0-9]*; do
    pid=${pid##*/}
    is_protected "$pid" && continue
    cmd=$(cmd_of "$pid")
    [ -n "$cmd" ] || continue
    case "$cmd" in *"$pat"*) ;; *) continue ;; esac
    undead "$pid" && printf '%s\n' "$pid"
  done
}

# A supervisor is a shell running the supervisor script. Not tmux, whatever its
# argv says; not this script; not an ancestor of it.
supervisor_pids() {
  local pid
  for pid in $(find_pids "$SUP_PATTERN"); do
    if ! is_shell "$pid"; then
      log "  ignoring pid=$pid as a supervisor: comm='$(comm_of "$pid")' is not a" \
        "shell (cmd: $(cmd_of "$pid" | cut -c1-100))"
      continue
    fi
    printf '%s\n' "$pid"
  done
}

# Children of $1 that have exited and not been reaped.
zombie_children_of() {
  local parent=$1 pid
  for pid in "$PROC"/[0-9]*; do
    pid=${pid##*/}
    [ "$(ppid_of "$pid")" = "$parent" ] || continue
    undead "$pid" && printf '%s\n' "$pid"
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

signal_pid() {
  local sig=$1 pid=$2
  # The last line of defence against the tmux-server incident: whatever any
  # matcher decided, these are never signalled.
  if is_protected "$pid"; then
    log "  REFUSING to send SIG$sig to $pid: it is this script, one of its" \
      "ancestors, or init"
    return 1
  fi
  if is_tmux "$pid"; then
    log "  REFUSING to send SIG$sig to $pid: it is tmux ('$(comm_of "$pid")')." \
      "Killing a tmux server tears down every session on that socket."
    return 1
  fi
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
  # SIGKILL cannot land on a process in uninterruptible sleep, which is where a
  # CUDA/NCCL teardown can park one. Say so precisely instead of "survived".
  log "  ERROR: $label pid=$pid survived SIGKILL (state=$(state_of "$pid")); if it" \
    "is D it is blocked in the kernel and no signal can reach it"
  return 1
}

# ----------------------------------------------------------------- stop logic --

freeze_supervisor() {
  local pids p st
  mapfile -t pids < <(supervisor_pids)
  if [ "${#pids[@]}" -eq 0 ]; then
    log "no supervisor process matching '$SUP_PATTERN' -- nothing to freeze"
    return 0
  fi
  for p in "${pids[@]}"; do
    log "supervisor pid=$p comm='$(comm_of "$p")' cmd=$(cmd_of "$p" | cut -c1-160)"
  done
  for p in "${pids[@]}"; do
    signal_pid STOP "$p"
    [ "$DRY_RUN" = "1" ] && continue
    sleep 1
    st=$(state_of "$p")
    if [ "$st" = "T" ]; then
      log "  supervisor pid=$p is STOPPED (state=T) -- restart loop frozen"
    else
      log "  WARNING: supervisor pid=$p state=$st (expected T)"
    fi
  done
  return 0
}

# A SIGSTOPped parent cannot call wait(), so children killed under it stay
# zombies. Let it run just long enough to reap them, watching for a relaunch, then
# freeze it again. The window is far shorter than its retry sleep, and any trainer
# that does appear is stopped, so "the supervisor cannot start a new attempt"
# still holds.
reap_frozen_supervisor() {
  local p=$1 waited=0 zc new q
  zc=$(zombie_children_of "$p" | tr '\n' ' ')
  if [ -z "${zc// /}" ]; then
    log "  supervisor pid=$p has no unreaped children"
    return 0
  fi
  log "  supervisor pid=$p has unreaped child(ren): $zc -- CONTinuing it for up to" \
    "${REAP_WINDOW}s so it can wait() them"
  [ "$DRY_RUN" = "1" ] && return 0
  signal_pid CONT "$p"
  while [ "$waited" -lt "$REAP_WINDOW" ]; do
    sleep 1
    waited=$((waited + 1))
    new=$(find_pids "$TRAINER_PATTERN" | tr '\n' ' ')
    if [ -n "${new// /}" ]; then
      log "  ALERT: supervisor spawned a trainer during the reap window ($new) --" \
        "re-freezing and stopping it"
      signal_pid STOP "$p"
      for q in $new; do
        stop_pid_gracefully "$q" "reap-window-trainer" || true
      done
      break
    fi
    zc=$(zombie_children_of "$p" | tr '\n' ' ')
    if [ -z "${zc// /}" ]; then
      log "  supervisor pid=$p reaped its children after ${waited}s"
      break
    fi
  done
  signal_pid STOP "$p"
  log "  supervisor pid=$p re-frozen (state=$(state_of "$p"))"
  return 0
}

kill_frozen_supervisor() {
  local pids p
  mapfile -t pids < <(supervisor_pids)
  if [ "${#pids[@]}" -eq 0 ]; then
    log "supervisor already gone"
    return 0
  fi
  for p in "${pids[@]}"; do
    reap_frozen_supervisor "$p"
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
  local pids leaves rest p q is_leaf rc=0 dead
  mapfile -t pids < <(find_pids "$TRAINER_PATTERN")
  mapfile -t dead < <(find_undead_pids "$TRAINER_PATTERN")
  [ "${#dead[@]}" -gt 0 ] &&
    log "trainer matches already dead (zombie/X), ignored: ${dead[*]}"
  if [ "${#pids[@]}" -eq 0 ]; then
    log "no live trainer process matching '$TRAINER_PATTERN' -- already stopped?"
    return 0
  fi
  for p in "${pids[@]}"; do
    log "trainer match pid=$p ppid=$(ppid_of "$p") state=$(state_of "$p")" \
      "cmd=$(cmd_of "$p" | cut -c1-120)"
  done
  mapfile -t leaves < <(leaf_pids "${pids[@]}")
  log "leaf trainer pid(s): ${leaves[*]:-none}"
  for p in "${leaves[@]}"; do
    stop_pid_gracefully "$p" "trainer(leaf)" || rc=1
  done

  # The rest of the match set is the `accelerate launch` parent, which carries
  # train.py in its own argv. Signalling only the leaf leaves it holding the GPU,
  # so it gets the same ladder -- named, from the ORIGINAL match set, so a parent
  # that is only reachable through a leaf that has now gone is still handled.
  rest=()
  for p in "${pids[@]}"; do
    is_leaf=0
    for q in "${leaves[@]}"; do [ "$p" = "$q" ] && is_leaf=1; done
    [ "$is_leaf" -eq 1 ] && continue
    rest+=("$p")
  done
  log "trainer launcher pid(s) (accelerate and friends): ${rest[*]:-none}"
  for p in "${rest[@]}"; do
    alive "$p" || {
      log "  launcher pid=$p already exited with its child (state=$(state_of "$p"))"
      continue
    }
    stop_pid_gracefully "$p" "trainer(launcher)" || rc=1
  done

  # Anything new that matched while the ladder ran.
  mapfile -t pids < <(find_pids "$TRAINER_PATTERN")
  for p in "${pids[@]}"; do
    log "  late trainer match pid=$p -- stopping it too"
    stop_pid_gracefully "$p" "trainer(late)" || rc=1
  done
  return "$rc"
}

# The previous version left no record of how it ended, because it killed the tmux
# server it was running in and died on that line. Say what is true afterwards.
verify_stop_complete() {
  local step=$1 t s zt rc=0
  mapfile -t t < <(find_pids "$TRAINER_PATTERN")
  mapfile -t s < <(supervisor_pids)
  mapfile -t zt < <(find_undead_pids "$TRAINER_PATTERN")
  log "STOP_VERDICT stopped_at_step=$step live_trainers=${#t[@]}" \
    "live_supervisors=${#s[@]} unreaped_trainers=${#zt[@]}"
  [ "${#t[@]}" -eq 0 ] || {
    log "  ERROR: trainer still live: ${t[*]} (states: $(for p in "${t[@]}"; do
      printf '%s=%s ' "$p" "$(state_of "$p")"
    done))"
    rc=1
  }
  [ "${#s[@]}" -eq 0 ] || {
    log "  ERROR: supervisor still live: ${s[*]}"
    rc=1
  }
  [ "${#zt[@]}" -eq 0 ] ||
    log "  note: ${#zt[@]} unreaped trainer entr(y|ies) remain (${zt[*]});" \
      "they hold no GPU memory and will be reaped when their parent exits"
  if [ "$DRY_RUN" != "1" ] && [ -f "$SENTINEL" ]; then
    {
      printf 'verified_at=%s\n' "$(date -Is)"
      printf 'live_trainers_after_stop=%s\n' "${#t[@]}"
      printf 'live_supervisors_after_stop=%s\n' "${#s[@]}"
      printf 'unreaped_trainer_entries=%s\n' "${#zt[@]}"
      printf 'stop_verdict=%s\n' "$([ "$rc" -eq 0 ] && echo clean || echo incomplete)"
    } >>"$SENTINEL"
  fi
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
      mapfile -t pids < <(supervisor_pids)
      for p in "${pids[@]}"; do
        log "ALERT: supervisor also back pid=$p -- killing"
        signal_pid KILL "$p"
      done
    fi
  done
  log "no resurrection during ${WATCH_AFTER}s window"
}

# ---------------------------------------------------------------------- main --

# `--list-matches`: print what the matcher sees and exit, signalling nothing. Two
# uses. On the box it answers "what would you touch?" before anyone arms this.
# Off the box, tests/test_stop_at_step_matcher.py points STOP_PROC_ROOT at a
# synthetic /proc and asserts these rules agree with lobora/procscan.py -- the
# bash rules are a second implementation only because the box has no checkout of
# this repo, and a second implementation that nothing compares is how the
# self-matching grep survived in two files at once.
if [ "${1:-}" = "--list-matches" ]; then
  # Diagnostics to stderr so stdout is exactly the sections, and so this mode
  # never has to create or write the real log file.
  log() { printf '%s\n' "$*" >&2; }
  emit_section() {
    local name=$1 p live=0 undead=0
    printf '###%s\n' "$name"
    for p in $2; do
      printf '%s %s\n' "$p" "$(state_of "$p")"
      live=$((live + 1))
    done
    for p in $3; do
      printf '%s %s\n' "$p" "$(state_of "$p")"
      undead=$((undead + 1))
    done
    printf 'OK %s %s\n' "$live" "$undead"
  }
  emit_section TRAINERS "$(find_pids "$TRAINER_PATTERN")" \
    "$(find_undead_pids "$TRAINER_PATTERN")"
  emit_section SUPERVISORS "$(supervisor_pids)" ""
  printf '###PROTECTED\n%s\n' "$(printf '%s' "$PROTECTED" | tr '\n' ' ')"
  exit 0
fi

log "=========================================================="
log "stop_at_step start: target_cumulative_step=$TARGET"
log "  lora_dir=$LORA_DIR"
log "  poll=${POLL}s stable_wait=${STABLE_WAIT}s quiesce=${QUIESCE}s dry_run=$DRY_RUN"
log "  sup_pattern='$SUP_PATTERN' trainer_pattern='$TRAINER_PATTERN'"
log "  counter=CUMULATIVE (step-N.safetensors filenames), NOT the tqdm in-pass bar"
log "  protected pids (never signalled: self, ancestors, init): $(
  printf '%s' "$PROTECTED" | tr '\n' ' '
)"
log "  tmux processes are never classified as supervisors and never signalled"
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
      verify_stop_complete "$step" ||
        log "WARNING: the stop is INCOMPLETE -- see the verdict above; something is" \
          "still live and may still be holding the GPU"
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
