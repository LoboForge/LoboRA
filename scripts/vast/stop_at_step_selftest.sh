#!/bin/bash
# stop_at_step_selftest.sh -- exercise stop_at_step.sh end to end WITHOUT touching
# the live training run.
#
# Everything under test is the real stop_at_step.sh next to this file, driven by env
# vars at a fake threshold against a fake checkpoint dir and stub processes that mimic
# the real shape: a replica of supervise_stage2.sh (unmodified except for its OUT
# path) supervising a two-level launcher/trainer pair, both of which carry the
# trainer pattern in argv so leaf detection is genuinely exercised.
#
# Run this ON THE BOX only: the final verdict asserts the live trainer and supervisor
# are still running, so it reports failures anywhere else.
#
# Scenario A: decoy attempt1_* ignored, truncated checkpoint rejected, valid
#             checkpoint accepted, graceful SIGINT stop, supervisor cannot restart.
# Scenario B: trainer ignores SIGINT and SIGTERM -> escalation to SIGKILL.
#
# Cleanup only ever signals PIDs this script itself recorded. No pkill/killall.
set -uo pipefail

ROOT=/tmp/stoptest
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WATCHER=${WATCHER:-$HERE/stop_at_step.sh}
SUPER_SRC=${SUPER_SRC:-$HERE/supervise_stage2.sh}
PY=${PY:-/workspace/venv/bin/python}

PASS=0
FAIL=0
TRACKED_PIDS=()

say() { printf '\n=== %s\n' "$*"; }
ok() {
  PASS=$((PASS + 1))
  printf 'PASS: %s\n' "$*"
}
bad() {
  FAIL=$((FAIL + 1))
  printf 'FAIL: %s\n' "$*"
}

check_absent() {
  local label=$1 file=$2 needle=$3
  if grep -q -- "$needle" "$file" 2>/dev/null; then
    bad "$label (unexpectedly found '$needle')"
  else
    ok "$label"
  fi
}

check_present() {
  local label=$1 file=$2 needle=$3
  if grep -q -- "$needle" "$file" 2>/dev/null; then
    ok "$label"
  else
    bad "$label (missing '$needle')"
  fi
}

cleanup() {
  local p
  for p in "${TRACKED_PIDS[@]:-}"; do
    [ -n "$p" ] || continue
    # A frozen process must be CONTinued or SIGKILL is the only thing it will act
    # on; kill -KILL works on stopped processes, so just use it directly.
    kill -KILL "$p" 2>/dev/null && printf 'cleanup: killed %s\n' "$p"
  done
}
trap cleanup EXIT

mk_safetensors() {
  # $1 path, $2 truncate_bytes (0 = leave complete)
  "$PY" - "$1" "$2" <<'PY'
import json
import os
import struct
import sys

path, trunc = sys.argv[1], int(sys.argv[2])
payload = b"\x00" * 4096
head = {
    "lora_unet_qkv_proj.lora_A.weight": {
        "dtype": "F32",
        "shape": [32, 32],
        "data_offsets": [0, len(payload)],
    },
    "__metadata__": {"note": "selftest stub"},
}
blob = json.dumps(head).encode()
with open(path, "wb") as fh:
    fh.write(struct.pack("<Q", len(blob)))
    fh.write(blob)
    fh.write(payload)
if trunc:
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - trunc)
PY
}

setup_case() {
  # $1 = case tag, $2 = "graceful"|"stubborn"
  # Declared separately: under `set -u` bash localises every name in a single
  # `local` before evaluating the right-hand sides, so `local tag=$1 d=$ROOT/$tag`
  # trips "tag: unbound variable".
  local tag=$1
  local mode=$2
  local d=$ROOT/$tag
  rm -rf "$d"
  mkdir -p "$d/lora"

  if [ "$mode" = "stubborn" ]; then
    cat >"$d/fake_trainer_$tag.py" <<'PY'
import signal
import sys
import time

signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("stub trainer up, ignoring INT/TERM", flush=True)
while True:
    time.sleep(1)
PY
  else
    cat >"$d/fake_trainer_$tag.py" <<'PY'
import signal
import sys
import time


def bye(signum, frame):
    print("stub trainer caught signal %d, flushing and exiting" % signum, flush=True)
    sys.exit(130)


signal.signal(signal.SIGINT, bye)
signal.signal(signal.SIGTERM, bye)
print("stub trainer up", flush=True)
while True:
    time.sleep(1)
PY
  fi

  # Mimics `accelerate launch`: the launcher's own argv also contains the trainer
  # path, which is exactly why the watcher needs leaf detection.
  cat >"$d/fake_launcher.py" <<'PY'
import subprocess
import sys

child = subprocess.Popen([sys.executable, sys.argv[1]])
rc = child.wait()
sys.exit(0 if rc == 0 else 1)
PY

  cat >"$d/stub_inner.sh" <<EOF
#!/bin/bash
exec $PY $d/fake_launcher.py $d/fake_trainer_$tag.py
EOF
  chmod +x "$d/stub_inner.sh"

  # Replica supervisor: byte-identical restart logic, only OUT is repointed.
  sed "s#^OUT=/workspace/output/anatomy_ref2va_a800#OUT=$d#" "$SUPER_SRC" \
    >"$d/supervise_replica_$tag.sh"
  chmod +x "$d/supervise_replica_$tag.sh"
  if grep -q "^OUT=$d\$" "$d/supervise_replica_$tag.sh"; then
    ok "$tag: replica supervisor repointed to sandbox dir"
  else
    bad "$tag: replica supervisor OUT rewrite failed"
  fi
}

start_replica() {
  local tag=$1
  local d=$ROOT/$tag
  env ANATOMY_INNER="$d/stub_inner.sh" \
    ANATOMY_LOG="$d/train.log" \
    ANATOMY_SUP="$d/supervisor.log" \
    ANATOMY_HB="$d/hb.txt" \
    ANATOMY_MAX_ATTEMPTS=8 \
    ANATOMY_RETRY_SLEEP=10 \
    setsid bash "$d/supervise_replica_$tag.sh" >"$d/replica_stdout.log" 2>&1 &
  local pid=$!
  TRACKED_PIDS+=("$pid")
  sleep 4
  printf '%s\n' "$pid"
}

# Extra env overrides arrive as "NAME=value" words in "$@". They must be passed to
# `env`, not used as shell assignment prefixes: bash only recognises a prefix
# assignment when the word is literal, so a "$@" expansion would be treated as the
# command name instead.
start_watcher() {
  local tag=$1
  local d=$ROOT/$tag
  shift
  env STOP_TARGET_STEP=2000 \
    STOP_LORA_DIR="$d/lora" \
    STOP_LOG="$d/stop_at_step.log" \
    STOP_SENTINEL="$d/STOP.sentinel" \
    STOP_POLL_SECS=8 \
    STOP_STABLE_WAIT=3 \
    STOP_QUIESCE_SECS=3 \
    STOP_SUP_PATTERN="supervise_replica_$tag.sh" \
    STOP_TRAINER_PATTERN="fake_trainer_$tag.py" \
    STOP_PY="$PY" \
    STOP_WATCH_AFTER="${WATCH_AFTER:-40}" \
    "$@" \
    setsid bash "$WATCHER" >"$d/watcher_stdout.log" 2>&1 &
  local pid=$!
  TRACKED_PIDS+=("$pid")
  printf '%s\n' "$pid"
}

pids_matching() {
  local pat=$1 pid cmd out=()
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)
    case "$cmd" in
      *selftest* | *stop_at_step*) continue ;;
      *"$pat"*) out+=("$pid") ;;
    esac
  done
  printf '%s\n' "${out[@]:-}"
}

count_matching() {
  local n
  n=$(pids_matching "$1" | grep -c '[0-9]' )
  printf '%s\n' "$n"
}

############################################################ preflight

say "preflight: syntax check on the watcher"
if bash -n "$WATCHER"; then ok "watcher parses"; else bad "watcher has a syntax error"; exit 1; fi

say "preflight: dry-run against the LIVE dir with an unreachable threshold"
# Proves the real patterns match the real processes and that nothing is signalled:
# threshold 999999 can never be met, and STOP_DRY_RUN blocks every kill anyway.
mkdir -p "$ROOT"
STOP_TARGET_STEP=999999 STOP_DRY_RUN=1 STOP_POLL_SECS=3 \
  STOP_LOG=$ROOT/live_dryrun.log timeout 10 bash "$WATCHER" >/dev/null 2>&1
check_present "live dry-run reads the cumulative counter" "$ROOT/live_dryrun.log" "counter=CUMULATIVE"
# The in-pass tqdm bar is in the tens right now; the cumulative counter is 600+.
# A number >= 600 therefore proves which of the two the watcher is reading.
CUM=$(grep -o "newest cumulative checkpoint right now: step-[0-9]*" "$ROOT/live_dryrun.log" |
  tail -1 | sed 's/.*step-//')
if [ -n "$CUM" ] && [ "$CUM" -ge 600 ]; then
  ok "live dry-run reads the CUMULATIVE checkpoint counter (step-$CUM >= 600, not the ~tens in-pass bar)"
else
  bad "live dry-run reported step-'${CUM:-none}', which is not the cumulative counter"
fi
check_absent "live dry-run sent no signals" "$ROOT/live_dryrun.log" "sent SIG"

############################################################ scenario A

say "scenario A: decoy + truncated + valid, graceful stop, no restart"
setup_case A graceful
DA=$ROOT/A

mk_safetensors "$DA/lora/step-1975.safetensors" 0
mk_safetensors "$DA/lora/attempt1_step-9999.safetensors" 0   # decoy, must be ignored
mk_safetensors "$DA/lora/step-2000.safetensors" 512           # truncated, must be rejected

SUP_A=$(start_replica A)
TRAINER_A=$(pids_matching fake_trainer_A.py | head -1)
if [ -n "$TRAINER_A" ]; then ok "A: stub trainer running (pid $TRAINER_A)"; else bad "A: stub trainer never started"; fi
if [ "$(count_matching fake_trainer_A.py)" -ge 2 ]; then
  ok "A: two-level launcher/trainer present, leaf detection will be exercised"
else
  bad "A: expected >=2 matching processes (launcher + trainer)"
fi

WATCH_A=$(start_watcher A)
sleep 20

check_absent "A: did NOT fire on the truncated step-2000" "$DA/stop_at_step.log" "TARGET REACHED"
check_present "A: rejected the truncated checkpoint" "$DA/stop_at_step.log" "header/payload check failed"
check_absent "A: ignored the attempt1_ decoy (never treated 9999 as newest)" "$DA/stop_at_step.log" "step-9999"
if [ -n "$(pids_matching fake_trainer_A.py | head -1)" ]; then
  ok "A: trainer still running while checkpoint was incomplete"
else
  bad "A: trainer was stopped before a valid checkpoint existed"
fi

say "scenario A: writing a COMPLETE step-2000"
mk_safetensors "$DA/lora/step-2000.safetensors" 0

for _ in $(seq 1 24); do
  grep -q "STOP_COMPLETE" "$DA/stop_at_step.log" 2>/dev/null && break
  sleep 5
done

check_present "A: verified the complete checkpoint" "$DA/stop_at_step.log" "VERIFIED complete"
check_present "A: fired on the cumulative target" "$DA/stop_at_step.log" "TARGET REACHED"
check_present "A: froze the supervisor before signalling" "$DA/stop_at_step.log" "restart loop frozen"
check_present "A: stopped the trainer with SIGINT" "$DA/stop_at_step.log" "sent SIGINT"
check_present "A: trainer exited on SIGINT (no SIGKILL needed)" "$DA/stop_at_step.log" "exited after SIGINT"
check_present "A: killed the frozen supervisor afterwards" "$DA/stop_at_step.log" "killing frozen supervisor"
check_present "A: wrote the sentinel" "$DA/stop_at_step.log" "wrote sentinel"
check_present "A: reached STOP_COMPLETE" "$DA/stop_at_step.log" "STOP_COMPLETE"

# Ordering matters: freeze must precede the first trainer signal.
FREEZE_LINE=$(grep -n "restart loop frozen" "$DA/stop_at_step.log" | head -1 | cut -d: -f1)
SIGNAL_LINE=$(grep -n "sent SIGINT" "$DA/stop_at_step.log" | head -1 | cut -d: -f1)
if [ -n "$FREEZE_LINE" ] && [ -n "$SIGNAL_LINE" ] && [ "$FREEZE_LINE" -lt "$SIGNAL_LINE" ]; then
  ok "A: supervisor frozen BEFORE the trainer was signalled (line $FREEZE_LINE < $SIGNAL_LINE)"
else
  bad "A: freeze/signal ordering wrong (freeze=$FREEZE_LINE signal=$SIGNAL_LINE)"
fi

if [ -z "$(pids_matching fake_trainer_A.py | head -1)" ]; then
  ok "A: no trainer process remains"
else
  bad "A: trainer process survived: $(pids_matching fake_trainer_A.py | tr '\n' ' ')"
fi
if [ -z "$(pids_matching supervise_replica_A.sh | head -1)" ]; then
  ok "A: no supervisor process remains"
else
  bad "A: supervisor survived: $(pids_matching supervise_replica_A.sh | tr '\n' ' ')"
fi

say "scenario A: waiting past RETRY_SLEEP to prove no resurrection"
sleep 25
ATTEMPTS=$(grep -c "ATTEMPT" "$DA/supervisor.log" 2>/dev/null || echo 0)
if [ "$ATTEMPTS" -le 1 ]; then
  ok "A: supervisor logged only attempt 1 -- restart prevented ($ATTEMPTS attempt lines)"
else
  bad "A: supervisor started another attempt ($ATTEMPTS attempt lines)"
fi
check_absent "A: supervisor never announced a retry sleep completing into attempt 2" "$DA/supervisor.log" "ATTEMPT 2/"
if [ -z "$(pids_matching fake_trainer_A.py | head -1)" ]; then
  ok "A: still no trainer 25s after the stop"
else
  bad "A: a trainer came back 25s after the stop"
fi
check_present "A: watcher confirmed no resurrection" "$DA/stop_at_step.log" "no resurrection"

############################################################ scenario B

say "scenario B: trainer ignores SIGINT/SIGTERM, escalation must reach SIGKILL"
setup_case B stubborn
DB=$ROOT/B
mk_safetensors "$DB/lora/step-2000.safetensors" 0
SUP_B=$(start_replica B)
if [ -n "$(pids_matching fake_trainer_B.py | head -1)" ]; then
  ok "B: stubborn stub trainer running"
else
  bad "B: stubborn stub trainer never started"
fi

WATCH_B=$(start_watcher B STOP_INT_GRACE=10 STOP_TERM_GRACE=10 STOP_KILL_WAIT=10)
for _ in $(seq 1 30); do
  grep -q "STOP_COMPLETE" "$DB/stop_at_step.log" 2>/dev/null && break
  sleep 5
done

check_present "B: escalated to SIGTERM" "$DB/stop_at_step.log" "escalating to SIGTERM"
check_present "B: escalated to SIGKILL" "$DB/stop_at_step.log" "escalating to SIGKILL"
check_present "B: trainer died on SIGKILL" "$DB/stop_at_step.log" "exited after SIGKILL"
check_absent "B: nothing survived SIGKILL" "$DB/stop_at_step.log" "survived SIGKILL"
check_present "B: reached STOP_COMPLETE" "$DB/stop_at_step.log" "STOP_COMPLETE"
if [ -z "$(pids_matching fake_trainer_B.py | head -1)" ]; then
  ok "B: no stubborn trainer remains"
else
  bad "B: stubborn trainer survived"
fi

############################################################ verdict

say "the live run must be untouched by all of the above"
if [ -n "$(pids_matching 'examples/minimax_h3/model_training/train.py' | head -1)" ]; then
  ok "live trainer still running (pids: $(pids_matching 'examples/minimax_h3/model_training/train.py' | tr '\n' ' '))"
else
  bad "LIVE TRAINER IS GONE -- investigate immediately"
fi
if [ -n "$(pids_matching supervise_stage2.sh | head -1)" ]; then
  ok "live supervisor still running"
else
  bad "LIVE SUPERVISOR IS GONE -- investigate immediately"
fi

say "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
