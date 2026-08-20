#!/bin/bash
# Launch lora_pull_watcher.py detached so it survives this terminal closing.
# Idempotent: the watcher itself refuses to start if a live one holds the lock, and
# takes over a lock whose owner is dead.
#
# Log, state and lock go to LPW_ARTIFACT_DIR (default ~/.lobora/lora_pull_watcher),
# never into this checkout.
#
# It pulls, and only pulls. There is no stop path anywhere in it, so leaving it
# running cannot end the run or the billing; stopping the instance stays a human
# decision. It does the final backfill itself when the run ends rather than
# handing that to another process.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ART=${LPW_ARTIFACT_DIR:-$HOME/.lobora/lora_pull_watcher}
PY=${LPW_PYTHON:-python3}

mkdir -p "$ART" || exit 1
export LPW_ARTIFACT_DIR="$ART"
cd "$ART" || exit 1
LOG=${LPW_LOG:-$ART/lora_pull_watcher.log}
LOCK=${LPW_LOCK:-$ART/lora_pull_watcher.lock}

setsid nohup "$PY" "$HERE/lora_pull_watcher.py" \
  >>"$ART/lora_pull_watcher.stdout" 2>&1 &
sleep 3

# $! is useless here: setsid forks, so the job this shell backgrounded has already
# exited and the watcher is reparented to init. The watcher writes its own pid to
# the lock file, and removes it on exit -- that is the thing to ask.
pid=$(cat "$LOCK" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  printf 'lora_pull_watcher running, pid %s\n' "$pid"
  printf 'log:   %s\n' "$LOG"
  printf 'state: %s/lora_pull_watcher.state.json\n' "$ART"
  printf 'stop:  kill %s   (pulls stop; nothing on the box is touched)\n' "$pid"
  exit 0
fi

# No live pid. Exiting at once is a legitimate outcome -- the run may already be
# over, or another watcher may hold the lock -- so report the verdict, not a guess.
verdict=$(grep -a 'FINAL:\|already running (pid' "$LOG" 2>/dev/null | tail -1)
if [ -n "$verdict" ]; then
  printf 'lora_pull_watcher exited immediately, on purpose:\n  %s\n' "$verdict"
  printf 'log: %s\n' "$LOG"
  exit 0
fi
printf 'lora_pull_watcher exited immediately with no verdict -- check %s\n' \
  "$ART/lora_pull_watcher.stdout"
exit 1
