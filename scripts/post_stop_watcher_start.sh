#!/bin/bash
# Launch post_stop_watcher.py detached so it survives this terminal closing.
# Idempotent: the watcher itself refuses to start if another one holds the lock.
#
# The log, state and lock go to PSW_ARTIFACT_DIR (default ~/.lobora/post_stop_watcher),
# never into this checkout. Run this on a machine that stays awake: a suspended
# laptop is a watcher that never fires, and the instance keeps billing.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ART=${PSW_ARTIFACT_DIR:-$HOME/.lobora/post_stop_watcher}
PY=${PSW_PYTHON:-python3}

mkdir -p "$ART" || exit 1
export PSW_ARTIFACT_DIR="$ART"
cd "$ART" || exit 1
setsid nohup "$PY" "$HERE/post_stop_watcher.py" \
  >>"$ART/post_stop_watcher.stdout" 2>&1 &
pid=$!
sleep 3
if kill -0 "$pid" 2>/dev/null; then
  printf 'post_stop_watcher started, pid %s\n' "$pid"
  printf 'log:   %s/post_stop_watcher.log\n' "$ART"
  printf 'state: %s/post_stop_watcher.state.json\n' "$ART"
  printf 'cancel: kill %s   (nothing is stopped on the box by cancelling)\n' "$pid"
  exit 0
fi
printf 'post_stop_watcher exited immediately -- check %s/post_stop_watcher.stdout\n' "$ART"
exit 1
