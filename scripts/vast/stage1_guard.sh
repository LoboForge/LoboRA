#!/usr/bin/env bash
# stage1_guard.sh -- refuse to start stage 1 when there is something to lose.
#
# WHY
#   Stage 1 encodes 963 clips into $CACHE. It costs about EIGHT HOURS of A6000
#   time, and on a rented box it is the only artefact that cannot be re-fetched:
#   the LoRA can be pulled, the dataset can be re-uploaded, the cache has to be
#   recomputed. It is also one tab-completion away -- `./run_anatomy_train.sh` is
#   the obvious thing to type, it is what you typed the first time, and it starts
#   stage 1 with no questions asked.
#
#   Two accidents this prevents:
#     1. Re-running the entrypoint while a populated cache exists. Stage 1 with an
#        unchanged geometry only verifies, so this is usually survivable -- but it
#        pins the GPU for the length of that verification, and if HEIGHT / WIDTH /
#        NUM_FRAMES differ from the cached geometry it re-encodes for real and the
#        old cache is gone.
#     2. Re-running it DURING a live stage-2 run. That is two jobs on one GPU:
#        stage 2 OOMs, the supervisor relaunches into a half-dead GPU, and hours
#        of paid training are lost.
#
# WHAT IT DOES NOT DO
#   It never writes, moves, re-caches or deletes anything under $CACHE. It counts
#   files and reads geometry. Refusal is the only action it has.
#
# HOW TO GET PAST IT
#   Deliberately, in two steps, with the number of hours in your hand:
#
#     STAGE1_FORCE=1 ./run_anatomy_train.sh          -> still refuses, and prints
#                                                       the exact token to use
#     STAGE1_FORCE=<token> ./run_anatomy_train.sh    -> proceeds
#
#   The token names the run and the cache size, so it cannot be copied out of a
#   chat log from last week and pasted into a different situation. A live trainer
#   is NOT overridable by any token: stop the run first, on purpose.
#
# EXIT CODES
#   0 clear to proceed   3 refused: cache exists   4 refused: training is live
set -uo pipefail

# Anything that looks like a cache entry. Stage 1 writes one tensor file per clip.
stage1_cache_entries() {
  local cache=$1
  [ -d "$cache" ] || {
    printf '0\n'
    return 0
  }
  find "$cache" -maxdepth 2 -type f \
    \( -name '*.pth' -o -name '*.pt' -o -name '*.safetensors' -o -name '*.npy' \) \
    2>/dev/null | wc -l
}

# Live trainer processes, counted by reading /proc directly. NOT with
# `ps | grep`: in a pipeline the child expands the glob and hands grep grep's own
# /proc entry, whose contents are by then grep's argv -- pattern included -- so a
# grep-based count never reaches zero and a guard built on it always refuses.
# Z/X entries are skipped: a zombie holds no GPU memory.
stage1_live_trainers() {
  local pat=$1 pid cmd state n=0
  for pid in /proc/[0-9]*; do
    pid=${pid##*/}
    [ "$pid" = "$$" ] && continue
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)
    [ -n "$cmd" ] || continue
    case "$cmd" in *stage1_guard* | *"$0"*) continue ;; esac
    case "$cmd" in *"$pat"*) ;; *) continue ;; esac
    state=$(sed 's/.*)//' "/proc/$pid/stat" 2>/dev/null | awk '{print $1}')
    case "$state" in Z | X | x | '') continue ;; esac
    n=$((n + 1))
  done
  printf '%s\n' "$n"
}

stage1_guard() {
  local cache=$1 run=$2 geometry=$3 pat=${4:-examples/minimax_h3/model_training/train.py}
  local entries trainers token
  # When sourced, $0 is the sourcing shell -- often literally "bash", which makes
  # the printed override line unusable. The entrypoint names itself.
  local entry=${STAGE1_ENTRYPOINT:-$0}

  trainers=$(stage1_live_trainers "$pat")
  if [ "$trainers" -gt 0 ]; then
    cat >&2 <<EOF

  REFUSING TO START STAGE 1: training is running right now.

  $trainers live process(es) match '$pat'.

  Stage 1 would put a second job on the same GPU. The running one OOMs, its
  supervisor reads that as a crash and relaunches into a half-dead GPU, and you
  lose hours of paid training -- for a stage that, if the cache is already there,
  had nothing to do anyway.

  No token overrides this. If you really mean it, stop the run first:
      bash scripts/vast/stop_at_step.sh   (or stop the supervisor deliberately)

EOF
    return 4
  fi

  entries=$(stage1_cache_entries "$cache")
  if [ "$entries" -eq 0 ]; then
    printf '[stage1_guard] %s is empty -- stage 1 has work to do, proceeding\n' "$cache"
    return 0
  fi

  token="i-know-this-recomputes-${entries}-cached-clips-for-${run}"
  if [ "${STAGE1_FORCE:-}" = "$token" ]; then
    printf '[stage1_guard] override accepted for %s (%s entries). Proceeding.\n' \
      "$cache" "$entries"
    return 0
  fi

  cat >&2 <<EOF

  REFUSING TO START STAGE 1: the cache is already populated.

    cache    : $cache
    entries  : $entries encoded clips
    geometry : $geometry
    cost     : this cache took about 8 hours of GPU time to build

  Stage 1 with UNCHANGED geometry only verifies -- it would still hold the GPU for
  the length of that verification and produce nothing new. Stage 1 with CHANGED
  height/width/num_frames re-encodes from scratch, and what is there now is gone.

  If you meant to TRAIN, stage 1 is not the entrypoint you want:
      bash scripts/vast/train_stage2.sh          one attempt, foreground
      bash scripts/vast/supervise_stage2.sh      unattended, with restarts

  If you really do mean to re-run stage 1, say so in a way you cannot do by
  accident:

      STAGE1_FORCE=$token \\
        $entry

  Nothing has been read, written or deleted under the cache.

EOF
  if [ -n "${STAGE1_FORCE:-}" ]; then
    printf '  (STAGE1_FORCE was set to "%s", which is not the token above. The\n' \
      "$STAGE1_FORCE" >&2
    printf '   token names this cache and this run on purpose, so a value copied\n' >&2
    printf '   from an older session cannot unlock a different situation.)\n\n' >&2
  fi
  return 3
}

# Callable directly -- `bash stage1_guard.sh` checks and reports without running
# anything -- as well as sourceable from an entrypoint.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck source=scripts/vast/h3_env.sh
  [ -f "$_HERE/h3_env.sh" ] && source "$_HERE/h3_env.sh"
  stage1_guard "${CACHE:-/workspace/output/h3_ref2va/split-cache}" \
    "${RUN:-h3_ref2va}" \
    "${HEIGHT:-?}x${WIDTH:-?}x${NUM_FRAMES:-?}"
  exit $?
fi
