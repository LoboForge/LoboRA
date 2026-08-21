#!/usr/bin/env bash
# procscan.sh -- the bash port of lobora/procscan.py. Source it; do not run it.
#
# WHY THERE IS A SECOND IMPLEMENTATION AT ALL
#   lobora/procscan.py is the canonical one, and the Python watchers ship it to
#   the box over SSH so the remote and local answers are literally the same code.
#   The box-side SHELL scripts cannot do that: they run on a machine with no
#   checkout of this repo, before any venv is guaranteed, and they are copied to
#   /workspace one file at a time. So the rules are restated here, ONCE, and every
#   bash caller sources this file instead of writing its own loop.
#
#   tests/test_stop_at_step_matcher.py runs these rules and the Python ones over
#   the same synthetic /proc and fails if they ever disagree. A second
#   implementation that nothing compares is exactly how the bug below survived in
#   two files at the same time.
#
# THE RULE THAT MATTERS
#   Never count processes with a pipeline:
#
#       grep -l "$pat" /proc/[0-9]*/cmdline | wc -l
#
#   In a pipeline the shell forks the child first and the CHILD expands the glob,
#   so grep is handed its own /proc/<pid>/cmdline -- which by then holds grep's
#   argv, search pattern included. It matches itself every time, the count is
#   permanently real+1, and any `== 0` test built on it is unsatisfiable. Splitting
#   the pattern with "" does not help: the shell joins it before execve.
#
#   So: read /proc/<pid>/cmdline directly, and exclude what cannot honestly be
#   counted -- the caller, its ancestors, tmux, and Z/X entries. A pid whose state
#   cannot be read counts as LIVE; every unknown must point at "still running".
#
# WHAT A CALLER MUST SET BEFORE SOURCING (or accept the defaults)
#   PROCFS     /proc root, overridden only by tests. Deliberately NOT called PROC:
#              a sourced file shares the caller's namespace, and the box's stage-1
#              entrypoint already used PROC for a model path. That collision made
#              this file scan a processor directory, find nothing, and report an
#              idle GPU -- a guard cheerfully answering "nothing is running" with
#              two trainers on the card. Names claimed here: PROCFS, SELF_TAG,
#              PROTECTED, and the functions below.
#   SELF_TAG   a string in the caller's own argv, so its subshells are skipped
#              (a forked subshell inherits the script's argv verbatim)

PROCFS=${PROCFS:-${STOP_PROC_ROOT:-/proc}}
# NEVER let this end up empty. `case "$cmd" in *""*)` matches EVERY process, so an
# empty tag silently skips the whole table and every count comes back 0 -- the
# same failure as the grep bug, in the opposite direction and just as quiet.
SELF_TAG=${SELF_TAG:-$(basename "${BASH_SOURCE[1]:-$0}")}
SELF_TAG=${SELF_TAG:-procscan-self-tag-unset}

cmd_of() { tr '\0' ' ' <"$PROCFS/$1/cmdline" 2>/dev/null; }

# From stat rather than /proc/<pid>/comm, so it is the same field the Python side
# reads: everything between the first '(' and the last ')'.
comm_of() {
  local s
  s=$(cat "$PROCFS/$1/stat" 2>/dev/null) || return 0
  s=${s#*(}
  printf '%s' "${s%)*}"
}

ppid_of() {
  # Everything after the last ')' is: state ppid ... -- the only parse that
  # survives a comm containing spaces or parentheses.
  sed 's/.*)//' "$PROCFS/$1/stat" 2>/dev/null | awk '{print $2}'
}

state_of() { sed 's/.*)//' "$PROCFS/$1/stat" 2>/dev/null | awk '{print $1}'; }

# Z (zombie) and X/x (dead) hold no GPU memory and cannot execute an instruction,
# but they answer `kill -0` and show up in `ps` indefinitely.
undead() {
  case "$(state_of "$1")" in Z | X | x) return 0 ;; *) return 1 ;; esac
}

alive() {
  case "$(state_of "$1")" in '' | Z | X | x) return 1 ;; *) return 0 ;; esac
}

# tmux keeps the argv of whatever forked the server, so a server started as
# `tmux new-session -d -s anatomy_train bash supervise_stage2.sh` has the
# supervisor pattern in its own cmdline. It is not a supervisor, and killing it
# takes down every session on that socket -- including the caller's own.
is_tmux() {
  case "$(comm_of "$1")" in tmux*) return 0 ;; esac
  case "$(cmd_of "$1")" in tmux | tmux\ * | */tmux\ *) return 0 ;; esac
  return 1
}

is_shell() {
  case "$(comm_of "$1")" in
    bash | sh | dash | zsh | ksh) return 0 ;;
    *) return 1 ;;
  esac
}

# The caller, everything that spawned it, and init. Over SSH the invoking shell's
# argv contains the pattern, because we are the ones who sent it.
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
PROTECTED=${PROTECTED:-$(protected_pids)}

is_protected() {
  case "
$PROTECTED
" in *"
$1
"*) return 0 ;; esac
  return 1
}

# Live pids whose cmdline contains $1.
find_pids() {
  local pat=$1 pid cmd
  for pid in "$PROCFS"/[0-9]*; do
    pid=${pid##*/}
    is_protected "$pid" && continue
    cmd=$(cmd_of "$pid")
    [ -n "$cmd" ] || continue
    if [ -n "$SELF_TAG" ]; then
      case "$cmd" in *"$SELF_TAG"*) continue ;; esac
    fi
    case "$cmd" in *"$pat"*) ;; *) continue ;; esac
    is_tmux "$pid" && continue
    undead "$pid" && continue
    printf '%s\n' "$pid"
  done
}

# Matches in Z/X state: visible in the audit trail, never signalled, never waited
# for.
find_undead_pids() {
  local pat=$1 pid cmd
  for pid in "$PROCFS"/[0-9]*; do
    pid=${pid##*/}
    is_protected "$pid" && continue
    cmd=$(cmd_of "$pid")
    [ -n "$cmd" ] || continue
    if [ -n "$SELF_TAG" ]; then
      case "$cmd" in *"$SELF_TAG"*) continue ;; esac
    fi
    case "$cmd" in *"$pat"*) ;; *) continue ;; esac
    undead "$pid" && printf '%s\n' "$pid"
  done
}

# wc, not `grep -c`: grep exits 1 on no match, which under `set -e` in a caller
# is an exit rather than a zero.
count_pids() { find_pids "$1" | wc -l | tr -d ' '; }

# Children of $1 that have exited and not been reaped.
zombie_children_of() {
  local parent=$1 pid
  for pid in "$PROCFS"/[0-9]*; do
    pid=${pid##*/}
    [ "$(ppid_of "$pid")" = "$parent" ] || continue
    undead "$pid" && printf '%s\n' "$pid"
  done
}
