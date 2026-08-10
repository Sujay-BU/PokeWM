#!/usr/bin/env bash
# Launch (or resume) a long training run, detached, with a PID file.
#
#   scripts/train.sh start [--preset laptop] [--logdir runs/overnight] [extra args...]
#   scripts/train.sh status
#   scripts/train.sh stop
#   scripts/train.sh watch
#
# `stop` sends SIGINT so the trainer checkpoints before exiting; `start` afterwards
# resumes from that checkpoint.
#
# Note the pattern-matching care in `stop`: `pkill -f pokewm.train` also matches the
# shell running this script, which kills the wrapper before it kills the job. The PID
# file avoids that entirely.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-pokewm}"
LOGDIR="${LOGDIR:-$REPO/runs/overnight}"
PIDFILE="$LOGDIR/train.pid"
OUTLOG="$LOGDIR/stdout.log"

activate() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

is_running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

cmd_start() {
  if is_running; then
    echo "already running (pid $(cat "$PIDFILE"))"; exit 0
  fi
  mkdir -p "$LOGDIR"
  activate
  cd "$REPO"
  # `setsid` puts the trainer in its own session and process group. `nohup` alone only
  # ignores SIGHUP, so a process-group kill of the launching shell takes the trainer with
  # it -- which silently killed a run mid-stride (no traceback, no shutdown log, just a
  # last metrics line) every time the controlling session went away.
  setsid nohup python -u -m pokewm.train --logdir "$LOGDIR" "$@" >>"$OUTLOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "started pid $(cat "$PIDFILE") -> $OUTLOG"
}

# Supervised mode: restart the trainer whenever it dies unexpectedly.
#
# A multi-day run has to survive things that are not bugs in the agent -- an OOM kill,
# a GPU driver fault, a transient CUDA error. The trainer checkpoints every 5 minutes
# and resumes from the checkpoint, so an automatic restart costs at most that much
# progress. `stop` removes the flag file, so a deliberate stop is never fought by the
# supervisor.
cmd_supervise() {
  # Re-exec detached unless already in our own session, for the same reason as `start`:
  # a supervisor that dies with the controlling shell supervises nothing.
  if [[ "${POKEWM_DETACHED:-0}" != "1" ]]; then
    mkdir -p "$LOGDIR"
    POKEWM_DETACHED=1 setsid nohup "$0" supervise "$@" \
      >>"$LOGDIR/supervisor.out" 2>&1 &
    echo "supervisor detached (pid $!) -> $LOGDIR/supervisor.out"
    return 0
  fi
  mkdir -p "$LOGDIR"
  local flag="$LOGDIR/supervise.on"
  : >"$flag"
  activate
  cd "$REPO"
  local n=0
  while [[ -f "$flag" ]]; do
    n=$((n + 1))
    echo "[supervisor] launch #$n at $(date -Is)" | tee -a "$OUTLOG"
    python -u -m pokewm.train --logdir "$LOGDIR" "$@" >>"$OUTLOG" 2>&1 &
    local pid=$!
    echo "$pid" >"$PIDFILE"
    # `set -e` would abort the supervisor the moment the trainer exits non-zero --
    # i.e. in exactly the case the supervisor exists to handle. Capture the status
    # explicitly instead of letting errexit see a failing command.
    local rc=0
    wait "$pid" || rc=$?
    rm -f "$PIDFILE"
    if [[ ! -f "$flag" ]]; then
      echo "[supervisor] stop requested; exiting" | tee -a "$OUTLOG"; break
    fi
    if [[ $rc -eq 0 ]]; then
      echo "[supervisor] trainer exited cleanly (rc=0); done" | tee -a "$OUTLOG"; break
    fi
    echo "[supervisor] trainer died rc=$rc; restarting in 20s" | tee -a "$OUTLOG"
    sleep 20
  done
  rm -f "$flag"
}

cmd_stop() {
  rm -f "$LOGDIR/supervise.on"   # tell any supervisor this is deliberate
  if ! is_running; then echo "not running"; exit 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  echo "sending SIGINT to $pid (it will checkpoint first)..."
  kill -INT "$pid"
  # Generous: the shutdown path joins the collector (up to 15 s) and then writes a
  # ~760 MB checkpoint plus the archive. Measured 6 s early in a run, but it grows with
  # the model and archive, and a 60 s budget started forcing SIGKILL -- which discards
  # the very checkpoint the graceful path exists to write.
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$pid" 2>/dev/null && { echo "still alive, SIGKILL"; kill -9 "$pid"; }
  rm -f "$PIDFILE"
  echo "stopped"
}

cmd_status() {
  if is_running; then echo "running (pid $(cat "$PIDFILE"))"; else echo "not running"; fi
  [[ -f "$LOGDIR/train.log" ]] && grep -v SDL2 "$LOGDIR/train.log" | tail -5
  [[ -f "$LOGDIR/events.jsonl" ]] && {
    echo "--- milestones ---"; tail -8 "$LOGDIR/events.jsonl"
  }
  true
}

case "${1:-status}" in
  start)     shift; cmd_start "$@" ;;
  supervise) shift; cmd_supervise "$@" ;;
  stop)      cmd_stop ;;
  status)    cmd_status ;;
  watch)     tail -f "$LOGDIR/train.log" ;;
  *) echo "usage: $0 {start|supervise|stop|status|watch}" >&2; exit 2 ;;
esac
