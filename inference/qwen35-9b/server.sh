#!/usr/bin/env bash
# Background control for serve.sh, by pidfile (never pkill -f / pgrep -f: they also match their own shell).
#
#   ./server.sh start [extra vllm args]   start in the background, wait until /v1/models answers
#   ./server.sh stop                      SIGTERM by pidfile, wait for a clean exit (no SIGKILL)
#   ./server.sh status                    pid + /v1/models
#   ./server.sh logs                      tail -f the log
#
# Env: PIDFILE (default logs/vllm.pid), LOGFILE (default logs/vllm.log), PORT (default 8000),
#      START_TIMEOUT (s, default 600), STOP_TIMEOUT (s, default 120); everything else goes to serve.sh.
# The DataMate resource sampler reads the vLLM pid from VLLM_PIDFILE: export VLLM_PIDFILE="$PIDFILE" for the harness.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="${PIDFILE:-$HERE/logs/vllm.pid}"
LOGFILE="${LOGFILE:-$HERE/logs/vllm.log}"
PORT="${PORT:-8000}"
export PORT
URL="http://localhost:$PORT/v1/models"

running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid="$(cat "$PIDFILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  start)
    if pid="$(running_pid)"; then
      echo "already running: pid $pid (pidfile $PIDFILE)"; exit 0
    fi
    if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
      echo "port $PORT already serves an API but $PIDFILE has no live pid: another server is running." >&2
      echo "Stop it by its own pidfile, or use another PORT=... (and PIDFILE=...)." >&2
      exit 1
    fi
    mkdir -p "$(dirname "$PIDFILE")" "$(dirname "$LOGFILE")"
    [ -f "$LOGFILE" ] && mv -f "$LOGFILE" "$LOGFILE.prev"
    # setsid: own session, survives the terminal; serve.sh execs vllm, so $! is the vLLM API server pid.
    setsid nohup "$HERE/serve.sh" "$@" >"$LOGFILE" 2>&1 </dev/null &
    echo $! >"$PIDFILE"
    echo "started pid $(cat "$PIDFILE"), log $LOGFILE; waiting for $URL ..."
    deadline=$((SECONDS + ${START_TIMEOUT:-600}))
    until curl -fsS -o /dev/null "$URL" 2>/dev/null; do
      if ! running_pid >/dev/null; then
        echo "vLLM exited during startup; last log lines:" >&2
        tail -n 40 "$LOGFILE" >&2
        rm -f "$PIDFILE"
        exit 1
      fi
      if [ "$SECONDS" -ge "$deadline" ]; then
        echo "not ready after ${START_TIMEOUT:-600}s (still running, see $LOGFILE)" >&2
        exit 1
      fi
      sleep 5
    done
    echo "ready: $URL"
    ;;
  stop)
    if ! pid="$(running_pid)"; then
      echo "not running (no live pid in $PIDFILE)"; rm -f "$PIDFILE"; exit 0
    fi
    kill -TERM "$pid"
    echo "SIGTERM sent to $pid; waiting up to ${STOP_TIMEOUT:-120}s for a clean shutdown ..."
    deadline=$((SECONDS + ${STOP_TIMEOUT:-120}))
    while kill -0 "$pid" 2>/dev/null; do
      if [ "$SECONDS" -ge "$deadline" ]; then
        # No automatic SIGKILL: a force-killed EngineCore was followed by a GPU fault (Xid 62/154) on the dev card.
        echo "pid $pid still alive after ${STOP_TIMEOUT:-120}s. Check $LOGFILE and nvidia-smi;" >&2
        echo "only if it is hung: kill -KILL $pid" >&2
        exit 1
      fi
      sleep 2
    done
    rm -f "$PIDFILE"
    echo "stopped"
    ;;
  status)
    if pid="$(running_pid)"; then echo "running: pid $pid"; else echo "not running (pidfile $PIDFILE)"; fi
    if curl -fsS "$URL" 2>/dev/null; then echo; else echo "no answer on $URL"; fi
    ;;
  logs)
    exec tail -n 100 -f "$LOGFILE"
    ;;
  *)
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    [ -z "$cmd" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ] || exit 2
    ;;
esac
