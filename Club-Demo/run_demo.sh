#!/usr/bin/env bash
# Локальный запуск демки «Резонанс»: backend (FastAPI/uvicorn) + статический фронт.
# Использование:  ./run_demo.sh   потом открыть  http://localhost:8000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONT_PORT="${FRONT_PORT:-8000}"
PY="${PYTHON:-python3}"

cleanup() {
  echo
  echo "[run_demo] остановка ..."
  [[ -n "${BACK_PID:-}" ]] && kill "$BACK_PID" 2>/dev/null || true
  [[ -n "${FRONT_PID:-}" ]] && kill "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[run_demo] backend: http://127.0.0.1:${BACKEND_PORT}  (uvicorn)"
( cd "$HERE/backend" && exec "$PY" -m uvicorn server:app --host 127.0.0.1 --port "$BACKEND_PORT" ) &
BACK_PID=$!

echo "[run_demo] frontend: http://localhost:${FRONT_PORT}  (http.server)"
( cd "$HERE" && exec "$PY" -m http.server "$FRONT_PORT" ) &
FRONT_PID=$!

echo "[run_demo] открой http://localhost:${FRONT_PORT}  (Ctrl+C — стоп)"
wait
