#!/usr/bin/env bash
# ---------------------------------------------------------------------
# One-command demo (Linux / macOS).
#
#   ./demo.sh                          first clip in data/videos
#   ./demo.sh data/videos/my_clip.mp4  a specific clip
#   ./demo.sh --multi-bus              3 buses -> cross-bus dedup visible
#   ./demo.sh --dashboard              NO model: load the shipped output
#                                      (outputs/runs/events.jsonl) and
#                                      start the web + mobile apps
#
# Demo accounts: admin@ traffic@ roads@ safety@ transport@ field@
# urbanintel.local.  The password below is for local demos only; set
# UIP_AUTH__DEMO_PASSWORD yourself to change it, and set
# auth.seed_demo_users: false in config/platform.yaml for real use.
# ---------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
export UIP_AUTH__DEMO_PASSWORD="${UIP_AUTH__DEMO_PASSWORD:-UrbanIntel@2026}"

if [ "${1:-}" = "--dashboard" ]; then
  shift
  echo "--- loading the shipped pipeline output (no model is run) ---"
  "$PY" scripts/import_events.py --sync
  echo
  echo "  Demo accounts : admin@ traffic@ roads@ safety@ transport@ field@urbanintel.local"
  echo "  Password      : $UIP_AUTH__DEMO_PASSWORD"
  exec "$PY" run_backend.py "$@"
fi

echo "--- checking the environment ---"
"$PY" scripts/check_setup.py || {
  echo
  echo "Setup check failed.  Try:  $PY -m pip install -r requirements.txt"
  exit 1
}

echo
echo "--- running the demo ---"
if [ $# -gt 0 ] && [ -f "$1" ]; then
  SRC="$1"; shift
  exec "$PY" scripts/run_demo.py --source "$SRC" "$@"
else
  exec "$PY" scripts/run_demo.py "$@"
fi
