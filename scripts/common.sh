# Shared environment for Energyinfra shell scripts.
# Source this from scripts under scripts/active/ or scripts/_legacy/:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
#
# Provides:
#   REPO_ROOT  — absolute path of the repository root (self-locating)
#   PY         — venv python (jetson_llm_env), falls back to system python3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/jetson_llm_env/bin/python3"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
    echo "[common.sh] WARNING: jetson_llm_env not found (run ./setup.sh first); using system python3" >&2
fi
