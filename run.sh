#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ and try again."
  exit 1
fi

VENV_DIR="$ROOT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  python -m pip install -r "$ROOT_DIR/requirements.txt"
else
  python -m pip install streamlit faster-whisper torch requests
fi

python -m streamlit run app.py
