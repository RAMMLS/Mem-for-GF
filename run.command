#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  /opt/homebrew/bin/python3.12 -m venv .venv
fi

if ! .venv/bin/python -c "import mem_for_gf" >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python main.py
