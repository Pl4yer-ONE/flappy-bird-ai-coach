#!/bin/bash
# Flappy Bird AI Coach - Run Script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

cd "$SCRIPT_DIR"

# Default to dashboard mode if no argument provided
if [ $# -eq 0 ]; then
    exec "$PYTHON" main.py --mode dashboard
else
    exec "$PYTHON" main.py "$@"
fi

