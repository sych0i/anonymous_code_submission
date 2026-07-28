#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON="$(command -v python3.12)"
    elif [[ -x "${HOME}/.local/bin/python3.12" ]]; then
        PYTHON="${HOME}/.local/bin/python3.12"
    else
        echo "Python 3.12 not found in PATH or ${HOME}/.local/bin." >&2
        exit 1
    fi
fi

"${PYTHON}" -m venv "${ROOT}/.venv"
"${ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${ROOT}/.venv/bin/python" -m pip install -r "${ROOT}/requirements.txt"

echo "Environment ready: ${ROOT}/.venv"
echo "Run: ${ROOT}/.venv/bin/python scripts/verify_package.py"
