#!/usr/bin/env bash
set -euo pipefail

MARKER="${HOME}/.python-fbas-devcontainer-setup"

if [ -f "$MARKER" ]; then
  exit 0
fi

if ! python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)
PY
then
  EXTRAS="${PYTHON_FBAS_EXTRAS:-.[dev,qbf]}"
  pip install -e "${EXTRAS}"
fi

touch "$MARKER"
