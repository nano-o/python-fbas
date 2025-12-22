#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/workspaces/python-fbas}
USER_HOME=${USER_HOME:-/home/${USERNAME:-developer}}
VENV_DIR=${VENV_DIR:-${USER_HOME}/.venv}
SETUP_MARKER=${SETUP_MARKER:-${USER_HOME}/.python-fbas-container-setup}

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  echo "Mount your repo to ${PROJECT_DIR} (see docker-compose.ai.yml)." >&2
  exec "$@"
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [ ! -f "${SETUP_MARKER}" ]; then
  python -m pip install --upgrade pip
  python -m pip install -e "${PROJECT_DIR}/${PYTHON_FBAS_EXTRAS:-.[dev]}"
  touch "${SETUP_MARKER}"
fi

exec "$@"
