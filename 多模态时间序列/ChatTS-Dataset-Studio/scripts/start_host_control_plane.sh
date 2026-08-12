#!/usr/bin/env bash
set -Eeuo pipefail

# Run Dataset Studio on the Docker host. The Studio is only a lightweight
# control plane; training and evaluation continue to run in their own containers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${PROJECT_ROOT}/configs/server.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

command -v docker >/dev/null 2>&1 || {
    echo "Docker CLI is unavailable. Run this script on the Docker host." >&2
    exit 1
}
docker info >/dev/null 2>&1 || {
    echo "Docker daemon is unavailable to the current host user." >&2
    exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python is unavailable: $PYTHON_BIN" >&2
    exit 1
}
[[ -f "$CONFIG_FILE" ]] || {
    echo "Dataset Studio configuration not found: $CONFIG_FILE" >&2
    echo "Copy configs/server.example.yaml to configs/server.yaml and set host paths first." >&2
    exit 1
}

echo "Starting ChatTS Dataset Studio on the Docker host"
echo "Configuration: $CONFIG_FILE"
echo "Containers visible to Docker:"
docker ps --format '  {{.Names}}\t{{.Status}}'

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PYTHON_BIN" -m chatts_dataset_studio serve -c "$CONFIG_FILE"
