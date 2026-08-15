#!/usr/bin/env bash
set -Eeuo pipefail

# Run Dataset Studio on the cluster login/control host. The selected backend is
# checked by the service: Docker can train+evaluate, while Slurm submits a
# trusted sbatch that runs training in Singularity.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${PROJECT_ROOT}/configs/server.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python is unavailable: $PYTHON_BIN" >&2
    exit 1
}
[[ -f "$CONFIG_FILE" ]] || {
    echo "Dataset Studio configuration not found: $CONFIG_FILE" >&2
    echo "Copy configs/server.example.yaml to configs/server.yaml and set host paths first." >&2
    exit 1
}

echo "Starting ChatTS Dataset Studio control plane"
echo "Configuration: $CONFIG_FILE"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "Docker backend is available; visible containers:"
    docker ps --format '  {{.Names}}\t{{.Status}}'
else
    echo "Docker backend is unavailable (Slurm can still be used if configured)."
fi
if command -v sbatch >/dev/null 2>&1; then
    echo "Slurm backend is available: $(command -v sbatch)"
else
    echo "Slurm backend is unavailable (Docker can still be used if configured)."
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PYTHON_BIN" -m chatts_dataset_studio serve -c "$CONFIG_FILE"
