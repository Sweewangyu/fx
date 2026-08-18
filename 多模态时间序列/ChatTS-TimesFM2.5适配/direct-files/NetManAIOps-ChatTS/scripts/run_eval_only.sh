#!/usr/bin/env bash
set -Eeuo pipefail

# Stable Dataset Studio entrypoint.  Keep the implementation name available to
# existing/manual callers while giving the control plane a concise fixed path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_standalone_eval.sh" "$@"
