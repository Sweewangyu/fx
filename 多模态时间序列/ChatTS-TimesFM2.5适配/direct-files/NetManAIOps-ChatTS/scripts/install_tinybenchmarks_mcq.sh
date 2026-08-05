#!/usr/bin/env bash
set -Eeuo pipefail

# Install only the Hugging Face evaluation backend.  Deliberately do not
# install lm-eval's vLLM extra: ChatTS currently pins vLLM 0.8.5, whereas the
# newest lm-evaluation-harness vLLM backend requires a much newer vLLM.

PYTHON_BIN="${PYTHON_BIN:-python}"
LM_EVAL_REF="${LM_EVAL_REF:-f4d4b3de3ee6741a7151a9fe74945ee515262f4c}"
TINYBENCHMARKS_REF="${TINYBENCHMARKS_REF:-e9a8b1031b0340571beb6c9ca3a27891be09a8fd}"
TINYBENCHMARKS_PKL_SHA256="${TINYBENCHMARKS_PKL_SHA256:-c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284}"
LM_EVAL_ROOT="${LM_EVAL_ROOT:-}"
TINYBENCHMARKS_ROOT="${TINYBENCHMARKS_ROOT:-}"

read -r -a PIP_ARGS <<< "${PIP_ARGS:-}"

if [[ -n "$LM_EVAL_ROOT" ]]; then
    if [[ ! -f "$LM_EVAL_ROOT/pyproject.toml" && ! -f "$LM_EVAL_ROOT/setup.py" ]]; then
        echo "Invalid LM_EVAL_ROOT: $LM_EVAL_ROOT" >&2
        exit 1
    fi
    LM_EVAL_SOURCE="$LM_EVAL_ROOT"
else
    LM_EVAL_SOURCE="git+https://github.com/EleutherAI/lm-evaluation-harness.git@${LM_EVAL_REF}"
fi

if [[ -n "$TINYBENCHMARKS_ROOT" ]]; then
    if [[ ! -f "$TINYBENCHMARKS_ROOT/setup.py" ]]; then
        echo "Invalid TINYBENCHMARKS_ROOT: $TINYBENCHMARKS_ROOT" >&2
        exit 1
    fi
    TINYBENCHMARKS_SOURCE="$TINYBENCHMARKS_ROOT"
else
    TINYBENCHMARKS_SOURCE="git+https://github.com/felipemaiapolo/tinyBenchmarks.git@${TINYBENCHMARKS_REF}#egg=tinyBenchmarks"
fi

echo "Installing tinyBenchmarks MCQ dependencies into: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "lm-evaluation-harness source: $LM_EVAL_SOURCE"
echo "tinyBenchmarks source:       $TINYBENCHMARKS_SOURCE"

"$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" \
    "accelerate>=0.26.0,<2" \
    "$LM_EVAL_SOURCE"

# The upstream setup.py does not include tinyBenchmarks.pkl in a normal wheel,
# while tb.evaluate() expects that calibration asset.  Editable installation
# keeps the pinned source tree and its official 5 MB calibration file intact.
"$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" -e "$TINYBENCHMARKS_SOURCE"

TINYBENCHMARKS_PKL_SHA256="$TINYBENCHMARKS_PKL_SHA256" "$PYTHON_BIN" - <<'PY'
import hashlib
from importlib import metadata
import os

import accelerate
import lm_eval
import tinyBenchmarks
import torch
import transformers


def version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


print("Installation verified:")
print("  torch                  =", torch.__version__)
print("  transformers           =", transformers.__version__)
print("  accelerate             =", accelerate.__version__)
print("  lm_eval                =", version("lm_eval"))
print("  tinyBenchmarks         =", version("tinyBenchmarks"))
calibration = __import__("pathlib").Path(tinyBenchmarks.__file__).with_name(
    "tinyBenchmarks.pkl"
)
if not calibration.is_file():
    raise SystemExit(
        "tinyBenchmarks calibration file is missing after installation: "
        f"{calibration}"
    )
actual_digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
expected_digest = os.environ["TINYBENCHMARKS_PKL_SHA256"]
if actual_digest != expected_digest:
    raise SystemExit(
        "tinyBenchmarks calibration SHA-256 mismatch: "
        f"expected {expected_digest}, got {actual_digest}. "
        "If you intentionally selected another official commit, set "
        "TINYBENCHMARKS_PKL_SHA256 to that commit's asset digest."
    )
print("  calibration asset      =", calibration)
print("  calibration SHA-256    =", actual_digest)
try:
    print("  vllm (left untouched)  =", version("vllm"))
except Exception:
    pass
PY

echo "Done. The first evaluation run downloads five tinyBenchmarks datasets unless they are already in HF_HOME."
