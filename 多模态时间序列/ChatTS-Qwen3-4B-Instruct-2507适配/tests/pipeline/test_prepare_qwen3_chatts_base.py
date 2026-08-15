from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_qwen3_chatts_base.py"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fixture(tmp_path: Path, *, conflicting_token: bool = False) -> tuple[Path, Path]:
    qwen = tmp_path / "qwen"
    template = tmp_path / "chatts-template"
    qwen.mkdir()
    template.mkdir()
    write_json(
        qwen / "config.json",
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "vocab_size": 151936,
            "tie_word_embeddings": True,
        },
    )
    added_tokens = [
        {
            "id": 151668,
            "content": "</think>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
    ]
    if conflicting_token:
        added_tokens.append(
            {
                "id": 151669,
                "content": "<occupied>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    write_json(qwen / "tokenizer.json", {"added_tokens": added_tokens, "model": {}})
    write_json(
        qwen / "tokenizer_config.json",
        {
            "added_tokens_decoder": {
                "151668": {
                    "content": "</think>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": False,
                }
            },
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "chat_template": "fixture-template",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        },
    )
    write_json(
        qwen / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 7},
            "weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"},
        },
    )
    (qwen / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    for filename in (
        "configuration_qwen3_ts.py",
        "modeling_qwen3_ts.py",
        "processing_qwen3_ts.py",
    ):
        (template / filename).write_text(f"# {filename}\n", encoding="utf-8")
    return qwen, template


def run_converter(qwen: Path, template: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--qwen-checkpoint",
            str(qwen),
            "--chatts-template",
            str(template),
            "--output-dir",
            str(output),
            "--weight-mode",
            "copy",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_qwen3_chatts_base(tmp_path: Path) -> None:
    qwen, template = make_fixture(tmp_path)
    output = tmp_path / "output"
    result = run_converter(qwen, template, output)
    assert result.returncode == 0, result.stderr

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen3TSForCausalLM"]
    assert config["model_type"] == "qwen3ts"
    assert config["hidden_size"] == config["ts"]["hidden_size"] == 2560
    assert config["ts_encoder_type"] == "native"
    assert config["ts_token_start_index"] == 151669
    assert config["ts_token_end_index"] == 151670
    assert config["tie_word_embeddings"] is True

    tokenizer = json.loads((output / "tokenizer.json").read_text(encoding="utf-8"))
    ids = {entry["content"]: entry["id"] for entry in tokenizer["added_tokens"]}
    assert ids["<ts>"] == 151669
    assert ids["<ts/>"] == 151670
    tokenizer_config = json.loads(
        (output / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["chat_template"] == "fixture-template"
    assert tokenizer_config["processor_class"] == "Qwen3TSProcessor"
    assert tokenizer_config["additional_special_tokens"][-2:] == ["<ts>", "<ts/>"]
    assert (output / "model-00001-of-00001.safetensors").read_bytes() == b"weights"
    assert (output / "CHATTS_BASE_MANIFEST.json").is_file()


def test_rejects_occupied_chatts_token_id(tmp_path: Path) -> None:
    qwen, template = make_fixture(tmp_path, conflicting_token=True)
    output = tmp_path / "output"
    result = run_converter(qwen, template, output)
    assert result.returncode != 0
    assert "already occupied" in result.stderr
    assert not output.exists()
