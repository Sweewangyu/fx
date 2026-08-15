#!/usr/bin/env python3
"""Create a ChatTS-compatible Qwen3 base without loading its weight tensors.

The Qwen language-model tensors are reused unchanged.  The output receives
ChatTS' Qwen3TS remote code, two time-series special tokens, and a Qwen3TS
configuration.  A subsequent ChatTS training run can then replace the native
placeholder encoder with Chronos-2, TimesFM, or Zeus.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


CODE_FILES = (
    "configuration_qwen3_ts.py",
    "modeling_qwen3_ts.py",
    "processing_qwen3_ts.py",
)
WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
TS_START_TOKEN = "<ts>"
TS_END_TOKEN = "<ts/>"
TS_START_ID = 151669
TS_END_ID = 151670


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-checkpoint",
        type=Path,
        required=True,
        help="Downloaded Qwen3 checkpoint directory.",
    )
    parser.add_argument(
        "--chatts-template",
        type=Path,
        required=True,
        help="Official ChatTS-Qwen3 directory containing the three remote-code files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weight-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Reuse immutable weight shards with hardlinks when possible, or copy them.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def prepare_staging_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    atexit.register(shutil.rmtree, staging, True)
    return staging


def is_weight_file(path: Path) -> bool:
    return path.suffix in {".safetensors", ".bin"} and not path.name.startswith(
        ("optimizer", "scheduler")
    )


def copy_or_link(source: Path, destination: Path, weight_mode: str) -> str:
    if is_weight_file(source) and weight_mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy-fallback"
    shutil.copy2(source, destination)
    return "copy"


def validate_qwen_config(config: dict[str, Any]) -> None:
    architecture = config.get("architectures")
    if architecture != ["Qwen3ForCausalLM"]:
        raise ValueError(
            "Expected a raw Qwen3ForCausalLM checkpoint, got "
            f"architectures={architecture!r}. Do not convert an existing ChatTS checkpoint."
        )
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Expected model_type=qwen3, got {config.get('model_type')!r}")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
        if not isinstance(config.get(key), int) or int(config[key]) <= 0:
            raise ValueError(f"Qwen config has no valid {key}: {config.get(key)!r}")
    if int(config["vocab_size"]) <= TS_END_ID:
        raise ValueError(
            f"Qwen vocabulary ({config['vocab_size']}) has no rows for ChatTS token ids "
            f"{TS_START_ID} and {TS_END_ID}."
        )


def patch_model_config(config: dict[str, Any], source: Path) -> dict[str, Any]:
    output = dict(config)
    hidden_size = int(output["hidden_size"])
    output["architectures"] = ["Qwen3TSForCausalLM"]
    output["model_type"] = "qwen3ts"
    output["auto_map"] = {
        "AutoConfig": "configuration_qwen3_ts.Qwen3TSConfig",
        "AutoModel": "modeling_qwen3_ts.Qwen3TSForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen3_ts.Qwen3TSForCausalLM",
        "AutoProcessor": "processing_qwen3_ts.Qwen3TSProcessor",
    }
    output["ignore_index"] = -100
    output["pad_token_id"] = int(output.get("pad_token_id") or 151643)
    output["ts_token_start_index"] = TS_START_ID
    output["ts_token_end_index"] = TS_END_ID
    output["ts_encoder_type"] = "native"
    output["ts"] = {
        "embedding_dim": 16,
        "hidden_size": hidden_size,
        "max_length": 32768,
        "max_sequence_length": 8192,
        "num_features": 2,
        "num_layers": 5,
        "patch_size": 8,
        "use_layer_norm": False,
        "use_position_embedding": True,
        "use_position_idx": False,
    }
    output["chatts_base_initialization"] = {
        "source_checkpoint": str(source),
        "language_weights_reused_without_conversion": True,
        "time_series_modules_initialized_during_training": True,
    }
    return output


def added_token(token: str, token_id: int) -> dict[str, Any]:
    return {
        "id": token_id,
        "content": token,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def patch_tokenizer_json(path: Path) -> None:
    payload = read_json(path)
    tokens = payload.get("added_tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"tokenizer.json has no added_tokens list: {path}")

    by_id = {int(item["id"]): item for item in tokens if isinstance(item, dict) and "id" in item}
    by_content = {
        str(item["content"]): item
        for item in tokens
        if isinstance(item, dict) and "content" in item
    }
    for token, token_id in ((TS_START_TOKEN, TS_START_ID), (TS_END_TOKEN, TS_END_ID)):
        id_owner = by_id.get(token_id)
        token_entry = by_content.get(token)
        if id_owner is not None and id_owner.get("content") != token:
            raise ValueError(
                f"Tokenizer id {token_id} is already occupied by {id_owner.get('content')!r}."
            )
        if token_entry is not None and int(token_entry.get("id", -1)) != token_id:
            raise ValueError(
                f"Tokenizer token {token!r} already uses id {token_entry.get('id')!r}."
            )
        if id_owner is None and token_entry is None:
            tokens.append(added_token(token, token_id))

    tokens.sort(key=lambda item: int(item["id"]))
    write_json(path, payload)


def tokenizer_decoder_entry(token: str) -> dict[str, Any]:
    return {
        "content": token,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }


def patch_tokenizer_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    decoder = payload.setdefault("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise ValueError(f"added_tokens_decoder is not an object: {path}")
    for token, token_id in ((TS_START_TOKEN, TS_START_ID), (TS_END_TOKEN, TS_END_ID)):
        current = decoder.get(str(token_id))
        if current is not None and current.get("content") != token:
            raise ValueError(
                f"Tokenizer id {token_id} is occupied by {current.get('content')!r} in {path}."
            )
        decoder[str(token_id)] = tokenizer_decoder_entry(token)

    special_tokens = payload.get("additional_special_tokens") or []
    if not isinstance(special_tokens, list):
        raise ValueError(f"additional_special_tokens is not a list: {path}")
    for token in (TS_START_TOKEN, TS_END_TOKEN):
        if token not in special_tokens:
            special_tokens.append(token)
    payload["additional_special_tokens"] = special_tokens
    payload["auto_map"] = {"AutoProcessor": "processing_qwen3_ts.Qwen3TSProcessor"}
    payload["processor_class"] = "Qwen3TSProcessor"
    write_json(path, payload)
    return payload


def patch_tokenizer_files(output_dir: Path) -> None:
    patch_tokenizer_json(output_dir / "tokenizer.json")
    tokenizer_config = patch_tokenizer_config(output_dir / "tokenizer_config.json")
    added_tokens_path = output_dir / "added_tokens.json"
    existing_added_tokens = read_json(added_tokens_path) if added_tokens_path.is_file() else {}
    existing_added_tokens[TS_START_TOKEN] = TS_START_ID
    existing_added_tokens[TS_END_TOKEN] = TS_END_ID
    write_json(
        added_tokens_path,
        existing_added_tokens,
    )
    special_tokens_path = output_dir / "special_tokens_map.json"
    special_tokens_map = read_json(special_tokens_path) if special_tokens_path.is_file() else {}
    special_tokens_map["additional_special_tokens"] = tokenizer_config[
        "additional_special_tokens"
    ]
    special_tokens_map.setdefault("eos_token", tokenizer_config.get("eos_token", "<|im_end|>"))
    special_tokens_map.setdefault("pad_token", tokenizer_config.get("pad_token", "<|endoftext|>"))
    write_json(
        special_tokens_path,
        special_tokens_map,
    )
    write_json(
        output_dir / "processor_config.json",
        {
            "auto_map": {"AutoProcessor": "processing_qwen3_ts.Qwen3TSProcessor"},
            "processor_class": "Qwen3TSProcessor",
        },
    )


def validate_weight_index(output_dir: Path) -> list[str]:
    for index_name in WEIGHT_INDEX_NAMES:
        index_path = output_dir / index_name
        if index_path.is_file():
            weight_map = read_json(index_path).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Weight index is empty: {index_path}")
            shard_names = sorted(set(str(value) for value in weight_map.values()))
            missing = [name for name in shard_names if not (output_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Weight index references missing shards: {missing}")
            return shard_names

    single_weights = [
        name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (output_dir / name).is_file()
    ]
    if single_weights:
        return single_weights
    raise FileNotFoundError(f"No model weights or weight index found under: {output_dir}")


def main() -> int:
    args = parse_args()
    qwen_dir = args.qwen_checkpoint.expanduser().resolve()
    template_dir = args.chatts_template.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if qwen_dir == output_dir:
        raise ValueError("--output-dir must differ from --qwen-checkpoint")
    if not qwen_dir.is_dir():
        raise FileNotFoundError(f"Qwen checkpoint directory not found: {qwen_dir}")
    if not template_dir.is_dir():
        raise FileNotFoundError(f"ChatTS template directory not found: {template_dir}")

    config = read_json(qwen_dir / "config.json")
    validate_qwen_config(config)
    for filename in CODE_FILES:
        if not (template_dir / filename).is_file():
            raise FileNotFoundError(f"ChatTS remote-code template missing: {template_dir / filename}")

    staging = prepare_staging_dir(output_dir)
    transfer_counts: dict[str, int] = {}
    for source in qwen_dir.iterdir():
        if not source.is_file():
            continue
        mode = copy_or_link(source, staging / source.name, args.weight_mode)
        transfer_counts[mode] = transfer_counts.get(mode, 0) + 1

    for filename in CODE_FILES:
        shutil.copy2(template_dir / filename, staging / filename)

    write_json(staging / "config.json", patch_model_config(config, qwen_dir))
    patch_tokenizer_files(staging)
    shard_names = validate_weight_index(staging)
    write_json(
        staging / "CHATTS_BASE_MANIFEST.json",
        {
            "format_version": 1,
            "architecture": "Qwen3TSForCausalLM",
            "qwen_source": str(qwen_dir),
            "chatts_code_template": str(template_dir),
            "hidden_size": int(config["hidden_size"]),
            "num_hidden_layers": int(config["num_hidden_layers"]),
            "vocab_size": int(config["vocab_size"]),
            "ts_token_start": {"token": TS_START_TOKEN, "id": TS_START_ID},
            "ts_token_end": {"token": TS_END_TOKEN, "id": TS_END_ID},
            "weight_shards": shard_names,
            "file_transfer_counts": transfer_counts,
            "language_weight_tensors_modified": False,
        },
    )

    if output_dir.exists():
        output_dir.rmdir()
    staging.rename(output_dir)
    print("ChatTS-compatible Qwen3 base prepared successfully")
    print(f"output: {output_dir}")
    print(f"hidden_size: {config['hidden_size']}")
    print(f"layers: {config['num_hidden_layers']}")
    print(f"weight shards: {len(shard_names)}")
    print("Next: use this output as MODEL_PATH for a fresh Stage1 -> Stage2 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
