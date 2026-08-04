from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
HF_FILES = ROOT / "hf_files"
SCRIPTS = ROOT / "scripts"


def tiny_chronos_config() -> dict:
    return {
        "architectures": ["Chronos2Model"],
        "chronos_config": {
            "context_length": 32,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 1,
            "output_patch_size": 16,
            "quantiles": [0.1, 0.5, 0.9],
            "time_encoding_scale": 32,
            "use_arcsinh": True,
            "use_reg_token": True,
        },
        "d_ff": 16,
        "d_kv": 8,
        "d_model": 8,
        "dropout_rate": 0.0,
        "feed_forward_proj": "relu",
        "initializer_factor": 0.05,
        "layer_norm_epsilon": 1e-6,
        "model_type": "t5",
        "num_heads": 1,
        "num_layers": 1,
        "pad_token_id": 0,
        "rope_theta": 10000.0,
        "vocab_size": 2,
    }


class SelfContainedModelTests(unittest.TestCase):
    def test_processor_is_exact_official_snapshot(self):
        digest = hashlib.sha256(
            (HF_FILES / "processing_qwen3_ts.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest,
            "cc6fb71c6e6a7c9cf4ef9e5df2b563cd9d0e201a8af8d4ec0199d2a71f640505",
        )

    def test_registered_backbone_and_forward_contract(self):
        sys.path.insert(0, str(ROOT))
        from hf_files.configuration_qwen3_ts import Qwen3TSConfig
        from hf_files.modeling_qwen3_ts import Qwen3TSForCausalLM
        from scripts.export_self_contained_chronos2 import expected_chronos2_keys

        chronos_config = tiny_chronos_config()
        config = Qwen3TSConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            ts_encoder_type="chronos2",
            chronos2_embedded=True,
            chronos2_config=chronos_config,
            projector_config={
                "input_hidden_size": 8,
                "activation": "gelu",
                "num_linear_layers": 2,
            },
            ts={"num_features": 2, "patch_size": 16, "max_sequence_length": 32},
            ts_token_start_index=30,
            ts_token_end_index=31,
        )
        model = Qwen3TSForCausalLM(config)
        state_keys = set(model.state_dict())
        self.assertTrue(
            any(key.startswith("ts_encoder.backbone.") for key in state_keys)
        )
        self.assertTrue(
            any(key.startswith("ts_encoder.projector.") for key in state_keys)
        )
        self.assertTrue(any(key.startswith("model.") for key in state_keys))
        self.assertEqual(
            set(model.ts_encoder.backbone.state_dict()),
            expected_chronos2_keys(chronos_config["num_layers"]),
        )

        values = torch.linspace(-1, 1, 17)
        encoded = torch.stack([values, torch.ones_like(values)], dim=-1).reshape(
            1, -1, 1
        )
        features, patch_count = model.ts_encoder(encoded)
        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(patch_count.tolist(), [2])

        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir, safe_serialization=True)
            reloaded = Qwen3TSForCausalLM.from_pretrained(save_dir)
            reloaded_keys = set(reloaded.state_dict())
            self.assertTrue(
                any(key.startswith("ts_encoder.backbone.") for key in reloaded_keys)
            )
            self.assertTrue(
                any(key.startswith("ts_encoder.projector.") for key in reloaded_keys)
            )

    def test_tensor_level_export(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            chatts = temp / "chatts"
            chronos = temp / "chronos"
            output = temp / "output"
            chatts.mkdir()
            chronos.mkdir()

            chatts_config = {
                "architectures": ["Qwen3TSForCausalLM"],
                "model_type": "qwen3ts",
                "hidden_size": 4,
                "ts_token_start_index": 10,
                "ts_token_end_index": 11,
                "ts": {
                    "num_features": 2,
                    "patch_size": 16,
                    "max_sequence_length": 8192,
                },
            }
            chronos_config = tiny_chronos_config()
            chronos_config["d_model"] = 768
            chronos_config["d_kv"] = 64
            chronos_config["d_ff"] = 3072
            chronos_config["num_heads"] = 12
            chronos_config["num_layers"] = 12
            chronos_config["chronos_config"]["context_length"] = 8192
            chronos_config["chronos_config"]["time_encoding_scale"] = 8192
            (chatts / "config.json").write_text(
                json.dumps(chatts_config), encoding="utf-8"
            )
            (chronos / "config.json").write_text(
                json.dumps(chronos_config), encoding="utf-8"
            )

            projector = {
                "ts_encoder.projector.input_norm.weight": torch.ones(768),
                "ts_encoder.projector.input_norm.bias": torch.zeros(768),
                "ts_encoder.projector.linear_in.weight": torch.zeros(4, 768),
                "ts_encoder.projector.linear_in.bias": torch.zeros(4),
                "ts_encoder.projector.linear_out.weight": torch.zeros(4, 4),
                "ts_encoder.projector.linear_out.bias": torch.zeros(4),
                "ts_encoder.projector.output_norm.weight": torch.ones(4),
                "ts_encoder.projector.output_norm.bias": torch.zeros(4),
                "model.embed_tokens.weight": torch.zeros(12, 4),
                "lm_head.weight": torch.zeros(12, 4),
            }
            save_file(projector, chatts / "model.safetensors")
            from scripts.export_self_contained_chronos2 import expected_chronos2_keys

            chronos_state = {
                key: torch.zeros(1)
                for key in expected_chronos2_keys(chronos_config["num_layers"])
            }
            chronos_state["shared.weight"] = torch.zeros(2, 768)
            save_file(chronos_state, chronos / "model.safetensors")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "export_self_contained_chronos2.py"),
                    "--chatts-checkpoint",
                    str(chatts),
                    "--chronos2-checkpoint",
                    str(chronos),
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "verify_self_contained_checkpoint.py"),
                    str(output),
                ],
                check=True,
            )
            merged_config = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(merged_config["chronos2_embedded"])
            self.assertNotIn("chronos2_model_name_or_path", merged_config)


if __name__ == "__main__":
    unittest.main()
