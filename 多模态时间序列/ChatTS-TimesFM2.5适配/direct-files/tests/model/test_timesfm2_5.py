from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from llamafactory.model.model_utils.timeseries import patch_timeseries_modules_for_lora
from llamafactory.model.model_utils.timesfm2_5 import (
    TimesFM2_5TimeSeriesEncoder,
    maybe_replace_with_timesfm2_5_encoder,
)


class _FakeTimesFMBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_weight = nn.Parameter(torch.ones(1))

    def forward(self, values: torch.Tensor, masks: torch.Tensor):
        patch_signal = values.mean(dim=-1, keepdim=True)
        output_embeddings = patch_signal.expand(-1, -1, 1280).contiguous()
        outputs = (output_embeddings, output_embeddings, values, values)
        return outputs, []


def _make_chatts_input(lengths: list[int]) -> torch.Tensor:
    max_length = max(lengths)
    encoded = []
    for length in lengths:
        values = torch.arange(max_length, dtype=torch.float32)
        mask = torch.zeros(max_length, dtype=torch.float32)
        mask[:length] = 1.0
        encoded.append(torch.stack([values, mask], dim=-1).reshape(-1, 1))
    return torch.stack(encoded)


class _DummyChatTS(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.ts_encoder = nn.Linear(2, hidden_size)
        self.embed = nn.Embedding(8, hidden_size)
        self.config = SimpleNamespace(model_type="qwen3ts")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed


def test_timesfm_encoder_preserves_chatts_contract(monkeypatch) -> None:
    encoder = TimesFM2_5TimeSeriesEncoder(
        llm_hidden_size=64,
        model_name_or_path="fake/timesfm",
    )
    fake_wrapper = SimpleNamespace(model=_FakeTimesFMBackbone())
    monkeypatch.setattr(encoder, "_load_timesfm", lambda: fake_wrapper)

    features, patch_cnt = encoder(_make_chatts_input([40, 65]))

    assert patch_cnt.tolist() == [2, 3]
    assert features.shape == (5, 64)
    assert torch.isfinite(features).all()
    assert not fake_wrapper.model.training
    assert all(not param.requires_grad for param in fake_wrapper.model.parameters())


def test_timesfm_backbone_is_not_serialized(monkeypatch) -> None:
    encoder = TimesFM2_5TimeSeriesEncoder(
        llm_hidden_size=32,
        model_name_or_path="fake/timesfm",
    )
    fake_wrapper = SimpleNamespace(model=_FakeTimesFMBackbone())
    monkeypatch.setattr(encoder, "_load_timesfm", lambda: fake_wrapper)
    encoder(_make_chatts_input([32]))

    state_keys = list(encoder.state_dict())
    assert state_keys
    assert all(key.startswith("projector.") for key in state_keys)
    assert not any("frozen_weight" in key for key in state_keys)


def test_timesfm_encoder_rejects_overlong_series() -> None:
    encoder = TimesFM2_5TimeSeriesEncoder(
        llm_hidden_size=32,
        model_name_or_path="fake/timesfm",
        context_limit=32,
    )

    try:
        encoder(_make_chatts_input([33]))
    except ValueError as exc:
        assert "at most 32" in str(exc)
    else:
        raise AssertionError("Expected an overlong TimesFM input to fail.")


def test_timesfm_projector_stays_fully_trainable_with_lora() -> None:
    model = _DummyChatTS()
    model.ts_encoder = TimesFM2_5TimeSeriesEncoder(64, "fake/timesfm")
    finetuning_args = SimpleNamespace(train_timeseries_modules=True, additional_target=None)

    targets = patch_timeseries_modules_for_lora(
        model,
        finetuning_args,
        ["q_proj", "linear_in", "linear_out"],
    )

    assert targets == ["q_proj"]
    assert finetuning_args.additional_target == ["ts_encoder.projector"]


def test_timesfm_projector_restores_for_stage_two(tmp_path) -> None:
    source_encoder = TimesFM2_5TimeSeriesEncoder(64, "saved/timesfm")
    for parameter in source_encoder.projector.parameters():
        nn.init.constant_(parameter, 0.125)

    checkpoint = {f"ts_encoder.{key}": value for key, value in source_encoder.state_dict().items()}
    save_file(checkpoint, tmp_path / "model.safetensors")

    model = _DummyChatTS()
    config = SimpleNamespace(
        hidden_size=64,
        model_type="qwen3ts",
        ts={"num_features": 2, "patch_size": 8},
        ts_encoder_type="timesfm2_5",
        timesfm_model_name_or_path="saved/timesfm",
    )
    model_args = SimpleNamespace(
        ts_encoder_type="auto",
        timesfm_model_name_or_path="wrong/default",
        model_name_or_path=str(tmp_path),
        use_unsloth=False,
    )

    maybe_replace_with_timesfm2_5_encoder(model, config, model_args)

    assert model.ts_encoder.model_name_or_path == "saved/timesfm"
    assert config.ts["patch_size"] == 32
    assert all(
        torch.allclose(parameter, torch.full_like(parameter, 0.125))
        for parameter in model.ts_encoder.projector.parameters()
    )
