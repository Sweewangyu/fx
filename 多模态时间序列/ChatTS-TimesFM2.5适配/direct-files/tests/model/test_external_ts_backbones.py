from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from llamafactory.model.model_utils.chronos2 import Chronos2TimeSeriesEncoder
from llamafactory.model.model_utils.timeseries_backbones import maybe_replace_timeseries_encoder
from llamafactory.model.model_utils.timesfm2_5 import TimesFM2_5TimeSeriesEncoder
from llamafactory.model.model_utils.zeus import ZeusTimeSeriesEncoder
from llamafactory.model.model_utils.zeus_modeling import ZeusConfig, ZeusForPrediction


def _make_chatts_input(lengths: list[int]) -> torch.Tensor:
    max_length = max(lengths)
    encoded = []
    for length in lengths:
        values = torch.arange(max_length, dtype=torch.float32)
        mask = torch.zeros(max_length, dtype=torch.float32)
        mask[:length] = 1.0
        encoded.append(torch.stack([values, mask], dim=-1).reshape(-1, 1))
    return torch.stack(encoded)


class _FakeChronos2Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_weight = nn.Parameter(torch.ones(1))

    def encode(self, context: torch.Tensor, group_ids: torch.Tensor):
        del group_ids
        patch_count = (context.size(1) + 15) // 16
        # The final two embeddings represent Chronos-2's REG and future token.
        context_tokens = torch.ones(1, patch_count, 768, device=context.device)
        special_tokens = torch.full((1, 2, 768), 99.0, device=context.device)
        outputs = SimpleNamespace(last_hidden_state=torch.cat((context_tokens, special_tokens), dim=1))
        return outputs, None, None, patch_count


class _FakeZeusBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_weight = nn.Parameter(torch.ones(1))
        self.last_inputs = None
        self.last_padding_mask = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        return_all_hidden_states: bool,
    ):
        assert return_all_hidden_states
        assert torch.count_nonzero(targets_mask) == 0
        self.last_inputs = inputs
        self.last_padding_mask = padding_mask
        patch_count = (inputs.size(1) + 31) // 32
        center = torch.ones(inputs.size(0), patch_count, 768, device=inputs.device)
        return {"all_hidden_states": [inputs, inputs, center, inputs, inputs]}


class _DummyChatTS(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.ts_encoder = nn.Linear(2, hidden_size)
        self.embed = nn.Embedding(8, hidden_size)
        self.config = SimpleNamespace(model_type="qwen3ts")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed


def test_chronos2_encoder_keeps_context_tokens_only(monkeypatch) -> None:
    encoder = Chronos2TimeSeriesEncoder(64, "fake/chronos2")
    backbone = _FakeChronos2Backbone()
    handle = SimpleNamespace(pipeline=SimpleNamespace(model=backbone))
    monkeypatch.setattr(encoder, "_load_chronos2", lambda: handle)

    features, patch_cnt = encoder(_make_chatts_input([17, 33]))

    assert patch_cnt.tolist() == [2, 3]
    assert features.shape == (5, 64)
    assert torch.isfinite(features).all()
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(key.startswith("projector.") for key in encoder.state_dict())


def test_zeus_encoder_uses_center_scale_and_masked_normalization(monkeypatch) -> None:
    encoder = ZeusTimeSeriesEncoder(64, "fake/zeus")
    backbone = _FakeZeusBackbone()
    handle = SimpleNamespace(model=backbone)
    monkeypatch.setattr(encoder, "_load_zeus", lambda: handle)

    features, patch_cnt = encoder(_make_chatts_input([31, 65]))

    assert patch_cnt.tolist() == [1, 3]
    assert features.shape == (4, 64)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(backbone.last_inputs[0, 31:]) == 0
    assert torch.count_nonzero(backbone.last_padding_mask[0, 31:]) == 0
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(key.startswith("projector.") for key in encoder.state_dict())


def test_zeus_eager_model_preserves_official_module_layout() -> None:
    config = ZeusConfig(
        hidden_size=[8, 16, 16, 16, 8],
        n_heads=[2, 2, 2, 2, 2],
        intermediate_size=[16, 32, 32, 32, 16],
        num_layers=[1, 1, 1, 1, 1],
        scales=[1, 2, 4, 2, 1],
        num_reg_tokens=1,
    )
    model = ZeusForPrediction(config).eval()
    inputs = torch.randn(2, 7, 1)
    mask = torch.ones(2, 7, 1, dtype=torch.int32)

    outputs = model(
        inputs,
        targets_mask=torch.zeros_like(mask),
        padding_mask=mask,
        return_all_hidden_states=True,
    )

    assert [tuple(hidden.shape) for hidden in outputs["all_hidden_states"]] == [
        (2, 8, 8),
        (2, 4, 16),
        (2, 2, 16),
        (2, 4, 16),
        (2, 8, 8),
    ]
    assert "encoders.2.layers.0.self_attn.out_proj.weight" in model.state_dict()
    assert "encoders.2.layers.0.self_attn.out_proj.bias" not in model.state_dict()


def test_cross_architecture_projector_is_not_restored(tmp_path) -> None:
    source_encoder = TimesFM2_5TimeSeriesEncoder(64, "saved/timesfm")
    checkpoint = {f"ts_encoder.{key}": value for key, value in source_encoder.state_dict().items()}
    save_file(checkpoint, tmp_path / "model.safetensors")

    model = _DummyChatTS()
    config = SimpleNamespace(
        hidden_size=64,
        model_type="qwen3ts",
        ts={"num_features": 2, "patch_size": 32},
        ts_encoder_type="timesfm2_5",
        timesfm_model_name_or_path="saved/timesfm",
    )
    model_args = SimpleNamespace(
        ts_encoder_type="chronos2",
        timesfm_model_name_or_path="saved/timesfm",
        chronos2_model_name_or_path="amazon/chronos-2",
        zeus_model_name_or_path="GestaltCog/zeus",
        model_name_or_path=str(tmp_path),
        use_unsloth=False,
    )

    maybe_replace_timeseries_encoder(model, config, model_args)

    assert isinstance(model.ts_encoder, Chronos2TimeSeriesEncoder)
    assert config.ts_encoder_type == "chronos2"
    assert config.ts["patch_size"] == 16


def test_chronos2_projector_restores_with_auto_stage_two(tmp_path) -> None:
    source_encoder = Chronos2TimeSeriesEncoder(64, "saved/chronos2")
    for parameter in source_encoder.projector.parameters():
        nn.init.constant_(parameter, 0.25)
    checkpoint = {f"ts_encoder.{key}": value for key, value in source_encoder.state_dict().items()}
    save_file(checkpoint, tmp_path / "model.safetensors")

    model = _DummyChatTS()
    config = SimpleNamespace(
        hidden_size=64,
        model_type="qwen3ts",
        ts={"num_features": 2, "patch_size": 16},
        ts_encoder_type="chronos2",
        chronos2_model_name_or_path="saved/chronos2",
    )
    model_args = SimpleNamespace(
        ts_encoder_type="auto",
        timesfm_model_name_or_path="google/timesfm-2.5-200m-pytorch",
        chronos2_model_name_or_path="wrong/default",
        zeus_model_name_or_path="GestaltCog/zeus",
        model_name_or_path=str(tmp_path),
        use_unsloth=False,
    )

    maybe_replace_timeseries_encoder(model, config, model_args)

    assert isinstance(model.ts_encoder, Chronos2TimeSeriesEncoder)
    assert model.ts_encoder.model_name_or_path == "saved/chronos2"
    assert all(
        torch.allclose(parameter, torch.full_like(parameter, 0.25))
        for parameter in model.ts_encoder.projector.parameters()
    )
