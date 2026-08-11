from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    source = (
        Path(__file__).parents[1]
        / "chatts"
        / "utils"
        / "inference_tinybenchmarks_mcq_vllm.py"
    )
    saved = {name: sys.modules.get(name) for name in ("chatts", "chatts.vllm", "chatts.vllm.chatts_vllm")}
    try:
        chatts = types.ModuleType("chatts")
        chatts.__path__ = []
        vllm_package = types.ModuleType("chatts.vllm")
        vllm_package.__path__ = []
        sys.modules["chatts"] = chatts
        sys.modules["chatts.vllm"] = vllm_package
        sys.modules["chatts.vllm.chatts_vllm"] = types.ModuleType(
            "chatts.vllm.chatts_vllm"
        )
        spec = importlib.util.spec_from_file_location("_tinybench_partition_under_test", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_tinybench_hash_partition_is_exact_disjoint_and_stratified() -> None:
    module = _load_module()
    rows = [
        {
            "id": f"row-{index}",
            "category": "a" if index < 50 else "b",
            "difficulty": "easy" if index % 2 else "hard",
        }
        for index in range(100)
    ]

    search = module.partition_task_rows("tinyMMLU", rows, "search-dev", 42)
    final = module.partition_task_rows("tinyMMLU", rows, "final-test", 42)
    search_ids = {row["id"] for row in search}
    final_ids = {row["id"] for row in final}

    assert len(search) == 20
    assert len(final) == 80
    assert search_ids.isdisjoint(final_ids)
    assert search_ids | final_ids == {row["id"] for row in rows}
    assert module.partition_task_rows("tinyMMLU", rows, "search-dev", 42) == search
    assert {row["category"] for row in search} == {"a", "b"}


def test_tinybench_all_partition_preserves_historical_rows() -> None:
    module = _load_module()
    rows = [{"id": "a"}, {"id": "b"}]
    assert module.partition_task_rows("tinyArc", rows, "all", 42) is rows
