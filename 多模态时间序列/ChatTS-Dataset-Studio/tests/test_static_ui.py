from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

STATIC_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "chatts_dataset_studio"
    / "static"
)


class _IdTreeParser(HTMLParser):
    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.ancestors: dict[str, tuple[str, ...]] = {}
        self._stack: list[tuple[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id is not None:
            self.ids.append(element_id)
            self.ancestors[element_id] = tuple(
                ancestor_id
                for _, ancestor_id in self._stack
                if ancestor_id is not None
            )
        if tag not in self._VOID_TAGS:
            self._stack.append((tag, element_id))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                return


def test_tsr_protocol_panel_is_inside_shared_evaluation_card() -> None:
    parser = _IdTreeParser()
    parser.feed((STATIC_ROOT / "index.html").read_text(encoding="utf-8"))

    duplicates = [value for value, count in Counter(parser.ids).items() if count > 1]
    assert duplicates == []
    assert parser.ids.index("standalone-evaluation-title") < parser.ids.index(
        "evaluation-config-card"
    ) < parser.ids.index("launch-guide-title")
    assert "evaluation-config-card" in parser.ancestors["tsr-protocol-panel"]
    for element_id in (
        "tsr-protocol-state",
        "tsr-protocol-disabled",
        "tsr-protocol-mode",
        "tsr-protocol-purpose",
        "tsr-prompt-preview",
        "tsr-max-model-len",
        "tsr-max-new-tokens",
        "tsr-batch-size",
        "tsr-request-chunk-size",
        "tsr-fixed-temperature",
        "tsr-fixed-top-p",
        "tsr-fixed-retries",
        "tsr-fixed-input-cutoff",
        "tsr-processed-input-budget",
        "tsr-fixed-native-thinking",
        "tsr-protocol-validation",
    ):
        assert "tsr-protocol-panel" in parser.ancestors[element_id]
    for element_id in (
        "benchmark-suites",
        "eval-max-samples",
        "eval-offline",
        "force-eval",
        "tiny-max-model-len",
        "haystack-max-model-len",
        "exam-max-model-len",
    ):
        assert "evaluation-config-card" in parser.ancestors[element_id]


def test_tsr_protocol_ui_uses_live_dom_values_and_safe_text_updates() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    for unsafe_api in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert unsafe_api not in javascript
    assert "function tsrProtocolValues()" in javascript
    assert "function tsrProtocolValidation()" in javascript
    assert "function renderTsrProtocol()" in javascript
    assert 'if (benchmarks.includes("tsrbench")) {' in javascript
    assert "Object.assign(payload, {" in javascript
    assert '$("#tsr-prompt-preview").textContent = protocol.outputInstruction;' in javascript
    assert '$("#tsr-protocol-body").hidden = !selected;' in javascript
    assert '$(selector).addEventListener("input", refreshProtocol);' in javascript
    assert '$(selector).addEventListener("change", refreshProtocol);' in javascript
    assert "evaluationProtocolValid" in javascript
    assert "submittedProtocol" in javascript
    assert ': "未选择 TSRBench";' in javascript
    assert "evaluation_protocol_id" in javascript


def test_tsr_protocol_ui_documents_runner_fixed_values() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    expected_mode_fragments = (
        'answer_only: {\n    maxModelLen: 12288,\n    maxNewTokens: 8,\n    batchSize: 16,',
        'official: {\n    maxModelLen: 12288,\n    maxNewTokens: 512,\n    batchSize: 1,',
        'json_reasoning: {\n    maxModelLen: 12288,\n    maxNewTokens: 256,\n    batchSize: 1,',
        'temperature: "1.0",\n    retries: "10",',
        'temperature: "0.0",\n    retries: "1",',
        'inputCutoff: "8,000 tokens",',
    )
    for fragment in expected_mode_fragments:
        assert fragment in javascript
    assert "Return exactly one uppercase option letter (A-G) and no other text." in javascript
    assert "<think>Your reasoning here (less than 2048 tokens)</think>" in javascript
    assert "Return exactly one valid JSON object with exactly two keys" in javascript
