# Search notes

## Scope

- Goal: architecture-only update to the ChatTS time-series encoder.
- Hard interface constraint: retain fixed patch size, per-sample patch count, one output token per patch, and the existing text-token replacement path.
- Preference: native time-series evidence; avoid presenting vision-origin architectures as if they were time-series papers.
- Evidence policy: primary paper/proceedings pages and official repositories first; no MDPI sources used.
- Cutoff: sources checked on 2026-07-25 (Asia/Shanghai).

## Query families

1. "time series patch encoder 2024 2025 ICLR ICML AAAI"
2. "time series Mamba patch bidirectional official code"
3. "time series LSTM patch encoder AAAI 2025"
4. "time series multimodal LLM question answering encoder 2025"
5. "general time series encoder multi task 2024 2025"
6. "ChatTS five layer MLP patch encoder"

## Direct evidence checks

- ChatTS paper §3.4.1 states fixed-size patches and a simple 5-layer MLP per patch. Its limitations section explicitly names more effective multimodal encoding/integration as future work.
- The local public implementation was inspected at work/ChatTS/chatts/vllm/chatts_vllm.py:61–193. It computes patch_cnt = ceil(valid_length / patch_size), pads the final patch with its last value, concatenates all patches, applies a shared MLP independently, and returns flat features plus patch_cnt.
- The official ChatTS-8B config was checked at https://huggingface.co/bytedance-research/ChatTS-8B/blob/main/config.json: patch_size=8, num_layers=5, hidden_size=4096, embedding_dim=16, use_position_embedding=true. From these public dimensions, the time encoder has about 67.8M parameters including its position table; this is a deterministic architecture calculation, not a benchmark result.
- ModernTCN official paper and repository were checked for large-kernel depthwise temporal convolution and its full multi-stage design. The report’s "lite" version is an explicit project adaptation, not a claim that the original paper used this exact ChatTS block.
- P-sLSTM official AAAI page and repository were checked for fixed patching, channel independence, and stacked sLSTM.
- Bi-Mamba+ arXiv v3 and official repository were checked. The report uses the channel-independent temporal-patch path. Publication status is kept as preprint.
- PatchTST, TimeMixer++, and ITFormer were checked on official ICLR/PMLR proceedings pages.
- OpenTSLM is treated as arXiv v3 in the evidence table; no unverified acceptance claim is used.

## Screening logic

Primary question: "Does the architecture mix information along temporal patches before the LLM while preserving N output tokens?"

- Yes + low dependency: shortlist.
- Yes + significant custom CUDA or preprint risk: shortlist-risk.
- Variable-axis rather than temporal-axis mixing: exclude-axis-mismatch.
- Requires new tokenizer/decoder/multiscale alignment: boundary or exclude-complex.
- Changes cross-modal integration rather than only encoder: boundary-nearest.

## Important non-claim

No reviewed paper directly proves that any shortlisted encoder improves ChatTS QA/reasoning under a controlled architecture-only replacement. Recommendations are evidence-backed compatibility hypotheses, not reported ChatTS results. The report does not invent accuracy, latency, or parameter outcomes.
