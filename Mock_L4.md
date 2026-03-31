# Phase 1 — L4 Mock Run Status

> **Date:** 2026-03-26
> **Hardware:** NVIDIA L4 23GB, x86_64, Ubuntu 22.04
> **Software:** vLLM 0.18.0, PyTorch 2.10+cu128, Python 3.10
> **Mock model:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

---

## Files Created

| File | Purpose | Reusable on GB10? |
|---|---|---|
| `scripts/serve_l4_mock.sh` | vLLM serve for Nano-4B on L4 | No — throwaway |
| `scripts/serve_gb10.sh` | spark-vllm-docker wrapper for 120B NVFP4 | Yes |
| `scripts/test_api.py` | API test client (chat, tool-use, streaming, reasoning, concurrent) | Yes — model-agnostic |
| `scripts/finetune_lora.py` | LoRA fine-tuning with HF PEFT | Yes — change `--mock` to default |

---

## Test Results

| Test | Status | Detail |
|---|---|---|
| Basic chat | **PASS** | 2.1 t/s single-user, security-themed prompt |
| Streaming | **PASS** | 0.10s TTFT, smooth SSE delivery |
| Reasoning | **PASS** | Model produces `<think>` blocks, step-by-step Apache CVE analysis |
| Tool use | **WARN** | Pipeline works (tools sent, parsed), but Nano-4B too small to reliably emit structured tool calls. Expected to pass on 120B Super |
| Concurrent (3 users) | **PASS** | ~67 t/s aggregate throughput |
| LoRA fine-tuning | **PASS** | 3 epochs, 5 samples, 87s, adapter saved (13MB) |
| LoRA serving (hot-swap) | **NOT TESTED** | Adapter ready at `scripts/lora-output/` |

---

## Issues Identified

### 1. torch.compile `FakeTensorMode` crash (resolved)

**Symptom:** vLLM server failed to start with:
```
AttributeError: <function standalone_compile at 0x...> does not have the attribute 'FakeTensorMode'
```

**Root cause:** Compatibility issue between vLLM 0.18.0's CUDAGraph compilation and this environment.

**Fix:** Added `--enforce-eager` to disable torch compile. Disables CUDAGraph optimization (slightly slower), but server runs fine. This flag is NOT needed on GB10 — the spark-vllm-docker image has a matching torch/CUDA stack.

---

### 2. `mamba-ssm` CUDA version mismatch (resolved)

**Symptom:** `finetune_lora.py --mock` fails with `ImportError: mamba-ssm is required`. Installing `mamba-ssm` then fails with CUDA 12.8 vs 13.0 mismatch.

**Root cause:** pip's build isolation pulled a different PyTorch version (compiled for CUDA 13.0) instead of using the installed one (CUDA 12.8).

**Fix:** `pip install mamba-ssm causal-conv1d --no-build-isolation` — forces build against the installed torch/CUDA 12.8 stack. Compilation takes ~10 min (builds CUDA kernels for sm_62 through sm_120).

---

### 3. Gradient checkpointing crash on hybrid Mamba+Attention (resolved)

**Symptom:** After fixing mamba-ssm, training crashed with:
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
Preceded by: `None of the inputs have requires_grad=True. Gradients will be None`

**Root cause:** LoRA targets only the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`). With gradient checkpointing, the non-LoRA Mamba layers between them break the gradient chain — inputs to checkpointed segments don't have `requires_grad=True`, so backward pass fails.

**Fix:** Added `model.enable_input_require_grads()` after model load in `finetune_lora.py`. This ensures all inputs to checkpointed layers carry gradient tracking, allowing gradients to flow through Mamba layers to reach the LoRA attention layers.

---

## Validation Coverage

| Capability | Validated on L4? | Tested on GB10? |
|---|---|---|
| vLLM serving pipeline | Yes | — |
| OpenAI-compatible API | Yes | — |
| Tool-use (structured calls) | Partial (pipeline only) | — |
| Reasoning mode (`<think>`) | Yes | — |
| Concurrent users | Yes (3 users) | — |
| LoRA fine-tuning | **Yes** | — |
| LoRA hot-swap serving | **No** | — |
| spark-vllm-docker | No (ARM64/sm_121 only) | — |
| NVFP4 quantization | No (B10-specific) | — |
| NVLink multi-node | No (single GPU) | — |

---

## Key Differences: L4 Mock vs GB10 Production

| | L4 Mock | GB10 Production |
|---|---|---|
| Model | Nemotron-3-Nano-4B-BF16 | Nemotron-3-Super-120B-A12B-NVFP4 |
| Architecture | Hybrid Mamba+Attention (4B) | Hybrid Mamba+Attention MoE (120B, 12B active) |
| Serving | Bare `vllm serve` | spark-vllm-docker container |
| Tool-call parser | `hermes` | `qwen3_coder` |
| Reasoning parser | Default (model's `<think>` tags) | Custom `super_v3` plugin |
| Quantization | BF16 (none) | NVFP4 |
| KV cache | auto | FP8 |
| MoE backend | N/A | CUTLASS |
| GPU memory | 23GB | 128GB unified |
| `--enforce-eager` | Required (torch.compile bug) | Not needed |
