# L4 Mock Validation Results

> **Date:** 2026-03-26
> **Hardware:** NVIDIA L4 23GB, x86_64, Ubuntu 22.04
> **Mock model:** nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

---

## Test Results

| Test | Status | Detail |
|------|--------|--------|
| Basic chat | **PASS** | 2.1 t/s single-user, security-themed prompt |
| Streaming | **PASS** | 0.10s TTFT, smooth SSE delivery |
| Reasoning | **PASS** | Model produces `<think>` blocks, step-by-step analysis |
| Tool use | **WARN** | Pipeline works (tools sent, parsed), but Nano-4B too small to reliably emit structured tool calls. Expected to pass on 120B |
| Concurrent (3 users) | **PASS** | ~67 t/s aggregate throughput |
| LoRA fine-tuning | **PASS** | 3 epochs, 5 samples, 87s, adapter saved (13MB) |
| LoRA serving (hot-swap) | **NOT TESTED** | Adapter ready at `lora-output/` |

---

## Issues Resolved

### 1. torch.compile FakeTensorMode crash

**Symptom:** vLLM server failed to start with `FakeTensorMode` AttributeError.

**Fix:** `--enforce-eager` to disable torch compile. Not needed on GB10 (spark-vllm-docker has matching stack).

### 2. mamba-ssm CUDA version mismatch

**Symptom:** `ImportError: mamba-ssm is required` then CUDA 12.8 vs 13.0 build mismatch.

**Fix:** `pip install mamba-ssm causal-conv1d --no-build-isolation` -- forces build against installed torch/CUDA.

### 3. Gradient chain broken through Mamba layers

**Symptom:** `RuntimeError: element 0 of tensors does not require grad`

**Root cause:** With gradient checkpointing, non-LoRA Mamba layers between attention layers break the gradient chain.

**Fix:** `model.enable_input_require_grads()` after model load.

---

## LoRA Training Output

```
trainable params: 3,178,496 || all params: 3,976,735,328 || trainable%: 0.0799
```

Adapter saved: 13MB (A and B matrices for q_proj, k_proj, v_proj, o_proj across all layers).

---

## L4 Mock vs GB10 Production

| | L4 Mock | GB10 Production |
|--|---------|-----------------|
| Model | Nemotron-3-Nano-4B-BF16 | Nemotron-3-Super-120B-A12B-NVFP4 |
| Architecture | Hybrid Mamba+Attention (4B) | Hybrid Mamba+Attention MoE (120B, 12B active) |
| Serving | Bare `vllm serve` | spark-vllm-docker container |
| Tool-call parser | `hermes` | `qwen3_coder` |
| Quantization | BF16 | NVFP4 |
| GPU memory | 23GB | 128GB unified |
