# Gemma-4-26B-A4B-it Deployment on Dell Pro Max GB10

**Date:** 2026-04-02
**Status:** Running

---

## Model Specs

| Spec | Value |
|---|---|
| Model | google/gemma-4-26B-A4B-it |
| Architecture | MoE — 128 experts, 8 active + 1 shared |
| Total / Active params | 25.2B / 3.8B |
| Precision | BF16 (~52 GB) |
| Context | 256K tokens |
| Modalities | Text + Vision |

## Performance on GB10

| Metric | Result |
|---|---|
| Decode speed | 24.4 tok/s |
| TTFT (after warmup) | 0.11s |
| Thinking overhead | None |

### vs Cascade-2-30B

| | Cascade-2 | Gemma-4 |
|---|---|---|
| Decode speed | 29.7 tok/s | 24.4 tok/s |
| TTFT (visible) | ~7s (thinking) | 0.11s |
| Thinking tokens | ~200 wasted | 0 |
| Quality | Good | On par or better |
| Vision | No | Yes |

Gemma-4 feels significantly faster in practice due to zero thinking overhead.

---

## How We Made It Work

### The Problem

Gemma-4 was released on 2026-04-02 (same day we tried to deploy). Multiple compatibility issues:

1. **transformers 5.5.0 required** — Google jumped from 4.x to 5.x, Gemma-4 architecture (`gemma4`) only exists in 5.5.0+
2. **vLLM native support merged same day** — PR #38826 merged April 2, no aarch64 nightly wheel available yet
3. **Container's vLLM (0.18.1/cu132)** had no Gemma-4 support, Transformers fallback had bugs (`NoneType` dtype error, `top_k` assertion failure)
4. **Nightly wheel (cu130)** caused ABI mismatch with container's PyTorch 2.11/cu132 (`libtorch_cuda.so` missing, undefined symbols)
5. **Building vLLM from source inside container** failed due to missing `numa.h` header
6. **`compressed-tensors`** requires `transformers<5`, so upgrading transformers in the existing container breaks NVFP4 quantization for Nemotron 120B

### What We Tried (and Failed)

| Attempt | Result |
|---|---|
| `pip install --upgrade transformers` | PyPI had 5.5.0 but broke `compressed-tensors` |
| `pip install --no-deps transformers` from git | Got 5.5.0.dev0, `huggingface_hub` import error |
| vLLM nightly wheel via pip index | Wrong index URL, couldn't find package |
| vLLM nightly wheel direct download (cu130) | ABI mismatch — downgraded PyTorch, broke `libtorch_cuda.so` |
| vLLM nightly wheel with `--no-deps` | `undefined symbol` — cu130 binary vs cu132 PyTorch |
| `pip install git+vllm` (build from source) | Missing `numa.h`, CUDA compile failed |
| Sed patching vLLM source in container | Fixed one bug, hit another, then another — whack-a-mole |

### The Solution: Dedicated Container

Built a **separate container** (`vllm-gemma4`) from source so Nemotron recipes on `vllm-node` stay untouched:

```bash
cd spark-vllm-docker/
./build-and-copy.sh --pre-transformers --vllm-ref main --tag vllm-gemma4
```

This builds:
- vLLM from latest `main` branch (has native Gemma-4 support)
- `--pre-transformers` enables transformers >= 5.0.0
- Compiled against the container's own CUDA 13.2 + PyTorch 2.11 — no ABI mismatch
- Tagged as `vllm-gemma4` — completely separate from `vllm-node`

### Running

```bash
cd spark-vllm-docker/
./run-recipe.sh gemma-4-26b --solo --setup
```

Recipe: `spark-vllm-docker/recipes/gemma-4-26b.yaml`
Container: `vllm-gemma4`

---

## Speed Improvement Options

| Approach | Expected tok/s | Notes |
|---|---|---|
| Current (BF16, single GB10) | 24.4 | Memory bandwidth limited |
| Dual GB10 TP=2 | ~40+ | Doubles bandwidth via NVLink |
| Quantized checkpoint (FP8/NVFP4) | ~40-48 | When available |
| Aggressive tuning (eager, fp8 KV) | ~27-29 | Marginal, not worth it |

---

## Key Lessons

1. **Day-one model deployment is painful** — toolchain (vLLM, transformers) needs time to catch up
2. **aarch64 (GB10/ARM) gets wheels later** — x86 nightly wheels come first
3. **Separate containers protect existing workloads** — `vllm-gemma4` vs `vllm-node`
4. **Building from source bypasses ABI hell** — compiles against your exact CUDA/PyTorch
5. **`--pre-transformers` breaks `compressed-tensors`** — never use it on a container that serves NVFP4 models
