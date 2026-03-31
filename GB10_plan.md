# GB10 Setup Plan

> Status: **Requirements complete** — ready for execution.
>
> This plan covers two models on separate GB10 units:
> - **Nemotron-3-Super-120B** (NVFP4) — primary production model
> - **Nemotron-3-Nano-30B** (FP8) — lightweight, high-throughput model

## Hardware

| Spec | Per Dell Pro Max GB10 | Paired (NVLink) |
|---|---|---|
| GPU | Blackwell B10 (sm_121) | 2x B10 |
| CPU | 72-core Grace (ARM64) | 144 cores |
| Memory | 128GB unified LPDDR5X | 256GB unified |
| Interconnect | NVLink-C2C (CPU↔GPU) | ConnectX NVLink (GPU↔GPU) |
| Power | ~120W | ~240W |
| OS | Ubuntu Linux (Dell OEM) | — |

## Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Model | Nemotron-3-Super-120B-A12B-**NVFP4** | Fits single GB10, proven stable, ~0.5pt avg quality loss vs BF16 |
| Upgrade path | FP8 on dual GB10 | If NVFP4 quality insufficient |
| Serving framework | vLLM (via Docker) | Best LoRA hot-swap, flexibility, community support |
| Docker setup | eugr/spark-vllm-docker | Handles sm_121 + ARM64, has NVFP4 Nemotron recipe |
| Fine-tuning method | LoRA (HuggingFace PEFT) | Most popular, proven ARM64, sufficient for ~thousands of examples |
| Fine-tuning domain | Security operations & red team (pentesting) | Offensive security datasets from HF + self-prepared |
| Serving target | API for 3-5 team members, agentic use | Low concurrency, single-node NVFP4 sufficient |
| Networking | User handles SSH/remote access | Office/lab network, direct internet |

---

## Phase 1 — Mock Runs on L4 (pre-arrival)

**Goal:** Validate the entire pipeline with a small model on the current dev server (x86_64, L4 23GB).

- [ ] 1.1 Install vLLM on L4 (x86_64, standard pip install)
- [ ] 1.2 Download & serve Nemotron-Nano-4B as mock model
- [ ] 1.3 Write and test API client script (OpenAI-compatible API)
- [ ] 1.4 Test agentic/tool-use workflows via API
- [ ] 1.5 Run LoRA fine-tuning on Nano with sample security dataset (HF PEFT)
- [ ] 1.6 Serve fine-tuned LoRA adapter alongside base model
- [ ] 1.7 Prepare GB10 deployment scripts (spark-vllm-docker configs)

**Deliverables:**
- `scripts/serve_l4_mock.sh` — L4 mock serving
- `scripts/serve_gb10.sh` — GB10 production serving (spark-vllm-docker)
- `scripts/finetune_lora.py` — LoRA training script
- `scripts/test_api.py` — API test client
- `configs/nemotron-super-nvfp4.yaml` — vLLM recipe for GB10

---

## Phase 2 — GB10 Hardware Setup (on arrival)

**Goal:** Get both Dell Pro Max GB10 units operational and verified.

- [ ] 2.1 Unbox, power on, verify Ubuntu boots on both units
- [ ] 2.2 Connect NVLink cable between both units
- [ ] 2.3 Verify GPU detected: `nvidia-smi` (should show B10, sm_121)
- [ ] 2.4 Verify NVLink: `nvidia-smi nvlink --status`
- [ ] 2.5 Install Docker + NVIDIA Container Runtime on both units
- [ ] 2.6 Set up passwordless SSH between both units (for Ray cluster)
- [ ] 2.7 Verify NVIDIA driver version is 580.x (NOT 590.x — causes CUDAGraph deadlock)
- [ ] 2.8 Pull or build spark-vllm-docker image on head node
- [ ] 2.9 Copy image to second node: `docker save | ssh node2 docker load`

**Verification checklist:**
```bash
# On each node:
nvidia-smi                          # Confirm B10 GPU, driver 580.x
nvidia-smi nvlink --status          # Confirm NVLink active
docker run --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi  # Confirm Docker GPU access
```

---

## Phase 3 — Model Deployment (NVFP4, single node)

**Goal:** Serve Nemotron-3-Super-120B-A12B-NVFP4 on one GB10 via vLLM.

- [ ] 3.1 Download model: `huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- [ ] 3.2 Launch vLLM via spark-vllm-docker recipe:
  ```bash
  ./run-recipe.sh nemotron-3-super-nvfp4 --solo --setup
  ```
- [ ] 3.3 Verify serving: `curl http://localhost:8000/v1/models`
- [ ] 3.4 Run API test client from Phase 1
- [ ] 3.5 Benchmark: single-user throughput (expect ~16-17 t/s)
- [ ] 3.6 Benchmark: concurrent users (3-5 users, expect ~40-60 t/s total)
- [ ] 3.7 Test reasoning mode on/off
- [ ] 3.8 Test tool-use/agentic capabilities

**Expected vLLM command (from recipe):**
```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --moe-backend cutlass \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.7 \
  --max-model-len 262144 \
  --max-num-seqs 10 \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --reasoning-parser nemotron_v3 \
  --mamba_ssm_cache_dtype float32 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 8000
```

---

## Phase 4 — LoRA Fine-tuning (security/pentest)

**Goal:** Fine-tune Nemotron for offensive security operations using LoRA.

- [ ] 4.1 Prepare training dataset:
  - Source offensive security datasets from HuggingFace
  - Add self-prepared pentest data
  - Format as instruction/response pairs
  - Target: ~thousands of examples
- [ ] 4.2 Configure LoRA:
  - Framework: HuggingFace PEFT
  - Rank: 16-64 (start with 32)
  - Target modules: attention layers (q_proj, k_proj, v_proj, o_proj)
  - Learning rate: 1e-4 to 2e-4
  - Enable gradient checkpointing (saves memory)
- [ ] 4.3 Run training on GB10
- [ ] 4.4 Evaluate: compare base vs fine-tuned on security tasks
- [ ] 4.5 Serve LoRA adapter via vLLM hot-swap:
  ```bash
  --enable-lora --lora-modules security-ops=/path/to/adapter
  ```
- [ ] 4.6 Iterate: adjust data, rank, learning rate based on results

---

## Phase 5 — FP8 Upgrade (optional, if needed)

**Goal:** Switch to FP8 for higher quality, using both GB10 nodes.

- [ ] 5.1 Download FP8 model: `huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`
- [ ] 5.2 Create FP8 recipe for spark-vllm-docker (based on NVFP4 recipe)
- [ ] 5.3 Launch in cluster mode with TP=2:
  ```bash
  ./run-recipe.sh nemotron-3-super-fp8 --setup
  # (cluster mode is default, TP=2)
  ```
- [ ] 5.4 Benchmark and compare quality vs NVFP4
- [ ] 5.5 Re-run LoRA fine-tuning on FP8 base if quality improvement confirmed

**Known risks:**
- Only 1 person has partially succeeded (23 t/s, some instability)
- Another attempt with TRT-LLM crashed (NCCL timeouts)
- FP8 recipe doesn't exist in spark-vllm-docker yet — we'd create it

---

## Phase 6 — Production Hardening

**Goal:** Make the setup reliable for daily team use.

- [ ] 6.1 Set up systemd service or Docker Compose for auto-start on boot
- [ ] 6.2 Configure model auto-download on first run
- [ ] 6.3 Set up basic monitoring (GPU util, memory, request latency)
- [ ] 6.4 Configure API authentication (API keys for team members)
- [ ] 6.5 Set up log rotation
- [ ] 6.6 Document team onboarding: API endpoint, auth, usage examples
- [ ] 6.7 Test failure recovery: what happens if one node reboots?

---

---

## Nemotron-3-Nano-30B-A3B-FP8 (Second GB10)

### Model Overview

| Spec | Value |
|---|---|
| Architecture | Mamba2-Transformer Hybrid MoE |
| Total params | 30B |
| Active params per token | ~3.5B (~10% of total) |
| Quantization | FP8 (selective — attention & Mamba layers kept in BF16) |
| FP8 weights size | ~30 GB |
| Context length | 256K default (up to 1M) |
| MoE layers | 23 layers, 128 routed experts + 1 shared per layer, 6 active |
| Attention layers | 6 (GQA, 2 groups) |
| Mamba-2 layers | 23 |

### GB10 Fit

- FP8 weights (~30 GB) fit easily in 128 GB unified memory
- Single node, TP=1 — no NVLink needed
- Higher `gpu_memory_utilization` (0.85) possible due to smaller model
- More headroom for KV cache → better concurrent performance

### Phase N1 — Deploy Nano-30B FP8

**Goal:** Serve Nemotron-3-Nano-30B-A3B-FP8 on the second GB10.

- [ ] N1.1 Download model:
  ```bash
  huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  ```
- [ ] N1.2 Launch via spark-vllm-docker recipe:
  ```bash
  cd spark-vllm-docker/
  ./run-recipe.sh nemotron-3-nano-fp8 --solo --setup
  ```
  Or use the wrapper script:
  ```bash
  ./scripts/serve_nano30b.sh --setup
  ```
- [ ] N1.3 Verify serving: `curl http://localhost:8000/v1/models`
- [ ] N1.4 Run API tests:
  ```bash
  python3 scripts/test_api.py --all --base-url http://<nano-gb10-ip>:8000/v1
  ```

**Recipe:** `spark-vllm-docker/recipes/nemotron-3-nano-fp8.yaml`

**vLLM command (from recipe):**
```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --moe-backend cutlass \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --host 0.0.0.0 --port 8000
```

### Phase N2 — Benchmark & Evaluate

**Goal:** Measure output speed and compare with 120B deployment.

- [ ] N2.1 Run throughput benchmark:
  ```bash
  python3 scripts/bench_nano30b.py --full --output bench_nano30b_results.json \
    --base-url http://<nano-gb10-ip>:8000/v1
  ```
- [ ] N2.2 Benchmark results to collect:
  - Time to first token (TTFT)
  - Single-user output tokens/sec
  - Concurrent output tokens/sec (1, 3, 5, 10 users)
  - Output speed at different generation lengths (64–1024 tokens)
  - Reasoning on vs off comparison
- [ ] N2.3 Compare quality vs 120B on security tasks (same eval prompts)
- [ ] N2.4 Document results in Notes.md

**Benchmark scripts:**
- `scripts/bench_nano30b.py` — dedicated throughput benchmark
- `scripts/test_api.py` — functional tests (also measures t/s)

### Phase N3 — LoRA Fine-tuning (if proceeding)

**Goal:** Fine-tune Nano-30B for security ops, same as 120B pipeline.

- [ ] N3.1 Run LoRA training with existing security dataset
- [ ] N3.2 Evaluate base vs fine-tuned on security tasks
- [ ] N3.3 Serve LoRA adapter:
  ```bash
  # Add to recipe or vllm command:
  --enable-lora --lora-modules security-ops=/path/to/adapter
  ```

### Key Differences: Nano-30B vs Super-120B

| Aspect | Super-120B (NVFP4) | Nano-30B (FP8) |
|---|---|---|
| Weights on disk | ~69.5 GB | ~30 GB |
| Active params/token | 12B | 3.5B |
| Quantization | 4-bit (NVFP4) | 8-bit (FP8) |
| Memory headroom | Tight (128 GB) | Plenty (128 GB) |
| Expected single-user t/s | ~16-17 | TBD (benchmark) |
| Quality | Higher (larger active set) | Lower but strong for size |
| Best for | Quality-critical tasks | High-throughput, lower-latency |

---

## Key References

| Resource | URL |
|---|---|
| Model (NVFP4) | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 |
| Model (FP8) | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 |
| spark-vllm-docker | https://github.com/eugr/spark-vllm-docker |
| vLLM GB10 support issue | https://github.com/vllm-project/vllm/issues/31128 |
| NVIDIA forum (NVFP4) | https://forums.developer.nvidia.com/t/nvidia-nemotron-3-super-120b-a12b-nvfp4/363175 |
| FlashAttention sm_121 | https://github.com/Dao-AILab/flash-attention/issues/1969 |
| Model (Nano FP8) | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 |
| Model (Nano NVFP4) | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 |
