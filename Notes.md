# GB10 Setup — Q&A Log

---

## Q1 — Primary Use Case & Model Targets

**Q (Claude):**
1. What's the primary use case — inference serving, fine-tuning, both?
2. Which Nemotron models specifically?
3. Are you planning to serve these models to other apps/users (API server), or is this for personal/team experimentation?

**A (User):**
- Target model: **NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** (start here), upgrade to **FP8** if quality not sufficient
- Use case: **Both inference and fine-tuning**
- Serving: **API for agentic use**, small team of 3-5 members
- Fine-tuning method: **LoRA only** (full fine-tune not feasible on 256GB)
- Fine-tuning domain: **Security operations & red team (pentesting)**
- Training data: **~thousands of examples** from HuggingFace offensive security datasets + self-prepared data
- Training framework: TBD — needs to be proven/popular (see Q12)
- Serving framework: **vLLM** (via Docker, using eugr/spark-vllm-docker)

### Model Research Summary

| Spec | Detail |
|---|---|
| Architecture | Hybrid — LatentMoE (Mamba-2 + MoE + Attention), 88 layers |
| Total params | 120B |
| Active params/token | 12B (512 routed experts, 22 selected + 1 shared) |
| FP8 size on disk | ~128.4 GB (26 safetensors shards) |
| Context length | 256K default, up to 1M tokens |
| Quantization | FP8 weights + activations + KV cache (via ModelOpt) |
| Languages | EN, FR, DE, IT, JA, ES, ZH |
| Reasoning | Configurable on/off thinking mode |
| License | NVIDIA Nemotron Open Model License (commercial OK) |

**Official HW requirements:**
- Minimum: 2x H100-80GB (Hopper)
- Single-GPU: 1x B200/B300 (Blackwell)
- Supported frameworks: vLLM (v0.17.1+), SGLang, TensorRT-LLM

---

## GB10 Feasibility Assessment

**Critical question: Can 2x Dell Pro Max GB10 run this model?**

**CORRECTION:** User's hardware is **Dell Pro Max GB10**, NOT NVIDIA DGX Spark.
- Same GB10 superchip (identical silicon, memory, NVLink)
- Key difference: Dell ships with **Ubuntu Linux**, NOT DGX OS
- No pre-installed NVIDIA AI Enterprise / NGC stack — must set up from scratch
- Dell ProSupport instead of NVIDIA direct support
- All community Docker repos (spark-vllm-docker, vllm-dgx-spark) were built for DGX Spark but should work on Dell Pro Max since hardware is identical

### Memory: YES — fits comfortably
- FP8 checkpoint = ~128 GB on disk
- 2x GB10 = 256 GB unified memory (NVLink connected)
- ~128 GB headroom for KV cache, SSM state, activations

### vLLM Model Code: YES — hardware-agnostic
- `nemotron_h` implementation has **zero GPU-type checks**
- TP uses standard vLLM parallel primitives (QKVParallelLinear, ColumnParallelLinear, etc.)
- Mamba-2 kernels are **Triton-based** (not CUDA C++), so they run on any GPU
- One minor note: `MambaMixer2` checks `is_blackwell` (SM 10.x family) for kernel tuning block sizes. GB10 (SM 12.1) will use the non-Blackwell path — still correct, just potentially suboptimal block sizes

### Compute Capability: DIFFERENT SM families
| GPU | SM | Family |
|---|---|---|
| B200 | sm_100 | 10.x (datacenter Blackwell) |
| B300 | sm_103 | 10.x (datacenter Blackwell) |
| **GB10 (DGX Spark)** | **sm_121** | **12.x (edge Blackwell)** |
| RTX 5090 | sm_120 | 12.x (consumer Blackwell) |

GB10 lacks `tcgen05` instruction and TMEM hardware from SM 10.x, but this does NOT block nemotron_h.

### Platform-level issues (NOT model-specific)
These affect ALL models on DGX Spark, not just NemotronH:
1. **FlashAttention lacks sm_121 support** — must use Triton attention backend as fallback ([flash-attention #1969](https://github.com/Dao-AILab/flash-attention/issues/1969))
2. **PyTorch pip wheels lack sm_121 kernels** — need NGC container or build from source ([vLLM #36821](https://github.com/vllm-project/vllm/issues/36821))
3. **Triton ptxas compatibility** — older Triton bundled ptxas doesn't recognize sm_121; workaround: symlink to system CUDA toolkit's ptxas ([vLLM #33857](https://github.com/vllm-project/vllm/issues/33857))
4. **ARM64 (aarch64)** — DGX Spark is ARM-based, adding another dimension for prebuilt package compatibility

### Community validation — CORRECTED

**Neither repo has run the exact FP8 model on GB10:**

| Repo | Nemotron-3-Super-120B tested? | Details |
|---|---|---|
| [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | **NVFP4 only** — NOT FP8 | Has recipe for `Nemotron-3-Super-120B-A12B-NVFP4`. No FP8 recipe exists. |
| [vllm-dgx-spark](https://github.com/mark-ramsey-ri/vllm-dgx-spark) | **No** — zero Nemotron support | No Nemotron models at all. Tested Llama, Qwen, Mistral, etc. |

### Broader community search (updated 2026-03-25)

**One person has attempted FP8 on dual GB10:**
- User **bugsareyummy** on [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvidia-nemotron-3-super-120b-a12b-nvfp4/363175?page=4) (comment #63)
- Used eugr/spark-vllm-docker cluster setup across 2 DGX Spark nodes
- Config: `--dtype auto --kv-cache-dtype fp8 -tp 2 --gpu-memory-utilization 0.7 --max-model-len 128000`
- Result: ~23 tokens/sec, but noted "config probably not optimal" and "strange output"
- **Partially working but not stable/reliable**

**Another user tried FP8 on dual GB10 with TensorRT-LLM:**
- [NVIDIA Forums](https://forums.developer.nvidia.com/t/multi-node-inference-crash-on-blackwell-gb10-memory-allocation-0x51-nccl-timeouts-tested-on-qwen-122b-nemotron-120b/363989)
- **Crashed** — NCCL timeouts and memory allocation failures

**What reliably works on GB10 today:**

| Variant | Framework | Status | Speed |
|---|---|---|---|
| **NVFP4** | vLLM (spark-vllm-docker) | Working | ~16-17 t/s |
| **GGUF Q4_K_M** | llama.cpp | Working | ~14-18 t/s |
| **GGUF Q4_K_M** | Ollama | Working | ~17 t/s |
| **FP8** (dual node) | vLLM | Partial — unstable | ~23 t/s |
| **FP8** (single node) | Any | **Not possible** — needs ~160GB, GB10 has 128GB | N/A |
| **BF16** | Any | **Not possible** — needs ~264GB | N/A |

### CRITICAL MEMORY CORRECTION

Earlier estimate of "128 GB on disk = 128 GB in memory" was **wrong**. The official vLLM cookbook states FP8 requires **~160 GB VRAM** (model weights + runtime overhead). This means:
- **Single GB10 (128 GB): NOT ENOUGH** — cannot run FP8
- **Dual GB10 (256 GB): FITS** — ~160 GB model + ~96 GB headroom for KV cache
- The one community attempt at dual-node FP8 partially worked but had issues

### Verdict: REVISED
- **Dual GB10 + FP8: Feasible but early-stage.** One person has done it with partial success (~23 t/s, some instability)
- **Single GB10 + FP8: Not possible** — insufficient memory
- **Safest path on GB10: NVFP4** — proven, stable, multiple community successes
- **FP8 on dual GB10: You would NOT be the first, but you'd be among the first few**
- Key references: vLLM issues [#31128](https://github.com/vllm-project/vllm/issues/31128), [#28589](https://github.com/vllm-project/vllm/issues/28589)

---

## Q2 — Networking & Access

**Q (Claude):**
4. Will the two GB10 units sit on the same local network? Dedicated switch or home router?
5. Need remote SSH access, or working on them directly?
6. Any existing infra to integrate with (NAS, K8s, monitoring)?

**A (User):**
- Both GB10s on desk, connected with **NVIDIA ConnectX cable** (NVLink bridge)
- **Remote SSH access needed** — user will handle SSH/network config themselves
- Office/lab network, GB10s have direct internet access
- No existing infra to integrate — standalone setup

---

## Q3 — Software Stack Preferences

**Q (Claude):**
7. Serving framework preference — vLLM, TensorRT-LLM, SGLang, or open to recommendation?
8. Container-based (Docker) or bare-metal installs?
9. Quantization — the model is already FP8. Want to also test BF16 (would need ~240GB, tight on 256GB)?

**A (User):**
- Serving framework: **vLLM** (agreed earlier)
- Docker vs bare-metal: answered below in Q5

---

## Q4 — GB10 Feasibility (NEW)

**Q (Claude):**
10. Are you aware that NVIDIA's official support matrix for this model lists H100/B200/B300 but **not** the GB10 (DGX Spark)? The memory fits (128GB into 256GB), but B10 GPU support in vLLM/TRT-LLM is unconfirmed. Are you prepared to be an early adopter / troubleshoot compatibility?
11. Have you checked if DGX Spark ships with a pre-configured NIM container catalog, or are you planning to set up serving from scratch?
12. Do you want to start with a smaller model (e.g., Nemotron-Mini) for validation, then scale up to the 120B?

**A (User):** _(pending)_

---

## Q5 — SM Version Explainer

**Q (User):** What is SM version? What does SM stand for?

**A (Claude):**
- **SM = Streaming Multiprocessor** — the fundamental compute unit inside an NVIDIA GPU
- Each SM contains CUDA cores, Tensor Cores, warp schedulers, shared memory/L1 cache
- **SM count** = how many compute units (more = faster): L4=58, GB10=84, B200=160
- **SM version** = architecture/instruction set (like x86 vs ARM): sm_80=Ampere, sm_90=Hopper, sm_100=Blackwell datacenter, sm_121=Blackwell edge (GB10)
- GB10 (sm_121) and B200 (sm_100) are both "Blackwell" but different SM families — software compiled for one won't automatically run on the other

---

## Q6 — Why Docker on DGX Spark?

**Q (User):** What are spark-vllm-docker and vllm-dgx-spark? Why run LLM in Docker?

**A (Claude):**

### The two repos

| | eugr/spark-vllm-docker | mark-ramsey-ri/vllm-dgx-spark |
|---|---|---|
| Approach | Builds vLLM + FlashInfer **from source** | Pulls NVIDIA's **prebuilt NGC container** |
| Base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` | `nvcr.io/nvidia/vllm:25.11-py3` |
| sm_121 patches | Yes — reverts broken PRs, Marlin kernel fixes | None — delegates to NGC |
| Distributed | Ray or PyTorch distributed (`--no-ray`) | Ray only |
| Multi-node | Autodiscovery + image distribution via SSH | SSH-based manual worker start |
| Models tested | Nemotron-3-Super-120B-NVFP4, Qwen3.5-397B, 13+ models | Llama-3.1-70B, Qwen2.5-72B, Mixtral-8x7B, 13 presets |
| Complexity | High (build pipeline, patches) | Low (pull-and-run) |

### Why Docker is practically required on DGX Spark

The reason is **not preference — it's necessity**:

1. **sm_121 + ARM64 = no prebuilt pip wheels work.** Standard `pip install vllm` gives you x86_64 binaries compiled for sm_80-sm_100. They won't run on GB10. You must either:
   - Build from source (painful, 6-stage Dockerfile in eugr's repo)
   - Use NVIDIA's NGC container (which has it pre-compiled for sm_121 + aarch64)
2. **Dependency hell.** vLLM on GB10 needs patched versions of FlashInfer, specific NCCL for RDMA, correct Triton with ptxas workarounds, CUDA 13.2 toolkit. Docker bundles all of this.
3. **Multi-node TP=2.** Both DGX Spark units need identical environments. Docker guarantees this — `docker save | docker load` across nodes.
4. **Driver compatibility.** eugr's repo warns: NVIDIA driver 580.x is required; driver 590.x causes CUDAGraph capture deadlock on GB10 unified memory.

**Bottom line:** Docker isn't about preference here — it's the practical way to get a working vLLM stack on sm_121 + ARM64 without spending days debugging build issues.

---

## Q7 — Hardware Correction & Repo Recommendation

**Q (User):** My device is Dell Pro Max GB10, not DGX Spark. Which repo should I use? Has either repo successfully run the exact FP8 model?

**A (Claude):**

### Dell Pro Max GB10 vs DGX Spark

| | Dell Pro Max GB10 | NVIDIA DGX Spark |
|---|---|---|
| Chip | GB10 (identical) | GB10 (identical) |
| Memory | 128GB unified (identical) | 128GB unified (identical) |
| NVLink pairing | Yes (identical) | Yes (identical) |
| **OS** | **Ubuntu Linux** | **DGX OS** (Ubuntu-based + NVIDIA stack) |
| **Software** | **Standard CUDA, no NGC pre-installed** | **NVIDIA AI Enterprise, NGC, NeMo pre-installed** |
| Support | Dell ProSupport | NVIDIA direct |

Hardware is identical — same chip, same memory, same NVLink. The only difference is software out-of-box.

### Which repo has run our exact model?

**Neither has run `Nemotron-3-Super-120B-A12B-FP8` on GB10.**

- **spark-vllm-docker** ran the **NVFP4** variant (different quantization) — close but not the same
- **vllm-dgx-spark** has zero Nemotron support

### Recommendation: Start with **eugr/spark-vllm-docker**

Reasons:
1. Already proved the Nemotron-3-Super-120B architecture works on GB10 (just NVFP4 not FP8)
2. Builds vLLM from source with sm_121 patches — handles all the hard platform issues
3. Has a recipe/mod system — we can create an FP8 recipe based on their NVFP4 one
4. More active maintenance and broader model support
5. Dell Pro Max runs same hardware — the Docker image should work identically

We would be among the first few to run FP8 on dual GB10 — one person has partially done it (see feasibility section above).

---

## Q8 — Docker Performance & FP8 vs NVFP4

**Q (User):** Will Docker affect performance vs bare-metal? What does FP8 offer compared to NVFP4?

**A (Claude):**

### Docker vs Bare-Metal: Zero GPU performance difference
- Docker uses Linux namespaces, not virtualization — same kernel, same driver
- NVIDIA Container Runtime passes GPU directly to container — CUDA calls hit the same driver
- `--network host` bypasses Docker networking for multi-node TP
- Only overhead: ~1-2% on disk I/O during model load (overlay2 filesystem)
- **The GPU literally doesn't know it's in a container**

### FP8 vs NVFP4 Comparison

| | FP8 (user's target) | NVFP4 (proven on GB10) |
|---|---|---|
| Bits per weight | 8 | 4 |
| Model size in memory | ~160 GB | ~64-80 GB |
| Single GB10 (128 GB) | **No — doesn't fit** | **Yes — fits** |
| Dual GB10 (256 GB) | Yes — fits with headroom | Yes — lots of headroom |
| Quality (MMLU-Pro) | 83.63 (near BF16) | Lower — more quantization loss |
| Inference speed | Fast | Faster — half the data movement |
| vLLM sm_121 support | Widely supported | Needs Marlin backend workaround |
| GB10 community status | Partial — 1 attempt, unstable | **Proven and stable** |

### Decision point for user
- FP8 on dual GB10: higher quality, but early-adopter territory
- NVFP4 on single or dual GB10: proven, stable, faster, but lower quality
- Could start with NVFP4 to validate setup, then try FP8

---

## Q9 — NVFP4 vs FP8 vs BF16 Accuracy Comparison

**Q (User):** How bad is NVFP4 accuracy compared to FP8 and BF16?

**A (Claude):** Surprisingly close. Official NVIDIA benchmarks from the model cards:

### Full Benchmark Table (source: NVIDIA HuggingFace model cards)

| Benchmark | BF16 | FP8 | NVFP4 | FP8 Δ | NVFP4 Δ |
|---|---|---|---|---|---|
| **General** | | | | | |
| MMLU-Pro | 83.73 | 83.63 | 83.33 | -0.10 | **-0.40** |
| **Reasoning** | | | | | |
| HMMT Feb25 (tools) | 94.73 | 94.38 | 95.36 | -0.35 | **+0.63** |
| GPQA (no tools) | 79.23 | 79.36 | 79.42 | +0.13 | **+0.19** |
| LiveCodeBench v6 | 78.69 | 78.44 | 78.44 | -0.25 | **-0.25** |
| LiveCodeBench v5 | 81.19 | 80.99 | 80.56 | -0.20 | **-0.63** |
| SciCode (subtask) | 42.05 | 41.38 | 40.83 | -0.67 | **-1.22** |
| HLE (no tools) | 18.26 | 17.42 | 17.42 | -0.84 | **-0.84** |
| **Agentic** | | | | | |
| Terminal Bench (hard) | 25.78 | 26.04 | 24.48 | +0.26 | **-1.30** |
| TauBench V2 (avg) | 61.15 | 61.07 | 60.46 | -0.08 | **-0.69** |
| **Chat & Instruction** | | | | | |
| IFBench (prompt) | 72.58 | 72.32 | 73.30 | -0.26 | **+0.72** |
| Scale AI Multi-Challenge | 55.23 | 54.35 | 52.80 | -0.88 | **-2.43** |
| Arena-Hard-V2 | 73.88 | 76.06 | 76.00 | +2.18 | **+2.12** |
| **Long Context** | | | | | |
| RULER-500 @ 128k | 96.79 | 96.85 | 95.99 | +0.06 | **-0.80** |
| RULER-500 @ 256k | 96.60 | 96.33 | 96.52 | -0.27 | **-0.08** |
| RULER-500 @ 512k | 96.09 | 95.66 | 96.23 | -0.43 | **+0.14** |
| **Multilingual** | | | | | |
| MMLU-ProX (avg) | 79.35 | 79.21 | 79.37 | -0.14 | **+0.02** |

### Key Takeaways

1. **NVFP4 is shockingly good** — worst case is Scale AI Multi-Challenge at -2.43 points. Most benchmarks within 1 point of BF16.
2. **NVFP4 actually BEATS BF16** on several benchmarks: HMMT (+0.63), GPQA (+0.19), IFBench (+0.72), Arena-Hard-V2 (+2.12)
3. **Why so good?** This model was **pre-trained in NVFP4 from scratch** — it's not post-hoc quantization. The model learned to work with 4-bit precision during training.
4. **FP8 is near-lossless** — max delta is -0.88, average ~0.3 points from BF16
5. **Average NVFP4 degradation across all benchmarks: ~0.5 points** — negligible for most use cases

### Implication for our setup
NVFP4 quality is NOT a meaningful sacrifice. The model was designed for NVFP4. Starting with NVFP4 gives us:
- Proven GB10 stability
- Fits on single GB10 (no multi-node complexity needed initially)
- Nearly identical quality to FP8/BF16
- Can always try FP8 later on dual node

---

## Q10 — NVFP4 Backend "Hacking" Concern

**Q (User):** You mentioned NVFP4 needs backend workarounds on GB10. Will this have adverse impacts?

**A (Claude):** It's NOT hacking — it's just selecting alternative backends via official vLLM flags.

### What the spark-vllm-docker recipe actually does

The actual recipe (from `recipes/nemotron-3-super-nvfp4.yaml`):
```
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --moe-backend cutlass \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --reasoning-parser nemotron_v3 \
  --mamba_ssm_cache_dtype float32 \
  --distributed-executor-backend ray
```

### What the "workaround" really is
Two flags selecting alternative backends:
1. `--attention-backend TRITON_ATTN` — because FlashAttention lacks sm_121 support. Triton attention is a built-in vLLM backend, same math.
2. `--moe-backend cutlass` — because FlashInfer MoE kernels don't support NVFP4 on sm_121. CUTLASS is NVIDIA's own kernel library, equally correct.

### Adverse impacts: None
- **Correctness:** Same math, different implementation → identical results
- **Accuracy:** No difference in model output
- **Speed:** Possibly slightly slower than FlashAttention (which is more hand-optimized), but CUTLASS/Triton are mature backends
- **Stability:** Actually MORE stable on GB10 — chosen because they work reliably on sm_121
- **No source patches:** The `nemotron-super` mod is even commented out in the current recipe

**Analogy:** Like choosing Firefox vs Chrome to open the same website. Different engine, same result.

---

## Q11 — NVFP4 on Dual GB10? & Token Speed Clarification

**Q (User):** Has anyone run NVFP4 on dual GB10? Why was FP8 faster?

**A (Claude):**

### Nobody runs NVFP4 on dual GB10 — because there's no reason to.

NVFP4 model weighs **~69.5 GB** and fits comfortably on a **single GB10 (128 GB)**. Adding a second node would only add inter-node communication overhead with no benefit.

### Corrected throughput comparison (apples-to-apples)

| Setup | Speed | GPUs | Source |
|---|---|---|---|
| NVFP4 single GB10, concurrency=1 | **~16-17 t/s** | 1 GPU | Multiple community reports |
| NVFP4 single GB10, concurrency=10 | **~62-63 t/s total** | 1 GPU | vnicolici, GitHub #125 |
| FP8 dual GB10, concurrency=1 | **~23 t/s** | 2 GPUs | bugsareyummy, NVIDIA forums |

**The FP8 "faster" number was misleading** — it used 2 GPUs vs 1 GPU for NVFP4. Per-GPU, NVFP4 is likely faster.

### Detailed single-node NVFP4 benchmarks

| Source | Backend | Concurrency | Speed |
|---|---|---|---|
| NVIDIA Forum (adi-sonusflow) | Marlin | 1 | 16.6 t/s |
| GitHub #93 (Sggin1) | Marlin | 1 | ~16-17 t/s |
| GitHub #125 (vnicolici) | CUTLASS | 1 | 15.9-16.0 t/s |
| GitHub #125 (vnicolici) | CUTLASS | 10 | 62-63 t/s total |
| Forum (giraudremi92) | llama-benchy | 1 | ~14.94 t/s |

### Implication for user's dual GB10 setup
With NVFP4 fitting on a single GB10:
- **Option A:** Run model on 1 GB10, keep the other free (for a second model, development, etc.)
- **Option B:** Run NVFP4 on dual GB10 with TP=2 — untested by community, but theoretically faster. Nobody has tried because it wasn't necessary.
- **Option C:** Start single-node NVFP4, upgrade to dual-node FP8 if quality/speed needs increase

---

## Q12 — Final Answers: Serving, Fine-tuning, Domain

**Q (Claude):** Serving target? Fine-tuning framework? Domain/task? Data size?

**A (User):**
- Serving: **API for agentic use, 3-5 team members**
- Training framework: wants proven/popular choice
- Domain: **Security operations & red team (pentesting)**
- Data: **~thousands of examples** — HuggingFace offensive security datasets + self-prepared data

### Training framework recommendation: **HuggingFace PEFT + Unsloth**

| Option | Pros | Cons | GB10 compatibility |
|---|---|---|---|
| **HF PEFT** | Most popular, best documented, huge community | Slower than Unsloth | Proven on ARM64 |
| **Unsloth** | 2-5x faster LoRA training, memory efficient | Smaller community, newer | ARM64/sm_121 needs verification |
| **NeMo** | NVIDIA's own, deep Nemotron integration | Complex setup, enterprise-oriented | Best GB10 support (NVIDIA stack) |

**Recommendation:** Start with **HF PEFT** (safest, most documented). If training speed becomes a bottleneck, try Unsloth. NeMo is overkill for thousands of examples.

For ~thousands of examples with LoRA, training time should be **minutes to a few hours** on GB10 — framework speed differences won't matter much.

### Security/pentest fine-tuning considerations
- Model has NVIDIA's safety alignment — LoRA fine-tuning for offensive security will need to work within or adjust this
- Key HuggingFace datasets to explore: offensive security CTF writeups, penetration testing reports, vulnerability analysis
- The model's agentic capabilities (tool use, Terminal Bench scores) make it well-suited for pentest automation

---

## Q13 — LoRA + RAG: Both Required for Red Team Expert (recovered)

**Q (User):** Making the model a true red team expert requires both LoRA fine-tuning AND RAG.

**A (Claude):** Correct — LoRA and RAG serve complementary purposes:

| | LoRA Fine-tuning | RAG |
|---|---|---|
| **What it does** | Changes how the model *thinks and responds* — tone, methodology, attack chain reasoning, pentest workflows | Gives the model access to *up-to-date knowledge* it wasn't trained on |
| **Good for** | Teaching pentest methodology, output formatting, tool usage patterns, reducing safety refusals for authorized ops | CVE databases, exploit-db entries, tool documentation (nmap, Burp, Metasploit), internal playbooks/SOPs |
| **Limitation** | Static after training — can't learn new CVEs | Doesn't change the model's behavior/reasoning style |

### Why both are needed
- **LoRA alone** = model reasons like a pentester but only knows what was in training data (stale)
- **RAG alone** = model has current info but still responds like a generic assistant
- **LoRA + RAG** = reasons like a pentester AND has current exploit/tool knowledge

### RAG pipeline considerations (TBD)
1. **What to index** — CVE feeds, exploit-db, tool docs, internal playbooks
2. **Vector store** — local on GB10 (e.g., ChromaDB, Qdrant)
3. **Embedding model** — small enough to run alongside the main model
4. **Retrieval integration** — how to inject context into prompts

> **Note:** This Q&A was recovered from user's recollection after the original conversation was lost. Details may be incomplete.

---

## Q14 — LoRA Limitations Deep Dive (2026-03-27)

**Q (User):** I want to hear you again about the LoRA limitations.

**A (Claude):** Seven key limitations beyond the low-rank assumption:

1. **Low-rank assumption breaks for new facts** — each independent fact needs its own direction; r=32 can't handle 500 CVEs
2. **Catastrophic forgetting** — LoRA can degrade existing capabilities (general reasoning, coding, fluency)
3. **Data quality amplification** — LoRA is extremely sensitive; one bad example disproportionately hurts
4. **Rank is a hard ceiling** — if true delta-W needs rank 200 and r=32, no amount of epochs fixes it
5. **Layer coverage tradeoffs** — targeting everything risks overfitting on small datasets
6. **No unlearning / selective editing** — can't surgically remove one bad pattern without full retrain
7. **Evaluation is hard** — subtle changes, model may look fine on 95% but fail on critical edge cases

Every limitation points the same direction: LoRA for behavior, RAG for facts.

---

## Q15 — Using Real Dataset for LoRA Training (2026-03-27)

**Q (User):** This is an example dataset: `pAILabs/infosec-security-qa`. Tell me how can I use it for fine tuning? Better to have a real example.

**A (Claude):** 11.2K Q&A pairs covering cybersecurity (phishing, ransomware, threat intel, incident response). Already has both question and answer — much better than prompt-only datasets.

Created `scripts/prepare_infosec_dataset.py` to download from HuggingFace, convert `{question, answer}` to `{instruction, input, output}` format, and split train/val. Three-command pipeline: prepare → train → eval.

Also referenced `yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K` (596K prompts, but no answers — would need synthetic response generation).

---

## Q16 — LoRA Math Pipeline & Embedding (2026-03-27)

**Q (User):** What happened under the hood mathematically? Do you need to embed the tuning data as well?

**A (Claude):** Full seven-stage walkthrough: Tokenization (BPE lookup, no math) → Embedding (frozen matrix inside model, not trained by LoRA) → Forward pass (x×W + x×A×B×α/r) → Loss (cross-entropy, next token prediction) → Backward pass (gradients for A and B only, chain rule through frozen layers) → Optimizer step (Adam, 18MB state vs 24GB for full fine-tune) → Repeat (~8,400 updates for 3 epochs on 11.2K samples).

**Key answer: No, you do NOT need to embed the training data separately.** The model's own frozen embedding layer handles it during the forward pass. No external vector DB, no sentence-transformers. Tokenizer converts text to IDs, embedding layer converts IDs to vectors, LoRA modifies what happens after embedding.

---

## Q17 — Evaluating LoRA Results (2026-03-27)

**Q (User):** How to evaluate the tuning result?

**A (Claude):** Three levels:

1. **Training loss** (automatic) — watch for healthy drop, overfitting (train↓ val↑), underfitting (plateau)
2. **Validation loss + perplexity** (added `--val-dataset` to `finetune_lora.py`) — pick checkpoint with lowest eval_loss, not lowest train_loss. PPL = e^loss, lower = better.
3. **Side-by-side comparison** (created `scripts/eval_lora.py`) — runs same questions through base model and LoRA model, computes perplexity on held-out set, saves results to JSON for human review.

Also: test non-security questions to check for catastrophic forgetting. Updated `finetune_lora.py` with `--val-dataset` flag and `eval_strategy="epoch"`.

---

## Q18 — Training Dataset Formats (2026-03-27)

**Q (User):** For LoRA dataset, if it is intended to improve function calling skills the dataset would be in JSON format; if it is intended to improve knowledge or switch personality, the dataset shall be in Q&A format. Am I generally right?

**A (Claude):** Yes. Dataset format matches the target behavior:

- **Q&A instruction** → behavior, style, reasoning (our infosec dataset)
- **Tool-call messages** → function calling reliability (need to build)
- **Preference pairs** → alignment/DPO/RLHF (maybe later)
- **Raw text** → domain knowledge injection via continued pre-training
- **Classification labels** → threat categorization
- **Token/span labels** → NER / IOC extraction
- **Multi-turn conversation** → dialogue flow
- **Embedding pairs** → RAG retrieval quality tuning

For our project: mix ~70% Q&A + ~30% tool-call conversations.

---

## Q19 — Raw Text Training vs RAG (2026-03-27)

**Q (User):** Raw text tuning is interesting — so I don't need any complicated data preparations and just feed texts to the LLM? How can it tune and how to evaluate? In addition, it looks like... just a RAG.

**A (Claude):** Correct on simplicity — just feed plain text, model trains via next-token prediction on every position. Called "continued pre-training" or "domain-adaptive pre-training."

Evaluation is hard: measure perplexity drop on domain text, check downstream task improvement (raw text + LoRA vs LoRA alone), probe questions.

**"It looks like RAG" — right, but key differences:**
- RAG is **grounded** (model reads actual source), raw text training is **fuzzy** (model may half-learn facts and hallucinate)
- RAG updates instantly (change a DB row), training requires hours
- RAG eats context window, training adds zero latency
- RAG capacity is unlimited (millions of docs), training capacity is limited by params

Raw text training is most useful when the base model genuinely lacks domain exposure (proprietary protocols, classified content). For public security knowledge, the 120B model likely already knows most of it — RAG is the better path for facts.

---
