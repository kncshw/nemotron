# LoRA + RAG High-Level Design

> **Date:** 2026-03-26 (updated 2026-03-27)
> **Context:** Nemotron-3-Super-120B on 2x GB10, security/pentest assistant
> **Status:** Design phase — LoRA mock validated on L4, data prep + eval scripts written, RAG not yet started

---

## Part 1: LLM Tool Calling — How It Works

### Q: When the LLM decides to emit tool calls, does it just return a JSON request and the client executes locally?

Yes. The LLM's only job is to produce structured JSON saying "please call function X with these arguments." It never executes anything.

**The workflow:**

```
You (client)                         LLM (vLLM server)
─────────────────────────────────────────────────────
1. Send messages + tool schemas ──►
                                     2. LLM decides: call tool or text?
                              ◄──── 3. Returns tool_calls[] OR text
4. If tool_calls:
   - Execute nmap_scan() locally
   - Execute search_cve() locally
5. Send tool results back      ──►
                                     6. LLM produces final answer
                              ◄──── 7. Returns text response
```

- The tool schemas (JSON descriptions of available functions) are sent alongside chat messages in the API request.
- With `tool_choice="auto"`, the LLM can either emit structured tool calls or respond with plain text.
- The LLM has no access to the OS, network, or any sandbox. It is pure text-in, text-out.
- The execution environment is entirely the client's decision: local OS (`subprocess.run`), sandboxed Docker container, remote API, or mock/stub.
- Think of it like a restaurant: the LLM is the customer placing an order (structured JSON). Your code is the kitchen that actually cooks the food.

### Q: Could the tool call test failure be caused by there being no actual nmap_scan available?

No. The test returned WARN because the small Nano-4B model chose to respond with plain text instead of emitting structured tool call JSON. Smaller models are often unreliable at producing the structured output format. The 120B Super model is expected to handle this correctly.

---

## Part 2: LoRA Fine-Tuning — Under the Surface

### Q: I know LoRA uses two low-rank matrices multiplied to approximate the tuning target, but how does it actually work?

**What a neural network layer is:**

A layer like `q_proj` is a matrix multiplication: `output = input x W`, where W is e.g. [3136, 5120] = ~16M parameters learned during pre-training. Full fine-tuning would update all 16M numbers — too much memory.

**What LoRA does:**

Instead of modifying W directly, LoRA learns a **change** (delta-W), represented cheaply as two small matrices:

```
W_original:  [3136 x 5120]   <- frozen, never changes

A:           [3136 x 32]      <- trainable  (rank r=32)
B:           [32 x 5120]      <- trainable  (rank r=32)

delta-W = A x B: [3136 x 5120]    <- same shape as W, built from only 32 columns/rows
```

During forward pass:

```
output = input x W_original  +  input x A x B
         -------------------    -----------------
         frozen (pre-trained)   learned adjustment
```

**Parameter savings:**

For `q_proj` [3136 x 5120]:
- Full fine-tuning: 16,056,320 parameters
- LoRA (r=32): 264,192 parameters (~1.6% of full)

Our mock training confirmed: `trainable params: 3,178,496 || all params: 3,976,735,328 || trainable%: 0.0799`

**What happens during training:**

```
Epoch loop:
  Forward pass:
    input --> W_original x input  --+
                                    +-- add --> output --> loss
    input --> A x (B x input)  ----+

  Backward pass:
    loss --> compute gradients for A and B only
             (W_original is frozen, no gradients for it)

  Optimizer step:
    Update A and B using their gradients
    (Adam only needs optimizer states for 3.1M params, not 4B)
```

**After training — what you get:**

The saved adapter (13MB) contains only A and B matrices. Two ways to use it:
- **Merge:** compute `W_new = W_original + A x B`, save a full model (permanent)
- **Hot-swap (vLLM):** keep W_original in memory, load the tiny adapter on the side, add A x B at runtime. Multiple LoRA adapters can share one base model.

**Rank r controls capacity:**

```
r=4:    Few trainable params, fast, limited expressiveness
r=16:   Good balance for simple domain adaptation
r=32:   Good for security domain shift (our setting)
r=64:   More capacity, more memory, diminishing returns
r=256:  Approaching full fine-tuning behavior
```

`lora_alpha=64` controls scaling: delta-W contribution is scaled by `alpha/r = 64/32 = 2x`.

---

### Q: A single [1,2]^T . [1,2] doesn't create new information — it's rank 1. So how could this tune the LLM?

This is exactly right for a single outer product. Three things make LoRA work despite this:

**1. It's not one vector pair — it's r=32 pairs summed:**

```
delta-W = A x B = a1.b1^T + a2.b2^T + a3.b3^T + ... + a32.b32^T
                  ------    ------    ------            --------
                  rank-1    rank-1    rank-1            rank-1

Together: rank-32 matrix, 32 independent directions
```

**2. Gradient descent FINDS the right directions:**

A and B don't stay random. Backpropagation rotates them to align with directions that reduce loss:

```
Epoch 1:  A,B point in random directions -> big loss
Epoch 2:  Gradients rotate A,B toward useful directions -> loss drops
Epoch 3:  A,B now span the subspace capturing "security expert behavior"
```

**3. The empirical discovery — fine-tuning is naturally low-rank:**

The LoRA paper (Hu et al., 2021) measured what actually changes during full fine-tuning:

```
delta-W = W_new - W_original

Finding: rank(delta-W) is extremely low relative to matrix size
```

The actual change matrix is almost entirely captured by its top ~10-64 singular values. LoRA isn't forcing a bad approximation — it's matching the structure that fine-tuning naturally has.

**Analogy:** The pre-trained model's knowledge is a position in a 16M-dimensional space. Fine-tuning for "security expert" means moving to a nearby point. The shift can be described as "move 3 steps in the 'security terminology' direction, 2 steps in the 'tool command syntax' direction..." — ~32 directions x magnitudes = full behavioral shift.

---

### Q: For new knowledge (new CVEs, specific target product info), how can we assume it's low-rank?

**We can't. This is LoRA's real weakness.**

The low-rank assumption holds for **behavioral/style adaptation** but breaks down for **new factual knowledge**:

```
Behavioral shift (low-rank, works):
  "Stop being a general assistant, start being a security expert"
  The model ALREADY knows about nmap, SQLi, privilege escalation
  Just amplifying existing directions in weight space

New knowledge (NOT necessarily low-rank, problematic):
  "CVE-2026-XXXX affects ProductX version 3.2.1 via buffer overflow in libfoo"
  This fact did NOT exist in pre-training
  Requires encoding new associations between unrelated concepts
  Each distinct fact may need its own direction
```

**Why new facts are hard for low-rank updates:**

Factual knowledge in MLPs is stored as key-value associations:

```
MLP learns:
  "if input looks like [pattern about Apache 2.4.49]" -> "retrieve [CVE-2021-41773]"
  "if input looks like [pattern about Log4j]"         -> "retrieve [CVE-2021-44228]"
```

Adding 500 new CVEs = 500+ independent associations. If r=32, you have 32 directions — not enough for 500 independent facts.

**Research confirms this:**

```
Task type                    LoRA vs Full fine-tuning
---------------------------------------------------------
Style/format change          ~Equal performance
Classification adaptation    ~Equal performance
New factual knowledge        LoRA significantly worse
Rare/specific facts          LoRA often fails to retain them
```

**Cranking up rank is a partial fix:**

r=256 or r=512 gives more capacity but approaches full fine-tuning cost, and still no guarantee for hundreds of distinct facts.

---

### Q: What are the full limitations of LoRA beyond the low-rank assumption?

Seven key limitations to keep in mind:

**1. Low-rank assumption breaks for new facts** (covered above)

Each new independent fact (CVE, product quirk) potentially needs its own direction in weight space. r=32 gives 32 directions — not enough for hundreds of distinct facts.

**2. Catastrophic forgetting**

LoRA can degrade existing capabilities. The base model's general reasoning, coding ability, or language fluency can subtly erode. Smaller datasets and more epochs make this worse — you're pulling the model toward your domain at the cost of general knowledge.

**3. Data quality amplification**

LoRA is extremely sensitive to training data quality because of the small trainable parameter count. A full fine-tune can absorb some noise — LoRA can't. One badly formatted tool-call example can disproportionately hurt structured output reliability. Every sample counts.

**4. Rank is a hard ceiling**

If the true delta-W needs rank 200 to be well-approximated and r=32, no amount of training epochs will fix it. You'll plateau early. But cranking r to 256+ defeats the purpose — approaching full fine-tuning costs.

**5. Layer coverage tradeoffs**

Targeting only attention layers misses factual knowledge in MLPs. Targeting everything (attention + Mamba + MLP) increases adapter size and training time, risking overfitting on small datasets.

**6. No unlearning / no selective editing**

Once a LoRA adapter bakes in a pattern, you can't surgically remove one fact or fix one bad behavior without retraining the whole adapter. Compare to RAG where you just update a database row.

**7. Evaluation is hard**

Changes are subtle. The model might look fine on 95% of prompts but fail on the exact edge cases you cared about. Without rigorous eval benchmarks built before training, you won't know what regressed.

**Bottom line:** Every limitation points the same direction — don't ask LoRA to memorize facts. Use it for behavioral shifts. Let RAG handle mutable, high-cardinality, fact-heavy knowledge.

---

## Part 3: Hybrid Architecture Layers — What to Tune

### Q: Can Mamba layers be LoRA-tuned?

Yes. Nemotron-3-Nano-4B (and Super-120B) is a hybrid Mamba+Attention model. The layer types:

**Attention layers:**
| Parameter | Shape | Purpose |
|---|---|---|
| `q_proj` | [5120, 3136] | Query projection |
| `k_proj` | [1024, 3136] | Key projection |
| `v_proj` | [1024, 3136] | Value projection |
| `o_proj` | [3136, 5120] | Output projection |

**Mamba layers:**
| Parameter | Shape | Purpose |
|---|---|---|
| `in_proj` | [17504, 3136] | Input projection (x, z, B, C, dt) |
| `out_proj` | [3136, 7680] | Output projection |
| `conv1d` | [9728, 1, 4] | Depthwise conv (not suitable for LoRA) |
| `dt_bias`, `A_log`, `D` | small 1D | SSM scalars (not suitable for LoRA) |

**MLP layers:**
| Parameter | Shape | Purpose |
|---|---|---|
| `up_proj` | [12544, 3136] | MLP up projection |
| `down_proj` | [3136, 12544] | MLP down projection |

`in_proj` and `out_proj` are large linear projections — perfectly suitable for LoRA, just like attention layers.

### Q: Do we need to fine-tune MLP layers?

Depends on the goal. Each layer type has a distinct role:

```
Attention (q/k/v/o_proj)    "What should I look at?"     Routing, relationships
Mamba (in_proj/out_proj)    "What sequential patterns?"  State-space processing
MLP (up_proj/down_proj)     "What does this mean?"       Factual knowledge storage
```

**Research shows factual knowledge lives primarily in MLP layers.** So:

- **Style/behavior shift only:** Attention + Mamba is enough
- **Knowledge injection:** Need MLP layers too

Tradeoff:

```
Target set                  Trainable params    Adapter size    What it learns
---------------------------------------------------------------------------------
Attention only              3.1M (0.08%)        ~13MB           Behavior/style
+ Mamba                     ~8M  (0.2%)         ~30MB           + Sequential patterns
+ Mamba + MLP               ~16M (0.4%)         ~60MB           + New knowledge
```

---

## Part 4: Architecture Decision

### Q: Given 2x GB10 (can't do full training), what's the strategy?

**Decision: LoRA for behavior, RAG for facts.**

```
LoRA (baked into weights)              RAG (retrieved at runtime)
------------------------------------   ------------------------------------
Tool-calling reliability               CVE database
Pentester reasoning patterns           Target product specs
Security-aware output style            Network topology / scan results
Report formatting                      Cheatsheets / playbooks
When to use which tool                 Historical vulnerability data
```

LoRA handles the low-rank stuff (behavior). RAG handles the high-rank stuff (facts). Each playing to its strength.

**LoRA target modules for security behavior tuning:**

```python
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention: how to reason
    "in_proj", "out_proj",                     # Mamba: sequential patterns
    "up_proj", "down_proj",                    # MLP: reinforce security awareness
]
```

**RAG provides facts at runtime** — same principle as the tool-calling architecture: the LLM decides what to look up, retrieval provides the actual facts. New CVE published? Add it to the database. No retraining needed.

---

## Part 5: L4 Mock Validation Status

Both LoRA blockers on L4 have been resolved:

1. **mamba-ssm CUDA mismatch:** Fixed with `pip install mamba-ssm causal-conv1d --no-build-isolation`
2. **Gradient chain broken through Mamba layers:** Fixed by adding `model.enable_input_require_grads()` in `finetune_lora.py`

LoRA mock training completed: 3 epochs, 5 samples, 87s, 13MB adapter saved to `scripts/lora-output/`.

---

## Part 6: Training Dataset Formats

### Q: For LoRA datasets, if it's intended to improve function calling the dataset would be in JSON format; if it's intended to improve knowledge or switch personality, the dataset shall be in Q&A format. Am I generally right?

Yes. The dataset format should match the behavior you want to train:

**Q&A / Instruction format → Knowledge & personality:**

```json
{
  "instruction": "You are a cybersecurity expert...",
  "input": "What is privilege escalation?",
  "output": "Privilege escalation is gaining higher-level permissions..."
}
```

The model learns how to reason, what tone to use, what depth to provide, what terminology to prefer. This is behavioral/style tuning — the sweet spot for LoRA.

**Tool-call format → Function calling:**

```json
{
  "messages": [
    {"role": "system", "content": "You have access to: nmap_scan(target, ports), search_cve(keyword)"},
    {"role": "user", "content": "Scan 192.168.1.0/24 for open web ports"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "nmap_scan", "arguments": "{\"target\": \"192.168.1.0/24\", \"ports\": \"80,443,8080\"}"}}
    ]},
    {"role": "tool", "content": "{\"hosts\": [{\"ip\": \"192.168.1.5\", \"ports\": [80, 443]}]}"},
    {"role": "assistant", "content": "Found 1 host with web services:\n- 192.168.1.5: ports 80, 443\n\nRecommend running service version detection next."}
  ]
}
```

The model learns when to call a tool vs respond directly, how to format the JSON, how to interpret results, and when to chain multiple calls.

**For our project:** Mix both — ~70% Q&A (infosec dataset) for security reasoning, ~30% tool-call conversations for structured output reliability.

### Q: Any other popular dataset formats designed for specific tuning/training tasks?

Yes — six more distinct formats beyond Q&A and tool-call:

**1. Preference pairs → RLHF / DPO (alignment tuning):**

```json
{
  "prompt": "How do I test for XSS?",
  "chosen": "To test for XSS in an authorized pentest: 1. Inject <script>alert(1)</script>...",
  "rejected": "Just inject scripts everywhere on any website you want to test..."
}
```

Teaches the model which response is better. Uses DPO or PPO training loops, not standard LoRA.

**2. Raw text / completion → Continued pre-training:**

```
Buffer overflow vulnerabilities occur when a program writes data beyond
the allocated memory buffer. In stack-based overflows, the attacker
overwrites the return address to redirect execution flow...
```

No structure at all — just raw documents. Makes the model "read" your security textbooks. Injects domain knowledge before fine-tuning.

**3. Classification / labeling → Sequence classification:**

```json
{"text": "Unusual outbound DNS traffic to known C2 domain", "label": "malware_communication"}
{"text": "Failed login from 5 different IPs in 10 seconds", "label": "brute_force"}
```

For training a classifier head. Uses `TaskType.SEQ_CLS` instead of `CAUSAL_LM`.

**4. Span extraction → NER / Token classification:**

```json
{
  "tokens":  ["CVE-2024-3094", "affects", "xz-utils", "version", "5.6.0"],
  "labels":  ["CVE_ID",        "O",       "PRODUCT",   "O",       "VERSION"]
}
```

For extracting CVE IDs, product names, IOCs from unstructured text. Uses `TaskType.TOKEN_CLS`.

**5. Multi-turn conversation → Chat fine-tuning:**

```json
{
  "messages": [
    {"role": "user", "content": "I found port 443 open on the target"},
    {"role": "assistant", "content": "Run service detection: nmap -sV -p 443 target"},
    {"role": "user", "content": "It's running Apache 2.4.49"},
    {"role": "assistant", "content": "Vulnerable to CVE-2021-41773 — path traversal..."}
  ]
}
```

Teaches conversational flow — context tracking, follow-ups, building on previous turns.

**6. Embedding / retrieval pairs → RAG embedding tuning:**

```json
{"query": "Apache path traversal exploit", "positive": "CVE-2021-41773 allows...", "negative": "Apache Kafka configuration guide..."}
```

For fine-tuning the embedding model (not the LLM), improving RAG retrieval quality.

**Summary table:**

| Format | Training Objective | Our Use Case |
|---|---|---|
| Instruction Q&A | Behavior, style, reasoning | **Yes — infosec dataset** |
| Tool-call messages | Function calling reliability | **Yes — need to build** |
| Preference pairs | Alignment (DPO/RLHF) | Maybe later — safety guardrails |
| Raw text | Domain knowledge injection | If 120B lacks security knowledge |
| Classification labels | Threat categorization | If we build a detection pipeline |
| Token/span labels | IOC extraction, NER | If we need structured entity parsing |
| Multi-turn conversation | Dialogue flow | Covered by tool-call format |
| Embedding pairs | RAG retrieval quality | When we build the RAG pipeline |

---

## Part 7: Raw Text Training (Continued Pre-Training)

### Q: Raw text training is interesting — so I don't need any complicated data preparation and just feed texts to the LLM? How can it tune and how to evaluate? In addition, it looks like... just a RAG.

**How it works — just next-token prediction:**

You literally feed plain text. No instruction/output pairs. The training objective is the same as pre-training:

```
Input text: "Buffer overflow vulnerabilities occur when a program writes data beyond the allocated buffer"

The model trains on every position:

Input:  "Buffer overflow vulnerabilities occur when a"
Target: "program"

Input:  "Buffer overflow vulnerabilities occur when a program"
Target: "writes"

Loss = how well it predicts each next token
```

No question, no answer, no instruction. The model just learns to predict security text better — absorbing terminology, concepts, relationships, and patterns. This is called **continued pre-training** or **domain-adaptive pre-training**.

**How to evaluate — the hard part:**

No "correct answer" to compare against. Three approaches:

1. **Perplexity drop on domain text:**

```
Before:  PPL = 18.3 on security corpus    ← model finds security text "surprising"
After:   PPL = 6.7  on security corpus    ← model finds it natural

Also check:
Before:  PPL = 5.2 on general text        ← baseline
After:   PPL = 5.4 on general text        ← still OK, didn't forget much
```

2. **Downstream task improvement** — after continued pre-training, do instruction tuning (LoRA Q&A) and measure if it performs better than LoRA alone:

```
Base model → LoRA on infosec Q&A → Eval score: 72%

vs.

Base model → Raw text on security docs → LoRA on infosec Q&A → Eval score: 81%
                                          ↑
                           model already "knows" the domain,
                           LoRA just steers the behavior
```

3. **Probe questions** — ask factual questions about content from the raw text. Unreliable though — which brings us to the key insight:

**"It looks like RAG" — right, but there are real differences:**

```
                    Raw Text Training              RAG
                    ──────────────────             ───────────
Where knowledge     Baked into weights             Retrieved at query time
lives               (permanent)                    (from vector DB)

Update cycle        Retrain (hours/days)           Update DB row (seconds)

Recall              Fuzzy — model may              Exact — retrieval finds
                    "forget" or hallucinate         the actual document

Capacity            Limited by rank/params          Limited by DB size
                    (can't absorb 10K docs          (can store millions)
                    reliably)

Latency             Zero — knowledge is             Adds retrieval step
                    in the forward pass             (~50-200ms)

Context window      Not consumed                    Eats context window
                    (it's in weights)               (retrieved chunks injected)

Hallucination       Can confidently state            Grounded — model sees
                    wrong "facts" it half-learned    the actual source text
```

The critical difference: **RAG is grounded, raw text training is not.** When the model absorbs a CVE description through training, it might later say "CVE-2024-3094 affects xz-utils version 5.6.1" when it was actually 5.6.0. It half-learned the fact. With RAG, the actual document is in the prompt.

**When raw text training makes sense:**

```
Good use cases:
  - Teaching the model a new LANGUAGE (security jargon, protocol specs)
  - Making the model comfortable with domain PATTERNS it rarely saw in pre-training
  - Preparing for downstream LoRA (the two-stage approach above)

Bad use cases:
  - Memorizing specific facts (CVE details, IP addresses, versions)
  - Anything that changes frequently
  - Anything where accuracy matters more than fluency
```

**Comparison of all approaches:**

```
                   Least effort                          Most effort
                   ───────────                           ──────────
                   RAG          LoRA (Q&A)    Raw text + LoRA    Full fine-tune
                   ─────────    ──────────    ─────────────────  ──────────────
Fact recall        Exact        Unreliable    Slightly better    Still unreliable
Behavior shift     None         Strong        Stronger           Strongest
Cost               Cheap        Moderate      Expensive          Too much for GB10
Update speed       Instant      Retrain       Retrain            Retrain
```

Raw text training is most useful when the base model genuinely lacks domain exposure — like training on classified network protocols or proprietary product internals that couldn't have been in pre-training data. For public security knowledge, the 120B model likely already knows most of it, making RAG the better path for facts.

---

## Part 8: Using a Real Dataset — pAILabs/infosec-security-qa

### Example dataset: pAILabs/infosec-security-qa

An 11.2K Q&A dataset from HuggingFace covering cybersecurity topics (phishing-as-a-service, ransomware, threat intelligence, incident response, digital forensics). Apache 2.0 license.

**Format:** `{question, answer}` — already has both sides of the conversation, unlike prompt-only datasets.

**Pipeline — three commands:**

```bash
cd ~/nemotron/scripts

# 1. Prepare data (downloads from HuggingFace, converts format, splits train/val)
python3 prepare_infosec_dataset.py --max-samples 500   # start small for mock

# 2. Train LoRA adapter
python3 finetune_lora.py --mock \
  --dataset data/infosec_train.json \
  --val-dataset data/infosec_val.json \
  --output-dir ./lora-infosec

# 3. Evaluate (side-by-side base vs LoRA + perplexity)
python3 eval_lora.py --mock \
  --adapter ./lora-infosec \
  --val-dataset data/infosec_val.json
```

**What happens under the hood mathematically:**

Stage 1 — **Tokenization** (text → numbers, dictionary lookup, no math):
```
"What are the key features of PhaaS platforms?"
→ [1724, 526, 279, 1820, 4519, 302, 2463, 29741, 15409, 30]
```

Stage 2 — **Embedding** (integer IDs → vectors, frozen lookup table inside model):
```
Embedding matrix E: [vocab_size × hidden_dim] = [256000 × 3136]
E[1724] → [0.12, -0.34, 0.87, ..., 0.21]   ← 3136-dim vector for "What"
Output shape: [seq_len × hidden_dim] = [10 × 3136]
```

This embedding matrix was learned during pre-training. **LoRA does not touch it.**

Stage 3 — **Forward pass through a layer** (where LoRA lives):
```
x = input vector: [3136]

q = x × W  +  x × A × B × (α/r)
    ─────     ─────────────────
    frozen    LoRA adjustment

Step by step:
  x × W:          [3136] × [3136 × 5120] = [5120]     ← base output
  x × A:          [3136] × [3136 × 32]   = [32]       ← compress to rank-32
  (x × A) × B:    [32]   × [32 × 5120]   = [5120]     ← expand back
  scale by α/r:   [5120] × (64/32)        = [5120]     ← scale factor 2.0
  q_final = base_output + scaled_lora_output
```

Stage 4 — **Loss computation** (how wrong was the prediction):
```
Each answer position → logits: [256000 values] → softmax → probabilities
Cross-entropy loss: L_i = -log(P(correct_token))
Total loss = average across all answer positions
```

Stage 5 — **Backward pass** (gradients for A and B only):
```
∂L/∂B = Aᵀ × xᵀ × (∂L/∂q) × (α/r)
∂L/∂A = xᵀ × (∂L/∂q) × Bᵀ × (α/r)
W gets NO gradients (frozen) → no update, no optimizer memory
```

Stage 6 — **Optimizer step** (Adam updates A and B):
```
Full fine-tune:  4B params × 3 copies (param + m + v) × 2 bytes = ~24 GB optimizer state
LoRA (r=32):     3.1M params × 3 copies × 2 bytes               = ~18 MB optimizer state
```

**You do NOT need to embed the training data separately.** The model's own frozen embedding layer handles it during the forward pass. No external vector DB, no sentence-transformers.

---

## Part 9: Evaluating LoRA Results

### Q: How to evaluate the tuning result?

Three levels of evaluation, from automatic to manual:

**Level 1 — Training metrics (automatic, already built in):**

```
Step 1:    loss = 2.847
Step 100:  loss = 1.234
Step 500:  loss = 0.891
Step 2800: loss = 0.652   ← end of epoch 1
```

What to watch for:
```
Healthy:     Loss drops steadily, flattens toward end
Overfitting: Train loss drops but val loss goes UP (memorized, can't generalize)
Underfitting: Loss plateaus high (rank too low or too few epochs)
```

**Level 2 — Validation loss (added to finetune_lora.py):**

```bash
python3 finetune_lora.py --mock \
  --dataset data/infosec_train.json \
  --val-dataset data/infosec_val.json
```

Output:
```
Epoch 1: train_loss=1.23, eval_loss=1.31    ← gap small, good
Epoch 2: train_loss=0.71, eval_loss=0.82    ← still tracking
Epoch 3: train_loss=0.41, eval_loss=0.85    ← gap widening = overfitting starting
```

**Decision rule:** Pick the checkpoint where eval_loss is lowest, not where train_loss is lowest.

**Perplexity** — exponential of eval_loss:
```
PPL = e^(eval_loss)
eval_loss = 1.5  →  PPL = 4.48  (choosing between ~4.5 tokens on average)
eval_loss = 0.8  →  PPL = 2.23  (much more confident)
```

**Level 3 — Side-by-side comparison (the one that actually matters):**

```bash
python3 eval_lora.py --mock --adapter ./lora-infosec --val-dataset data/infosec_val.json
```

Runs same questions through base model and LoRA model:
```
Q: "How would you use nmap to discover services?"

Base:  "Nmap is a network scanning tool. You can use it to..."
       (generic, vague, textbook)

LoRA:  "For service discovery: nmap -sV -sC -p- target
       Version detection (-sV), default scripts (-sC), all ports (-p-).
       For stealth, use -sS with --scan-delay 500ms to avoid IDS..."
       (specific, actionable, practitioner-level)
```

**What "good" looks like:**

| Signal | Good | Bad |
|---|---|---|
| Train loss | Drops steadily | Stuck high or drops to ~0 |
| Eval loss | Tracks train loss, small gap | Diverges upward |
| Perplexity | LoRA PPL < Base PPL on security text | LoRA PPL higher |
| Side-by-side | More specific, actionable, structured | Generic or gibberish |
| General knowledge | Still intact | Forgot non-security answers |

**Important:** Also test non-security questions ("Explain photosynthesis", "Write Python hello world") to check for catastrophic forgetting. If those degrade, reduce epochs or lower learning rate.

---

## Part 10: Scripts Status

| Script | Purpose | Status |
|---|---|---|
| `scripts/prepare_infosec_dataset.py` | Download infosec-security-qa, convert format, train/val split | Created 2026-03-27 |
| `scripts/finetune_lora.py` | LoRA training with validation eval | Updated 2026-03-27 (added --val-dataset, eval_strategy) |
| `scripts/eval_lora.py` | Side-by-side base vs LoRA comparison + perplexity | Created 2026-03-27 |

**Not yet covered:**
- Tool-call training data (infosec dataset is Q&A only, no structured tool-call examples)
- Expanded TARGET_MODULES (still only q/k/v/o_proj; design calls for Mamba + MLP)
- vLLM hot-swap serving test (serving adapter with --enable-lora)

---

## Next Steps

- [ ] Run `prepare_infosec_dataset.py` to download and convert training data
- [ ] Run LoRA mock training on L4 with infosec dataset
- [ ] Run `eval_lora.py` to validate adapter quality
- [ ] Build tool-call training dataset for function calling patterns
- [ ] Update `finetune_lora.py` TARGET_MODULES to include Mamba + MLP layers
- [ ] Re-run LoRA mock with expanded targets
- [ ] Test LoRA hot-swap serving on L4 (`--enable-lora`)
- [ ] Scaffold RAG pipeline (vector DB, embedding model, retrieval chain)
- [ ] Populate RAG with CVE data, cheatsheets, product knowledge
- [ ] Validate full stack on GB10
