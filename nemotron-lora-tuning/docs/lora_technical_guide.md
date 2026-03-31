# LoRA Technical Guide

How LoRA fine-tuning works, from math to practice, in the context of Nemotron-3 hybrid Mamba+Attention models.

---

## What LoRA Does

A neural network layer like `q_proj` is a matrix multiplication: `output = input x W`, where W is e.g. [3136 x 5120] = ~16M parameters. Full fine-tuning would update all 16M -- too much memory.

LoRA learns a **change** (delta-W), represented as two small matrices:

```
W_original:  [3136 x 5120]   <- frozen, never changes

A:           [3136 x 32]      <- trainable  (rank r=32)
B:           [32 x 5120]      <- trainable  (rank r=32)

delta-W = A x B: [3136 x 5120]    <- same shape as W, built from only 32 columns/rows
```

During forward pass:

```
output = input x W_original  +  input x A x B x (alpha/r)
         -------------------    -------------------------
         frozen (pre-trained)   learned adjustment
```

Parameter savings for `q_proj` [3136 x 5120]:
- Full fine-tuning: 16,056,320 parameters
- LoRA (r=32): 264,192 parameters (~1.6% of full)

---

## Why Low-Rank Works

The LoRA paper (Hu et al., 2021) measured what actually changes during full fine-tuning:

```
delta-W = W_new - W_original
Finding: rank(delta-W) is extremely low relative to matrix size
```

The actual change matrix is almost entirely captured by its top ~10-64 singular values. Fine-tuning for a new domain is like moving to a nearby point in weight space -- the shift can be described with ~32 directions and magnitudes.

With r=32, you get 32 independent direction vectors summed together:

```
delta-W = a1.b1^T + a2.b2^T + ... + a32.b32^T
```

Gradient descent rotates these vectors to align with the directions that reduce loss on your training data.

---

## The Seven-Stage Pipeline

What happens when you train on an infosec Q&A sample:

**Stage 1 -- Tokenization** (text to numbers, dictionary lookup):
```
"What are the key features of PhaaS platforms?"
-> [1724, 526, 279, 1820, 4519, 302, 2463, 29741, 15409, 30]
```

**Stage 2 -- Embedding** (IDs to vectors, frozen lookup table inside model):
```
Embedding matrix E: [vocab_size x hidden_dim] = [256000 x 3136]
E[1724] -> [0.12, -0.34, 0.87, ..., 0.21]   <- 3136-dim vector for "What"
```
LoRA does NOT touch the embedding matrix.

**Stage 3 -- Forward pass** (where LoRA lives):
```
x = input vector: [3136]

q = x * W  +  x * A * B * (alpha/r)

Step by step:
  x * W:          [3136] x [3136 x 5120] = [5120]     <- base output
  x * A:          [3136] x [3136 x 32]   = [32]       <- compress to rank-32
  (x * A) * B:    [32]   x [32 x 5120]   = [5120]     <- expand back
  scale by a/r:   [5120] x (64/32)        = [5120]     <- scale factor 2.0
  q_final = base_output + scaled_lora_output
```

**Stage 4 -- Loss** (how wrong was the prediction):
```
Each position -> logits: [256000 values] -> softmax -> probabilities
Cross-entropy loss: L_i = -log(P(correct_token))
Total loss = average across all answer positions
```

**Stage 5 -- Backward pass** (gradients for A and B only):
```
dL/dB = A^T x x^T x (dL/dq) x (alpha/r)
dL/dA = x^T x (dL/dq) x B^T x (alpha/r)
W gets NO gradients (frozen)
```

**Stage 6 -- Optimizer step** (Adam updates A and B):
```
Full fine-tune:  4B params x 3 copies (param + m + v) x 2 bytes = ~24 GB optimizer state
LoRA (r=32):     3.1M params x 3 copies x 2 bytes               = ~18 MB optimizer state
```

**Stage 7 -- Repeat** (~8,400 updates for 3 epochs on 11.2K samples with batch=4, grad_accum=4).

---

## After Training: Using the Adapter

The saved adapter (~13MB) contains only A and B matrices. Two deployment modes:

1. **Merge:** `W_new = W_original + A x B` -- permanent, baked into weights
2. **Hot-swap (vLLM):** Keep W_original in memory, load the tiny adapter on the side. Multiple LoRA adapters can share one base model via `--enable-lora`.

---

## Hybrid Mamba+Attention: Special Considerations

Nemotron-3 is a hybrid model with three layer types:

| Layer Type | Parameters | Role |
|-----------|-----------|------|
| Attention (q/k/v/o_proj) | Routing, relationships | "What should I look at?" |
| Mamba (in_proj/out_proj) | Sequential patterns | "What patterns in sequence?" |
| MLP (up_proj/down_proj) | Factual knowledge storage | "What does this mean?" |

**Key issue:** LoRA targets attention layers, but Mamba layers sit between them. With gradient checkpointing, the gradient chain breaks through non-LoRA Mamba layers.

**Fix:** `model.enable_input_require_grads()` ensures all inputs carry gradient tracking through Mamba layers.

**Target module strategy (GB10 default uses all three):**

| Target set | Trainable params | Adapter size | What it learns |
|-----------|-----------------|-------------|---------------|
| Attention only | ~3.1M (0.08%) | ~13MB | Behavior/style |
| + Mamba | ~8M (0.2%) | ~30MB | + Sequential patterns |
| **+ Mamba + MLP (default)** | **~16M (0.4%)** | **~60MB** | **+ Domain knowledge** |

GB10's 128GB unified memory easily handles the full target set. The expanded coverage gives the adapter access to factual knowledge stored in MLP layers alongside behavioral shifts in attention and sequential patterns in Mamba.

---

## LoRA Limitations

1. **Low-rank assumption breaks for new facts** -- each independent fact needs its own direction; r=32 can't handle 500 CVEs
2. **Catastrophic forgetting** -- LoRA can degrade existing capabilities
3. **Data quality amplification** -- small param count means every bad sample hurts disproportionately
4. **Rank is a hard ceiling** -- if true delta-W needs rank 200 and r=32, no amount of epochs fixes it
5. **No selective editing** -- can't surgically remove one bad pattern without full retrain
6. **Evaluation is hard** -- subtle changes, model may look fine on 95% but fail on edge cases

**Bottom line:** LoRA for behavior, RAG for facts.

---

## LoRA vs RAG: When to Use Which

LoRA and RAG are complementary -- each has a distinct role. Using only one leaves a gap.

### Why both are needed

- **LoRA alone** = model reasons like a pentester but only knows what was in pre-training data (stale)
- **RAG alone** = model has current info but still responds like a generic assistant
- **LoRA + RAG** = reasons like a pentester AND has current exploit/tool knowledge

### Head-to-head comparison

```
                        LoRA Fine-tuning                RAG
                        ────────────────                ───
Where knowledge         Baked into weights              Retrieved at query time
lives                   (permanent)                     (from vector DB)

Update cycle            Retrain (hours)                 Update DB row (seconds)

Recall                  Fuzzy -- model may              Exact -- retrieval finds
                        half-learn or hallucinate       the actual document

Capacity                Limited by rank/params          Limited by DB size
                        (can't absorb 10K docs          (can store millions)
                        reliably)

Latency                 Zero -- knowledge is            Adds retrieval step
                        in the forward pass             (~50-200ms)

Context window          Not consumed                    Eats context window
                        (it's in weights)               (retrieved chunks injected)

Hallucination           Can confidently state           Grounded -- model sees
                        wrong "facts" it half-learned   the actual source text
```

### What goes where

| LoRA (baked into weights) | RAG (retrieved at runtime) |
|---------------------------|---------------------------|
| Tool-calling reliability | CVE database |
| Pentester reasoning patterns | Target product specs |
| Security-aware output style | Network topology / scan results |
| Report formatting | Cheatsheets / playbooks |
| When to use which tool | Historical vulnerability data |
| Attack chain methodology | Exploit-db entries |

### Why not raw text training instead of RAG?

We considered continued pre-training (feeding raw security docs) as an alternative to RAG. The critical difference: **RAG is grounded, raw text training is not.** When the model absorbs a CVE description through training, it might later say "CVE-2024-3094 affects xz-utils version 5.6.1" when it was actually 5.6.0. It half-learned the fact. With RAG, the actual document is in the prompt.

Raw text training makes sense only when the base model genuinely lacks domain exposure (proprietary protocols, classified content). For public security knowledge, the 120B model likely already knows most of it.

### Comparison of all approaches

```
                   Least effort                                Most effort
                   ───────────                                 ──────────
                   RAG          LoRA (Q&A)    Raw text + LoRA    Full fine-tune
                   ─────────    ──────────    ─────────────────  ──────────────
Fact recall        Exact        Unreliable    Slightly better    Still unreliable
Behavior shift     None         Strong        Stronger           Strongest
Cost               Cheap        Moderate      Expensive          Too much for GB10
Update speed       Instant      Retrain       Retrain            Retrain
```

### Our strategy for GB10

**LoRA for behavior, RAG for facts. Each playing to its strength.**

1. LoRA fine-tune with ~70% Q&A (infosec reasoning) + ~30% tool-call (structured output)
2. RAG pipeline for mutable, high-cardinality knowledge (CVEs, exploit-db, tool docs, internal playbooks)
3. No raw text continued pre-training -- the 120B model already has broad security knowledge

---

## Rank Selection Guide

```
r=4:    Few trainable params, fast, limited expressiveness
r=16:   Good balance for simple domain adaptation
r=32:   Good for security domain shift (our setting)
r=64:   More capacity, more memory, diminishing returns
r=256:  Approaching full fine-tuning behavior
```

`lora_alpha` controls scaling: delta-W contribution is scaled by `alpha/r`. With alpha=64 and r=32, the scaling factor is 2.0.

---

## Evaluation Guide

Three levels:

**Level 1 -- Training loss** (automatic):
- Healthy: drops steadily, flattens toward end
- Overfitting: train loss drops but val loss goes UP
- Underfitting: loss plateaus high

**Level 2 -- Validation loss + perplexity:**
- Pick checkpoint where eval_loss is lowest (not train_loss)
- PPL = e^(eval_loss) -- lower means more confident predictions

**Level 3 -- Side-by-side comparison** (what actually matters):
- Run same questions through base and LoRA model
- Check for: more specific/actionable answers, correct tool usage
- Also test non-security questions to detect catastrophic forgetting
