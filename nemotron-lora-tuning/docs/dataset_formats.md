# Training Dataset Formats

Dataset format should match the target behavior you want to train. This guide covers all major formats and when to use each.

---

## 1. Q&A Instruction Format (Knowledge & Personality)

```json
{
  "instruction": "You are a cybersecurity expert...",
  "input": "What is privilege escalation?",
  "output": "Privilege escalation is gaining higher-level permissions..."
}
```

The model learns how to reason, what tone to use, what depth to provide, what terminology to prefer. This is behavioral/style tuning -- the sweet spot for LoRA.

**Source:** [pAILabs/infosec-security-qa](https://huggingface.co/datasets/pAILabs/infosec-security-qa) (11.2K pairs, Apache 2.0)

---

## 2. Tool-Call Format (Function Calling)

```json
{
  "messages": [
    {"role": "system", "content": "You have access to: nmap_scan(target, ports), search_cve(keyword)"},
    {"role": "user", "content": "Scan 192.168.1.0/24 for open web ports"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "nmap_scan", "arguments": "{\"target\": \"192.168.1.0/24\", \"ports\": \"80,443,8080\"}"}}
    ]},
    {"role": "tool", "content": "{\"hosts\": [{\"ip\": \"192.168.1.5\", \"ports\": [80, 443]}]}"},
    {"role": "assistant", "content": "Found 1 host with web services..."}
  ]
}
```

The model learns when to call a tool vs respond directly, how to format the JSON, how to interpret results, and when to chain multiple calls.

---

## 3. Preference Pairs (RLHF / DPO Alignment)

```json
{
  "prompt": "How do I test for XSS?",
  "chosen": "To test for XSS in an authorized pentest: 1. Inject <script>alert(1)</script>...",
  "rejected": "Just inject scripts everywhere on any website you want to test..."
}
```

Teaches the model which response style is better. Uses DPO or PPO training loops, not standard LoRA SFT.

---

## 4. Raw Text (Continued Pre-Training)

```
Buffer overflow vulnerabilities occur when a program writes data beyond
the allocated memory buffer. In stack-based overflows, the attacker
overwrites the return address to redirect execution flow...
```

No structure -- just raw documents. The model trains via next-token prediction on every position. Called "continued pre-training" or "domain-adaptive pre-training."

Best for: teaching domain language/patterns the base model has never seen (proprietary protocols, classified content). For public security knowledge, the 120B model likely already knows most of it.

---

## 5. Classification Labels (Sequence Classification)

```json
{"text": "Unusual outbound DNS traffic to known C2 domain", "label": "malware_communication"}
{"text": "Failed login from 5 different IPs in 10 seconds", "label": "brute_force"}
```

For training a classifier head. Uses `TaskType.SEQ_CLS` instead of `CAUSAL_LM`.

---

## 6. Span Extraction (NER / Token Classification)

```json
{
  "tokens":  ["CVE-2024-3094", "affects", "xz-utils", "version", "5.6.0"],
  "labels":  ["CVE_ID",        "O",       "PRODUCT",   "O",       "VERSION"]
}
```

For extracting CVE IDs, product names, IOCs from unstructured text.

---

## 7. Multi-Turn Conversation (Chat Fine-Tuning)

```json
{
  "messages": [
    {"role": "user", "content": "I found port 443 open on the target"},
    {"role": "assistant", "content": "Run service detection: nmap -sV -p 443 target"},
    {"role": "user", "content": "It's running Apache 2.4.49"},
    {"role": "assistant", "content": "Vulnerable to CVE-2021-41773..."}
  ]
}
```

Teaches conversational flow -- context tracking, follow-ups, building on previous turns.

---

## 8. Embedding Pairs (RAG Retrieval Tuning)

```json
{"query": "Apache path traversal exploit", "positive": "CVE-2021-41773 allows...", "negative": "Apache Kafka configuration guide..."}
```

For fine-tuning the embedding model (not the LLM), improving RAG retrieval quality.

---

## Summary

| Format | Training Objective | Our Use Case |
|--------|-------------------|-------------|
| Instruction Q&A | Behavior, style, reasoning | **Yes -- infosec dataset** |
| Tool-call messages | Function calling reliability | **Yes -- need to build** |
| Preference pairs | Alignment (DPO/RLHF) | Maybe later |
| Raw text | Domain knowledge injection | If 120B lacks security knowledge |
| Classification labels | Threat categorization | If we build detection pipeline |
| Token/span labels | IOC extraction, NER | If we need entity parsing |
| Multi-turn conversation | Dialogue flow | Covered by tool-call format |
| Embedding pairs | RAG retrieval quality | When we build RAG pipeline |

**Our recommended mix:** ~70% Q&A + ~30% tool-call conversations.
