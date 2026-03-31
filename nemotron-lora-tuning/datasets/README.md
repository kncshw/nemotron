# Training Datasets

This directory contains sample datasets in two formats for LoRA fine-tuning.

## Q&A Format (`qa/`)

**Purpose:** Train the model's security reasoning, tone, and methodology.

Each sample has three fields:

```json
{
  "instruction": "System prompt setting the persona",
  "input": "User question",
  "output": "Expected assistant response"
}
```

- Use with `prepare_infosec_dataset.py` to download the full 11.2K sample [pAILabs/infosec-security-qa](https://huggingface.co/datasets/pAILabs/infosec-security-qa) dataset
- The included `sample_infosec_qa.json` contains 5 hand-crafted examples for testing

## Tool-Call Format (`toolcall/`)

**Purpose:** Train the model to reliably emit structured function calls and interpret tool results.

Each sample is a multi-turn conversation with tool interactions:

```json
{
  "messages": [
    {"role": "system", "content": "...tools available..."},
    {"role": "user", "content": "user request"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "...", "arguments": "..."}}]},
    {"role": "tool", "name": "...", "content": "tool result JSON"},
    {"role": "assistant", "content": "final response incorporating tool results"}
  ]
}
```

- `sample_toolcall.json` contains 3 examples demonstrating single-call, sequential, and parallel tool use patterns
- Production datasets should include 100+ examples for reliable tool-calling behavior

## Recommended Mix

For a security-focused fine-tune: ~70% Q&A + ~30% tool-call conversations.

## Why Two Formats?

| Format | What It Trains | LoRA Sweet Spot |
|--------|---------------|-----------------|
| Q&A instruction | Behavior, reasoning style, domain knowledge emphasis | Yes -- behavioral shift is low-rank |
| Tool-call messages | When to call tools, JSON structure, result interpretation | Yes -- output format is a style shift |

Both are behavioral changes (how the model responds), not factual knowledge injection. This is exactly what LoRA excels at. For factual knowledge (CVE databases, product specs), use RAG instead.
