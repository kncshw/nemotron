# Nemotron LoRA Fine-Tuning Samples

End-to-end LoRA fine-tuning pipeline for NVIDIA Nemotron-3-Super-120B, targeting security operations and penetration testing.

## Quick Start (GB10)

```bash
# 1. Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes mamba-ssm causal-conv1d openai

# 2. Prepare dataset (downloads 11.2K samples from HuggingFace)
python src/prepare_infosec_dataset.py

# 3. Train LoRA adapter on GB10 (Nemotron-3-Super-120B-A12B-NVFP4)
python src/finetune_lora.py \
  --dataset data/infosec_train.json \
  --val-dataset data/infosec_val.json \
  --output-dir ./lora-infosec

# 4. Evaluate (base vs LoRA side-by-side + perplexity)
python src/eval_lora.py \
  --adapter ./lora-infosec \
  --val-dataset data/infosec_val.json

# 5. Test API endpoint (requires running vLLM server)
python src/test_api.py --all
```

For mock runs on smaller GPUs (e.g., L4), add `--mock` to steps 3 and 4.

## Repo Structure

```
nemotron-lora-tuning/
├── README.md                 # This file
├── datasets/
│   ├── README.md             # Dataset format documentation
│   ├── qa/                   # Q&A instruction format (behavior/reasoning)
│   │   └── sample_infosec_qa.json
│   └── toolcall/             # Tool-call format (function calling)
│       └── sample_toolcall.json
├── src/
│   ├── prepare_infosec_dataset.py   # Download & convert HF datasets
│   ├── finetune_lora.py             # LoRA training with HF PEFT
│   ├── eval_lora.py                 # Evaluation (side-by-side + perplexity)
│   └── test_api.py                  # API test client (chat, tools, streaming)
├── docs/
│   ├── lora_technical_guide.md      # How LoRA works under the hood
│   ├── dataset_formats.md           # All training data formats explained
│   └── mock_validation.md           # L4 mock test results
└── .gitignore
```

## Dataset Formats

Two dataset types are included as samples:

| Format | File | Purpose | Count |
|--------|------|---------|-------|
| Q&A instruction | `datasets/qa/sample_infosec_qa.json` | Security reasoning & behavior | 5 samples |
| Tool-call messages | `datasets/toolcall/sample_toolcall.json` | Function calling reliability | 3 samples |

For production training, use `prepare_infosec_dataset.py` to download the full [pAILabs/infosec-security-qa](https://huggingface.co/datasets/pAILabs/infosec-security-qa) dataset (11.2K Q&A pairs).

## Hardware

| Environment | Model | GPU | Purpose |
|-------------|-------|-----|---------|
| **Production (GB10)** | Nemotron-3-Super-120B-A12B-NVFP4 | Dell Pro Max GB10 128GB | **Training & serving** |
| Mock (L4) | Nemotron-3-Nano-4B-BF16 | NVIDIA L4 23GB | Pipeline validation only |

## LoRA Configuration (GB10 defaults)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 32 | Good capacity for security domain shift |
| Alpha | 64 | 2x scaling factor (alpha/r = 2.0) |
| Target modules | q/k/v/o_proj, in/out_proj, up/down_proj | Full coverage: Attention + Mamba + MLP |
| Max seq length | 4096 | GB10 has ample memory for longer contexts |
| Batch size | 4 | Per-device batch size |
| Grad accumulation | 4 | Effective batch = 16 |
| Dropout | 0.05 | Light regularization |

## Architecture Decision

**LoRA for behavior, RAG for facts.**

- LoRA handles: pentester reasoning, tool-call reliability, output style, report formatting
- RAG handles: CVE databases, exploit details, target product specs, playbooks

See [docs/lora_technical_guide.md](docs/lora_technical_guide.md) for the full technical explanation.

## License

Training scripts: MIT. Dataset samples are for educational/authorized security testing purposes.
