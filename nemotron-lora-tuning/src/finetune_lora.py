#!/usr/bin/env python3
"""
LoRA fine-tuning for Nemotron-3-Super-120B on Dell Pro Max GB10.

GB10 run:       python3 finetune_lora.py --dataset data/infosec_train.json
Mock run (L4):  python3 finetune_lora.py --mock --dataset data/infosec_train.json

Configurable via CLI args or by editing the defaults below.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ---- Config ----

MOCK_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LR = 1.5e-4
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_GRAD_ACCUM = 4

# Attention + Mamba + MLP: full coverage for behavior + sequential patterns + knowledge
# See docs/lora_technical_guide.md for rationale on each layer type.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention: reasoning & routing
    "in_proj", "out_proj",                     # Mamba: sequential pattern adaptation
    "up_proj", "down_proj",                    # MLP: domain knowledge reinforcement
]


# ---- Dataset Preparation ----

def build_sample_security_dataset() -> list[dict]:
    """Sample security dataset for mock testing. Replace with real data for production."""
    samples = [
        {
            "instruction": "You are a penetration testing assistant.",
            "input": "How would you enumerate subdomains for a target domain?",
            "output": (
                "Subdomain enumeration can be done through several methods:\n"
                "1. **Passive**: Use tools like Subfinder, Amass, or crt.sh to find subdomains from public sources.\n"
                "2. **Active brute-force**: Use tools like ffuf or gobuster with a wordlist against the DNS.\n"
                "3. **DNS zone transfer**: Attempt `dig axfr @ns.target.com target.com` (rarely works but worth trying).\n"
                "4. **Search engines**: Google dork `site:*.target.com` or use Shodan.\n"
                "5. **Certificate transparency logs**: Query crt.sh or certspotter for issued certificates."
            ),
        },
        {
            "instruction": "You are a penetration testing assistant.",
            "input": "Explain how to test for SQL injection in a login form.",
            "output": (
                "To test for SQL injection in a login form:\n"
                "1. **Basic test**: Enter `' OR '1'='1` in the username field.\n"
                "2. **Error-based**: Enter a single quote `'` and observe if the application returns a database error.\n"
                "3. **Time-based blind**: Use `' OR SLEEP(5)--` to detect injection via response delay.\n"
                "4. **Union-based**: Try `' UNION SELECT NULL,NULL--` incrementing NULLs to find column count.\n"
                "5. **Automated**: Run sqlmap against the form: `sqlmap -u 'http://target/login' --data='user=test&pass=test' --batch`.\n"
                "Always ensure you have written authorization before testing."
            ),
        },
        {
            "instruction": "You are a penetration testing assistant.",
            "input": "What is privilege escalation on Linux and how do you check for it?",
            "output": (
                "Privilege escalation is gaining higher-level permissions than initially granted. Common checks:\n"
                "1. **SUID binaries**: `find / -perm -4000 -type f 2>/dev/null` — look for unusual SUID binaries.\n"
                "2. **Sudo misconfig**: `sudo -l` — check what commands the current user can run as root.\n"
                "3. **Kernel exploits**: `uname -r` and check against known exploits (e.g., DirtyPipe, DirtyCow).\n"
                "4. **Writable paths**: Check if /etc/passwd, cron directories, or service configs are writable.\n"
                "5. **Automated**: Run LinPEAS or linux-exploit-suggester for comprehensive enumeration.\n"
                "6. **Capabilities**: `getcap -r / 2>/dev/null` — check for binaries with dangerous capabilities."
            ),
        },
        {
            "instruction": "You are a penetration testing assistant.",
            "input": "How do you perform a basic network pivot after compromising a host?",
            "output": (
                "After compromising a host with access to an internal network:\n"
                "1. **Discover internal networks**: `ip addr`, `ip route`, `arp -a` to map connected subnets.\n"
                "2. **SSH tunnel**: `ssh -D 9050 user@compromised` creates a SOCKS proxy for pivoting.\n"
                "3. **Chisel**: Run `chisel server -p 8080 --reverse` on attacker, `chisel client attacker:8080 R:socks` on target.\n"
                "4. **Port forwarding**: `ssh -L 8080:internal-target:80 user@compromised` for specific services.\n"
                "5. **Scan internal network**: Use proxychains with nmap: `proxychains nmap -sT -Pn 10.0.0.0/24`.\n"
                "Ensure all pivoting activities are within the authorized scope."
            ),
        },
        {
            "instruction": "You are a penetration testing assistant.",
            "input": "Describe how to extract credentials from a Windows machine during a pentest.",
            "output": (
                "Credential extraction on Windows (post-exploitation, with admin access):\n"
                "1. **Mimikatz**: `sekurlsa::logonpasswords` to dump plaintext passwords and NTLM hashes from memory.\n"
                "2. **SAM database**: `reg save HKLM\\SAM sam.bak` + `reg save HKLM\\SYSTEM sys.bak`, then extract offline.\n"
                "3. **LSASS dump**: Use procdump or comsvcs.dll MiniDump, then parse with pypykatz offline.\n"
                "4. **Cached credentials**: `cachedump` or mimikatz `lsadump::cache` for domain cached creds.\n"
                "5. **DPAPI**: Decrypt saved browser passwords, WiFi keys, and other DPAPI-protected secrets.\n"
                "6. **Kerberos tickets**: `sekurlsa::tickets /export` for pass-the-ticket attacks.\n"
                "Document all extracted credentials and report them. Never exfiltrate beyond the assessment scope."
            ),
        },
    ]
    return samples


def format_for_training(samples: list[dict], tokenizer, max_seq_len: int) -> Dataset:
    """Format instruction/input/output samples into tokenized dataset."""
    formatted = []
    for s in samples:
        # Use chat template if available, otherwise simple format
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": s["instruction"]},
                {"role": "user", "content": s["input"]},
                {"role": "assistant", "content": s["output"]},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"### System:\n{s['instruction']}\n\n### User:\n{s['input']}\n\n### Assistant:\n{s['output']}"
        else:
            text = f"### System:\n{s['instruction']}\n\n### User:\n{s['input']}\n\n### Assistant:\n{s['output']}"

        tokenized = tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        formatted.append(tokenized)

    return Dataset.from_list(formatted)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Nemotron")
    parser.add_argument("--mock", action="store_true", help="Use Nano-4B on L4 (mock run)")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON dataset file (instruction/input/output format)")
    parser.add_argument("--val-dataset", type=str, default=None, help="Path to JSON validation dataset for eval loss")
    parser.add_argument("--output-dir", type=str, default="./lora-output", help="Output directory for adapter")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R, help=f"LoRA rank (default: {DEFAULT_LORA_R})")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM})")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN, help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LEN})")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit quantization (saves memory)")
    args = parser.parse_args()

    # Select model
    if args.model:
        model_name = args.model
    elif args.mock:
        model_name = MOCK_MODEL
    else:
        model_name = DEFAULT_MODEL

    print(f"=== LoRA Fine-tuning ===")
    print(f"  Model: {model_name}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Target modules: {TARGET_MODULES}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Output: {args.output_dir}")

    # Load dataset
    if args.dataset:
        print(f"  Loading dataset from {args.dataset}")
        with open(args.dataset) as f:
            samples = json.load(f)
    else:
        print("  Using built-in sample security dataset (5 examples)")
        samples = build_sample_security_dataset()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if args.load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    # Hybrid Mamba+Attention models need this so gradients flow through
    # non-LoRA Mamba layers to reach the LoRA-targeted attention layers.
    model.enable_input_require_grads()

    # Apply LoRA
    print("Applying LoRA config...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = format_for_training(samples, tokenizer, args.max_seq_len)
    print(f"  Training examples: {len(train_dataset)}")

    # Prepare validation dataset (if provided)
    val_dataset = None
    if args.val_dataset:
        print(f"  Loading validation dataset from {args.val_dataset}")
        with open(args.val_dataset) as f:
            val_samples = json.load(f)
        val_dataset = format_for_training(val_samples, tokenizer, args.max_seq_len)
        print(f"  Validation examples: {len(val_dataset)}")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print("Starting training...")
    trainer.train()

    # Save adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n=== Training Complete ===")
    print(f"Adapter saved to: {args.output_dir}")
    print(f"\nTo serve with vLLM:")
    print(f"  --enable-lora --lora-modules security-ops={os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
