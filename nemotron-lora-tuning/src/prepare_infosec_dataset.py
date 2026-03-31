#!/usr/bin/env python3
"""
Download pAILabs/infosec-security-qa from HuggingFace and convert to
the instruction/input/output JSON format expected by finetune_lora.py.

Usage:
  pip install datasets
  python3 prepare_infosec_dataset.py                    # full dataset (~11.2K)
  python3 prepare_infosec_dataset.py --max-samples 500  # subset for testing
  python3 prepare_infosec_dataset.py --val-split 0.1    # 90/10 train/val split

Output:
  data/infosec_train.json   — training set
  data/infosec_val.json     — validation set (if --val-split > 0)
"""

import argparse
import json
import os
import random

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a cybersecurity expert and penetration testing assistant. "
    "Provide detailed, accurate, and actionable security guidance. "
    "Always emphasize authorized testing and responsible disclosure."
)


def convert_sample(row: dict) -> dict:
    """Convert HuggingFace row {question, answer} to {instruction, input, output}."""
    return {
        "instruction": SYSTEM_PROMPT,
        "input": row["question"],
        "output": row["answer"],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare infosec-security-qa for LoRA training")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total samples (default: use all ~11.2K)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction held out for validation (default: 0.1)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Step 1: Download from HuggingFace ---
    print("Downloading pAILabs/infosec-security-qa from HuggingFace...")
    ds = load_dataset("pAILabs/infosec-security-qa", split="train")
    print(f"  Downloaded {len(ds)} rows")

    # --- Step 2: Convert format ---
    print("Converting to instruction/input/output format...")
    samples = [convert_sample(row) for row in ds]

    # Shuffle before any subsetting or splitting
    random.shuffle(samples)

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"  Subsetted to {len(samples)} samples")

    # --- Step 3: Train/val split ---
    if args.val_split > 0:
        val_size = int(len(samples) * args.val_split)
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]
    else:
        train_samples = samples
        val_samples = []

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    # --- Step 4: Save ---
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "infosec_train.json")
    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"  Saved {train_path}")

    if val_samples:
        val_path = os.path.join(args.output_dir, "infosec_val.json")
        with open(val_path, "w") as f:
            json.dump(val_samples, f, indent=2)
        print(f"  Saved {val_path}")

    # --- Preview ---
    print(f"\n--- Sample preview ---")
    s = train_samples[0]
    print(f"  Q: {s['input'][:100]}...")
    print(f"  A: {s['output'][:100]}...")
    print(f"\nDone! Next step:")
    print(f"  python3 finetune_lora.py --mock --dataset {train_path}")


if __name__ == "__main__":
    main()
