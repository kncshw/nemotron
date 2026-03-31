#!/usr/bin/env python3
"""
Evaluate LoRA adapter quality by comparing base model vs fine-tuned responses.

Usage:
  GB10:   python3 eval_lora.py --adapter ./lora-infosec --val-dataset data/infosec_val.json
  Mock:   python3 eval_lora.py --mock --adapter ./lora-infosec

Runs the same prompts through base model and LoRA-adapted model side by side.
Also computes perplexity on held-out validation data.
"""

import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MOCK_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

SYSTEM_PROMPT = (
    "You are a cybersecurity expert and penetration testing assistant. "
    "Provide detailed, accurate, and actionable security guidance."
)

# Default eval questions — covering different security domains
DEFAULT_EVAL_QUESTIONS = [
    # Factual knowledge
    "What is a SQL injection attack and how do you test for it?",
    # Procedural reasoning
    "Walk me through the steps of performing a network penetration test.",
    # Tool usage
    "How would you use nmap to discover services on a target network?",
    # Incident response
    "You've detected lateral movement in a Windows Active Directory environment. What do you do?",
    # Defensive
    "What are the best practices for hardening a Linux web server?",
    # Recent/specific
    "Explain how a buffer overflow exploit works at the memory level.",
]


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model for a given question."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{question}\n\n### Assistant:\n"
    else:
        prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{question}\n\n### Assistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the new tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def compute_perplexity(model, tokenizer, texts: list[str], max_len: int = 2048) -> float:
    """Compute perplexity on a list of texts. Lower = model predicts these texts better."""
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.item()
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += loss * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter")
    parser.add_argument("--mock", action="store_true", help="Use Nano-4B")
    parser.add_argument("--model", type=str, default=None, help="Override base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--eval-file", type=str, default=None,
                        help="JSON file with custom eval questions (list of strings)")
    parser.add_argument("--val-dataset", type=str, default=None,
                        help="Validation JSON for perplexity (instruction/input/output format)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.model:
        model_name = args.model
    elif args.mock:
        model_name = MOCK_MODEL
    else:
        model_name = DEFAULT_MODEL

    # Load eval questions
    if args.eval_file:
        with open(args.eval_file) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_EVAL_QUESTIONS

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model responses ---
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()

    print("\n" + "=" * 70)
    print("GENERATING BASE MODEL RESPONSES")
    print("=" * 70)
    base_responses = []
    for i, q in enumerate(questions):
        print(f"\n--- Q{i+1}: {q}")
        resp = generate_response(base_model, tokenizer, q, args.max_new_tokens)
        base_responses.append(resp)
        print(f"--- Base: {resp[:300]}{'...' if len(resp) > 300 else ''}")

    # Perplexity on validation set (base)
    base_ppl = None
    if args.val_dataset:
        with open(args.val_dataset) as f:
            val_samples = json.load(f)
        val_texts = [f"{s['input']}\n{s['output']}" for s in val_samples[:100]]
        base_ppl = compute_perplexity(base_model, tokenizer, val_texts)
        print(f"\nBase model perplexity on val set: {base_ppl:.2f}")

    # --- LoRA model responses ---
    print(f"\nLoading LoRA adapter: {args.adapter}")
    lora_model = PeftModel.from_pretrained(base_model, args.adapter)
    lora_model.eval()

    print("\n" + "=" * 70)
    print("GENERATING LoRA MODEL RESPONSES")
    print("=" * 70)
    lora_responses = []
    for i, q in enumerate(questions):
        print(f"\n--- Q{i+1}: {q}")
        resp = generate_response(lora_model, tokenizer, q, args.max_new_tokens)
        lora_responses.append(resp)
        print(f"--- LoRA: {resp[:300]}{'...' if len(resp) > 300 else ''}")

    # Perplexity on validation set (LoRA)
    lora_ppl = None
    if args.val_dataset:
        lora_ppl = compute_perplexity(lora_model, tokenizer, val_texts)
        print(f"\nLoRA model perplexity on val set: {lora_ppl:.2f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if base_ppl and lora_ppl:
        print(f"\nPerplexity (lower = better):")
        print(f"  Base:  {base_ppl:.2f}")
        print(f"  LoRA:  {lora_ppl:.2f}")
        delta = ((lora_ppl - base_ppl) / base_ppl) * 100
        print(f"  Delta: {delta:+.1f}%")

    print(f"\nSide-by-side responses saved to: eval_results.json")

    # Save full results for review
    results = []
    for i, q in enumerate(questions):
        results.append({
            "question": q,
            "base_response": base_responses[i],
            "lora_response": lora_responses[i],
        })

    output_path = os.path.join(os.path.dirname(args.adapter), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
