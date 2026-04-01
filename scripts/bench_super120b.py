#!/usr/bin/env python3
"""
Throughput benchmark for Nemotron-3-Super-120B-A12B on GB10.

Measures:
  - Time to first token (TTFT)
  - Output tokens/sec (single user)
  - Output tokens/sec (concurrent users)
  - Throughput at different output lengths
  - Throughput with reasoning on vs off

Usage:
    python3 bench_super120b.py                          # quick benchmark
    python3 bench_super120b.py --full                   # full benchmark suite
    python3 bench_super120b.py --concurrent 1 3 5       # custom concurrency levels
    python3 bench_super120b.py --base-url http://host:8000/v1
    python3 bench_super120b.py --output results.json    # save results to file
"""

import argparse
import json
import time
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def get_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="not-needed")


def get_model(client: OpenAI) -> str:
    models = client.models.list()
    return models.data[0].id


# ---------- Single request with detailed timing ----------

def timed_request(client, model, messages, max_tokens, temperature=0.7, stream=True):
    """Run a single request and return detailed timing info."""
    start = time.time()
    first_token_time = None
    token_count = 0

    if stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        chunks = []
        for chunk in resp:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                chunks.append(chunk.choices[0].delta.content)
                token_count += 1
        text = "".join(chunks)
        end = time.time()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        end = time.time()
        first_token_time = end  # no streaming, so TTFT ~ total time
        token_count = resp.usage.completion_tokens
        text = resp.choices[0].message.content

    total_time = end - start
    ttft = (first_token_time - start) if first_token_time else total_time
    generation_time = end - first_token_time if first_token_time else total_time
    tps = token_count / generation_time if generation_time > 0 else 0

    return {
        "tokens": token_count,
        "total_time": round(total_time, 3),
        "ttft": round(ttft, 3),
        "generation_time": round(generation_time, 3),
        "tokens_per_sec": round(tps, 1),
        "text_preview": text[:200] if text else "",
    }


# ---------- Benchmark: Single-user output speed ----------

def bench_single_user(client, model, max_tokens=256, runs=3):
    print(f"\n{'='*60}")
    print(f"  SINGLE USER — {max_tokens} max tokens, {runs} runs")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": "You are a helpful cybersecurity assistant."},
        {"role": "user", "content": "Explain the OWASP Top 10 vulnerabilities with brief descriptions for each."},
    ]

    results = []
    for i in range(runs):
        r = timed_request(client, model, messages, max_tokens)
        results.append(r)
        print(f"  Run {i+1}: {r['tokens']} tokens | TTFT {r['ttft']:.2f}s | {r['tokens_per_sec']:.1f} t/s | total {r['total_time']:.1f}s")

    avg_tps = statistics.mean(r["tokens_per_sec"] for r in results)
    avg_ttft = statistics.mean(r["ttft"] for r in results)
    print(f"  ——————————————")
    print(f"  AVG: {avg_tps:.1f} t/s | TTFT {avg_ttft:.2f}s")

    return {
        "test": "single_user",
        "max_tokens": max_tokens,
        "runs": results,
        "avg_tokens_per_sec": round(avg_tps, 1),
        "avg_ttft": round(avg_ttft, 3),
    }


# ---------- Benchmark: Output length scaling ----------

def bench_output_lengths(client, model):
    print(f"\n{'='*60}")
    print(f"  OUTPUT LENGTH SCALING")
    print(f"{'='*60}")

    lengths = [64, 128, 256, 512, 1024]
    messages = [
        {"role": "user", "content": "Write a detailed guide on network penetration testing methodology. Be thorough and cover all phases."},
    ]

    results = []
    for length in lengths:
        r = timed_request(client, model, messages, length)
        results.append({"max_tokens": length, **r})
        print(f"  {length:>5} tokens: {r['tokens']:>5} generated | {r['tokens_per_sec']:.1f} t/s | TTFT {r['ttft']:.2f}s")

    return {"test": "output_length_scaling", "results": results}


# ---------- Benchmark: Concurrent users ----------

def bench_concurrent(client, model, concurrency_levels=None):
    if concurrency_levels is None:
        concurrency_levels = [1, 3, 5]

    print(f"\n{'='*60}")
    print(f"  CONCURRENT USERS — levels: {concurrency_levels}")
    print(f"{'='*60}")

    messages = [
        {"role": "user", "content": "What is SQL injection? Explain with an example in 3-4 sentences."},
    ]

    all_results = []
    for n in concurrency_levels:
        def _run(user_id):
            return timed_request(client, model, messages, max_tokens=128, stream=False)

        with ThreadPoolExecutor(max_workers=n) as pool:
            start = time.time()
            futures = [pool.submit(_run, i) for i in range(n)]
            user_results = [f.result() for f in as_completed(futures)]
            wall_time = time.time() - start

        total_tokens = sum(r["tokens"] for r in user_results)
        aggregate_tps = total_tokens / wall_time
        avg_latency = statistics.mean(r["total_time"] for r in user_results)
        avg_per_user_tps = statistics.mean(r["tokens_per_sec"] for r in user_results)

        result = {
            "concurrency": n,
            "wall_time": round(wall_time, 2),
            "total_tokens": total_tokens,
            "aggregate_tps": round(aggregate_tps, 1),
            "avg_per_user_tps": round(avg_per_user_tps, 1),
            "avg_latency": round(avg_latency, 2),
        }
        all_results.append(result)
        print(f"  {n} users: {aggregate_tps:.1f} t/s aggregate | {avg_per_user_tps:.1f} t/s per-user | {avg_latency:.1f}s avg latency")

    return {"test": "concurrent", "results": all_results}


# ---------- Benchmark: Reasoning on vs off ----------

def bench_reasoning(client, model):
    print(f"\n{'='*60}")
    print(f"  REASONING ON vs OFF")
    print(f"{'='*60}")

    prompt = "A web server runs Apache 2.4.49. What vulnerability should I check for, what is the CVE, and how would I exploit it?"

    # Without reasoning (greedy)
    messages_no_reason = [{"role": "user", "content": prompt}]
    r_off = timed_request(client, model, messages_no_reason, max_tokens=256, temperature=0.0)
    print(f"  Reasoning OFF: {r_off['tokens']} tokens | {r_off['tokens_per_sec']:.1f} t/s | TTFT {r_off['ttft']:.2f}s")

    # With reasoning (temperature=1.0 as recommended)
    messages_reason = [{"role": "user", "content": f"Think step by step: {prompt}"}]
    r_on = timed_request(client, model, messages_reason, max_tokens=512, temperature=1.0)
    print(f"  Reasoning ON:  {r_on['tokens']} tokens | {r_on['tokens_per_sec']:.1f} t/s | TTFT {r_on['ttft']:.2f}s")

    return {
        "test": "reasoning_comparison",
        "reasoning_off": r_off,
        "reasoning_on": r_on,
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Nemotron-3-Super-120B-A12B")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--concurrent", type=int, nargs="+", metavar="N", help="Concurrency levels to test")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for single-user test")
    args = parser.parse_args()

    client = get_client(args.base_url)
    print(f"Connecting to {args.base_url}...")
    model = get_model(client)
    print(f"Model: {model}\n")

    all_results = {"model": model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Always run single-user benchmark
    all_results["single_user_256"] = bench_single_user(client, model, max_tokens=256, runs=args.runs)

    if args.full:
        all_results["single_user_512"] = bench_single_user(client, model, max_tokens=512, runs=args.runs)
        all_results["output_lengths"] = bench_output_lengths(client, model)
        all_results["reasoning"] = bench_reasoning(client, model)

    # Concurrent test
    levels = args.concurrent or ([1, 3, 5] if args.full else [1, 3])
    all_results["concurrent"] = bench_concurrent(client, model, levels)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    s = all_results["single_user_256"]
    print(f"  Single user (256 tok): {s['avg_tokens_per_sec']} t/s avg, TTFT {s['avg_ttft']}s")
    c = all_results["concurrent"]["results"]
    for r in c:
        print(f"  {r['concurrency']} concurrent: {r['aggregate_tps']} t/s aggregate, {r['avg_latency']}s avg latency")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
