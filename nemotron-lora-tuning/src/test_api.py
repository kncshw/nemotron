#!/usr/bin/env python3
"""
API test client for Nemotron vLLM endpoint (OpenAI-compatible).
Works with both L4 mock (Nano-4B) and GB10 production (Super-120B).

Usage:
    python3 test_api.py                    # basic chat test
    python3 test_api.py --all              # run all tests
    python3 test_api.py --tool-use         # tool-use test only
    python3 test_api.py --reasoning        # reasoning mode test
    python3 test_api.py --streaming        # streaming test
    python3 test_api.py --concurrent N     # concurrent user simulation
    python3 test_api.py --base-url URL     # custom endpoint
"""

import argparse
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def get_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="not-needed")


def get_model(client: OpenAI) -> str:
    """Auto-detect the served model name."""
    models = client.models.list()
    model_id = models.data[0].id
    print(f"  Model: {model_id}")
    return model_id


# ---------- Test: Basic Chat ----------

def test_chat(client: OpenAI, model: str):
    print("\n=== Test: Basic Chat ===")
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful cybersecurity assistant."},
            {"role": "user", "content": "Explain what a reverse shell is in 2-3 sentences."},
        ],
        max_tokens=256,
        temperature=0.7,
    )
    elapsed = time.time() - start
    msg = resp.choices[0].message.content
    tokens = resp.usage.completion_tokens
    print(f"  Response ({tokens} tokens, {elapsed:.1f}s, {tokens/elapsed:.1f} t/s):")
    print(f"  {msg[:1000]}")
    return True


# ---------- Test: Tool Use ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nmap_scan",
            "description": "Run an nmap scan against a target host",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target IP or hostname"},
                    "ports": {"type": "string", "description": "Port range, e.g. '1-1000'"},
                    "scan_type": {
                        "type": "string",
                        "enum": ["syn", "tcp", "udp", "service"],
                        "description": "Type of scan to run",
                    },
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cve",
            "description": "Search for CVE vulnerabilities by keyword or product name",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (product, CVE ID, or keyword)"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Filter by severity",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def test_tool_use(client: OpenAI, model: str):
    print("\n=== Test: Tool Use ===")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a penetration testing assistant. Use the available tools when appropriate."},
            {"role": "user", "content": "I need to scan 192.168.1.100 for open ports and check if any services have known CVEs."},
        ],
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=512,
    )
    choice = resp.choices[0]
    if choice.message.tool_calls:
        print(f"  Tool calls made: {len(choice.message.tool_calls)}")
        for tc in choice.message.tool_calls:
            print(f"    -> {tc.function.name}({tc.function.arguments})")
        return True
    else:
        print(f"  No tool calls (model responded with text instead)")
        print(f"  Response: {choice.message.content[:3000]}")
        return False


# ---------- Test: Streaming ----------

def test_streaming(client: OpenAI, model: str):
    print("\n=== Test: Streaming ===")
    start = time.time()
    token_count = 0
    first_token_time = None

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "List 5 common web application vulnerabilities with one-line descriptions."},
        ],
        max_tokens=256,
        stream=True,
    )

    chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            chunks.append(chunk.choices[0].delta.content)
            token_count += 1

    elapsed = time.time() - start
    ttft = (first_token_time - start) if first_token_time else 0
    text = "".join(chunks)
    print(f"  TTFT: {ttft:.2f}s | Total: {elapsed:.1f}s | ~{token_count} chunks")
    print(f"  {text[:1000]}")
    return True


# ---------- Test: Reasoning ----------

def test_reasoning(client: OpenAI, model: str):
    print("\n=== Test: Reasoning Mode ===")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Think step by step: A web server is running Apache 2.4.49. What vulnerability should I check for and why?"},
        ],
        max_tokens=512,
        temperature=0.6,
    )
    msg = resp.choices[0].message.content
    print(f"  Response ({resp.usage.completion_tokens} tokens):")
    print(f"  {msg[:1000]}")
    return True


# ---------- Test: Concurrent Users ----------

def _single_request(client: OpenAI, model: str, user_id: int) -> dict:
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"User {user_id}: What is SQL injection? Answer in 2 sentences."},
        ],
        max_tokens=128,
    )
    elapsed = time.time() - start
    tokens = resp.usage.completion_tokens
    return {"user": user_id, "tokens": tokens, "time": elapsed, "tps": tokens / elapsed}


def test_concurrent(client: OpenAI, model: str, num_users: int = 3):
    print(f"\n=== Test: Concurrent Users ({num_users}) ===")
    with ThreadPoolExecutor(max_workers=num_users) as pool:
        futures = {pool.submit(_single_request, client, model, i): i for i in range(num_users)}
        results = []
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            print(f"  User {r['user']}: {r['tokens']} tokens in {r['time']:.1f}s ({r['tps']:.1f} t/s)")

    total_tokens = sum(r["tokens"] for r in results)
    max_time = max(r["time"] for r in results)
    print(f"  Aggregate: {total_tokens} tokens, {total_tokens/max_time:.1f} t/s effective throughput")
    return True


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Test vLLM Nemotron API endpoint")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--chat", action="store_true", help="Basic chat test")
    parser.add_argument("--tool-use", action="store_true", help="Tool-use test")
    parser.add_argument("--streaming", action="store_true", help="Streaming test")
    parser.add_argument("--reasoning", action="store_true", help="Reasoning mode test")
    parser.add_argument("--concurrent", type=int, metavar="N", help="Concurrent user test")
    args = parser.parse_args()

    # Default to chat if nothing specified
    if not any([args.all, args.chat, args.tool_use, args.streaming, args.reasoning, args.concurrent]):
        args.chat = True

    client = get_client(args.base_url)
    print(f"Connecting to {args.base_url}...")
    model = get_model(client)

    results = {}
    if args.all or args.chat:
        results["chat"] = test_chat(client, model)
    if args.all or args.tool_use:
        results["tool_use"] = test_tool_use(client, model)
    if args.all or args.streaming:
        results["streaming"] = test_streaming(client, model)
    if args.all or args.reasoning:
        results["reasoning"] = test_reasoning(client, model)
    if args.all or args.concurrent:
        n = args.concurrent if args.concurrent else 3
        results["concurrent"] = test_concurrent(client, model, n)

    print("\n=== Summary ===")
    for test, passed in results.items():
        status = "PASS" if passed else "WARN"
        print(f"  {test}: {status}")


if __name__ == "__main__":
    main()
