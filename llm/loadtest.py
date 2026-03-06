#!/usr/bin/env python3
"""
Load test script for vLLM.
Ramps concurrency from 1 → 1000, measuring total and per-thread tok/s at each level.
Outputs a PNG graph when complete.
"""

import json
import time
import sys
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ────────────────────────────────────────────────────────────
HOST      = "http://localhost:8000"
MODEL     = None   # auto-discovered
PROMPT    = (
    "Explain the concept of entropy in thermodynamics. "
    "Keep your answer to about 100 words."
)
MAX_TOKENS  = 150     # keep consistent across all threads
TEMPERATURE = 0.0     # deterministic → stable token counts
TIMEOUT_S   = 120     # per-request HTTP timeout

# Concurrency levels to test (log-ish ramp 1 → 1000)
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]

OUTPUT_PNG = "loadtest_results.png"
# ─────────────────────────────────────────────────────────────────────────────


def discover_model() -> str:
    resp = requests.get(f"{HOST}/v1/models", timeout=10)
    resp.raise_for_status()
    items = resp.json().get("data") or resp.json().get("models", [])
    if not items:
        raise RuntimeError("No models reported by server")
    first = items[0]
    return first.get("id") or first.get("model") if isinstance(first, dict) else first


def single_request(barrier: threading.Barrier):
    """
    Wait at the barrier then fire one non-streaming request.
    Returns (completion_tokens: int, elapsed_s: float) or None on error.
    """
    barrier.wait()  # all threads start together
    url     = f"{HOST}/v1/chat/completions"
    payload = {
        "model":    MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream":      False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    try:
        resp    = requests.post(url, json=payload, timeout=TIMEOUT_S)
        elapsed = time.monotonic() - t0
        if resp.status_code != 200:
            return None
        tokens = resp.json().get("usage", {}).get("completion_tokens", 0)
        return (tokens, elapsed)
    except Exception:
        return None


def run_level(n: int):
    """
    Spawn n threads simultaneously, collect results.
    Returns (total_tps, avg_individual_tps, success_count, error_count).
    """
    barrier = threading.Barrier(n)
    results = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(single_request, barrier) for _ in range(n)]
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)

    errors = n - len(results)
    if not results:
        return None, None, 0, errors

    # per-thread tok/s
    individual_tps = [tok / elapsed for tok, elapsed in results if elapsed > 0]
    avg_tps = statistics.mean(individual_tps) if individual_tps else 0.0

    # total tok/s = all tokens produced / wall-clock span (max latency across threads)
    total_tokens = sum(tok for tok, _ in results)
    max_elapsed  = max(elapsed for _, elapsed in results)
    total_tps    = total_tokens / max_elapsed if max_elapsed > 0 else 0.0

    return total_tps, avg_tps, len(results), errors


def plot(levels, total_tps_list, avg_tps_list):
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        levels, total_tps_list,
        "o-", color="tab:blue", linewidth=2.5, markersize=7,
        label="Total tok/s  (all threads combined)",
    )
    ax.plot(
        levels, avg_tps_list,
        "s--", color="tab:orange", linewidth=2.5, markersize=7,
        label="Individual tok/s  (avg per thread)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Concurrent Threads", fontsize=13)
    ax.set_ylabel("Tokens / Second", fontsize=13)
    ax.set_title(f"vLLM Load Test  —  {MODEL}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, which="major", alpha=0.35)

    # annotate each point with its value
    for x, y in zip(levels, total_tps_list):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
    for x, y in zip(levels, avg_tps_list):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="tab:orange")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"\n  Graph saved → {OUTPUT_PNG}")


def main():
    global MODEL

    print("Discovering model...")
    MODEL = discover_model()
    print(f"  Model : {MODEL}")
    print(f"  Levels: {LEVELS}")
    print(f"  Prompt: {PROMPT[:60]}...")
    print(f"  Max tokens per request: {MAX_TOKENS}\n")

    header = f"{'Threads':>8}  {'Total tok/s':>12}  {'Indiv tok/s':>12}  {'OK':>5}  {'Err':>5}"
    print(header)
    print("-" * len(header))

    levels_done   = []
    total_tps_list = []
    avg_tps_list   = []

    for n in LEVELS:
        total_tps, avg_tps, ok, err = run_level(n)
        if total_tps is None:
            print(f"{n:>8}  {'FAILED':>12}  {'':>12}  {ok:>5}  {err:>5}")
            continue
        levels_done.append(n)
        total_tps_list.append(total_tps)
        avg_tps_list.append(avg_tps)
        print(f"{n:>8}  {total_tps:>12.1f}  {avg_tps:>12.2f}  {ok:>5}  {err:>5}")

    if not levels_done:
        print("\nNo successful results — nothing to plot.")
        sys.exit(1)

    plot(levels_done, total_tps_list, avg_tps_list)


if __name__ == "__main__":
    main()
