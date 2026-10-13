# SPDX-License-Identifier: Apache-2.0
"""Solo warm single-token latency vs. a k-way concurrent batch, by document length.

Companion to ``bench_uq_endpoint.py``. That benchmark's ``primed_split``
"parallel" phase showed per-request latency growing with the shared
document's length N (e.g. mean ~0.48s at N=128 vs ~0.77s at N=8192, inside a
k=32 concurrent batch, all with ``cached_tokens`` ~= N). ``decode_tok_per_sec.py``
shows decode speed itself doesn't depend on N, so that growth isn't a
per-request attention-over-cache cost. This script isolates the real cause:
is a *solo* warm single-token request also slower at larger N, or is the
growth only visible when many requests contend for the same cached document
at once?

For each N: warm the document, then measure (a) one solo UQ-style
``max_tokens=1`` request on a fresh task against it, and (b) a k-way
concurrent batch of the same kind of request against the same document.

Finding (recorded in docs/UQ_ENDPOINT_BENCHMARK.md): solo latency is cheap
and flat (~0.12-0.17s) regardless of N. Only the batched case grows with N
(batch mean 0.50s -> 0.84s from N=128 to N=8192). The growth is batch
contention over the shared KV cache (aggregate memory bandwidth scales with
batch_size x N), not a per-request cost -- correcting an earlier assumption
that attention over a long cache was expensive per request.

Example::

    python solo_vs_batch_latency.py --n 128,2048,8192 --batch-k 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_uq_endpoint import (
    CLOSE,
    DEFAULT_ENDPOINT,
    OPENER,
    SUFFIX,
    _build_filler_pool,
    _discover_model,
    _post,
    _sample,
    _tokenize,
)


def _complete(endpoint, token, model, prompt_ids, timeout=120):
    payload = {"model": model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0.0}
    t0 = time.perf_counter()
    out = _post(endpoint, "/v1/completions", token, payload, timeout=timeout)
    latency_s = time.perf_counter() - t0
    usage = out.get("usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"latency_s": latency_s, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached}


def run_one(endpoint, token, model, pool, opener_ids, close_ids, suffix_ids, n_doc, batch_k):
    nonce = _sample(pool, 8)
    doc = _sample(pool, n_doc)
    shared_prefix = opener_ids + nonce + doc

    # Warm the document (throwaway, not timed).
    _complete(endpoint, token, model, shared_prefix + _sample(pool, 64) + close_ids + suffix_ids)

    # Solo warm request: a brand-new task, same cached document, no concurrency.
    solo = _complete(endpoint, token, model, shared_prefix + _sample(pool, 64) + close_ids + suffix_ids)

    # Batch of batch_k warm requests: same cached document, concurrent.
    batch_prompts = [shared_prefix + _sample(pool, 64) + close_ids + suffix_ids for _ in range(batch_k)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_k) as ex:
        batch = [f.result() for f in [ex.submit(_complete, endpoint, token, model, p) for p in batch_prompts]]
    batch_wall_s = time.perf_counter() - t0
    lats = [r["latency_s"] for r in batch]

    return {
        "n": n_doc,
        "solo_s": solo["latency_s"],
        "solo_cached": solo["cached_tokens"],
        "batch_wall_s": batch_wall_s,
        "batch_mean_s": sum(lats) / len(lats),
        "batch_min_s": min(lats),
        "batch_max_s": max(lats),
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="128,2048,8192", help="Document sizes (tokens)")
    ap.add_argument("--batch-k", type=int, default=32, help="Concurrent requests in the batch arm")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--output", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Set {args.token_env} in the environment")

    ns = [int(x) for x in args.n.split(",") if x]
    model = _discover_model(args.endpoint, token)
    pool = _build_filler_pool(args.endpoint, token)
    opener_ids = _tokenize(args.endpoint, token, OPENER)
    close_ids = _tokenize(args.endpoint, token, CLOSE)
    suffix_ids = _tokenize(args.endpoint, token, SUFFIX)

    results = []
    print(
        f"{'N':>6} {'solo_s':>8} {'cached':>8}  |  "
        f"{'batch_wall_s':>12} {'batch_mean_s':>12} {'batch_min_s':>12} {'batch_max_s':>12}"
    )
    print("-" * 80)
    for n_doc in ns:
        r = run_one(args.endpoint, token, model, pool, opener_ids, close_ids, suffix_ids, n_doc, args.batch_k)
        results.append(r)
        print(
            f"{r['n']:>6} {r['solo_s']:>8.3f} {r['solo_cached']!s:>8}  |  "
            f"{r['batch_wall_s']:>12.3f} {r['batch_mean_s']:>12.3f} {r['batch_min_s']:>12.3f} {r['batch_max_s']:>12.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"endpoint": args.endpoint, "model": model, "results": results}, indent=1))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
