# SPDX-License-Identifier: Apache-2.0
"""Aggregate vs. per-request decode tok/sec as concurrency grows.

Companion to ``bench_uq_endpoint.py`` and ``solo_vs_batch_latency.py``. The
latter shows that per-request latency against a shared cached document grows
with concurrency (batch contention over the shared KV cache); this script
puts a number on the resulting throughput/latency tradeoff: for a fixed,
pre-warmed document of length N, how do aggregate throughput (total tokens
generated across all k concurrent requests / wall-clock) and per-request
throughput (``gen_tokens / mean individual latency``) move as k grows?

Each of the k concurrent requests generates ``--gen-tokens`` real tokens
(``ignore_eos=True``) against the same cached document, so this measures
genuine decode throughput, not single-token prefill latency.

Finding (recorded in docs/UQ_ENDPOINT_BENCHMARK.md): aggregate tok/sec scales
up with k (near-linearly through k=8, sub-linearly beyond), while
per-request tok/sec degrades (e.g. at N=8192: 38 tok/s solo -> 19 tok/s at
k=32). The larger N, the earlier this degradation sets in and the lower the
achieved aggregate throughput at a given k -- the standard throughput/latency
tradeoff of batched serving, sharpened by a longer shared KV cache.

Example::

    python tok_per_sec_by_concurrency.py --n 128,8192 --k 1,2,4,8,16,32
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
    _build_filler_pool,
    _discover_model,
    _post,
    _sample,
    _tokenize,
)

GEN_PROMPT = "<|start_of_role|>assistant<|end_of_role|>"


def _complete(endpoint, token, model, prompt_ids, max_tokens, timeout=180):
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    out = _post(endpoint, "/v1/completions", token, payload, timeout=timeout)
    latency_s = time.perf_counter() - t0
    usage = out.get("usage", {}) or {}
    return latency_s, usage.get("completion_tokens") or 0


def run_n(endpoint, token, model, pool, opener_ids, close_ids, gen_ids, n_doc, ks, gen_tokens):
    nonce = _sample(pool, 8)
    doc = _sample(pool, n_doc)
    shared_prefix = opener_ids + nonce + doc

    # Warm the document (throwaway, not timed).
    _complete(endpoint, token, model, shared_prefix + _sample(pool, 64) + close_ids + gen_ids, max_tokens=1)

    rows = []
    for k in ks:
        prompts = [shared_prefix + _sample(pool, 64) + close_ids + gen_ids for _ in range(k)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as ex:
            results = [
                f.result() for f in [ex.submit(_complete, endpoint, token, model, p, gen_tokens) for p in prompts]
            ]
        wall_s = time.perf_counter() - t0
        total_toks = sum(toks for _, toks in results)
        mean_req_s = sum(lat for lat, _ in results) / len(results)
        rows.append(
            {
                "n": n_doc,
                "k": k,
                "wall_s": wall_s,
                "agg_tok_s": total_toks / wall_s,
                "mean_req_s": mean_req_s,
                "per_req_tok_s": gen_tokens / mean_req_s,
            }
        )
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="128,8192", help="Document sizes (tokens)")
    ap.add_argument("--k", default="1,2,4,8,16,32", help="Concurrency levels")
    ap.add_argument("--gen-tokens", type=int, default=64, help="Tokens to force-generate per request")
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
    ks = [int(x) for x in args.k.split(",") if x]
    model = _discover_model(args.endpoint, token)
    pool = _build_filler_pool(args.endpoint, token)
    opener_ids = _tokenize(args.endpoint, token, OPENER)
    close_ids = _tokenize(args.endpoint, token, CLOSE)
    gen_ids = _tokenize(args.endpoint, token, GEN_PROMPT)

    all_rows = []
    for n_doc in ns:
        print(f"\nN={n_doc}")
        print(f"{'k':>4} {'wall_s':>8} {'agg_tok/s':>10} {'mean_req_s':>11} {'per_req_tok/s':>13}")
        print("-" * 55)
        rows = run_n(args.endpoint, token, model, pool, opener_ids, close_ids, gen_ids, n_doc, ks, args.gen_tokens)
        all_rows.extend(rows)
        for r in rows:
            print(
                f"{r['k']:>4} {r['wall_s']:>8.3f} {r['agg_tok_s']:>10.1f} "
                f"{r['mean_req_s']:>11.3f} {r['per_req_tok_s']:>13.1f}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"endpoint": args.endpoint, "model": model, "rows": all_rows}, indent=1))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
