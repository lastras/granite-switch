# SPDX-License-Identifier: Apache-2.0
"""Decode throughput (tok/sec) at varying document (context) lengths.

Companion to ``bench_uq_endpoint.py``, written to answer a question its
numbers raised: that benchmark's single-token (``max_tokens=1``) UQ-readout
latency grows with the shared document's length N even when the document is
fully cached. Is that attention-over-a-long-cache cost showing up per
request, or something else?

This script isolates decode speed from that question directly: for each N,
it warms the document's prefix cache with one throwaway request, then times
a *solo* request that generates ``--gen-tokens`` real tokens
(``ignore_eos=True``, so it always generates exactly that many, never
stopping early on EOS) on top of the now-cached document, and reports
``completion_tokens / latency`` as tok/sec.

Finding (recorded in docs/UQ_ENDPOINT_BENCHMARK.md): tok/sec stays flat
(~38-40) from N=0 to N=8192. Decode speed does not depend on cached context
length, which rules out "attention over the cache costs more per request as
N grows" as the explanation for bench_uq_endpoint.py's N-dependent latency.
See ``solo_vs_batch_latency.py`` for the actual explanation (batch contention
over the shared KV cache, not a per-request cost).

Example::

    python decode_tok_per_sec.py --n 0,128,512,2048,8192 --gen-tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
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


def _complete(endpoint, token, model, prompt_ids, max_tokens, timeout=120):
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
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "latency_s": latency_s,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "finish_reason": out["choices"][0].get("finish_reason"),
    }


def run_one(endpoint, token, model, pool, opener_ids, close_ids, gen_ids, n_doc, gen_tokens):
    nonce = _sample(pool, 8)
    doc = _sample(pool, n_doc)
    shared_prefix = opener_ids + nonce + doc

    # Warm the document (throwaway, not timed).
    _complete(endpoint, token, model, shared_prefix + _sample(pool, 64) + close_ids + gen_ids, max_tokens=1)

    # Timed: a fresh task on the now-cached document, forced to generate
    # exactly gen_tokens.
    prompt_ids = shared_prefix + _sample(pool, 64) + close_ids + gen_ids
    result = _complete(endpoint, token, model, prompt_ids, max_tokens=gen_tokens)
    result["n"] = n_doc
    result["tok_per_s"] = (
        result["completion_tokens"] / result["latency_s"] if result["completion_tokens"] else float("nan")
    )
    return result


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="0,128,512,2048,8192", help="Document sizes (tokens)")
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
    model = _discover_model(args.endpoint, token)
    pool = _build_filler_pool(args.endpoint, token)
    opener_ids = _tokenize(args.endpoint, token, OPENER)
    close_ids = _tokenize(args.endpoint, token, CLOSE)
    gen_ids = _tokenize(args.endpoint, token, GEN_PROMPT)
    print(f"model={model}  pool={len(pool)} ids  opener={len(opener_ids)} close={len(close_ids)} gen={len(gen_ids)}")

    results = []
    print(f"\n{'N':>6} {'latency_s':>10} {'completion_tok':>14} {'prompt_tok':>10} {'cached':>8} {'tok/s':>8}")
    print("-" * 62)
    for n_doc in ns:
        r = run_one(args.endpoint, token, model, pool, opener_ids, close_ids, gen_ids, n_doc, args.gen_tokens)
        results.append(r)
        print(
            f"{r['n']:>6} {r['latency_s']:>10.3f} {r['completion_tokens']!s:>14} "
            f"{r['prompt_tokens']!s:>10} {r['cached_tokens']!s:>8} {r['tok_per_s']:>8.1f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"endpoint": args.endpoint, "model": model, "results": results}, indent=1))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
