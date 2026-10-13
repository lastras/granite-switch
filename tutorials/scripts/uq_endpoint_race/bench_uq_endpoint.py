# SPDX-License-Identifier: Apache-2.0
"""Single-token UQ scoring throughput against a live HF Inference Endpoint.

The scenario: one long shared **document** (N tokens), then **k** short
independent **assertions** about it (fixed at ``--task-size`` tokens each),
each scored by its own call to the ``uncertainty`` aLoRA. N and k are swept;
the per-task size stays fixed. The target metric is end-to-end wall-clock for
all k scores, but this script also reports how that time splits between
prefilling the shared document once and fanning the k task scores out over
it -- because whether truly-concurrent requests that share an uncached prefix
actually share the prefill (vLLM's cascade/prefix-cache behavior) or
redundantly recompute it is the open question this is built to measure.

The trick: ``uncertainty`` emits a tiny fixed JSON scaffold, ``{"score":
"D"}``, where D is a single calibrated digit. Both the control token and the
scaffold are deterministic, so instead of letting the model generate them,
this script prefills them and asks for exactly one generated token (the
digit), via ``/v1/completions`` with ``prompt`` as token ids and
``max_tokens=1``. The prompt shape, per task::

    <|start_of_role|>user<|end_of_role|>
      <nonce> <N random document tokens> <task-size random assertion tokens>
    <|end_of_text|>
    <|uncertainty|>assistant<|end_of_role|>{"score": "

``opener + nonce + document`` is the shared prefix a cell wants cached across
its k requests; ``assertion + close + suffix`` is the per-task tail. The
nonce is a handful of random tokens unique to each (N, k, mode) trial, so no
two trials' shared-prefix content ever collides and contaminates each other's
cache -- the only lever available, since vLLM exposes no HTTP cache-flush call.

Scaffold token ids (the role markers, the control token, the JSON opener) are
discovered at run time via the endpoint's own ``/tokenize`` and
``/detokenize`` routes, not hardcoded, so a different checkpoint/vocab fails
loudly (an assertion) rather than silently mis-scoring. Document and
assertion content is deliberately random, drawn from a filler token pool
(itself discovered by tokenizing a fixed paragraph and dropping any token
whose decoded text contains a control-token marker) -- this is a throughput
benchmark, not a correctness one.

Two modes run for every (N, k) cell:

  cold_parallel    Fire all k task requests simultaneously (same nonce'd
                    document, distinct random assertions) via a thread pool
                    sized to k. One number: wall-clock from just-before-submit
                    to last-response -- what actually happens if k requests
                    arrive from an external caller, with no sequencing we impose.

  primed_split      A different nonce. Send task #0 alone first (pays the
                    document's cost once; timed as ``prefill_s``), then once
                    it returns, fire tasks #1..k-1 in parallel (timed as
                    ``parallel_s``, should be small if caching under
                    concurrency works). ``primed_total_s = prefill_s +
                    parallel_s``.

Dependency-free (stdlib only): no local tokenizer/transformers download needed.

Examples::

    python bench_uq_endpoint.py --n 128 --k 1 --repeats 1
    python bench_uq_endpoint.py --n 128,2048 --k 1,4 --repeats 1
    python bench_uq_endpoint.py --output out/uq_race.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_ENDPOINT = "https://6ac93f78e4d5812f4c1a9c3a.endpoints.huggingface.cloud"

# The aLoRA placement convention this mirrors (control token in place of the
# role marker that would otherwise open the assistant turn) is confirmed by
# this project's own granite_switch/conversation.py; here it is reproduced
# directly as fixed strings since this script has no local tokenizer for the
# composed checkpoint and only needs one fixed turn, not multi-turn history.
OPENER = "<|start_of_role|>user<|end_of_role|>"
CLOSE = "<|end_of_text|>\n"
SUFFIX = '<|uncertainty|>assistant<|end_of_role|>{"score": "'
UNCERTAINTY_TOKEN = "<|uncertainty|>"

NONCE_LEN = 8  # random tokens prepended per trial so no two trials' shared
# document content can ever collide, guaranteeing a cold cache per trial.

# Neutral, non-meaningful content for the filler token pool. Its content
# never matters -- only that it tokenizes to a reasonable diversity of plain
# (non-control) tokens to sample from.
FILLER_PARAGRAPH = (
    "The committee reviewed the quarterly budget and found that spending on "
    "maintenance had increased modestly while travel costs declined. Staff "
    "proposed a revised schedule for equipment inspections, noting that most "
    "delays traced back to parts availability rather than labor shortages. "
    "The warehouse team suggested consolidating orders across departments to "
    "reduce shipping fees, and the finance office agreed to pilot the change "
    "for one quarter before expanding it further. Overall attendance at the "
    "monthly meeting was steady, and no major objections were raised to the "
    "proposed changes. A follow-up review is scheduled for next spring."
)


# ── HTTP plumbing (stdlib only, mirrors tests/integration/_conversation_concurrency_worker.py) ──
def _request(method, endpoint, path, token, payload=None, timeout=120):
    url = endpoint.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {body[:500]}") from None


def _get(endpoint, path, token, timeout=30):
    return _request("GET", endpoint, path, token, timeout=timeout)


def _post(endpoint, path, token, payload, timeout=120):
    return _request("POST", endpoint, path, token, payload, timeout=timeout)


def _discover_model(endpoint, token):
    data = _get(endpoint, "/v1/models", token)
    models = data.get("data", [])
    if not models:
        raise RuntimeError("endpoint reports no served models at /v1/models")
    return models[0]["id"]


def _tokenize(endpoint, token, text):
    # Bare {"prompt": ...} -- verified live to return a clean tokenization with
    # no injected special tokens (e.g. no BOS), which this script's whole
    # approach of tokenizing fragments separately and concatenating ids
    # depends on. Adding "model" or "add_special_tokens" here is unverified
    # against this endpoint, so this mirrors the live-confirmed shape exactly.
    out = _post(endpoint, "/tokenize", token, {"prompt": text})
    return out["tokens"]


def _detokenize(endpoint, token, ids):
    out = _post(endpoint, "/detokenize", token, {"tokens": ids})
    return out["prompt"]


# ── Scaffold + filler pool (startup, not timed) ──────────────────────────────
class Scaffold:
    """Fixed token ids around the random content, discovered at run time."""

    def __init__(self, endpoint, token):
        self.opener = _tokenize(endpoint, token, OPENER)
        self.close = _tokenize(endpoint, token, CLOSE)
        self.suffix = _tokenize(endpoint, token, SUFFIX)

        unc = _tokenize(endpoint, token, UNCERTAINTY_TOKEN)
        if len(unc) != 1:
            raise RuntimeError(
                f"{UNCERTAINTY_TOKEN!r} tokenized to {len(unc)} tokens, expected "
                "exactly 1 -- wrong checkpoint/vocab for this endpoint?"
            )
        self.uncertainty_id = unc[0]
        if self.suffix[0] != self.uncertainty_id or (self.suffix.count(self.uncertainty_id) != 1):
            raise RuntimeError(
                "SUFFIX does not start with exactly one <|uncertainty|> control "
                f"token (got ids {self.suffix[:3]}..., expected a leading "
                f"{self.uncertainty_id} occurring once) -- wrong checkpoint/vocab?"
            )


def _build_filler_pool(endpoint, token):
    """Token ids from a fixed neutral paragraph, minus any decoding to a control token."""
    ids = _tokenize(endpoint, token, FILLER_PARAGRAPH)
    pool = [tid for tid in ids if "<|" not in _detokenize(endpoint, token, [tid])]
    if not pool:
        raise RuntimeError("filler pool is empty after dropping control tokens")
    return pool


# ── Prompt construction ──────────────────────────────────────────────────────
def _sample(pool, n):
    return random.choices(pool, k=n)


def build_shared_prefix(scaffold, pool, n_doc):
    """``opener + nonce + document`` -- the prefix a cell wants cached across its k requests."""
    return scaffold.opener + _sample(pool, NONCE_LEN) + _sample(pool, n_doc)


def build_full_prompt(scaffold, shared_prefix, pool, task_size):
    """``shared_prefix + assertion + close + suffix`` -- one full request's token ids."""
    return shared_prefix + _sample(pool, task_size) + scaffold.close + scaffold.suffix


# ── One scoring call ──────────────────────────────────────────────────────────
def _complete(endpoint, token, model, prompt_ids, timeout=120):
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    out = _post(endpoint, "/v1/completions", token, payload, timeout=timeout)
    latency_s = time.perf_counter() - t0
    choice = out["choices"][0]
    usage = out.get("usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "latency_s": latency_s,
        "text": choice.get("text", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
    }


def _check_digits(results):
    for r in results:
        d = r["text"].strip()
        if len(d) != 1 or not d.isdigit():
            print(f"  WARNING: non-digit readout {d!r}", file=sys.stderr)


# ── The two modes ─────────────────────────────────────────────────────────────
def run_cold_parallel(endpoint, token, model, scaffold, pool, n_doc, k, task_size):
    """Fire all k task requests on a fresh, never-before-seen document, at once."""
    shared_prefix = build_shared_prefix(scaffold, pool, n_doc)
    prompts = [build_full_prompt(scaffold, shared_prefix, pool, task_size) for _ in range(k)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=k) as ex:
        results = [f.result() for f in [ex.submit(_complete, endpoint, token, model, p) for p in prompts]]
    cold_e2e_s = time.perf_counter() - t0
    return cold_e2e_s, results


def run_primed_split(endpoint, token, model, scaffold, pool, n_doc, k, task_size):
    """A different fresh document: send task #0 alone, then #1..k-1 in parallel."""
    shared_prefix = build_shared_prefix(scaffold, pool, n_doc)

    prompt0 = build_full_prompt(scaffold, shared_prefix, pool, task_size)
    t0 = time.perf_counter()
    result0 = _complete(endpoint, token, model, prompt0)
    prefill_s = time.perf_counter() - t0

    rest_prompts = [build_full_prompt(scaffold, shared_prefix, pool, task_size) for _ in range(k - 1)]
    if rest_prompts:
        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(rest_prompts)) as ex:
            rest_results = [
                f.result() for f in [ex.submit(_complete, endpoint, token, model, p) for p in rest_prompts]
            ]
        parallel_s = time.perf_counter() - t1
    else:
        rest_results, parallel_s = [], 0.0

    return prefill_s, parallel_s, prefill_s + parallel_s, [result0, *rest_results]


# ── One (N, k) cell, repeated and medianed ───────────────────────────────────
def run_cell(endpoint, token, model, scaffold, pool, n_doc, k, task_size, repeats):
    cold_vals, prefill_vals, parallel_vals, primed_vals = [], [], [], []
    records = {"cold_parallel": [], "primed_split": []}

    for _ in range(repeats):
        cold_e2e_s, cold_results = run_cold_parallel(endpoint, token, model, scaffold, pool, n_doc, k, task_size)
        cold_vals.append(cold_e2e_s)
        records["cold_parallel"].append(cold_results)
        _check_digits(cold_results)

        prefill_s, parallel_s, primed_total_s, primed_results = run_primed_split(
            endpoint, token, model, scaffold, pool, n_doc, k, task_size
        )
        prefill_vals.append(prefill_s)
        parallel_vals.append(parallel_s)
        primed_vals.append(primed_total_s)
        records["primed_split"].append(primed_results)
        _check_digits(primed_results)

    return {
        "n": n_doc,
        "k": k,
        "cold_e2e_s": statistics.median(cold_vals),
        "prefill_s": statistics.median(prefill_vals),
        "parallel_s": statistics.median(parallel_vals),
        "primed_total_s": statistics.median(primed_vals),
        "records": records,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────
def _ratio(cell):
    return cell["cold_e2e_s"] / cell["primed_total_s"] if cell["primed_total_s"] > 0 else float("nan")


def print_table(cells):
    print("\n" + "=" * 92)
    print(
        f"{'N':>6} {'k':>4} {'cold_e2e_s':>11} {'prefill_s':>10} "
        f"{'parallel_s':>11} {'primed_total_s':>15} {'cold/primed':>12}"
    )
    print("-" * 92)
    for cell in cells:
        print(
            f"{cell['n']:>6} {cell['k']:>4} {cell['cold_e2e_s']:>11.3f} "
            f"{cell['prefill_s']:>10.3f} {cell['parallel_s']:>11.3f} "
            f"{cell['primed_total_s']:>15.3f} {_ratio(cell):>12.2f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="128,512,2048,8192", help="Document sizes (tokens)")
    ap.add_argument("--k", default="1,2,4,8,16,32", help="Parallel per-task assertions")
    ap.add_argument("--task-size", type=int, default=64, help="Tokens per assertion (fixed, not swept)")
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per (N,k) cell per mode; report median")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--token-env", default="HF_TOKEN", help="Env var holding the endpoint token")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON dump path")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(
            f"Set {args.token_env} in the environment to the endpoint's token (never pass it on the command line)"
        )

    ns = [int(x) for x in args.n.split(",") if x]
    ks = [int(x) for x in args.k.split(",") if x]
    random.seed(args.seed)

    print(f"Endpoint: {args.endpoint}")
    model = _discover_model(args.endpoint, token)
    print(f"Model: {model}")

    scaffold = Scaffold(args.endpoint, token)
    print(
        f"Scaffold ids: opener={len(scaffold.opener)} close={len(scaffold.close)} "
        f"suffix={len(scaffold.suffix)} (uncertainty token id={scaffold.uncertainty_id})"
    )

    pool = _build_filler_pool(args.endpoint, token)
    print(f"Filler pool: {len(pool)} usable token ids (control tokens excluded)")

    cells = []
    for n_doc in ns:
        for k in ks:
            print(
                f"\n== N={n_doc} k={k} (task_size={args.task_size}, repeats={args.repeats}) ==",
                flush=True,
            )
            cell = run_cell(args.endpoint, token, model, scaffold, pool, n_doc, k, args.task_size, args.repeats)
            cells.append(cell)
            print(
                f"  cold_e2e={cell['cold_e2e_s']:.3f}s  prefill={cell['prefill_s']:.3f}s  "
                f"parallel={cell['parallel_s']:.3f}s  primed_total={cell['primed_total_s']:.3f}s  "
                f"cold/primed={_ratio(cell):.2f}",
                flush=True,
            )

    print_table(cells)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"endpoint": args.endpoint, "model": model, "cells": cells}, indent=1))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
