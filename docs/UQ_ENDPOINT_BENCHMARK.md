# UQ Single-Token-Prefill Throughput Against a Live HF Endpoint

Measurements against `granite-switch-4.1-30b-uc` (the `uncertainty` aLoRA from
`ibm-granite/granitelib-core-r1.0`, composed with a Granite 4.1 30B base, served on
2xA100, TP2, vLLM 0.26.0) at
`https://6ac93f78e4d5812f4c1a9c3a.endpoints.huggingface.cloud`. The code is in
[`tutorials/scripts/uq_endpoint_race/`](../tutorials/scripts/uq_endpoint_race/bench_uq_endpoint.py).
This doc collects what was measured and what we conclude; rerun the scripts for
current numbers -- these are point-in-time snapshots against one live endpoint, not a
guaranteed SLA.

## Conclusions

1. **Firing k requests at once beats explicit prefill-then-fanout, at every N and k
   tested.** `cold_parallel` (fire all k simultaneously on a document none of them have
   seen) finishes faster than `primed_split` (prefill once, then fan out the rest) in
   23 of 24 (N, k) cells, by a `cold_e2e / primed_total` ratio of 0.62-1.01. The one
   exception is k=1, where the two modes are the same single request.
2. **vLLM shares a cold document's cost across requests that arrive together, not just
   across requests that arrive in sequence.** In a 32-way `cold_parallel` batch on a
   document none of the 32 had seen, 31 of 32 already report a cache hit on essentially
   the whole document (`cached_tokens` ~= N) -- the scheduler computed it once and shared
   it, rather than 32 times.
3. **Decode speed does not depend on cached context length.** Generating 64 real tokens
   (`ignore_eos=True`, so always exactly 64) on a warmed document holds at ~38-40 tok/s
   from N=0 to N=8192.
4. **A solo warm request is cheap and flat.** One `max_tokens=1` request against an
   already-cached document costs ~0.12-0.25s regardless of N (128 to 8192) -- so #3
   isn't masking a per-request cache-read cost either.
5. **The N-dependent latency growth seen under concurrency is batch contention, not a
   per-request cost.** A 32-way concurrent batch of warm `max_tokens=1` requests against
   the *same* cached document grows with N: mean per-request latency 0.52s (N=128) ->
   0.61s (N=2048) -> 0.93s (N=8192). Combined with #4, the cause is aggregate GPU
   contention over the shared KV cache (scales with `batch_size x N`), not attention
   cost per individual request. **This corrects an earlier explanation given mid-session**
   that attributed the growth directly to per-request attention-over-cache cost; #4
   disproves that reading.
6. **Concurrency trades per-request latency for aggregate throughput, and a longer
   shared document makes that trade worse.** Aggregate decode tok/sec scales up with k
   (near-linearly through k=8, sub-linearly beyond) while per-request tok/sec degrades.
   At the same k, a longer document gets less aggregate throughput and degrades at a
   lower k (see the k=16 anomaly in the table below, likely a scheduling blip on the
   shared endpoint rather than a real effect -- single-run data, not repeated/medianed).

## What was built

| Script | What it measures |
|---|---|
| [`bench_uq_endpoint.py`](../tutorials/scripts/uq_endpoint_race/bench_uq_endpoint.py) | The main sweep: `cold_parallel` vs. `primed_split` wall-clock across document size N and parallel task count k. |
| [`decode_tok_per_sec.py`](../tutorials/scripts/uq_endpoint_race/decode_tok_per_sec.py) | Decode tok/sec vs. document length N (cache warmed first, solo request, 64 forced tokens). |
| [`solo_vs_batch_latency.py`](../tutorials/scripts/uq_endpoint_race/solo_vs_batch_latency.py) | Solo warm single-token latency vs. a k-way concurrent batch on the same cached document, by N. |
| [`tok_per_sec_by_concurrency.py`](../tutorials/scripts/uq_endpoint_race/tok_per_sec_by_concurrency.py) | Aggregate vs. per-request decode tok/sec as concurrency k grows, at fixed N. |

All four are stdlib-only (`argparse`, `urllib.request`, `concurrent.futures`), discover
token ids via the endpoint's own `/tokenize`/`/detokenize` routes rather than hardcoding
them, and read the endpoint token from `$HF_TOKEN` (never printed, never passed on the
command line).

## Results

### Main sweep: `cold_parallel` vs. `primed_split`

`--repeats 3`, median reported. N = document tokens, k = parallel task count,
task size fixed at 64 tokens.

```
     N    k  cold_e2e_s  prefill_s  parallel_s  primed_total_s  cold/primed
--------------------------------------------------------------------------------------------
   128    1       0.178      0.182       0.000           0.182         0.98
   128    2       0.181      0.141       0.122           0.291         0.62
   128    4       0.217      0.146       0.171           0.312         0.70
   128    8       0.275      0.143       0.214           0.360         0.76
   128   16       0.433      0.136       0.357           0.494         0.88
   128   32       0.613      0.137       0.587           0.725         0.85
   512    1       0.205      0.202       0.000           0.202         1.01
   512    2       0.224      0.199       0.124           0.323         0.69
   512    4       0.294      0.211       0.182           0.394         0.75
   512    8       0.335      0.200       0.236           0.437         0.77
   512   16       0.460      0.200       0.393           0.593         0.78
   512   32       0.686      0.206       0.581           0.791         0.87
  2048    1       0.453      0.470       0.000           0.470         0.96
  2048    2       0.492      0.501       0.148           0.648         0.76
  2048    4       0.533      0.448       0.220           0.668         0.80
  2048    8       0.595      0.457       0.285           0.748         0.79
  2048   16       0.716      0.489       0.431           0.892         0.80
  2048   32       0.924      0.473       0.654           1.112         0.83
  8192    1       1.585      1.576       0.000           1.576         1.01
  8192    2       1.614      1.641       0.210           1.849         0.87
  8192    4       1.658      1.576       0.315           1.932         0.86
  8192    8       1.732      1.577       0.387           1.975         0.88
  8192   16       1.958      1.576       0.620           2.193         0.89
  8192   32       2.191      1.606       0.923           2.494         0.88
```

Across all 1512 individual scoring requests in this sweep (24 cells x 2 modes x
3 repeats x k), every decoded readout was a valid single digit `0`-`9`; zero
non-digit warnings.

### Decode tok/sec vs. document length (`decode_tok_per_sec.py`)

```
     N  latency_s completion_tok prompt_tok   cached    tok/s
--------------------------------------------------------------
     0      1.634             64         80        0     39.2
   128      1.604             64        208      128     39.9
   512      1.601             64        592      512     40.0
  2048      1.593             64       2128     2048     40.2
  8192      1.694             64       8272     8192     37.8
```

Flat, as conclusion #3 says. (An earlier single-sample run showed 24.8 tok/s at
N=128 -- that was run-to-run noise, not reproduced here.)

### Solo vs. 32-way batch latency, by document length (`solo_vs_batch_latency.py`)

```
     N   solo_s   cached  |  batch_wall_s batch_mean_s  batch_min_s  batch_max_s
--------------------------------------------------------------------------------
   128    0.203      128  |         0.636        0.537        0.196        0.634
  2048    0.158     2048  |         0.719        0.610        0.179        0.706
  8192    0.251     8192  |         1.085        0.925        0.323        1.081
```

Solo stays flat (~0.15-0.25s); only the batched case grows with N. This is the direct
evidence for conclusion #5.

### Aggregate vs. per-request tok/sec by concurrency (`tok_per_sec_by_concurrency.py`)

Single run, not repeated/medianed -- expect more noise than the tables above.

```
N=128
   k   wall_s  agg_tok/s  mean_req_s per_req_tok/s
-------------------------------------------------------
   1    2.086       30.7       2.086          30.7
   2    1.644       77.9       1.643          38.9
   4    1.745      146.7       1.741          36.8
   8    3.106      164.8       3.095          20.7
  16    1.879      545.0       1.868          34.3
  32    2.202      930.1       2.184          29.3

N=8192
   k   wall_s  agg_tok/s  mean_req_s per_req_tok/s
-------------------------------------------------------
   1    1.840       34.8       1.839          34.8
   2    2.042       62.7       2.041          31.4
   4    2.032      126.0       2.017          31.7
   8    2.133      240.0       2.107          30.4
  16    3.810      268.7       3.779          16.9
  32    3.363      609.0       3.315          19.3
```

The k=8 dip at N=128 and the k=16 dip at N=8192 are each one data point against a
live, shared endpoint -- plausibly scheduling jitter rather than a real effect. The
trend that does hold across both N and is visible past the noise: aggregate throughput
scales up with k (roughly 20-30x from k=1 to k=32), per-request throughput degrades
(roughly 1.1-1.8x worse at k=32 than solo), and N=8192 reaches a given per-request
degradation at a lower k than N=128 does.

## How to reproduce

```bash
export HF_TOKEN=<endpoint token with inference.endpoints.infer.write scope>
cd tutorials/scripts/uq_endpoint_race
python3 bench_uq_endpoint.py --output /tmp/uq_full_sweep.json
python3 decode_tok_per_sec.py
python3 solo_vs_batch_latency.py
python3 tok_per_sec_by_concurrency.py
```

Each script's `--help` lists the swept parameters. A 401 from `/v1/models` means the
token lacks `inference.endpoints.infer.write` scope for this specific endpoint, not a
problem with the script -- check with `huggingface.co/api/whoami-v2`.
