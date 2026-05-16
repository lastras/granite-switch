# SPDX-License-Identifier: Apache-2.0
"""Shared benchmarking utilities for the fused linear+LoRA kernel study.

Provides timing wrappers, roofline calculators, table formatters, GPU detection,
and standard projection configs for Granite 8B.
"""

import dataclasses
import math
from typing import Optional

import torch
import triton


# ---------------------------------------------------------------------------
# GPU specs
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GPUSpec:
    name: str
    hbm_bw_tb_s: float        # HBM bandwidth in TB/s
    bf16_tflops_tc: float      # BF16 tensor-core peak TFLOPS
    l2_cache_mb: float         # L2 cache in MB
    num_sms: int               # Number of SMs


GPU_SPECS = {
    "H100": GPUSpec("H100 SXM", hbm_bw_tb_s=3.35, bf16_tflops_tc=1341.0, l2_cache_mb=50, num_sms=132),
    "L40S": GPUSpec("L40S", hbm_bw_tb_s=0.864, bf16_tflops_tc=729.0, l2_cache_mb=192, num_sms=142),
}


def detect_gpu() -> GPUSpec:
    """Detect the current GPU and return its spec, falling back to H100."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — benchmarks require a GPU")
    name = torch.cuda.get_device_name(0).upper()
    if "H100" in name or "H200" in name:
        return GPU_SPECS["H100"]
    if "L40S" in name or "L40" in name:
        return GPU_SPECS["L40S"]
    # Fallback: return H100 with a warning
    print(f"[bench_utils] Unknown GPU '{torch.cuda.get_device_name(0)}', using H100 spec as fallback")
    return GPU_SPECS["H100"]


# ---------------------------------------------------------------------------
# Granite 8B projection configurations
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProjectionConfig:
    """A single linear projection's dimensions."""
    name: str
    in_features: int
    out_features: int


# The four projection shapes in Granite 8B (fused QKV, O, fused gate_up, down)
GRANITE_8B_PROJECTIONS = [
    ProjectionConfig("qkv",     in_features=4096, out_features=5120),
    ProjectionConfig("o_proj",  in_features=4096, out_features=4096),
    ProjectionConfig("gate_up", in_features=4096, out_features=28672),
    ProjectionConfig("down",    in_features=14336, out_features=4096),
]

# Standard batch sizes for benchmarking
DECODE_BATCH_SIZES = [1, 8, 32, 64, 128, 256]
PREFILL_SEQ_LENS = [128, 256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------

def compute_gemm_flops(M: int, N: int, K: int) -> int:
    """FLOPs for a GEMM: 2*M*N*K (multiply-add)."""
    return 2 * M * N * K


def compute_gemm_bytes(M: int, N: int, K: int, dtype_bytes: int = 2) -> int:
    """Bytes transferred for a GEMM (read A, B, write C)."""
    return (M * K + K * N + M * N) * dtype_bytes


def compute_lora_flops(M: int, K: int, rank: int, N: int) -> int:
    """FLOPs for LoRA: shrink (2*M*rank*K) + expand (2*M*N*rank)."""
    return 2 * M * rank * K + 2 * M * N * rank


def compute_lora_bytes(M: int, K: int, rank: int, N: int, dtype_bytes: int = 2) -> int:
    """Bytes for LoRA shrink+expand (reads x, A, intermediate, B; writes intermediate, output-delta)."""
    shrink_bytes = (M * K + rank * K + M * rank) * dtype_bytes
    expand_bytes = (M * rank + N * rank + M * N) * dtype_bytes
    return shrink_bytes + expand_bytes


def arithmetic_intensity(flops: int, bytes_transferred: int) -> float:
    """FLOPs / byte — the operational intensity."""
    return flops / bytes_transferred if bytes_transferred > 0 else float("inf")


def roofline_bound(flops: int, bytes_transferred: int, gpu: GPUSpec) -> tuple[float, float, str]:
    """Return (achievable_tflops, percent_of_peak, bound_type).

    bound_type is 'compute' or 'memory'.
    """
    ai = arithmetic_intensity(flops, bytes_transferred)
    ridge_point = (gpu.bf16_tflops_tc * 1e12) / (gpu.hbm_bw_tb_s * 1e12)  # FLOP/byte
    if ai >= ridge_point:
        achievable = gpu.bf16_tflops_tc
        bound = "compute"
    else:
        achievable = ai * gpu.hbm_bw_tb_s  # TFLOPS
        bound = "memory"
    pct = (achievable / gpu.bf16_tflops_tc) * 100.0
    return achievable, pct, bound


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def bench_fn(fn, warmup: int = 25, rep: int = 100, quantiles=None):
    """Benchmark a callable using triton's do_bench. Returns median ms."""
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    if quantiles is not None:
        return ms  # returns tuple of quantile values
    return ms


def latency_to_tflops(ms: float, flops: int) -> float:
    """Convert latency (ms) and FLOPs to TFLOPS."""
    if ms <= 0:
        return float("inf")
    return flops / (ms * 1e-3) / 1e12


def latency_to_bandwidth(ms: float, bytes_transferred: int) -> float:
    """Convert latency (ms) and bytes to bandwidth in GB/s."""
    if ms <= 0:
        return float("inf")
    return bytes_transferred / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

def check_close(
    name: str,
    result: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Check closeness and print max abs error. Returns True if close."""
    max_abs_err = (result - reference).abs().max().item()
    close = torch.allclose(result, reference, atol=atol, rtol=rtol)
    status = "PASS" if close else "FAIL"
    print(f"  [{status}] {name}: max_abs_err={max_abs_err:.6f} (atol={atol}, rtol={rtol})")
    return close


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table(headers: list[str], rows: list[list], title: Optional[str] = None):
    """Print a markdown-formatted table."""
    if title:
        print(f"\n### {title}\n")
    # Compute column widths
    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        str_row = [str(v) for v in row]
        str_rows.append(str_row)
        for i, v in enumerate(str_row):
            widths[i] = max(widths[i], len(v))

    def fmt_row(vals):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in str_rows:
        print(fmt_row(row))
    print()


def fmt_us(ms: float) -> str:
    """Format milliseconds as microseconds string."""
    return f"{ms * 1000:.1f}"


def fmt_tflops(tflops: float) -> str:
    """Format TFLOPS with 1 decimal."""
    return f"{tflops:.1f}"


def fmt_pct(pct: float) -> str:
    """Format percentage with 1 decimal."""
    return f"{pct:.1f}%"


def fmt_overhead(base_ms: float, with_lora_ms: float) -> str:
    """Format overhead percentage."""
    if base_ms <= 0:
        return "N/A"
    overhead = ((with_lora_ms - base_ms) / base_ms) * 100.0
    return f"{overhead:+.1f}%"
