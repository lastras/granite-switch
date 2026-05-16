# SPDX-License-Identifier: Apache-2.0
"""Step 2: GEMM + Separate LoRA Kernels.

Measures the overhead of LoRA as separate kernels (simulates current
vLLM/SGLang approach with 3 kernel launches: base GEMM, shrink, expand).

Usage:
    python kernels/step2_separate.py
"""

import torch
import triton
import triton.language as tl

from bench_utils import (
    DECODE_BATCH_SIZES,
    GRANITE_8B_PROJECTIONS,
    PREFILL_SEQ_LENS,
    bench_fn,
    check_close,
    compute_gemm_bytes,
    compute_gemm_flops,
    compute_lora_flops,
    detect_gpu,
    fmt_overhead,
    fmt_pct,
    fmt_tflops,
    fmt_us,
    latency_to_tflops,
    print_table,
    roofline_bound,
)
from step1_baseline import triton_matmul


# ---------------------------------------------------------------------------
# Triton LoRA shrink kernel: intermediate = x @ lora_A.T
# x: [M, K], lora_A: [rank, K] -> intermediate: [M, rank]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_stages=5, num_warps=2),
    ],
    key=["M", "RANK", "K"],
)
@triton.jit
def lora_shrink_kernel(
    X_ptr, A_ptr, Out_ptr,
    M, RANK: tl.constexpr, K,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_om, stride_or,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute Out[M, rank] = X[M, K] @ A[rank, K].T"""
    pid_m = tl.program_id(0)
    # rank is small enough to process in one tile for rank <= 64
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = tl.arange(0, RANK)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    a_ptrs = A_ptr + (offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak)

    acc = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        a_mask = (offs_r[:, None] < RANK) & ((k_offset + offs_k[None, :]) < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        acc = tl.dot(x_tile, tl.trans(a_tile), acc)
        x_ptrs += BLOCK_K * stride_xk
        a_ptrs += BLOCK_K * stride_ak

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_r[None, :] * stride_or)
    out_mask = (offs_m[:, None] < M) & (offs_r[None, :] < RANK)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# ---------------------------------------------------------------------------
# Triton LoRA expand kernel: output += intermediate @ lora_B.T
# intermediate: [M, rank], lora_B: [N, rank] -> output: [M, N] (in-place add)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "RANK"],
)
@triton.jit
def lora_expand_add_kernel(
    Inter_ptr, B_ptr, Out_ptr,
    M, N, RANK: tl.constexpr,
    stride_im, stride_ir,
    stride_bn, stride_br,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Compute Out[M, N] += Inter[M, rank] @ B[N, rank].T"""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)

    # Load intermediate tile [BLOCK_M, rank] — rank fits in one load
    inter_ptrs = Inter_ptr + (offs_m[:, None] * stride_im + offs_r[None, :] * stride_ir)
    inter_mask = (offs_m[:, None] < M) & (offs_r[None, :] < RANK)
    inter_tile = tl.load(inter_ptrs, mask=inter_mask, other=0.0)

    # Load B tile [BLOCK_N, rank]
    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br)
    b_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Compute: inter [BLOCK_M, rank] @ b.T [rank, BLOCK_N]
    delta = tl.dot(inter_tile, tl.trans(b_tile))

    # Load existing output and add
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    existing = tl.load(out_ptrs, mask=out_mask, other=0.0)
    result = existing.to(tl.float32) + delta
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=out_mask)


def triton_lora_shrink(x: torch.Tensor, lora_A: torch.Tensor) -> torch.Tensor:
    """Compute x @ lora_A.T. x: [M, K], lora_A: [rank, K] -> [M, rank]."""
    M, K = x.shape
    rank = lora_A.shape[0]
    intermediate = torch.empty((M, rank), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    lora_shrink_kernel[grid](
        x, lora_A, intermediate,
        M, rank, K,
        x.stride(0), x.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        intermediate.stride(0), intermediate.stride(1),
    )
    return intermediate


def triton_lora_expand_add(intermediate: torch.Tensor, lora_B: torch.Tensor, output: torch.Tensor):
    """Compute output += intermediate @ lora_B.T in-place."""
    M, rank = intermediate.shape
    N = lora_B.shape[0]
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    lora_expand_add_kernel[grid](
        intermediate, lora_B, output,
        M, N, rank,
        intermediate.stride(0), intermediate.stride(1),
        lora_B.stride(0), lora_B.stride(1),
        output.stride(0), output.stride(1),
    )


# ---------------------------------------------------------------------------
# Combined separate-kernels path
# ---------------------------------------------------------------------------

def separate_gemm_lora(
    x: torch.Tensor, W: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor,
) -> torch.Tensor:
    """3-kernel LoRA: base GEMM + shrink + expand."""
    output = triton_matmul(x, W)
    intermediate = triton_lora_shrink(x, lora_A)
    triton_lora_expand_add(intermediate, lora_B, output)
    return output


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_correctness():
    """Verify separate LoRA kernels produce correct results."""
    print("=" * 70)
    print("Step 2: Correctness Verification")
    print("=" * 70)
    all_pass = True
    for rank in [32, 64]:
        for proj in GRANITE_8B_PROJECTIONS:
            M = 128
            K = proj.in_features
            N = proj.out_features
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            lora_A = torch.randn(rank, K, device="cuda", dtype=torch.bfloat16) * 0.01
            lora_B = torch.randn(N, rank, device="cuda", dtype=torch.bfloat16) * 0.01

            # Reference
            ref = torch.mm(x, W.T) + torch.mm(torch.mm(x, lora_A.T), lora_B.T)

            # Kernel
            out = separate_gemm_lora(x, W, lora_A, lora_B)

            ok = check_close(f"{proj.name} rank={rank}", out, ref)
            if not ok:
                all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks():
    """Run benchmarks comparing base GEMM vs GEMM+separate LoRA."""
    gpu = detect_gpu()
    print("=" * 70)
    print(f"Step 2: GEMM + Separate LoRA Kernels — {gpu.name}")
    print(f"  BF16 peak: {gpu.bf16_tflops_tc} TFLOPS, HBM BW: {gpu.hbm_bw_tb_s} TB/s")
    print("=" * 70)

    for rank in [32, 64]:
        print(f"\n{'='*70}")
        print(f"  LoRA rank = {rank}")
        print(f"{'='*70}")

        for proj in GRANITE_8B_PROJECTIONS:
            K = proj.in_features
            N = proj.out_features
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            lora_A = torch.randn(rank, K, device="cuda", dtype=torch.bfloat16) * 0.01
            lora_B = torch.randn(N, rank, device="cuda", dtype=torch.bfloat16) * 0.01

            headers = [
                "M (tokens)", "Base GEMM (us)", "GEMM+LoRA (us)", "Overhead",
                "Shrink (us)", "Expand (us)", "LoRA total (us)",
                "Base TFLOPS", "Total TFLOPS",
            ]
            rows = []

            all_m = sorted(set(DECODE_BATCH_SIZES + PREFILL_SEQ_LENS))

            for M in all_m:
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                base_flops = compute_gemm_flops(M, N, K)
                lora_flops = compute_lora_flops(M, K, rank, N)
                total_flops = base_flops + lora_flops

                # Base GEMM only
                base_ms = bench_fn(lambda: triton_matmul(x, W))

                # Individual LoRA kernels
                shrink_ms = bench_fn(lambda: triton_lora_shrink(x, lora_A))
                output_buf = triton_matmul(x, W)
                expand_ms = bench_fn(lambda: triton_lora_expand_add(
                    torch.randn(M, rank, device="cuda", dtype=torch.bfloat16), lora_B, output_buf
                ))
                lora_total_ms = shrink_ms + expand_ms

                # Full 3-kernel path
                total_ms = bench_fn(lambda: separate_gemm_lora(x, W, lora_A, lora_B))

                base_tflops = latency_to_tflops(base_ms, base_flops)
                total_tflops = latency_to_tflops(total_ms, total_flops)

                rows.append([
                    str(M),
                    fmt_us(base_ms),
                    fmt_us(total_ms),
                    fmt_overhead(base_ms, total_ms),
                    fmt_us(shrink_ms),
                    fmt_us(expand_ms),
                    fmt_us(lora_total_ms),
                    fmt_tflops(base_tflops),
                    fmt_tflops(total_tflops),
                ])

            print_table(headers, rows, title=f"{proj.name}: ({K} -> {N}), rank={rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_correctness()
    print()
    run_benchmarks()
