# SPDX-License-Identifier: Apache-2.0
"""Step 3: Fused GEMM+LoRA — Single Adapter.

First fusion attempt: all tokens use the same adapter (no routing complexity).
A single Triton kernel computes output = x @ W.T + x @ A.T @ B.T by
accumulating both the base and LoRA-A products from the same x tiles in
the inner K loop, then multiplying the LoRA-A result by B after the loop.

Usage:
    python kernels/step3_fused.py
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
    compute_gemm_flops,
    compute_lora_flops,
    detect_gpu,
    fmt_overhead,
    fmt_pct,
    fmt_tflops,
    fmt_us,
    latency_to_tflops,
    print_table,
)
from step1_baseline import triton_matmul
from step2_separate import separate_gemm_lora


# ---------------------------------------------------------------------------
# Fused GEMM+LoRA kernel (single adapter, all tokens)
#
# output[m, n] = x[m, :] @ W[n, :].T + x[m, :] @ A[:, :].T @ B[n, :].T
#
# Strategy:
#   - Tile over M (tokens) and N (output features) as normal GEMM
#   - Inner K loop accumulates BOTH:
#       base_acc [BLOCK_M, BLOCK_N] from x @ W.T
#       lora_a_acc [BLOCK_M, RANK] from x @ A.T
#   - After K loop: multiply lora_a_acc @ B_tile.T [RANK, BLOCK_N]
#     and add to base_acc
#   - RANK is a constexpr — fits in registers (32 or 64 values per row)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        # Small M for decode
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "RANK"],
)
@triton.jit
def fused_gemm_lora_kernel(
    # Pointers
    X_ptr, W_ptr, A_ptr, B_ptr, Out_ptr,
    # Dimensions
    M, N, K,
    RANK: tl.constexpr,
    # Strides for X [M, K]
    stride_xm, stride_xk,
    # Strides for W [N, K]
    stride_wn, stride_wk,
    # Strides for A [rank, K]
    stride_ar, stride_ak,
    # Strides for B [N, rank]
    stride_bn, stride_br,
    # Strides for Out [M, N]
    stride_om, stride_on,
    # Meta
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused: Out[M, N] = X[M, K] @ W[N, K].T + X[M, K] @ A[rank, K].T @ B[N, rank].T"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, RANK)

    # Pointers for K-loop
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    a_ptrs = A_ptr + (offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak)

    # Accumulators
    base_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    lora_a_acc = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    # K-loop: accumulate both base GEMM and LoRA-A
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        w_mask = (offs_n[:, None] < N) & ((k_offset + offs_k[None, :]) < K)
        a_mask = (offs_r[:, None] < RANK) & ((k_offset + offs_k[None, :]) < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)          # [BLOCK_M, BLOCK_K]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)          # [BLOCK_N, BLOCK_K]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)          # [RANK, BLOCK_K]

        # Base GEMM: x_tile @ w_tile.T -> [BLOCK_M, BLOCK_N]
        base_acc = tl.dot(x_tile, tl.trans(w_tile), base_acc)

        # LoRA shrink: x_tile @ a_tile.T -> [BLOCK_M, RANK]
        lora_a_acc = tl.dot(x_tile, tl.trans(a_tile), lora_a_acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        a_ptrs += BLOCK_K * stride_ak

    # After K-loop: LoRA expand — lora_a_acc @ B_tile.T
    # Load B tile for this output tile: B[offs_n, :] -> [BLOCK_N, RANK]
    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br)
    b_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_N, RANK]

    # lora_a_acc [BLOCK_M, RANK] @ b_tile.T [RANK, BLOCK_N] -> [BLOCK_M, BLOCK_N]
    lora_delta = tl.dot(lora_a_acc.to(tl.bfloat16), tl.trans(b_tile))

    # Combine
    result = base_acc + lora_delta

    # Store
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=out_mask)


def fused_gemm_lora(
    x: torch.Tensor, W: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor,
) -> torch.Tensor:
    """Fused GEMM+LoRA: x @ W.T + x @ A.T @ B.T in a single kernel.

    Args:
        x: [M, K] input
        W: [N, K] base weight
        lora_A: [rank, K] LoRA down-projection
        lora_B: [N, rank] LoRA up-projection (pre-scaled by alpha/rank)
    """
    M, K = x.shape
    N = W.shape[0]
    rank = lora_A.shape[0]
    output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_gemm_lora_kernel[grid](
        x, W, lora_A, lora_B, output,
        M, N, K, rank,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        lora_B.stride(0), lora_B.stride(1),
        output.stride(0), output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_correctness():
    """Verify fused kernel matches reference computation."""
    print("=" * 70)
    print("Step 3: Correctness Verification")
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

            # Fused kernel
            out = fused_gemm_lora(x, W, lora_A, lora_B)

            ok = check_close(f"{proj.name} rank={rank}", out, ref)
            if not ok:
                all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks():
    """Compare fused kernel to step 1 (base GEMM) and step 2 (separate LoRA)."""
    gpu = detect_gpu()
    print("=" * 70)
    print(f"Step 3: Fused GEMM+LoRA (Single Adapter) — {gpu.name}")
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
                "M (tokens)",
                "Base GEMM (us)",
                "Separate (us)",
                "Fused (us)",
                "Fused vs Base",
                "Fused vs Separate",
                "Fused TFLOPS",
                "% peak",
            ]
            rows = []

            all_m = sorted(set(DECODE_BATCH_SIZES + PREFILL_SEQ_LENS))

            for M in all_m:
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                total_flops = compute_gemm_flops(M, N, K) + compute_lora_flops(M, K, rank, N)

                # Base GEMM only
                base_ms = bench_fn(lambda: triton_matmul(x, W))

                # Separate 3-kernel
                sep_ms = bench_fn(lambda: separate_gemm_lora(x, W, lora_A, lora_B))

                # Fused single kernel
                fused_ms = bench_fn(lambda: fused_gemm_lora(x, W, lora_A, lora_B))

                fused_tflops = latency_to_tflops(fused_ms, total_flops)
                fused_pct = fused_tflops / gpu.bf16_tflops_tc * 100

                rows.append([
                    str(M),
                    fmt_us(base_ms),
                    fmt_us(sep_ms),
                    fmt_us(fused_ms),
                    fmt_overhead(base_ms, fused_ms),
                    fmt_overhead(sep_ms, fused_ms),
                    fmt_tflops(fused_tflops),
                    fmt_pct(fused_pct),
                ])

            print_table(headers, rows, title=f"{proj.name}: ({K} -> {N}), rank={rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_correctness()
    print()
    run_benchmarks()
