# SPDX-License-Identifier: Apache-2.0
"""Step 1: Bare GEMM Baseline.

Establishes the performance ceiling — how fast is a pure linear projection?
Implements a Triton matmul kernel and compares against torch.mm (cuBLAS).

Usage:
    python kernels/step1_baseline.py
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
    detect_gpu,
    fmt_overhead,
    fmt_pct,
    fmt_tflops,
    fmt_us,
    latency_to_tflops,
    print_table,
    roofline_bound,
)


# ---------------------------------------------------------------------------
# Triton GEMM kernel: output = x @ W.T
# x: [M, K], W: [N, K] (row-major, transposed access), output: [M, N]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        # Configs tuned for small M (decode)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers
    X_ptr, W_ptr, Out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute output[M, N] = X[M, K] @ W[N, K].T"""
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

    # Pointers to first tiles
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        # Masks
        x_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        w_mask = (offs_n[:, None] < N) & ((k_offset + offs_k[None, :]) < K)
        # Load tiles
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # GEMM: x_tile [BLOCK_M, BLOCK_K] @ w_tile.T [BLOCK_K, BLOCK_N]
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Write output
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def triton_matmul(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Compute x @ W.T using Triton. x: [M, K], W: [N, K] -> output: [M, N]."""
    assert x.dtype == torch.bfloat16 and W.dtype == torch.bfloat16
    M, K = x.shape
    N, K2 = W.shape
    assert K == K2
    output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        x, W, output,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        output.stride(0), output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_correctness():
    """Verify Triton matmul matches torch.mm for all projection shapes."""
    print("=" * 70)
    print("Step 1: Correctness Verification")
    print("=" * 70)
    all_pass = True
    for proj in GRANITE_8B_PROJECTIONS:
        M = 128
        K = proj.in_features
        N = proj.out_features
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.mm(x, W.T)
        out = triton_matmul(x, W)
        ok = check_close(f"{proj.name} ({K}x{N})", out, ref)
        if not ok:
            all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks():
    """Run GEMM benchmarks for all projection shapes and batch sizes."""
    gpu = detect_gpu()
    print("=" * 70)
    print(f"Step 1: Bare GEMM Baseline — {gpu.name}")
    print(f"  BF16 peak: {gpu.bf16_tflops_tc} TFLOPS, HBM BW: {gpu.hbm_bw_tb_s} TB/s")
    print("=" * 70)

    for proj in GRANITE_8B_PROJECTIONS:
        K = proj.in_features
        N = proj.out_features
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        headers = ["M (tokens)", "cuBLAS (us)", "Triton (us)", "cuBLAS TFLOPS", "Triton TFLOPS",
                    "% peak (cuBLAS)", "% peak (Triton)", "Triton vs cuBLAS", "Bound"]
        rows = []

        # Combine decode batch sizes and prefill sequence lengths
        all_m = sorted(set(DECODE_BATCH_SIZES + PREFILL_SEQ_LENS))

        for M in all_m:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            flops = compute_gemm_flops(M, N, K)
            mem_bytes = compute_gemm_bytes(M, N, K, dtype_bytes=2)
            _, _, bound = roofline_bound(flops, mem_bytes, gpu)

            # cuBLAS
            cublas_ms = bench_fn(lambda: torch.mm(x, W.T))
            cublas_tflops = latency_to_tflops(cublas_ms, flops)
            cublas_pct = cublas_tflops / gpu.bf16_tflops_tc * 100

            # Triton
            triton_ms = bench_fn(lambda: triton_matmul(x, W))
            triton_tflops = latency_to_tflops(triton_ms, flops)
            triton_pct = triton_tflops / gpu.bf16_tflops_tc * 100

            rows.append([
                str(M),
                fmt_us(cublas_ms),
                fmt_us(triton_ms),
                fmt_tflops(cublas_tflops),
                fmt_tflops(triton_tflops),
                fmt_pct(cublas_pct),
                fmt_pct(triton_pct),
                fmt_overhead(cublas_ms, triton_ms),
                bound,
            ])

        print_table(headers, rows, title=f"{proj.name}: ({K} -> {N})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_correctness()
    print()
    run_benchmarks()
