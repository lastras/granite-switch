# SPDX-License-Identifier: Apache-2.0
"""Step 5: Packed Projections — QKV and gate_up.

Handles fused QKV (3 output slices) and gate_up (2 output slices) cases
where lora_A is shared across slices but lora_B differs per slice.

The key insight: in packed projections, the LoRA down-projection (A) is
the same for all output slices because it only depends on the input.
We compute x @ A.T once, then apply different B matrices for each slice.

Usage:
    python kernels/step5_packed.py
"""

import torch
import triton
import triton.language as tl

from bench_utils import (
    DECODE_BATCH_SIZES,
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
from step3_fused import fused_gemm_lora


# ---------------------------------------------------------------------------
# Packed projection configs
# ---------------------------------------------------------------------------

# QKV: shared A, separate B for Q (4096), K (1024), V (1024) = total out 5120
QKV_CONFIG = {
    "name": "qkv",
    "in_features": 4096,
    "total_out": 5120,
    "slices": [4096, 1024, 1024],  # Q, K, V output sizes (after GQA: 32*128, 8*128, 8*128)
    "slice_names": ["Q", "K", "V"],
}

# gate_up: shared A, separate B for gate (14336) and up (14336) = total out 28672
GATE_UP_CONFIG = {
    "name": "gate_up",
    "in_features": 4096,
    "total_out": 28672,
    "slices": [14336, 14336],
    "slice_names": ["gate", "up"],
}


# ---------------------------------------------------------------------------
# Fused GEMM+LoRA kernel for packed projections (shared A)
#
# The kernel computes one output slice at a time but shares the LoRA-A
# accumulation across all slices by computing it once and reusing.
#
# For the packed case, we tile over the TOTAL output dimension and determine
# which slice each N-tile belongs to, loading the appropriate B slice.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K", "RANK"],
)
@triton.jit
def fused_gemm_lora_packed_kernel(
    # Pointers
    X_ptr, W_ptr, A_ptr, B_ptr, Out_ptr,
    # B slice offset table: [num_slices] cumulative offsets into B's N dimension
    B_offsets_ptr,
    # Dimensions
    M, N, K,
    RANK: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    # Strides for X [M, K]
    stride_xm, stride_xk,
    # Strides for W [N, K] (N is the total packed output dimension)
    stride_wn, stride_wk,
    # Strides for A [rank, K] (shared across slices)
    stride_ar, stride_ak,
    # Strides for B_packed [total_N, rank] (concatenated slices)
    stride_bn, stride_br,
    # Strides for Out [M, N]
    stride_om, stride_on,
    # Meta
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused GEMM+LoRA for packed projections.

    W is [total_N, K] (packed QKV or gate_up).
    A is [rank, K] (shared across all output slices).
    B is [total_N, rank] (packed — each slice's B occupies its own N range).

    The kernel tiles over the full N dimension; the LoRA-B applied for each
    N-tile is just B[offs_n, :] since B is already packed contiguously.
    """
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

    # K-loop: accumulate base GEMM and shared LoRA-A
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        w_mask = (offs_n[:, None] < N) & ((k_offset + offs_k[None, :]) < K)
        a_mask = (offs_r[:, None] < RANK) & ((k_offset + offs_k[None, :]) < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        base_acc = tl.dot(x_tile, tl.trans(w_tile), base_acc)
        lora_a_acc = tl.dot(x_tile, tl.trans(a_tile), lora_a_acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        a_ptrs += BLOCK_K * stride_ak

    # After K-loop: apply LoRA-B for this N-tile
    # B is packed: B[offs_n, :] gives the correct slice's B values
    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br)
    b_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # lora_a_acc [BLOCK_M, RANK] @ b_tile.T [RANK, BLOCK_N]
    lora_delta = tl.dot(lora_a_acc.to(tl.bfloat16), tl.trans(b_tile))
    result = base_acc + lora_delta

    # Store
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=out_mask)


def fused_gemm_lora_packed(
    x: torch.Tensor,
    W: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B_packed: torch.Tensor,
) -> torch.Tensor:
    """Fused GEMM+LoRA for packed projections with shared A.

    Args:
        x: [M, K] input
        W: [total_N, K] packed base weight (e.g., QKV or gate_up fused)
        lora_A: [rank, K] shared LoRA down-projection
        lora_B_packed: [total_N, rank] packed B (slices concatenated along N)
    """
    M, K = x.shape
    N = W.shape[0]
    rank = lora_A.shape[0]
    num_slices = 1  # Not used in the kernel but kept for API clarity
    output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    # B offsets (not actually needed since B is contiguously packed)
    b_offsets = torch.zeros(1, device=x.device, dtype=torch.int64)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_gemm_lora_packed_kernel[grid](
        x, W, lora_A, lora_B_packed, output,
        b_offsets,
        M, N, K, rank, num_slices,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        lora_B_packed.stride(0), lora_B_packed.stride(1),
        output.stride(0), output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Separate-slice baseline: compute each slice independently with step3 kernel
# ---------------------------------------------------------------------------

def separate_slice_gemm_lora(
    x: torch.Tensor,
    W: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B_slices: list[torch.Tensor],
    slice_sizes: list[int],
) -> torch.Tensor:
    """Baseline: run fused GEMM+LoRA separately per slice.

    W is split into slices along N, each with its own B but shared A.
    """
    M, K = x.shape
    total_N = sum(slice_sizes)
    output = torch.empty((M, total_N), device=x.device, dtype=torch.bfloat16)

    offset = 0
    for i, (slice_n, b_slice) in enumerate(zip(slice_sizes, lora_B_slices)):
        W_slice = W[offset : offset + slice_n]
        out_slice = fused_gemm_lora(x, W_slice, lora_A, b_slice)
        output[:, offset : offset + slice_n] = out_slice
        offset += slice_n

    return output


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_correctness():
    """Verify packed kernel matches reference."""
    print("=" * 70)
    print("Step 5: Correctness Verification")
    print("=" * 70)
    all_pass = True

    for config in [QKV_CONFIG, GATE_UP_CONFIG]:
        K = config["in_features"]
        total_N = config["total_out"]
        slices = config["slices"]

        for rank in [32, 64]:
            M = 128
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(total_N, K, device="cuda", dtype=torch.bfloat16)
            lora_A = torch.randn(rank, K, device="cuda", dtype=torch.bfloat16) * 0.01

            # Create packed B: [total_N, rank]
            lora_B_packed = torch.randn(total_N, rank, device="cuda", dtype=torch.bfloat16) * 0.01

            # Also create slice list for separate baseline
            lora_B_slices = []
            offset = 0
            for s in slices:
                lora_B_slices.append(lora_B_packed[offset : offset + s].contiguous())
                offset += s

            # Reference: x @ W.T + x @ A.T @ B_packed.T
            ref = torch.mm(x, W.T) + torch.mm(torch.mm(x, lora_A.T), lora_B_packed.T)

            # Packed fused kernel
            out_packed = fused_gemm_lora_packed(x, W, lora_A, lora_B_packed)
            ok = check_close(f"{config['name']} packed rank={rank}", out_packed, ref)
            if not ok:
                all_pass = False

            # Separate-slice baseline
            out_sep = separate_slice_gemm_lora(x, W, lora_A, lora_B_slices, slices)
            ok = check_close(f"{config['name']} separate rank={rank}", out_sep, ref)
            if not ok:
                all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks():
    """Benchmark packed fused kernel vs separate-slice handling."""
    gpu = detect_gpu()
    print("=" * 70)
    print(f"Step 5: Packed Projections (Shared LoRA-A) — {gpu.name}")
    print(f"  BF16 peak: {gpu.bf16_tflops_tc} TFLOPS, HBM BW: {gpu.hbm_bw_tb_s} TB/s")
    print("=" * 70)

    for config in [QKV_CONFIG, GATE_UP_CONFIG]:
        K = config["in_features"]
        total_N = config["total_out"]
        slices = config["slices"]
        num_slices = len(slices)

        for rank in [32, 64]:
            print(f"\n{'='*70}")
            print(f"  {config['name']}: ({K} -> {total_N}), {num_slices} slices {slices}, rank={rank}")
            print(f"{'='*70}")

            W = torch.randn(total_N, K, device="cuda", dtype=torch.bfloat16)
            lora_A = torch.randn(rank, K, device="cuda", dtype=torch.bfloat16) * 0.01
            lora_B_packed = torch.randn(total_N, rank, device="cuda", dtype=torch.bfloat16) * 0.01

            lora_B_slices = []
            offset = 0
            for s in slices:
                lora_B_slices.append(lora_B_packed[offset : offset + s].contiguous())
                offset += s

            headers = [
                "M (tokens)",
                "Base GEMM (us)",
                "Separate slices (us)",
                "Packed fused (us)",
                "Packed vs Base",
                "Packed vs Separate",
                "Packed TFLOPS",
                "% peak",
            ]
            rows = []

            all_m = sorted(set(DECODE_BATCH_SIZES + PREFILL_SEQ_LENS))

            for M in all_m:
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

                # FLOPs: base GEMM + LoRA (shared A across slices = 1 shrink + N_slices expands)
                base_flops = compute_gemm_flops(M, total_N, K)
                # LoRA with shared A: shrink once (M, K, rank) + expand over total_N
                lora_flops = 2 * M * rank * K + 2 * M * total_N * rank
                total_flops = base_flops + lora_flops

                # Base GEMM only (no LoRA)
                base_ms = bench_fn(lambda: triton_matmul(x, W))

                # Separate slices (step3 fused per slice)
                sep_ms = bench_fn(
                    lambda: separate_slice_gemm_lora(x, W, lora_A, lora_B_slices, slices)
                )

                # Packed fused (single kernel, shared A)
                packed_ms = bench_fn(lambda: fused_gemm_lora_packed(x, W, lora_A, lora_B_packed))

                packed_tflops = latency_to_tflops(packed_ms, total_flops)
                packed_pct = packed_tflops / gpu.bf16_tflops_tc * 100

                rows.append([
                    str(M),
                    fmt_us(base_ms),
                    fmt_us(sep_ms),
                    fmt_us(packed_ms),
                    fmt_overhead(base_ms, packed_ms),
                    fmt_overhead(sep_ms, packed_ms),
                    fmt_tflops(packed_tflops),
                    fmt_pct(packed_pct),
                ])

            print_table(headers, rows, title=f"{config['name']}: ({K} -> {total_N}), rank={rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_correctness()
    print()
    run_benchmarks()
