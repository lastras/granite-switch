# SPDX-License-Identifier: Apache-2.0
"""Step 4: Fused GEMM+LoRA — Per-Token Adapter Routing.

Adds per-token adapter selection to the fused kernel. Each token in a batch
may use a different adapter (or no adapter at all, index 0 = base only).

Two strategies:
  A) Load all adapters: each tile handles mixed adapters by loading all
     lora_A weights and selecting per-row via adapter_indices.
  B) Sorted tokens: pre-sort tokens by adapter so each tile is pure
     single-adapter, then scatter-back output.

Usage:
    python kernels/step4_routing.py
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
from step3_fused import fused_gemm_lora


# ---------------------------------------------------------------------------
# Strategy A: Fused GEMM+LoRA with per-token adapter selection
#
# Each tile loads x tiles and W tiles as normal. For LoRA, we load lora_A
# for ALL adapters and use adapter_indices to select per-row.
#
# lora_A: [num_adapters, rank, K] — adapter 0..num_adapters-1
# lora_B: [num_adapters, N, rank]
# adapter_indices: [M] — 0 = base only, 1+ = adapter index (1-based)
#
# For the K-loop, we accumulate lora_A for each adapter separately,
# then mask-select per row. Since num_adapters is small (4) and rank is
# modest (32-64), we keep separate accumulators per adapter.
#
# After the K-loop, for each row we pick the right lora_a_acc based on
# adapter_indices, multiply by B, and add to base_acc.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K", "RANK", "NUM_ADAPTERS"],
)
@triton.jit
def fused_gemm_lora_routed_kernel(
    # Pointers
    X_ptr, W_ptr, A_ptr, B_ptr, Idx_ptr, Out_ptr,
    # Dimensions
    M, N, K,
    RANK: tl.constexpr,
    NUM_ADAPTERS: tl.constexpr,
    # Strides for X [M, K]
    stride_xm, stride_xk,
    # Strides for W [N, K]
    stride_wn, stride_wk,
    # Strides for A [num_adapters, rank, K]
    stride_aa, stride_ar, stride_ak,
    # Strides for B [num_adapters, N, rank]
    stride_ba, stride_bn, stride_br,
    # Strides for Out [M, N]
    stride_om, stride_on,
    # Meta
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused GEMM+LoRA with per-token adapter routing (Strategy A).

    adapter_indices[t] == 0 means base-only (no LoRA for that token).
    adapter_indices[t] == i (1-based) selects adapter i-1 (0-indexed into A/B).
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

    # Load adapter indices for this tile
    idx_ptrs = Idx_ptr + offs_m
    idx_mask = offs_m < M
    adapter_idx = tl.load(idx_ptrs, mask=idx_mask, other=0)  # [BLOCK_M], 0-based: 0=no adapter

    # Check if any token in this tile needs LoRA
    any_lora = tl.max(adapter_idx) > 0

    # Pointers for K-loop
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # Accumulators
    base_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Single LoRA-A accumulator — we'll use the most common adapter in the tile
    # For simplicity with constexpr RANK, accumulate for all tokens then mask
    lora_a_acc = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    if any_lora:
        # Find the dominant adapter (most frequent non-zero) for A loading
        # For strategy A, we iterate over adapters after K-loop
        # During K-loop, accumulate x @ A.T for each adapter's A weights
        # Since adapters may differ per-row, we accumulate ONE set of
        # lora_a products and handle multi-adapter via iteration after K-loop

        # For the common case (all tokens same adapter), this is efficient.
        # For mixed case, we iterate over active adapters.

        # K-loop: accumulate base GEMM and all-tokens LoRA-A for adapter 0 (the first active one)
        # We'll handle multi-adapter after the K-loop via iteration
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_offset = k * BLOCK_K
            x_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
            w_mask = (offs_n[:, None] < N) & ((k_offset + offs_k[None, :]) < K)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
            base_acc = tl.dot(x_tile, tl.trans(w_tile), base_acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # After K-loop: compute LoRA contribution per adapter
        # Re-read x for LoRA-A computation (we need the full K dimension again)
        # This is the cost of routing — we re-read x tiles
        for adapter_i in range(NUM_ADAPTERS):
            # Check if any token uses this adapter (1-based, so adapter_i+1)
            has_adapter = tl.max((adapter_idx == (adapter_i + 1)).to(tl.int32)) > 0
            if has_adapter:
                # Compute x @ A[adapter_i].T for all tokens
                a_base = A_ptr + adapter_i * stride_aa
                a_ptrs_inner = a_base + (offs_r[:, None] * stride_ar + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
                x_ptrs_inner = X_ptr + (offs_m[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :] * stride_xk)

                lora_a_local = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BLOCK_K)):
                    k_offset = k * BLOCK_K
                    x_mask_inner = (offs_m[:, None] < M) & ((k_offset + tl.arange(0, BLOCK_K)[None, :]) < K)
                    a_mask_inner = (offs_r[:, None] < RANK) & ((k_offset + tl.arange(0, BLOCK_K)[None, :]) < K)
                    x_tile_inner = tl.load(x_ptrs_inner, mask=x_mask_inner, other=0.0)
                    a_tile_inner = tl.load(a_ptrs_inner, mask=a_mask_inner, other=0.0)
                    lora_a_local = tl.dot(x_tile_inner, tl.trans(a_tile_inner), lora_a_local)
                    x_ptrs_inner += BLOCK_K * stride_xk
                    a_ptrs_inner += BLOCK_K * stride_ak

                # Load B[adapter_i] for this output tile
                b_base = B_ptr + adapter_i * stride_ba
                b_ptrs = b_base + (offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br)
                b_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

                # lora_a_local @ B.T -> [BLOCK_M, BLOCK_N]
                lora_delta = tl.dot(lora_a_local.to(tl.bfloat16), tl.trans(b_tile))

                # Mask: only apply to rows that use this adapter
                row_mask = (adapter_idx == (adapter_i + 1))[:, None]  # [BLOCK_M, 1]
                base_acc += tl.where(row_mask, lora_delta, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
    else:
        # Pure base-only path — no LoRA overhead
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_offset = k * BLOCK_K
            x_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
            w_mask = (offs_n[:, None] < N) & ((k_offset + offs_k[None, :]) < K)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
            base_acc = tl.dot(x_tile, tl.trans(w_tile), base_acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

    # Store output
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, base_acc.to(tl.bfloat16), mask=out_mask)


def fused_gemm_lora_routed(
    x: torch.Tensor,
    W: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_indices: torch.Tensor,
) -> torch.Tensor:
    """Fused GEMM+LoRA with per-token routing (Strategy A).

    Args:
        x: [M, K] input
        W: [N, K] base weight
        lora_A: [num_adapters, rank, K]
        lora_B: [num_adapters, N, rank] (pre-scaled)
        adapter_indices: [M] int64, 0=base, 1+=adapter (1-based)
    """
    M, K = x.shape
    N = W.shape[0]
    num_adapters = lora_A.shape[0]
    rank = lora_A.shape[1]
    output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    fused_gemm_lora_routed_kernel[grid](
        x, W, lora_A, lora_B, adapter_indices, output,
        M, N, K, rank, num_adapters,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        lora_A.stride(0), lora_A.stride(1), lora_A.stride(2),
        lora_B.stride(0), lora_B.stride(1), lora_B.stride(2),
        output.stride(0), output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Strategy B: Sort tokens by adapter, run single-adapter fused kernel per
# group, scatter-back results.
# ---------------------------------------------------------------------------

def fused_gemm_lora_sorted(
    x: torch.Tensor,
    W: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_indices: torch.Tensor,
) -> torch.Tensor:
    """Strategy B: sort tokens by adapter, run fused kernel per group.

    Args:
        x: [M, K] input
        W: [N, K] base weight
        lora_A: [num_adapters, rank, K]
        lora_B: [num_adapters, N, rank] (pre-scaled)
        adapter_indices: [M] int64, 0=base, 1+=adapter (1-based)
    """
    M, K = x.shape
    N = W.shape[0]

    # Sort tokens by adapter index
    sorted_idx = torch.argsort(adapter_indices, stable=True)
    x_sorted = x[sorted_idx]
    idx_sorted = adapter_indices[sorted_idx]

    output_sorted = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    # Find group boundaries
    unique_adapters, counts = torch.unique_consecutive(idx_sorted, return_counts=True)

    offset = 0
    for adapter_id, count in zip(unique_adapters.tolist(), counts.tolist()):
        x_group = x_sorted[offset : offset + count]
        if adapter_id == 0:
            # Base-only group
            out_group = triton_matmul(x_group, W)
        else:
            # LoRA group — use step3 fused kernel with single adapter
            a_idx = adapter_id - 1  # Convert to 0-based
            out_group = fused_gemm_lora(x_group, W, lora_A[a_idx], lora_B[a_idx])
        output_sorted[offset : offset + count] = out_group
        offset += count

    # Scatter back to original order
    output = torch.empty_like(output_sorted)
    output[sorted_idx] = output_sorted
    return output


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def reference_routed_lora(
    x: torch.Tensor,
    W: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation: per-token LoRA with torch operations."""
    output = torch.mm(x, W.T)
    for t in range(x.shape[0]):
        idx = adapter_indices[t].item()
        if idx > 0:
            a = lora_A[idx - 1]  # [rank, K]
            b = lora_B[idx - 1]  # [N, rank]
            intermediate = x[t:t+1] @ a.T  # [1, rank]
            output[t:t+1] += intermediate @ b.T  # [1, N]
    return output


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_correctness():
    """Verify both strategies match reference."""
    print("=" * 70)
    print("Step 4: Correctness Verification")
    print("=" * 70)
    all_pass = True
    num_adapters = 4
    for rank in [32, 64]:
        # Use a smaller projection for verification speed
        K, N = 4096, 4096
        M = 64
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        lora_A = torch.randn(num_adapters, rank, K, device="cuda", dtype=torch.bfloat16) * 0.01
        lora_B = torch.randn(num_adapters, N, rank, device="cuda", dtype=torch.bfloat16) * 0.01

        # Test several routing patterns
        patterns = {
            "all_base": torch.zeros(M, device="cuda", dtype=torch.int64),
            "all_adapter_1": torch.ones(M, device="cuda", dtype=torch.int64),
            "mixed_50_50": torch.cat([
                torch.zeros(M // 2, dtype=torch.int64),
                torch.ones(M // 2, dtype=torch.int64),
            ]).cuda(),
            "all_different": torch.arange(M, device="cuda", dtype=torch.int64) % (num_adapters + 1),
        }

        for pattern_name, adapter_indices in patterns.items():
            ref = reference_routed_lora(x, W, lora_A, lora_B, adapter_indices)

            # Strategy A
            out_a = fused_gemm_lora_routed(x, W, lora_A, lora_B, adapter_indices)
            ok = check_close(f"Strategy A, rank={rank}, {pattern_name}", out_a, ref)
            if not ok:
                all_pass = False

            # Strategy B
            out_b = fused_gemm_lora_sorted(x, W, lora_A, lora_B, adapter_indices)
            ok = check_close(f"Strategy B, rank={rank}, {pattern_name}", out_b, ref)
            if not ok:
                all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def make_adapter_indices(M: int, pattern: str, num_adapters: int = 4) -> torch.Tensor:
    """Create adapter index tensors for different routing patterns."""
    if pattern == "all_base":
        return torch.zeros(M, device="cuda", dtype=torch.int64)
    elif pattern == "single_adapter":
        return torch.ones(M, device="cuda", dtype=torch.int64)
    elif pattern == "mixed_50_50":
        idx = torch.zeros(M, device="cuda", dtype=torch.int64)
        idx[M // 2:] = 1
        return idx
    elif pattern == "all_different":
        return (torch.arange(M, device="cuda", dtype=torch.int64) % (num_adapters + 1))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def run_benchmarks():
    """Benchmark Strategy A vs B across routing patterns."""
    gpu = detect_gpu()
    num_adapters = 4
    print("=" * 70)
    print(f"Step 4: Per-Token Adapter Routing — {gpu.name}")
    print(f"  BF16 peak: {gpu.bf16_tflops_tc} TFLOPS, HBM BW: {gpu.hbm_bw_tb_s} TB/s")
    print(f"  num_adapters: {num_adapters}")
    print("=" * 70)

    patterns = ["all_base", "single_adapter", "mixed_50_50", "all_different"]

    for rank in [32, 64]:
        print(f"\n{'='*70}")
        print(f"  LoRA rank = {rank}")
        print(f"{'='*70}")

        for proj in GRANITE_8B_PROJECTIONS:
            K = proj.in_features
            N = proj.out_features
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            lora_A = torch.randn(num_adapters, rank, K, device="cuda", dtype=torch.bfloat16) * 0.01
            lora_B = torch.randn(num_adapters, N, rank, device="cuda", dtype=torch.bfloat16) * 0.01

            headers = [
                "M", "Pattern",
                "Base (us)", "Step3 fused (us)", "Strategy A (us)", "Strategy B (us)",
                "A vs Base", "A vs Step3", "B vs Base", "B vs Step3",
            ]
            rows = []

            all_m = sorted(set(DECODE_BATCH_SIZES + PREFILL_SEQ_LENS))

            for M in all_m:
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                # Single-adapter weights for step3 comparison
                single_A = lora_A[0]  # [rank, K]
                single_B = lora_B[0]  # [N, rank]

                # Base GEMM
                base_ms = bench_fn(lambda: triton_matmul(x, W))

                # Step3 fused (single adapter, all tokens)
                step3_ms = bench_fn(lambda: fused_gemm_lora(x, W, single_A, single_B))

                for pattern in patterns:
                    adapter_indices = make_adapter_indices(M, pattern, num_adapters)

                    # Strategy A
                    a_ms = bench_fn(lambda: fused_gemm_lora_routed(x, W, lora_A, lora_B, adapter_indices))

                    # Strategy B
                    b_ms = bench_fn(lambda: fused_gemm_lora_sorted(x, W, lora_A, lora_B, adapter_indices))

                    rows.append([
                        str(M), pattern,
                        fmt_us(base_ms), fmt_us(step3_ms), fmt_us(a_ms), fmt_us(b_ms),
                        fmt_overhead(base_ms, a_ms), fmt_overhead(step3_ms, a_ms),
                        fmt_overhead(base_ms, b_ms), fmt_overhead(step3_ms, b_ms),
                    ])

            print_table(headers, rows, title=f"{proj.name}: ({K} -> {N}), rank={rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_correctness()
    print()
    run_benchmarks()
