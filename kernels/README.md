# Fused Linear+LoRA Kernel Study

## What this is

A benchmarking study to determine whether fusing the base linear projection and its LoRA
correction into a single Triton kernel can eliminate the ~30% overhead that vLLM incurs from
separate LoRA kernel launches (shrink + expand at every adapted linear layer).

## Current state

**All 6 files are written and syntax-verified. Nothing has been run yet.**
The scripts require a CUDA GPU with Triton installed. They were authored on a Mac (CPU-only)
and need to be executed on a GPU node (H100 or L40S preferred).

## How to run

From the repo root, with a Python environment that has `torch` and `triton` installed:

```bash
# Run each step sequentially — each builds on prior steps' kernels
cd /path/to/granite-switch
PYTHONPATH=kernels python kernels/step1_baseline.py
PYTHONPATH=kernels python kernels/step2_separate.py
PYTHONPATH=kernels python kernels/step3_fused.py
PYTHONPATH=kernels python kernels/step4_routing.py
PYTHONPATH=kernels python kernels/step5_packed.py
```

Each script:
1. Runs correctness verification (compares kernel output to torch.mm reference, prints PASS/FAIL)
2. Prints benchmark tables in markdown format

If any step fails correctness checks, fix that step before moving on.

## What to do next (for the GPU session)

### Phase 1: Run and capture results

1. Run all 5 steps in order, capturing stdout to files:
   ```bash
   for step in 1 2 3 4 5; do
     PYTHONPATH=kernels python kernels/step${step}_*.py 2>&1 | tee kernels/results_step${step}.txt
   done
   ```

2. If any correctness check says FAIL, debug that kernel before proceeding.
   Common bf16 issues: accumulator precision (should use fp32 accumulators, which these do),
   tile boundary masking, stride calculations.

3. If Triton autotuning fails or a kernel crashes, check:
   - `RANK` constexpr must be a power of 2 (32 and 64 are fine)
   - Shared memory limits — step4 Strategy A loads per-adapter weights per tile
   - `tl.dot` requires both operands' inner dimension to be >= 16

### Phase 2: Analyze results

Key questions to answer from the benchmark tables:

1. **Step 2 vs Step 1**: What is the raw overhead of separate LoRA kernels?
   - At M=1 (decode): expect kernel launch overhead to dominate
   - At M=2048 (prefill): expect overhead to be proportionally smaller

2. **Step 3 vs Step 1**: Does the fused kernel match bare GEMM speed?
   - The extra register pressure from `lora_a_acc [BLOCK_M, RANK]` may reduce occupancy
   - If fused is >5% slower than base GEMM, the LoRA-A accumulation is hurting

3. **Step 3 vs Step 2**: How much does fusion save?
   - This is the main payoff — should eliminate 2 kernel launch overheads + memory round-trips

4. **Step 4, all_base pattern**: When no tokens need LoRA, does the routed kernel match bare GEMM?
   - The `any_lora` tile-level check should skip all LoRA work

5. **Step 4, Strategy A vs B**: Which routing approach wins?
   - Strategy A (inline): better for mixed batches, avoids sort+scatter overhead
   - Strategy B (sorted): better if sort is cheap and tiles are pure single-adapter
   - At rank 64 with 4 adapters, check if Strategy A hits shared memory limits

6. **Step 5**: Does shared lora_A across QKV/gate_up slices help vs separate slices?
   - The packed kernel reads x once for LoRA-A; the separate-slice baseline reads x per slice

### Phase 3: Iterate on kernels

Likely issues and fixes:

- **Poor autotuning**: The autotune configs may not cover the sweet spot for your GPU.
  Add/remove configs based on which ones win. Check `matmul_kernel.best_config` etc.

- **Step 4 Strategy A re-reads x**: The current implementation re-reads x tiles from global
  memory for each active adapter's LoRA-A computation (separate K-loop per adapter). This is
  a known inefficiency. A better approach: accumulate x @ A_i.T for ALL adapters in a single
  K-loop using separate accumulators (costs `NUM_ADAPTERS * BLOCK_M * RANK` registers). With
  4 adapters, rank 32, BLOCK_M=64, that's 8192 fp32 values = 32KB registers, which may or
  may not fit depending on the SM's register file pressure.

- **Step 5 packed kernel**: Currently identical to step 3's kernel since B is contiguously
  packed along N. The advantage is conceptual (one kernel launch vs num_slices launches).
  The real win shows at small M where launch overhead matters.

### Phase 4: Write up findings

After benchmarking, update this README with:
- Actual numbers for each step on H100 and/or L40S
- Which strategy wins for step 4
- Whether fusion is worth pursuing for production (in `granite_switch.vllm`)
- Specific recommendations for next steps

## File dependency graph

```
bench_utils.py          (standalone — GPU spec, roofline, formatting)
    ↑
step1_baseline.py       (Triton GEMM + cuBLAS comparison)
    ↑
step2_separate.py       (imports triton_matmul from step1)
    ↑
step3_fused.py          (imports triton_matmul from step1, separate_gemm_lora from step2)
    ↑
step4_routing.py        (imports triton_matmul from step1, fused_gemm_lora from step3)
step5_packed.py         (imports triton_matmul from step1, fused_gemm_lora from step3)
```

## Target dimensions (Granite 8B)

| Projection | In | Out | Notes |
|---|---|---|---|
| QKV (fused) | 4096 | 5120 | 32 heads × 128 + 2 × 8 heads × 128 |
| O | 4096 | 4096 | |
| gate_up (fused) | 4096 | 28672 | 2 × 14336 |
| down | 14336 | 4096 | |

LoRA ranks: 32, 64. num_adapters: 4. All bf16. lora_B pre-scaled by alpha/rank.

## Adapter index convention

- `adapter_indices[t] == 0` → base model only (no LoRA for this token)
- `adapter_indices[t] == i` (1-based) → apply adapter `i-1` (0-indexed into lora_A/B arrays)

This matches Granite Switch's convention (0 = no adapter, 1+ = adapter).
