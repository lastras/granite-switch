# SPDX-License-Identifier: Apache-2.0
"""Verify that a hidden control token creates a transparent gap in attention.

The upstream model processes a contiguous N-token sequence. The switch model
processes the same N content tokens with a hidden control token inserted,
giving N+1 total tokens. With zero LoRA weights and SingleSwitch's hidden_count
closing the RoPE gap, the logits at corresponding visible positions should
match within FP tolerance.

The hiding mechanism itself is exact on CPU:
- exp(finfo.min) = 0.0 exactly → hidden token gets zero softmax weight
- Control dims: Q_ctrl * K_ctrl = 1.0 * 0.0 = 0.0 → no score change
- V_control = 0.0 → zero contribution to attention output

The ~1e-7 tolerance comes from different softmax window sizes at
corresponding positions. Switch position k+1 computes softmax over k+2
entries (including the ~0 hidden token), while upstream position k computes
softmax over k+1 entries. Although the hidden entry contributes exactly 0.0
to the denominator, SDPA's fused softmax kernel processes different-length
reductions with different FP accumulation order. Positions before the
control token are bit-exact (same causal window in both models).

Attention-only models only — Mamba layers do not support KV hiding (the hidden
control token would flow through conv1d and SSM state, corrupting subsequent
positions). Only dense (attention-only) configs from GRANITE4_MINI are tested.

SingleSwitch: hidden_count = (adapter_indices > 0).long() — fires once,
so 0 before control token and 1 at/after (see issue #16).
"""

import pytest
import torch
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridForCausalLM,
)

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

from tests.shared.granite4_equivalence import (
    augment_cfg_with_adapters,
    transfer_weights,
    zero_lora_weights,
    GRANITE4_MINI,
)
from tests.shared.gap_equivalence import (
    ATTN_ONLY_NAMES,
    make_gapped_inputs,
    extract_visible_batched,
)

# Softmax window-size tolerance (see module docstring).
# Observed max: ~1e-7 across all configs/positions/seeds.
# Use 5e-7 (≈5x margin) to accommodate variation.
_ATOL = 5e-7


# ── Helpers ────────────────────────────────────────────────────────


def _make_gap_pair(cfg_dict):
    """Create upstream + 1-adapter switch model pair with zero LoRA weights.

    SingleSwitch: adapter_token_ids=[101], 101 is adapter_0 (KV-hidden).
    """
    torch.manual_seed(0)
    upstream = GraniteMoeHybridForCausalLM(
        GraniteMoeHybridConfig(**cfg_dict)
    ).eval()

    switch_cfg_dict = augment_cfg_with_adapters(cfg_dict, num_adapters=1)
    switch = GraniteSwitchForCausalLM(
        GraniteSwitchConfig(**switch_cfg_dict)
    ).eval()

    # Transfer base weights (non-strict: LoRA/switch params left unloaded)
    unloaded = transfer_weights(upstream.state_dict(), switch.state_dict())

    # Verify unloaded params are only LoRA and switch related
    for name in unloaded:
        assert any(k in name for k in (
            "lora_A", "lora_B", "switch", "adapter_token_ids",
            "token_to_group_mask", "adapter_hiding_matrix",
        )), f"Unexpected unloaded parameter: {name}"

    # Zero all LoRA weights defensively
    zero_lora_weights(switch)

    return upstream, switch


def _assert_gap_equivalence(name, upstream, switch, seq_len, ctrl_pos, seed=42):
    """Run forward pass and assert visible logits match within tolerance."""
    upstream_ids, switch_ids = make_gapped_inputs(seq_len, ctrl_pos, seed)

    with torch.no_grad():
        upstream_out = upstream(input_ids=upstream_ids, use_cache=False)
        switch_out = switch(input_ids=switch_ids, use_cache=False)

    visible = extract_visible_batched(switch_out.logits, ctrl_pos)

    torch.testing.assert_close(
        visible, upstream_out.logits,
        atol=_ATOL, rtol=0.0,
        msg=f"{name}: visible logits diverge (seq={seq_len}, ctrl={ctrl_pos})",
    )


# ── Test class: KV Hiding Gap Equivalence ─────────────────────────


class TestKVHidingGapEquivalence:
    """Verify hidden control token creates a transparent gap.

    The upstream model processes N contiguous tokens. The switch model
    processes the same N tokens with a hidden control token inserted (N+1
    total). Visible-position logits match within BLAS gemm tolerance.
    """

    @pytest.fixture(params=ATTN_ONLY_NAMES)
    def model_pair(self, request):
        model_name = request.param
        upstream, switch = _make_gap_pair(GRANITE4_MINI[model_name])
        return model_name, upstream, switch

    def test_gap_short(self, model_pair):
        """Short sequence (16 tokens), control token at position 2."""
        name, upstream, switch = model_pair
        _assert_gap_equivalence(name, upstream, switch, seq_len=16, ctrl_pos=2)

    def test_gap_long(self, model_pair):
        """Longer sequence (64 tokens), control token at position 8."""
        name, upstream, switch = model_pair
        _assert_gap_equivalence(name, upstream, switch, seq_len=64, ctrl_pos=8)

    def test_ctrl_at_position_1(self, model_pair):
        """Control token at position 1.

        With SingleSwitch, position 0 has no special role (no counting
        anchor needed). ctrl_pos=0 is tested separately in
        test_multiple_ctrl_positions as a NaN regression guard.
        """
        name, upstream, switch = model_pair
        _assert_gap_equivalence(name, upstream, switch, seq_len=16, ctrl_pos=1)

    def test_ctrl_near_end(self, model_pair):
        """Control token near the end of the sequence (pos=seq_len-2)."""
        name, upstream, switch = model_pair
        _assert_gap_equivalence(name, upstream, switch, seq_len=16, ctrl_pos=14)

    @pytest.mark.parametrize("ctrl_pos", [0, 1, 2, 4, 8, 14])
    def test_multiple_ctrl_positions(self, model_pair, ctrl_pos):
        """Sweep control token across multiple positions.

        ctrl_pos=0 is a regression guard for the NaN bug fixed in PR #87:
        when the control token sits at position 0 with no other causal key,
        softmax([-inf]) = NaN unless q_control is zeroed for group members.
        """
        name, upstream, switch = model_pair
        _assert_gap_equivalence(name, upstream, switch, seq_len=16, ctrl_pos=ctrl_pos)


# ── Test class: Adapter Indices Sanity ────────────────────────────


class TestAdapterIndicesSanity:
    """Verify adapter_indices correctness with a single hidden control token.

    Uses a single config (4.0-350m) to check that:
    - Positions before the control token have adapter_indices=0 (base)
    - Positions at and after the control token have adapter_indices=1
    """

    @pytest.fixture
    def model(self):
        cfg_dict = GRANITE4_MINI["4.0-350m"]
        _, switch = _make_gap_pair(cfg_dict)
        return switch

    def _run(self, model, ctrl_pos, seed=42):
        """Run forward pass and return adapter_indices."""
        _, switch_ids = make_gapped_inputs(seq_len=16, ctrl_pos=ctrl_pos, seed=seed)
        with torch.no_grad():
            model(input_ids=switch_ids, use_cache=False)
        return model.model._last_adapter_indices

    def test_adapter_indices_before_ctrl(self, model):
        """Positions before control token should be base (0)."""
        ctrl_pos = 4
        ai = self._run(model, ctrl_pos)
        assert (ai[:, :ctrl_pos] == 0).all(), (
            f"Pre-control positions should be base, got {ai[:, :ctrl_pos]}"
        )

    def test_adapter_indices_at_and_after_ctrl(self, model):
        """Positions at and after control token should be adapter_0 (1)."""
        ctrl_pos = 4
        ai = self._run(model, ctrl_pos)
        assert (ai[:, ctrl_pos:] == 1).all(), (
            f"Post-control positions should be adapter_0 (1), got {ai[:, ctrl_pos:]}"
        )

    def test_adapter_indices_sweep(self, model):
        """Sweep ctrl_pos and verify adapter_indices boundary."""
        for ctrl_pos in [1, 2, 4, 8, 14]:
            ai = self._run(model, ctrl_pos, seed=ctrl_pos)
            assert (ai[:, :ctrl_pos] == 0).all(), (
                f"ctrl_pos={ctrl_pos}: pre-ctrl should be 0, got {ai[:, :ctrl_pos]}"
            )
            assert (ai[:, ctrl_pos:] == 1).all(), (
                f"ctrl_pos={ctrl_pos}: post-ctrl should be 1, got {ai[:, ctrl_pos:]}"
            )
