# SPDX-License-Identifier: Apache-2.0
"""NaN regression tests — control token at sequence position 0 (HF backend).

HF-specific unit tests for GraniteLoRAEmbeddedAttention._expand_with_control_dimensions
(batch/seq tensor layout: [batch, seq, heads, head_dim]) plus shared SDPANaNCases.

Note: model-level finiteness tests are not included here — the NaN bug only manifests
in vLLM's FlashAttention path, not in HF's stable softmax. See tests/vllm/ for those.
"""

import types

import torch

from granite_switch.hf.core.lora import GraniteLoRAEmbeddedAttention

from tests.shared.position_zero_nan_cases import SDPANaNCases


# ── Helpers ────────────────────────────────────────────────────────


def _stub(num_heads=4, num_kv_heads=1, control_dims=1):
    """Minimal namespace satisfying _expand_with_control_dimensions's self usage."""
    return types.SimpleNamespace(
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        control_dims=control_dims,
    )


def _expand(stub, q, k, v, membership, suppression):
    return GraniteLoRAEmbeddedAttention._expand_with_control_dimensions(
        stub, q, k, v, membership, suppression
    )



# ════════════════════════════════════════════════════════════════════
# 1. HF-specific unit tests: _expand_with_control_dimensions
#    Tensor layout: [batch, seq_len, num_heads, head_dim]
# ════════════════════════════════════════════════════════════════════


class TestExpandControlDimensions:
    """Direct tests of _expand_with_control_dimensions (HF tensor layout).

    token_group_membership=True marks the control token itself.
    query_group_suppression=True marks adapter-generated tokens that suppress
    the group — these are NOT group members and must keep q_control=1.
    """

    _HEAD_DIM = 32

    def _qkv(self, stub, seq_len):
        q = torch.randn(1, seq_len, stub.num_heads, self._HEAD_DIM)
        k = torch.randn(1, seq_len, stub.num_key_value_heads, self._HEAD_DIM)
        v = torch.randn(1, seq_len, stub.num_key_value_heads, self._HEAD_DIM)
        return q, k, v

    # ── fix: control token must have q_control = 0 ──────────────────

    def test_control_token_q_hide_zero_at_position_zero(self):
        """Core fix: control token at pos 0 must not activate Q-side hiding.

        Before the fix q_control was 1.0 unconditionally, so
        softmax([−∞]) = NaN when it had no other causal keys.
        """
        stub = _stub()
        membership = torch.ones(1, 1, 1, dtype=torch.bool)
        suppression = torch.ones(1, 1, 1, dtype=torch.bool)
        q, k, v = self._qkv(stub, seq_len=1)

        q_exp, _, _ = _expand(stub, q, k, v, membership, suppression)

        q_ctrl = q_exp[0, 0, :, self._HEAD_DIM:]
        assert q_ctrl.eq(0).all(), f"Control token at pos 0: q_control must be 0, got {q_ctrl}"

    def test_control_token_q_hide_zero_at_later_position(self):
        """Control token q_control is 0 regardless of its sequence position."""
        stub = _stub()
        membership = torch.zeros(1, 5, 1, dtype=torch.bool)
        membership[0, 3, 0] = True
        suppression = torch.ones(1, 5, 1, dtype=torch.bool)
        q, k, v = self._qkv(stub, seq_len=5)

        q_exp, _, _ = _expand(stub, q, k, v, membership, suppression)

        assert q_exp[0, 3, :, self._HEAD_DIM:].eq(0).all(), "Control token at pos 3: q_control must be 0"

    # ── adapter-generated tokens must still suppress the control token ──

    def test_adapter_generated_tokens_q_hide_one(self):
        """Adapter-generated tokens (non-members) keep q_control=1 to hide the control token."""
        stub = _stub()
        membership = torch.zeros(1, 5, 1, dtype=torch.bool)
        membership[0, 0, 0] = True   # control token at pos 0
        suppression = torch.ones(1, 5, 1, dtype=torch.bool)
        q, k, v = self._qkv(stub, seq_len=5)

        q_exp, _, _ = _expand(stub, q, k, v, membership, suppression)

        assert q_exp[0, 0, :, self._HEAD_DIM:].eq(0).all(), "Control token: q_control must be 0"
        for pos in range(1, 5):
            assert q_exp[0, pos, :, self._HEAD_DIM:].eq(1).all(), (
                f"Adapter-generated token at pos {pos}: q_control must be 1"
            )

    # ── k-side unchanged by fix ──────────────────────────────────────

    def test_k_side_finfo_min_for_control_token(self):
        """K-side branding is unaffected by the fix — control token gets finfo.min."""
        stub = _stub()
        membership = torch.ones(1, 1, 1, dtype=torch.bool)
        q, k, v = self._qkv(stub, seq_len=1)

        _, k_exp, _ = _expand(stub, q, k, v, membership, None)

        expected_min = torch.finfo(k.dtype).min
        k_ctrl = k_exp[0, 0, :, self._HEAD_DIM:]
        torch.testing.assert_close(k_ctrl, torch.full_like(k_ctrl, expected_min))

    def test_k_side_zero_for_adapter_generated_tokens(self):
        """Adapter-generated tokens have k_control=0."""
        stub = _stub()
        q, k, v = self._qkv(stub, seq_len=3)

        _, k_exp, _ = _expand(stub, q, k, v, torch.zeros(1, 3, 1, dtype=torch.bool), None)

        assert k_exp[:, :, :, self._HEAD_DIM:].eq(0).all()

    # ── v-side and no-mask baseline ──────────────────────────────────

    def test_v_control_always_zero(self):
        """V control dimensions are always zero."""
        stub = _stub()
        q, k, v = self._qkv(stub, seq_len=3)
        _, _, v_exp = _expand(
            stub, q, k, v,
            torch.ones(1, 3, 1, dtype=torch.bool),
            torch.ones(1, 3, 1, dtype=torch.bool),
        )
        assert v_exp[:, :, :, self._HEAD_DIM:].eq(0).all()

    def test_both_none_leaves_all_control_dims_zero(self):
        """With both tensors None, all control dims remain zero."""
        stub = _stub(control_dims=2)
        q, k, v = self._qkv(stub, seq_len=4)
        q_exp, k_exp, v_exp = _expand(stub, q, k, v, None, None)
        assert q_exp[..., self._HEAD_DIM:].eq(0).all()
        assert k_exp[..., self._HEAD_DIM:].eq(0).all()
        assert v_exp[..., self._HEAD_DIM:].eq(0).all()

    # ── multiple groups ──────────────────────────────────────────────

    def test_multiple_groups_independent(self):
        """Control token of group 0 only zeroes q_control for group 0."""
        stub = _stub(control_dims=2)
        membership = torch.zeros(1, 1, 2, dtype=torch.bool)
        membership[0, 0, 0] = True
        suppression = torch.ones(1, 1, 2, dtype=torch.bool)
        q, k, v = self._qkv(stub, seq_len=1)

        q_exp, _, _ = _expand(stub, q, k, v, membership, suppression)
        q_ctrl = q_exp[0, 0, :, self._HEAD_DIM:]

        assert q_ctrl[:, 0].eq(0).all(), "Group 0 dim must be 0 (control token is a member)"
        assert q_ctrl[:, 1].eq(1).all(), "Group 1 dim must be 1 (control token is not a member)"

    def test_original_qkv_dimensions_preserved(self):
        """Original Q/K/V dimensions are unchanged; only control dims appended."""
        stub = _stub(control_dims=3)
        q, k, v = self._qkv(stub, seq_len=5)
        q_exp, k_exp, v_exp = _expand(stub, q, k, v, None, None)
        torch.testing.assert_close(q_exp[..., : self._HEAD_DIM], q)
        torch.testing.assert_close(k_exp[..., : self._HEAD_DIM], k)
        torch.testing.assert_close(v_exp[..., : self._HEAD_DIM], v)


# ════════════════════════════════════════════════════════════════════
# 2. Shared SDPA cases
# ════════════════════════════════════════════════════════════════════


class TestSDPANaN(SDPANaNCases):
    pass


