# SPDX-License-Identifier: Apache-2.0
"""Tests for the K-side hiding constant used in control-dimension masking.

The hiding mechanism sets K[d_g] = finfo(dtype).min for tokens in hiding group g.
This test validates that this constant behaves correctly across all supported
floating-point types:

1. exp(constant) == 0  (softmax produces zero weight)
2. Accumulation of multiple constants (multiple groups) also exponentiates to zero
3. Adding realistic finite attention scores to the constant still exponentiates to zero
4. 0 * constant does NOT produce NaN (critical: Q[d_g]=0 for non-hiding adapters)

Safety margin reporting (how large a positive value must be added before exp produces
a nonzero result) is part of the builder's verbose output, not tested here.
"""

import pytest
import torch

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
DTYPE_IDS = ["float16", "bfloat16", "float32", "float64"]

# Maximum number of hiding groups we might realistically accumulate in a dot product
MAX_GROUPS = 32



def hiding_constant(dtype: torch.dtype) -> torch.Tensor:
    """The K-side hiding constant: the most negative finite value in the dtype."""
    return torch.tensor(torch.finfo(dtype).min, dtype=dtype)


@pytest.fixture(params=DTYPES, ids=DTYPE_IDS)
def dtype(request):
    return request.param


class TestHidingConstantExponentiation:
    """exp(hiding_constant) must be exactly zero — this is what makes softmax
    assign zero attention weight to hidden tokens."""

    def test_exp_of_constant_is_zero(self, dtype):
        c = hiding_constant(dtype)
        assert torch.exp(c).item() == 0.0

    def test_exp_of_sum_of_constants_is_zero(self, dtype):
        """A token in multiple groups: dot product accumulates multiple constants."""
        c = hiding_constant(dtype)
        for n in [2, 4, MAX_GROUPS]:
            accum = torch.zeros(1, dtype=dtype)
            for _ in range(n):
                accum = accum + c
            assert torch.exp(accum).item() == 0.0, f"Failed for {n} groups"

    def test_exp_of_constant_plus_finite_is_zero(self, dtype):
        """Normal attention score added to the constant must still exponentiate to zero."""
        c = hiding_constant(dtype)
        for score in [0.0, 10.0, 100.0, 1000.0]:
            s = torch.tensor(score, dtype=dtype)
            result = torch.exp(c + s)
            assert result.item() == 0.0, f"Failed for score={score}"


class TestHidingConstantNoNaN:
    """0 * hiding_constant must NOT produce NaN. This is the scenario where
    Q[d_g] = 0 (adapter does not hide group g) and K[d_g] = constant."""

    def test_zero_times_constant_is_not_nan(self, dtype):
        c = hiding_constant(dtype)
        zero = torch.tensor(0.0, dtype=dtype)
        result = zero * c
        assert not result.isnan().item()

    def test_zero_times_constant_does_not_corrupt_dot_product(self, dtype):
        """In a realistic dot product, 0 * constant contributions must not
        change the result compared to a clean dot product without control dims."""
        torch.manual_seed(42)
        head_dim = 128
        control_dims = 4
        total_dim = head_dim + control_dims

        Q = torch.randn(total_dim, dtype=dtype)
        K = torch.randn(total_dim, dtype=dtype)

        c = hiding_constant(dtype)
        # Token is in groups 0 and 2
        K[head_dim + 0] = c
        K[head_dim + 1] = 0.0
        K[head_dim + 2] = c
        K[head_dim + 3] = 0.0

        # Query does NOT hide any group
        Q[head_dim:] = 0.0

        dot_with_ctrl = torch.dot(Q, K)
        dot_clean = torch.dot(Q[:head_dim], K[:head_dim])

        assert not dot_with_ctrl.isnan().item()
        assert torch.isclose(dot_with_ctrl, dot_clean, atol=1e-2)


class TestHidingConstantSoftmax:
    """End-to-end: softmax assigns exactly zero weight to hidden positions."""

    def test_softmax_zero_weight_for_hidden(self, dtype):
        scores = torch.tensor([5.0, 3.0, 7.0], dtype=dtype)
        c = hiding_constant(dtype)
        scores_with_hidden = scores.clone()
        scores_with_hidden[1] = scores_with_hidden[1] + c  # hide position 1

        sm = torch.softmax(scores_with_hidden, dim=0)
        assert sm[1].item() == 0.0
        # Non-hidden positions should get all the probability mass
        assert sm[0].item() > 0.0
        assert sm[2].item() > 0.0
        assert abs(sm.sum().item() - 1.0) < 1e-3


