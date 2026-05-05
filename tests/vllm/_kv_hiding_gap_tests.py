# SPDX-License-Identifier: Apache-2.0
"""KV hiding gap equivalence tests (inner file — run by test_kv_hiding_gap_equivalence.py).

Verify that a hidden control token creates a transparent gap in vLLM.
Requires CUDA GPU and vLLM installed.
"""

import pytest
import torch

from tests.shared.granite4_equivalence import GRANITE4_MINI
from tests.shared.gap_equivalence import extract_visible_flat


_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm import LLM  # noqa: F401
        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

_CFG_NAME = "4.0-350m"


@pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)
class TestKVHidingGapEquivalence:

    @pytest.fixture
    def gap_runner(self, tmp_path):
        cfg_dict = GRANITE4_MINI[_CFG_NAME]

        def run(seq_len, ctrl_pos):
            from tests.shared.vllm_equivalence import run_gap_equivalence
            return run_gap_equivalence(
                cfg_dict,
                seq_len=seq_len, ctrl_pos=ctrl_pos,
                tmpdir=tmp_path,
            )

        return run

    def _assert_gap(self, run, seq_len, ctrl_pos, atol=0, rtol=0):
        upstream_lp, switch_lp = run(seq_len, ctrl_pos)
        visible_lp = extract_visible_flat(switch_lp, ctrl_pos)

        torch.testing.assert_close(
            visible_lp, upstream_lp,
            atol=atol, rtol=rtol,
            msg=(
                f"{_CFG_NAME}: visible logprobs diverge "
                f"(seq={seq_len}, ctrl={ctrl_pos})"
            ),
        )

    def test_gap_short(self, gap_runner):
        self._assert_gap(gap_runner, seq_len=16, ctrl_pos=2)

    def test_gap_ctrl_at_1(self, gap_runner):
        self._assert_gap(gap_runner, seq_len=16, ctrl_pos=1)

    def test_gap_near_end(self, gap_runner):
        self._assert_gap(gap_runner, seq_len=16, ctrl_pos=14)
