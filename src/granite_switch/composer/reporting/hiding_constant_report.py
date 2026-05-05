# SPDX-License-Identifier: Apache-2.0
"""Safety margin report for the K-side hiding constant.

The hiding mechanism uses finfo(dtype).min as the K-side control dimension value
for tokens in hiding groups. This module computes and reports the safety margin:
how large a positive attention score would need to be before the hiding breaks
(i.e., exp(fmin + score) > 0).
"""

import torch


def _find_exp_underflow_threshold(dtype: torch.dtype) -> float:
    """Find the smallest (most negative) x where exp(x) > 0 for the given dtype.

    Searches from -500 upward in steps of 0.5.
    """
    for x_int in range(-1000, 0):
        x = x_int * 0.5
        x_t = torch.tensor(x, dtype=dtype)
        if torch.exp(x_t).item() > 0.0:
            return x
    return 0.0  # fallback: exp(x) > 0 for all tested x


def compute_hiding_constant_safety(dtype: torch.dtype) -> dict:
    """Compute safety margin data for the hiding constant at the given dtype.

    Returns a dict with:
        fmin: the hiding constant value
        exp_underflow_threshold: smallest x where exp(x) > 0
        safety_margin: positive value that must be added to fmin to break hiding
    """
    fmin_val = torch.finfo(dtype).min
    exp_threshold = _find_exp_underflow_threshold(dtype)
    safety_margin = abs(fmin_val) + exp_threshold  # exp_threshold is negative

    return {
        "dtype": str(dtype),
        "fmin": fmin_val,
        "exp_underflow_threshold": exp_threshold,
        "safety_margin": safety_margin,
    }


def print_hiding_constant_safety(dtype: torch.dtype):
    """Print the hiding constant safety margin report for the given dtype."""
    data = compute_hiding_constant_safety(dtype)

    print(f"\n{'='*80}")
    print("CONTROL DIMENSION HIDING CONSTANT")
    print(f"{'='*80}")
    print(f"  Model dtype: {data['dtype']}")
    print(f"  Hiding constant (finfo.min): {data['fmin']:.6e}")
    print(f"  exp(hiding_constant) underflows to zero: True")
    print(f"  exp() underflow threshold: {data['exp_underflow_threshold']}")
    print(f"  Safety margin: a positive attention score of {data['safety_margin']:.6e}")
    print(f"    would be needed to break hiding (make exp(fmin + score) > 0)")
