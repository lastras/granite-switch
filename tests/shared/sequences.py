# SPDX-License-Identifier: Apache-2.0
"""Shared sequence builders and ground truth computation for SingleSwitch tests.

Used by both HF and vLLM tests to generate control token sequences
and compute expected results.
"""

import random


def build_sequence_with_control(
    num_controls,
    adapters,
    text_token,
    p=0.5,
    seed=42,
):
    """Build a sequence with control tokens and geometric gaps.

    Creates a deterministically random (seeded) sequence with:
    - A text token at position 0
    - Control tokens with gaps between them
    - Gaps drawn from Geometric(p): each text position has
      probability p of ending the gap

    Args:
        num_controls: Total number of control tokens
        adapters: List of adapter token IDs (e.g. [101, 102])
        text_token: Regular text token ID
        p: Probability of ending each text gap position
        seed: RNG seed for reproducibility

    Returns:
        seq: List of token IDs
        post_start: Index of first post-init token in seq (always 1)
    """
    rng = random.Random(seed)

    seq = [text_token]
    adapter_cycle = 0
    remaining = num_controls
    while remaining > 0:
        burst = min(rng.randint(1, 5), remaining)
        for _ in range(burst):
            seq.append(adapters[adapter_cycle % len(adapters)])
            adapter_cycle += 1
            remaining -= 1
        if remaining > 0:
            gap = 0
            while rng.random() > p:
                gap += 1
            for _ in range(gap):
                seq.append(text_token)
        else:
            for _ in range(5):
                seq.append(text_token)

    return seq, 1


def compute_single_switch_ground_truth(seq, adapters):
    """Compute ground truth for SingleSwitch.

    SingleSwitch uses differential-gain attention: control tokens get K=+gain,
    non-control tokens get K=-gain. With sufficient gain, softmax concentrates
    entirely on control tokens.

    Args:
        seq: List of token IDs
        adapters: List of adapter token IDs (length = num_adapters).
                  adapters[i] activates adapter i+1 (1-indexed output).

    Returns:
        active_adapter: [seq_len] expected adapter index at each position.
                        0 before first control token, then the adapter ID.
    """
    seq_len = len(seq)
    active_adapter = [0] * seq_len

    last_adapter = 0  # 0 = base (no control token seen yet)

    for pos in range(seq_len):
        tok = seq[pos]
        if tok in adapters:
            idx = adapters.index(tok)
            last_adapter = idx + 1  # 1-indexed
        active_adapter[pos] = last_adapter

    return active_adapter


# Backward compatibility alias
compute_basic_switch_ground_truth = compute_single_switch_ground_truth
