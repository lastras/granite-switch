# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementations for Granite Switch (HuggingFace).

This package provides the SingleSwitch mechanism for adapter selection.
"""

from .single import SingleSwitch

__all__ = [
    "SingleSwitch",
    "create_switch",
]


def create_switch(config, layer_idx=0):
    """Factory function to create SingleSwitch based on config.

    Args:
        config: GraniteSwitchConfig
        layer_idx: Layer index for cache management (default: 0)

    Returns:
        SingleSwitch instance
    """
    return SingleSwitch(
        num_adapters=config.num_adapters,
        config=config,
        control_token_gain=getattr(config, "control_token_gain", 15.0),
        switch_head_dim=config.switch_head_dim,
        layer_idx=layer_idx,
    )
