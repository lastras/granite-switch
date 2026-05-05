# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementations for Granite Switch (vLLM).

This package provides the SingleSwitch mechanism for adapter selection.
"""

from .single import SingleSwitch

__all__ = [
    "SingleSwitch",
    "create_switch",
]


def create_switch(config, vllm_config=None):
    """Factory function to create SingleSwitch based on config.

    Args:
        config: GraniteSwitchConfig
        vllm_config: vLLM configuration (for vLLM implementation)

    Returns:
        SingleSwitch instance
    """
    return SingleSwitch(
        num_adapters=config.num_adapters,
        vllm_config=vllm_config,
        control_token_gain=getattr(config, "control_token_gain", 15.0),
        switch_head_dim=config.switch_head_dim,
        config=config,
    )
