# SPDX-License-Identifier: Apache-2.0
"""Compose utilities for Granite Switch models.

This module provides utilities for composing Granite Switch checkpoints from
base models and LoRA adapters.
"""

from .weight_remapper import AdapterRemapper, RemapResult
from .compose_utils import GraniteSwitchComposer
from .arch import (
    ArchDescriptor,
    ModuleDescriptor,
    resolve_arch,
    granite_dense_arch,
    granite_moe_hybrid_arch,
)

__all__ = [
    "AdapterRemapper",
    "RemapResult",
    "GraniteSwitchComposer",
    "ArchDescriptor",
    "ModuleDescriptor",
    "resolve_arch",
    "granite_dense_arch",
    "granite_moe_hybrid_arch",
]
