# SPDX-License-Identifier: Apache-2.0
"""Size FlashAttention 3's ahead-of-time schedule for the layers it serves.

vLLM builds one FlashAttention metadata builder per KV-cache group and sizes
FA3's ahead-of-time scheduler metadata from the *model config*: its query heads,
KV heads and head size. A Granite Switch checkpoint has attention layers whose
shape differs from the model config (the switch's counting and memory heads,
and Shadow Residual's doubled-query decoder attention). For their groups the
schedule is computed for the wrong shape: eager steps raise
``scheduler_metadata must have shape (metadata_size)`` and CUDA-graph steps
compute wrong attention without an error. It shows only in small,
prefix-cached steps, such as one conversation or one game served alone, so
parity checks on batches of fresh prompts pass while the served model misreads
its context.

The patch reads the shape from the group's own layers. A group whose layers
disagree keeps FA3 but lets it schedule each call (no ahead-of-time metadata).
Other models are untouched.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
# "patched", or why not: a vLLM whose internals differ leaves the bug in place,
# so it is said, not swallowed.
STATUS = "not installed"
_ATTRS = ("num_heads_q", "num_heads_kv", "headdim")


def patch_flash_attn_schedule() -> str:
    """Install the patch (idempotent); returns :data:`STATUS`. Where vLLM's
    internals differ it warns and changes nothing."""
    global STATUS
    if STATUS == "patched":
        return STATUS
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.attention.backends import flash_attn as fa
    except ImportError as e:
        STATUS = f"unavailable: {e}"
        logger.warning("Granite Switch FA3 schedule patch not installed (%s)", STATUS)
        return STATUS
    builder = getattr(fa, "FlashAttentionMetadataBuilder", None)
    if builder is None:
        STATUS = "unavailable: no FlashAttentionMetadataBuilder"
        logger.warning("Granite Switch FA3 schedule patch not installed (%s)", STATUS)
        return STATUS
    if getattr(builder, "_granite_switch_schedule", False):
        STATUS = "patched"
        return STATUS
    original = builder.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        names = ("kv_cache_spec", "layer_names", "vllm_config", "device")
        bound = {**dict(zip(names, args)), **kwargs}
        vllm_config, layer_names = bound.get("vllm_config"), bound.get("layer_names")
        hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        if getattr(hf, "model_type", None) != "granite_switch" or not getattr(
            self, "aot_schedule", False
        ):
            return
        if os.environ.get("GRANITE_SWITCH_FA3_AOT", "1") == "0":
            self.aot_schedule = False  # FA3 schedules every call itself
            return
        if not all(hasattr(self, a) for a in _ATTRS):
            # A vLLM that sizes the schedule elsewhere: schedule each call.
            logger.warning(
                "Granite Switch: FA3 metadata builder has no %s; ahead-of-time "
                "schedule disabled for this group",
                "/".join(_ATTRS),
            )
            self.aot_schedule = False
            return
        layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
        shapes = {(a.num_heads, a.num_kv_heads, a.head_size) for a in layers.values()}
        if len(shapes) == 1:
            self.num_heads_q, self.num_heads_kv, self.headdim = shapes.pop()
        elif shapes:
            self.aot_schedule = False

    builder.__init__ = __init__
    builder._granite_switch_schedule = True
    STATUS = "patched"
    return STATUS
