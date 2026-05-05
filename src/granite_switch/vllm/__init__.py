# SPDX-License-Identifier: Apache-2.0
"""vLLM backend for Granite Switch model."""

__version__ = "0.1.0"

# Export main classes
from granite_switch.config import GraniteSwitchConfig
from .granite_switch_model import GraniteSwitchForCausalLM, GraniteSwitchModel
from .switch import SingleSwitch

# Export core components (for advanced use)
from .core import (
    GraniteLoRAEmbeddedAttention,
    GraniteSwitchDecoderLayer,
    SwitchedLoRALinear,
)

__all__ = [
    # Main API
    "GraniteSwitchConfig",
    "GraniteSwitchModel",
    "GraniteSwitchForCausalLM",
    "SingleSwitch",
    "register",
    # Core components (advanced)
    "SwitchedLoRALinear",
    "GraniteLoRAEmbeddedAttention",
    "GraniteSwitchDecoderLayer",
]

# Register config with transformers AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("granite_switch", GraniteSwitchConfig)
except Exception:
    # Registration may fail if already registered or transformers not available
    pass


def register():
    """Register the GraniteSwitch model with vLLM.

    This function is called by vLLM's plugin system on startup.
    It must be re-entrant (can be called multiple times safely).
    """
    from vllm import ModelRegistry

    # Register config with transformers AutoConfig
    try:
        from transformers import AutoConfig
        AutoConfig.register("granite_switch", GraniteSwitchConfig)
    except Exception:
        pass

    # Register custom ModelArchConfigConvertor so vLLM sees the correct
    # KV cache head size.  When adapters use control_dims, the decoder
    # attention stores expanded vectors (projection_head_dim + control_dims)
    # in the KV cache.
    try:
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,
            ModelArchConfigConvertorBase,
        )

        class _GraniteSwitchArchConfigConvertor(ModelArchConfigConvertorBase):
            def get_num_hidden_layers(self) -> int:
                cfg = self.hf_text_config
                num_layers = super().get_num_hidden_layers()
                if getattr(cfg, "num_adapters", 0) > 0:
                    # GraniteSwitch configs include one SingleSwitch KV-cache
                    # placeholder before the decoder layers.  vLLM discovers the
                    # switch Attention module separately for KV allocation, but
                    # PP layer slicing must only count physical decoder layers.
                    return max(0, num_layers - 1)
                return num_layers

            def get_head_size(self) -> int:
                cfg = self.hf_text_config
                if hasattr(cfg, 'expanded_head_dim'):
                    return cfg.expanded_head_dim
                # Fallback for configs without the property
                base = super().get_head_size()
                num_adapters = getattr(cfg, "num_adapters", 0)
                control_dims = getattr(cfg, "control_dims", 32)
                if num_adapters > 0 and control_dims > 0:
                    return base + control_dims
                return base

        MODEL_ARCH_CONFIG_CONVERTORS["granite_switch"] = (
            _GraniteSwitchArchConfigConvertor
        )
    except ImportError:
        pass

    # Only register if not already registered
    if "GraniteSwitchForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "GraniteSwitchForCausalLM",
            "granite_switch.vllm.granite_switch_model:GraniteSwitchForCausalLM",
        )
        print("✓ GraniteSwitchForCausalLM registered with vLLM")
