"""SD3 Region-Guided Dual Adapter clean-room scaffold."""

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.conditions import Box, GenerationCondition, build_rg_map
from sd3_rgda.timestep_gate import TimestepGate

__all__ = [
    "Box",
    "DualAdapterBlock",
    "DualAdapterConfig",
    "GenerationCondition",
    "TimestepGate",
    "build_rg_map",
]
