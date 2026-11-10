"""Reference Python / NumPy implementations for MiniMax-H3 / FastH3."""

from tools.ref.h3_ref import (
    StepPlan,
    block,
    dit_forward,
    euler_step,
    final_layer,
    refiner_block,
    sampler_plan,
    stochastic_step,
)
from tools.ref.vsa_ref import (
    VsaCfg,
    VsaGeometry,
    build_vsa_geometry,
    vsa_attention,
)

__all__ = [
    "StepPlan",
    "VsaCfg",
    "VsaGeometry",
    "block",
    "build_vsa_geometry",
    "dit_forward",
    "euler_step",
    "final_layer",
    "refiner_block",
    "sampler_plan",
    "stochastic_step",
    "vsa_attention",
]
