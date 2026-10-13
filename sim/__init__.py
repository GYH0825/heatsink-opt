# -*- coding: utf-8 -*-
"""散热器热-几何优化的物理仿真层。"""

from .config import (  # noqa: F401
    CHIP_POWER,
    DEPTH,
    H_CONVECTION,
    HEIGHT,
    K_ALUMINUM,
    MASS_LIMIT,
    MIN_GAP,
    NX,
    NY,
    PARAM_LABEL,
    PARAM_SPEC,
    RHO_ALUMINUM,
    T_AMBIENT,
    T_MAX_LIMIT,
    WIDTH,
)
from .geometry import HeatSink, build_mask, check_manufacturable, fin_gap, mass  # noqa: F401
from .thermal import ThermalResult, solve  # noqa: F401

__all__ = [
    "HeatSink", "build_mask", "check_manufacturable", "fin_gap", "mass",
    "ThermalResult", "solve",
]
