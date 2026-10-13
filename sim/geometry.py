# -*- coding: utf-8 -*-
"""散热器几何：设计参数 → 实体掩码，以及可制造性检查。

布局约定（单位：米）
    y 从 0 开始向上，基板占据 y ∈ [0, base_thickness]
    翅片从基板顶面向上生长，顶端为 base_thickness + fin_height
    翅片沿 x 方向等间距排布，两端留边与片间间隙相等：

        gap = (WIDTH - n_fins * fin_thickness) / (n_fins + 1)

    这样布局是关于中心对称的，且 gap 就是需要满足 MIN_GAP 的量。
"""

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass(frozen=True)
class HeatSink:
    """一组散热器设计参数。"""

    n_fins: int
    fin_height: float
    fin_thickness: float
    base_thickness: float

    def as_array(self):
        return np.array([self.n_fins, self.fin_height,
                         self.fin_thickness, self.base_thickness], dtype=float)

    def describe(self):
        return (f"{self.n_fins} 片 × 高 {self.fin_height * 1e3:.1f} mm "
                f"× 厚 {self.fin_thickness * 1e3:.1f} mm，"
                f"基板 {self.base_thickness * 1e3:.1f} mm")


def fin_gap(hs):
    """翅片间隙 [m]；间隙为负说明翅片总宽已超出基板。"""
    return (C.WIDTH - hs.n_fins * hs.fin_thickness) / (hs.n_fins + 1)


def check_manufacturable(hs):
    """返回 (是否可行, 原因)。不可行时原因用于日志与约束惩罚。"""
    if hs.n_fins < 1:
        return False, "翅片数量必须为正"
    gap = fin_gap(hs)
    if gap < C.MIN_GAP:
        return False, (f"翅片间隙 {gap * 1e3:.2f} mm 低于制造下限 "
                       f"{C.MIN_GAP * 1e3:.1f} mm")
    if hs.base_thickness + hs.fin_height > C.HEIGHT + 1e-12:
        return False, (f"基板+翅片总高 {(hs.base_thickness + hs.fin_height) * 1e3:.1f} mm "
                       f"超出外形高度 {C.HEIGHT * 1e3:.0f} mm")
    if hs.base_thickness <= 0 or hs.fin_height <= 0 or hs.fin_thickness <= 0:
        return False, "几何尺寸必须为正"
    return True, ""


def build_mask(hs, nx=C.NX, ny=C.NY):
    """生成实体掩码。返回 (mask, reason)：

        mask : bool 数组，形状 (ny, nx)，True 表示该节点属于金属实体
        reason : 不可行时为原因字符串，可行时为空串

    节点使用格心坐标（第 i 列中心 x = (i+0.5)*dx），避免边界落在整数网格上
    时产生半格歧义。
    """
    ok, reason = check_manufacturable(hs)
    if not ok:
        return None, reason

    dx = C.WIDTH / nx
    dy = C.HEIGHT / ny

    xs = (np.arange(nx) + 0.5) * dx      # (nx,)
    ys = (np.arange(ny) + 0.5) * dy      # (ny,)

    mask = np.zeros((ny, nx), dtype=bool)

    # --- 基板：横跨整个宽度 ---
    base_rows = ys <= hs.base_thickness
    mask[base_rows, :] = True

    # --- 翅片：等间距居中排布 ---
    gap = fin_gap(hs)
    fin_top = hs.base_thickness + hs.fin_height
    for k in range(hs.n_fins):
        x_lo = gap + k * (hs.fin_thickness + gap)
        x_hi = x_lo + hs.fin_thickness
        cols = (xs >= x_lo) & (xs <= x_hi)
        rows = (ys > hs.base_thickness) & (ys <= fin_top)
        mask[np.ix_(rows, cols)] = True

    if not mask.any():
        return None, "实体掩码为空"
    return mask, ""


def solid_area(mask, nx=C.NX, ny=C.NY):
    """实体截面积 [m²]（二维截面内的金属面积）。"""
    dx = C.WIDTH / nx
    dy = C.HEIGHT / ny
    return float(mask.sum()) * dx * dy


def mass(hs, mask, nx=C.NX, ny=C.NY):
    """散热器质量 [kg] = 截面积 × 深度 × 密度。"""
    return solid_area(mask, nx, ny) * C.DEPTH * C.RHO_ALUMINUM
