# -*- coding: utf-8 -*-
"""二维稳态热传导求解器（有限差分离散 + 稀疏直接求解）。

控制方程（截面内，无内热源）：
        ∇·(k∇T) = 0

边界条件：
    基板底面      —— 给定热流 q'（芯片发热，单位深度线功率密度）
    其余金属表面  —— 自然对流  -k ∂T/∂n = h (T - T_amb)

离散：格心有限体积法。对每个金属节点 i，
    Σ_j  G_ij (T_j - T_i) + Σ_face h·L (T_amb - T_i) + Q_i = 0

其中固-固面 G_ij = k·L/d，暴露面（邻居为空气或域外）按对流处理，
L 为面长度、d 为节点间距。整理成稀疏线性方程组 A·T = b 后直接求解。

单位深度约定：所有功率与导热都按 1 m 深度计算，最后再按 DEPTH 换算回真实器件。
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from . import config as C
from .geometry import build_mask, mass as compute_mass, solid_area


@dataclass
class ThermalResult:
    """一次热仿真的结果。"""

    feasible: bool                 # 几何是否可制造
    reason: str = ""               # 不可行原因

    t_max: float = float("nan")    # °C   最高温度
    t_ambient: float = C.T_AMBIENT
    r_th: float = float("nan")     # °C/W 结到环境的热阻
    mass: float = float("nan")     # kg
    conv_area: float = float("nan")  # m²  参与对流的表面积
    solid_area: float = float("nan")  # m²  金属截面积

    t_field: object = None         # (ny, nx) 温度场，空气处为 NaN；绘图用
    mask: object = None            # (ny, nx) 实体掩码；绘图用

    @property
    def delta_t(self):
        """结温升 [K]。"""
        return self.t_max - self.t_ambient

    @property
    def summary(self):
        if not self.feasible:
            return f"不可行：{self.reason}"
        return (f"T_max={self.t_max:.1f}°C  R_th={self.r_th:.3f}°C/W  "
                f"m={self.mass * 1e3:.1f}g  A_conv={self.conv_area * 1e4:.1f}cm²")


def solve(hs, nx=C.NX, ny=C.NY):
    """求解给定散热器设计的稳态温度场。

    参数变化会改变实体掩码，因此每次都要重建稀疏矩阵——这是本项目的性能热点。
    """
    mask, reason = build_mask(hs, nx, ny)
    if mask is None:
        return ThermalResult(feasible=False, reason=reason)

    dx = C.WIDTH / nx
    dy = C.HEIGHT / ny
    q_line = C.flux_per_depth()      # W/m 单位深度线功率密度

    # ---------------- 编号 ----------------
    node_id = -np.ones((ny, nx), dtype=np.int64)
    n_solid = int(mask.sum())
    node_id[mask] = np.arange(n_solid)

    js, xs_idx = np.nonzero(mask)          # 每个金属节点的行列索引
    own_id = node_id[js, xs_idx]

    diag = np.zeros(n_solid)
    rhs = np.zeros(n_solid)
    off_rows, off_cols, off_vals = [], [], []
    conv_face_len = 0.0                    # 累计暴露面长度（截面内，单位深度）

    # ---------------- 四个方向 ----------------
    # (dj, di, 节点间距 d, 面长度 L)
    for dj, di, d, L in ((0, 1, dx, dy), (0, -1, dx, dy),
                         (1, 0, dy, dx), (-1, 0, dy, dx)):
        nj = js + dj
        ni = xs_idx + di
        in_bounds = (nj >= 0) & (nj < ny) & (ni >= 0) & (ni < nx)

        is_solid = np.zeros(n_solid, dtype=bool)
        is_solid[in_bounds] = mask[nj[in_bounds], ni[in_bounds]]

        # --- 固-固导热 ---
        if is_solid.any():
            g_cond = C.K_ALUMINUM * L / d
            diag[is_solid] += g_cond
            off_rows.append(own_id[is_solid])
            off_cols.append(node_id[nj[is_solid], ni[is_solid]])
            off_vals.append(np.full(int(is_solid.sum()), -g_cond))

        # --- 暴露面：自然对流 ---
        exposed = ~is_solid
        if exposed.any():
            g_conv = C.H_CONVECTION * L
            diag[exposed] += g_conv
            rhs[exposed] += g_conv * C.T_AMBIENT
            conv_face_len += float(exposed.sum()) * L

    # ---------------- 底部热源 ----------------
    # 芯片功率均匀施加于基板底面，每个底排节点的入热 = q' · dx / WIDTH
    bottom = js == 0
    if bottom.any():
        rhs[bottom] += q_line * dx / C.WIDTH

    # ---------------- 组装并求解 ----------------
    rows = np.concatenate([own_id] + off_rows)
    cols = np.concatenate([own_id] + off_cols)
    vals = np.concatenate([diag] + off_vals)
    a_mat = coo_matrix((vals, (rows, cols)), shape=(n_solid, n_solid)).tocsr()

    t_solid = spsolve(a_mat, rhs)

    # ---------------- 后处理 ----------------
    t_field = np.full((ny, nx), np.nan)
    t_field[mask] = t_solid
    t_max = float(t_solid.max())

    # 单位深度结果换算回真实器件深度
    r_th = (t_max - C.T_AMBIENT) / q_line / C.DEPTH

    return ThermalResult(
        feasible=True,
        t_max=t_max,
        r_th=r_th,
        mass=compute_mass(hs, mask, nx, ny),
        conv_area=conv_face_len * C.DEPTH,
        solid_area=solid_area(mask, nx, ny),
        t_field=t_field,
        mask=mask,
    )
