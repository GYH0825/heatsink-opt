# -*- coding: utf-8 -*-
"""优化问题的定义：参数空间、约束、目标函数与评估缓存。

约束处理采用 Deb 可行性规则（不做人工调参的罚函数）：
    可行解 优于 不可行解
    两个可行解之间    —— 比较目标值（热阻）
    两个不可行解之间  —— 比较约束违反度
等价于对 (是否不可行, 违反度, 目标值) 做字典序比较，因此可以直接用元组当排序键，
不需要担心惩罚系数的量级选择。
"""

from dataclasses import dataclass, field

import numpy as np

from sim import config as C
from sim.geometry import HeatSink, build_mask, check_manufacturable, mass as compute_mass
from sim.thermal import solve

# 设计变量离散化精度：0.1 mm。低于此精度的差异在制造上没有意义，
# 同时它也让评估结果可以安全地做缓存去重。
DISCRETIZE = 1e-4  # m


def discretize(hs):
    """把连续尺寸量化到制造精度，得到可缓存的规范形式。"""
    return HeatSink(
        n_fins=int(hs.n_fins),
        fin_height=round(hs.fin_height / DISCRETIZE) * DISCRETIZE,
        fin_thickness=round(hs.fin_thickness / DISCRETIZE) * DISCRETIZE,
        base_thickness=round(hs.base_thickness / DISCRETIZE) * DISCRETIZE,
    )


@dataclass
class Evaluation:
    """一次设计的评估结果。"""

    hs: HeatSink
    geom_feasible: bool            # 几何是否可制造
    reason: str                    # 几何不可行原因

    r_th: float = float("inf")     # °C/W 热阻（越小越好）
    t_max: float = float("nan")    # °C
    mass: float = float("nan")     # kg
    violation: float = float("inf")  # 约束违反度，0 表示全部满足
    t_field: object = None         # 温度场，绘图用
    mask: object = None

    @property
    def feasible(self):
        return self.geom_feasible and self.violation <= 0.0

    @property
    def key(self):
        """字典序比较键，直接 min() 即可实现 Deb 可行性规则。"""
        if not self.geom_feasible:
            return (2, float("inf"), float("inf"))
        return (0 if self.feasible else 1, self.violation, self.r_th)

    @property
    def summary(self):
        if not self.geom_feasible:
            return f"几何不可行（{self.reason}）"
        tag = "可行" if self.feasible else "越界"
        return (f"[{tag}] T_max={self.t_max:.1f}°C  R_th={self.r_th:.3f}°C/W  "
                f"m={self.mass * 1e3:.1f}g  违反度={self.violation:.3f}")


def constraint_violation(t_max, mass):
    """归一化约束违反度：温度越界与质量越界之和，0 表示满足全部约束。"""
    v_temp = max(0.0, (t_max - C.T_MAX_LIMIT) / (C.T_MAX_LIMIT - C.T_AMBIENT))
    v_mass = max(0.0, (mass - C.MASS_LIMIT) / C.MASS_LIMIT)
    return v_temp + v_mass


class Problem:
    """散热器优化问题：采样 + 评估 + 缓存。"""

    def __init__(self):
        self.cache = {}
        self.n_eval = 0        # 真实求解次数
        self.n_hit = 0         # 缓存命中次数
        self.t_evals = []      # 每次真实求解的耗时（绘图/统计用）

    # ------------------------------------------------------------ 采样

    def sample_random(self, rng):
        """在参数空间内均匀采样，并通过拒绝采样保证几何可制造。"""
        for _ in range(200):
            vals = []
            for _name, lo, hi, is_int in C.PARAM_SPEC:
                v = rng.uniform(lo, hi)
                vals.append(int(round(v)) if is_int else v)
            hs = discretize(HeatSink(*vals))
            if check_manufacturable(hs)[0]:
                return hs
        # 兜底：一个确定可行的保守设计
        return HeatSink(6, 0.020, 0.003, 0.004)

    # ------------------------------------------------------------ 评估

    def evaluate(self, hs):
        """评估一个设计；几何不可行时直接返回，不进入热求解。"""
        import time

        hs = discretize(hs)
        key = (hs.n_fins, hs.fin_height, hs.fin_thickness, hs.base_thickness)
        cached = self.cache.get(key)
        if cached is not None:
            self.n_hit += 1
            return cached

        t0 = time.perf_counter()
        ok, reason = check_manufacturable(hs)
        if not ok:
            ev = Evaluation(hs=hs, geom_feasible=False, reason=reason)
        else:
            mask, mreason = build_mask(hs)
            if mask is None:
                ev = Evaluation(hs=hs, geom_feasible=False, reason=mreason)
            else:
                res = solve(hs)
                ev = Evaluation(
                    hs=hs,
                    geom_feasible=True,
                    reason="",
                    r_th=res.r_th,
                    t_max=res.t_max,
                    mass=res.mass,
                    violation=constraint_violation(res.t_max, res.mass),
                    t_field=res.t_field,
                    mask=mask,
                )
        self.t_evals.append(time.perf_counter() - t0)
        self.n_eval += 1
        self.cache[key] = ev
        return ev

    # ------------------------------------------------------------ 统计

    def cache_stats(self):
        total = self.n_eval + self.n_hit
        hit_rate = self.n_hit / total if total else 0.0
        avg_ms = np.mean(self.t_evals) * 1e3 if self.t_evals else 0.0
        return dict(n_eval=self.n_eval, n_hit=self.n_hit,
                    hit_rate=hit_rate, avg_ms=avg_ms)
