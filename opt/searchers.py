# -*- coding: utf-8 -*-
"""搜索算法：随机搜索基线 vs 自适应邻域搜索。

预算统一以「设计评估次数」计。为了让两种算法在完全相同的条件下比较，
每次运行都新建一个 Problem（各自独立的评估缓存）：缓存命中只会让某次评估
变便宜，而不会让算法白拿别人已经算过的结果。

设计变量归一化到 [0,1]^4 之后再扰动，这样「步长」对四个量纲不同的参数
（片数 3–15、尺寸 2–40 mm）具有一致的含义。
"""

import time
from dataclasses import dataclass, field

import numpy as np

from sim import config as C
from sim.geometry import HeatSink, check_manufacturable
from opt.problem import Evaluation, discretize

_PARAM = C.PARAM_SPEC

# 自适应邻域搜索的超参数
INIT_SAMPLES = 24        # 初始随机采样数，用来定位第一个锚点
CAND_PER_ROUND = 6       # 每轮在锚点邻域内生成并评估的候选数
STEP_INIT = 0.25         # 初始步长（归一化空间占比）
STEP_MIN = 0.01
STEP_MAX = 0.60
STALL_LIMIT = 5          # 连续多少轮无改善后重启
RESTART_STEP = 0.35      # 重启时的步长


# ---------------------------------------------------------------- 编码

def to_unit(hs):
    """设计参数 → 归一化坐标 [0,1]^4。"""
    u = np.empty(len(_PARAM))
    for i, (name, lo, hi, _is_int) in enumerate(_PARAM):
        u[i] = (getattr(hs, name) - lo) / (hi - lo)
    return np.clip(u, 0.0, 1.0)


def from_unit(u):
    """归一化坐标 → 设计参数；整数维取整，全部夹紧到参数边界内。"""
    kwargs = {}
    for i, (name, lo, hi, is_int) in enumerate(_PARAM):
        v = lo + float(np.clip(u[i], 0.0, 1.0)) * (hi - lo)
        kwargs[name] = int(round(v)) if is_int else v
    return discretize(HeatSink(**kwargs))


def _perturb(hs, step, rng):
    """在归一化空间里做高斯扰动，再映射回设计参数。"""
    u = to_unit(hs) + step * rng.standard_normal(len(_PARAM))
    return from_unit(u)


# ---------------------------------------------------------------- 结果记录

@dataclass
class RunResult:
    """一次完整搜索过程的记录。"""

    name: str
    seed: int
    n_eval: int                     # 评估次数（= 预算）
    n_hit: int                      # 其中命中缓存、未真正求解的次数
    wall_time: float                # 墙钟耗时 [s]
    best: Evaluation                # 按 Deb 可行性规则得到的最优设计

    rth_history: np.ndarray = None  # 每步之后「已找到的最优可行热阻」，暂无可行解为 inf
    viol_history: np.ndarray = None  # 每步之后「最小约束违反度」
    hs_history: list = field(default_factory=list)

    @property
    def label(self):
        return f"{self.name}(seed={self.seed})"


class _Tracker:
    """推入每一次评估，维护两个独立的运行最优：最优可行热阻、最小违反度。"""

    def __init__(self):
        self.best = None
        self.best_rth = float("inf")
        self.best_viol = float("inf")
        self.k = 0
        self.rth_history = []
        self.viol_history = []
        self.hs_history = []

    def push(self, ev):
        if ev.geom_feasible:
            if ev.violation < self.best_viol:
                self.best_viol = ev.violation
            if ev.feasible and ev.r_th < self.best_rth:
                self.best_rth = ev.r_th
        if self.best is None or ev.key < self.best.key:
            self.best = ev
        self.k += 1
        self.rth_history.append(self.best_rth)
        self.viol_history.append(self.best_viol)
        self.hs_history.append(self.best.hs)

    def finish(self, problem, name, seed, wall_time):
        return RunResult(
            name=name,
            seed=seed,
            n_eval=self.k,
            n_hit=problem.n_hit,
            wall_time=wall_time,
            best=self.best,
            rth_history=np.asarray(self.rth_history, dtype=float),
            viol_history=np.asarray(self.viol_history, dtype=float),
            hs_history=self.hs_history,
        )


# ---------------------------------------------------------------- 基线

def random_search(problem, budget, seed, name="随机搜索"):
    """基线：在参数空间内独立均匀采样，不做任何利用。"""
    rng = np.random.default_rng(seed)
    tracker = _Tracker()
    t0 = time.perf_counter()
    for _ in range(budget):
        tracker.push(problem.evaluate(problem.sample_random(rng)))
    return tracker.finish(problem, name, seed, time.perf_counter() - t0)


# ---------------------------------------------------------------- 自研算法

def adaptive_search(problem, budget, seed, name="自适应邻域搜索"):
    """自适应邻域搜索。

    每轮在锚点周围生成 CAND_PER_ROUND 个候选：
        有改善 → 锚点前移，步长收缩（利用，step × 0.75）
        无改善 → 步长放大（探索，step × 1.25）；连续 STALL_LIMIT 轮无改善
                 则把锚点拉回当前全局最优并重置步长，避免在死区里空转。
    """
    rng = np.random.default_rng(seed)
    tracker = _Tracker()
    t0 = time.perf_counter()

    # --- 初始随机采样：定位第一个锚点 ---
    anchor = None
    for _ in range(min(INIT_SAMPLES, budget)):
        ev = problem.evaluate(problem.sample_random(rng))
        tracker.push(ev)
        if anchor is None or ev.key < anchor.key:
            anchor = ev

    step = STEP_INIT
    stall = 0
    while tracker.k < budget:
        cands = []
        for _ in range(CAND_PER_ROUND):
            if tracker.k >= budget:
                break
            hs = _perturb(anchor.hs, step, rng)
            if not check_manufacturable(hs)[0]:
                # 扰动落进了几何不可行区（如翅片总宽超限），改取随机点
                hs = problem.sample_random(rng)
            ev = problem.evaluate(hs)
            tracker.push(ev)
            cands.append(ev)

        if not cands:
            break

        best_cand = min(cands, key=lambda e: e.key)
        if best_cand.key < anchor.key:
            anchor = best_cand
            step = max(STEP_MIN, step * 0.75)
            stall = 0
        else:
            step = min(STEP_MAX, step * 1.25)
            stall += 1
            if stall >= STALL_LIMIT:
                anchor = tracker.best
                step = RESTART_STEP
                stall = 0

    return tracker.finish(problem, name, seed, time.perf_counter() - t0)
