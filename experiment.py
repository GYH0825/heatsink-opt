# -*- coding: utf-8 -*-
"""对比实验：随机搜索基线 vs 自适应邻域搜索。

在相同的评估预算下各跑若干随机种子，产出：
    results/convergence.png       收敛曲线（最优可行热阻 / 约束违反度）
    results/comparison.png        最终结果柱状对比（热阻 / 质量）
    results/temperature_field.png 基准设计与两种算法最优解的温度场
    results/results.csv           每次运行的完整记录

运行（在「项目文件」目录下）：
    python experiment.py
"""

import csv
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from sim import config as C
from sim.geometry import HeatSink, fin_gap
from sim.thermal import solve
from opt.problem import Problem
from opt.searchers import adaptive_search, random_search

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 实验设置
BUDGET = 400                      # 每种算法每个种子的评估预算
SEEDS = (0, 1, 2, 3, 4)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 对照用的基准设计：散热性能不错，但质量明显超标，用来体现约束的作用
REFERENCE = HeatSink(10, 0.030, 0.003, 0.005)

COLORS = ("#2c6fbb", "#d1721f")


# ---------------------------------------------------------------- 绘图工具

def _band(ax, runs, attr, color, floor=None):
    """把多个种子的一条历史曲线画成 均值 ± 标准差 的带状图。"""
    data = np.array([getattr(r, attr) for r in runs], dtype=float)
    data[~np.isfinite(data)] = np.nan
    if floor is not None:
        data = np.where(np.isfinite(data), np.maximum(data, floor), np.nan)

    ready = np.all(np.isfinite(data), axis=0)
    if not ready.any():
        return
    start = int(np.argmax(ready))
    x = np.arange(data.shape[1])[start:]
    mean = np.nanmean(data[:, start:], axis=0)
    std = np.nanstd(data[:, start:], axis=0)
    ax.plot(x, mean, color=color, lw=1.9)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, lw=0)


def plot_convergence(results, path):
    """左图：最优可行热阻随评估次数的下降；右图：约束违反度随评估次数的下降。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for (algo, runs), color in zip(results.items(), COLORS):
        _band(axes[0], runs, "rth_history", color)
        axes[0].plot([], [], color=color, lw=1.9,
                     label=f"{algo}（{len(runs)} 个种子的均值±标准差）")
        _band(axes[1], runs, "viol_history", color, floor=1e-4)

    axes[0].set_xlabel("设计评估次数")
    axes[0].set_ylabel("最优可行热阻 $R_{th}$ [°C/W]")
    axes[0].set_title("目标函数收敛（越低越好）")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("设计评估次数")
    axes[1].set_ylabel("最小约束违反度（0 以 $10^{-4}$ 显示）")
    axes[1].set_yscale("log")
    axes[1].set_title("约束满足过程（前 60 次评估）")

    for ax in axes:
        ax.grid(alpha=0.3, lw=0.6)
    axes[0].set_xlim(0, BUDGET)
    axes[1].set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(results, path):
    """柱状对比：最终最优热阻与对应质量。"""
    labels = list(results.keys())
    rth = [np.mean([r.best.r_th for r in runs]) for runs in results.values()]
    rth_std = [np.std([r.best.r_th for r in runs]) for runs in results.values()]
    mass = [np.mean([r.best.mass * 1e3 for r in runs]) for runs in results.values()]
    mass_std = [np.std([r.best.mass * 1e3 for r in runs]) for runs in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xs = np.arange(len(labels))

    axes[0].bar(xs, rth, yerr=rth_std, capsize=5, color=COLORS, width=0.55)
    axes[0].set_ylabel("最优可行热阻 $R_{th}$ [°C/W]")
    axes[0].set_title("相同预算下的最终热阻")

    axes[1].bar(xs, mass, yerr=mass_std, capsize=5, color=COLORS, width=0.55)
    axes[1].axhline(C.MASS_LIMIT * 1e3, color="#c0392b", ls="--", lw=1.2)
    axes[1].text(-0.42, C.MASS_LIMIT * 1e3 + 2, f"质量上限 {C.MASS_LIMIT * 1e3:.0f} g",
                 color="#c0392b", va="bottom", ha="left", fontsize=9)
    axes[1].set_ylabel("散热器质量 [g]")
    axes[1].set_title("最终设计的质量（约束上限 150 g）")
    axes[1].set_ylim(0, 172)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_temperature_fields(designs, path):
    """温度场对比：基准设计 + 各算法的一个代表性最优解。

    designs 是 (标签, 设计) 的列表；三幅图共用同一色标，便于直接比高低。
    """
    n = len(designs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6))
    if n == 1:
        axes = [axes]

    solved = [(label, hs, solve(hs)) for label, hs in designs]
    vmax = max(res.t_max for _l, _h, res in solved)
    extent = [0, C.WIDTH * 1e3, 0, C.HEIGHT * 1e3]
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    image = None
    for ax, (label, hs, res) in zip(axes, solved):
        field = np.ma.masked_invalid(res.t_field)
        image = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                          vmin=C.T_AMBIENT, vmax=vmax, interpolation="nearest")
        ok = res.t_max <= C.T_MAX_LIMIT and res.mass <= C.MASS_LIMIT
        ax.set_title(
            f"{label}\n"
            f"{hs.n_fins} 片 · 翅高 {hs.fin_height * 1e3:.1f} · "
            f"翅厚 {hs.fin_thickness * 1e3:.1f} · 基板 {hs.base_thickness * 1e3:.1f} mm\n"
            f"翅片间隙 {fin_gap(hs) * 1e3:.2f} mm\n"
            f"$T_{{max}}$={res.t_max:.1f}°C  $R_{{th}}$={res.r_th:.3f}°C/W  "
            f"m={res.mass * 1e3:.1f}g（{'可行' if ok else '越界'}）",
            fontsize=8.5)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.subplots_adjust(wspace=0.32)
    fig.text(0.5, 0.005,
             f"自然对流 h={C.H_CONVECTION:.0f} W/(m²·K)，环境 {C.T_AMBIENT:.0f}°C，"
             f"芯片 {C.CHIP_POWER:.0f} W，白色区域为空气",
             ha="center", fontsize=9)
    fig.colorbar(image, ax=axes, label="温度 [°C]", shrink=0.85)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def run_algorithms():
    results = {}
    for algo_name, fn in (("随机搜索", random_search), ("自适应邻域搜索", adaptive_search)):
        runs = []
        for seed in SEEDS:
            problem = Problem()          # 每个种子独立的缓存，保证比较公平
            run = fn(problem, BUDGET, seed, name=algo_name)
            stats = problem.cache_stats()
            print(f"  {run.label:<28} 真实求解 {stats['n_eval']:>3} 次"
                  f"（缓存命中 {stats['n_hit']:>3}）"
                  f"  单次 {stats['avg_ms']:.2f} ms"
                  f"  用时 {run.wall_time:.2f} s"
                  f"  →  {run.best.summary}")
            runs.append(run)
        results[algo_name] = runs
    return results


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "算法", "种子", "评估次数", "缓存命中", "墙钟耗时_s",
            "翅片数量", "翅片高度_mm", "翅片厚度_mm", "基板厚度_mm",
            "T_max_C", "热阻_C_per_W", "质量_g", "约束违反度", "是否可行",
            "几何不可行原因",
        ])
        for algo, runs in results.items():
            for run in runs:
                ev = run.best
                writer.writerow([
                    algo, run.seed, run.n_eval, run.n_hit, f"{run.wall_time:.3f}",
                    ev.hs.n_fins, f"{ev.hs.fin_height * 1e3:.2f}",
                    f"{ev.hs.fin_thickness * 1e3:.2f}",
                    f"{ev.hs.base_thickness * 1e3:.2f}",
                    f"{ev.t_max:.2f}", f"{ev.r_th:.4f}", f"{ev.mass * 1e3:.2f}",
                    f"{ev.violation:.4f}", int(ev.feasible), ev.reason,
                ])


def print_summary(results):
    print("\n" + "=" * 78)
    print("结果汇总（每个算法 {} 个种子 × {} 次评估）".format(len(SEEDS), BUDGET))
    print("=" * 78)

    ref = solve(REFERENCE)
    print(f"基准设计           {REFERENCE.describe()}")
    print(f"                  {ref.summary}")

    best_overall = None
    for algo, runs in results.items():
        rths = np.array([r.best.r_th for r in runs])
        masses = np.array([r.best.mass for r in runs]) * 1e3
        feas = sum(1 for r in runs if r.best.feasible)
        print(f"\n{algo}")
        print(f"  最优热阻      {rths.mean():.3f} ± {rths.std():.3f} °C/W"
              f"   （最好 {rths.min():.3f}）")
        print(f"  对应质量      {masses.mean():.1f} ± {masses.std():.1f} g")
        print(f"  找到可行解    {feas}/{len(runs)}")
        for r in runs:
            if best_overall is None or r.best.key < best_overall.best.key:
                best_overall = r

    if best_overall is not None:
        print(f"\n全局最优（{best_overall.label}）")
        print(f"  {best_overall.best.hs.describe()}")
        print(f"  {best_overall.best.summary}")

    base = np.mean([r.best.r_th for r in results["随机搜索"]])
    ours = np.mean([r.best.r_th for r in results["自适应邻域搜索"]])
    if base > 0:
        print(f"\n自适应邻域搜索相对随机搜索基线：热阻降低 {(base - ours) / base * 100:.1f}%")
    return best_overall


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 78)
    print("散热器几何参数优化：随机搜索 vs 自适应邻域搜索")
    print(f"网格 {C.NX}×{C.NY}   预算 {BUDGET} 次评估/种子   种子 {list(SEEDS)}")
    print("=" * 78)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results = run_algorithms()
    elapsed = time.perf_counter() - t0

    best_overall = print_summary(results)
    print(f"\n实验总耗时 {elapsed:.1f} s")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plot_convergence(results, os.path.join(RESULT_DIR, "convergence.png"))
        plot_comparison(results, os.path.join(RESULT_DIR, "comparison.png"))

    designs = [("基准设计（未优化）", REFERENCE)]
    for algo, runs in results.items():
        best_run = min(runs, key=lambda r: r.best.key)   # 该算法跨种子的最优
        designs.append((f"{algo}最优（seed={best_run.seed}）", best_run.best.hs))
    plot_temperature_fields(designs, os.path.join(RESULT_DIR, "temperature_field.png"))

    csv_path = os.path.join(RESULT_DIR, "results.csv")
    write_csv(results, csv_path)

    print("\n已输出：")
    for name in ("convergence.png", "comparison.png", "temperature_field.png", "results.csv"):
        print(f"  results/{name}")
    return best_overall


if __name__ == "__main__":
    main()
