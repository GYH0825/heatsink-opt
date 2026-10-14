# -*- coding: utf-8 -*-
"""MCP 工具层：把散热器的仿真与优化能力暴露给 Agnes Harness（AGH）。

通过 stdio 与 AGH 通信，因此本模块**禁止向 stdout 打印任何内容**
（stdout 被 JSON-RPC 独占），调试信息一律走日志（stderr）。

六个工具的划分对应赛事要求的三步闭环：

    任务规划    get_problem_spec
    能力调用    simulate_design / search_designs / render_temperature_field
    反馈与验证  verify_design / compare_methods

单独运行本模块会用 stdio 启动一个 MCP server，由 AGH 作为客户端拉起。
"""

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from sim import config as C
from sim.geometry import HeatSink, fin_gap
from sim.thermal import solve
from opt.problem import Problem, constraint_violation, discretize
from opt.searchers import adaptive_search, random_search

server = MCPServer(
    name="heatsink-optimizer",
    title="散热器几何参数闭环优化",
    version="1.0.0",
    instructions=(
        "针对自然对流条件下的铝制板翅散热器做几何参数优化。"
        "典型流程：先调用 get_problem_spec 了解参数空间与约束，"
        "用 simulate_design 试算设计，用 search_designs 做搜索，"
        "再对得到的方案调用 verify_design 做网格无关性复核；"
        "若复核不通过或结果不理想，调整参数后重新搜索。"
        "所有长度输入输出统一使用毫米。"
    ),
)

# 六个工具的 MCP 注解：它们都只在自身进程内做数值计算，不修改调用方的工作区
# （render_temperature_field 只在本工具自己的 results/ 目录落一张输出图）。
# 声明为只读 / 幂等 / 封闭域后，AGH 会把它们判为非破坏性工具，无需人工批准即可调用
# （判定见 AGH packages/base/src/mcp/register.ts 对 readOnlyHint 的处理）。
_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)

# 跨工具调用复用的评估缓存：AGH 反复询问同一设计时不必重新求解。
_CACHE = {}

# 参数对外呈现用的名字与单位换算（内部是 SI，对 AGH 一律用毫米/片）
_VAR_OUT = {
    "n_fins": ("n_fins", "片", 1.0, True),
    "fin_height": ("fin_height_mm", "mm", 1e3, False),
    "fin_thickness": ("fin_thickness_mm", "mm", 1e3, False),
    "base_thickness": ("base_thickness_mm", "mm", 1e3, False),
}


def _make_hs(n_fins, fin_height_mm, fin_thickness_mm, base_thickness_mm):
    """把毫米输入转成内部 SI 单位的 HeatSink，并量化到制造精度。"""
    return discretize(HeatSink(
        n_fins=int(n_fins),
        fin_height=float(fin_height_mm) / 1e3,
        fin_thickness=float(fin_thickness_mm) / 1e3,
        base_thickness=float(base_thickness_mm) / 1e3,
    ))


def _describe(hs, res, violation):
    return {
        "design": {
            "n_fins": hs.n_fins,
            "fin_height_mm": round(hs.fin_height * 1e3, 3),
            "fin_thickness_mm": round(hs.fin_thickness * 1e3, 3),
            "base_thickness_mm": round(hs.base_thickness * 1e3, 3),
            "fin_gap_mm": round(fin_gap(hs) * 1e3, 3),
        },
        "t_max_c": round(res.t_max, 2),
        "r_th_c_per_w": round(res.r_th, 4),
        "mass_g": round(res.mass * 1e3, 2),
        "conv_area_cm2": round(res.conv_area * 1e4, 2),
        "constrained": {
            "t_max_limit_c": C.T_MAX_LIMIT,
            "mass_limit_g": C.MASS_LIMIT * 1e3,
            "t_max_ok": bool(res.t_max <= C.T_MAX_LIMIT),
            "mass_ok": bool(res.mass <= C.MASS_LIMIT),
        },
        "violation": round(violation, 4),
        "feasible": bool(violation <= 0.0),
        "summary": res.summary,
    }


# ---------------------------------------------------------------- 任务规划

@server.tool(description="获取优化问题的完整定义：设计变量与其取值范围（毫米）、"
                         "物理常量、散热器外形、网格分辨率、约束上限与目标函数。"
                         "做任何规划之前先调用它。",
             annotations=_READ_ONLY)
def get_problem_spec() -> dict:
    return {
        "objective": "最小化结到环境的热阻 R_th [°C/W]；约束不满足时按违反度优先处理",
        "variables": [
            {"name": _VAR_OUT[name][0],
             "min": round(lo * _VAR_OUT[name][2], 4),
             "max": round(hi * _VAR_OUT[name][2], 4),
             "unit": _VAR_OUT[name][1],
             "integer": bool(_VAR_OUT[name][3])}
            for name, lo, hi, _is_int in C.PARAM_SPEC
        ],
        "variables_note": "对外全部使用毫米与片；内部一律按 SI 计算。"
                          "simulate_design / verify_design 的参数名与本列表一致",
        "constraints": {
            "t_max_limit_c": C.T_MAX_LIMIT,
            "mass_limit_g": C.MASS_LIMIT * 1e3,
            "min_fin_gap_mm": C.MIN_GAP * 1e3,
            "envelope_mm": {"width": C.WIDTH * 1e3, "height": C.HEIGHT * 1e3,
                            "depth": C.DEPTH * 1e3},
        },
        "physics": {
            "material": "铝合金",
            "k_w_per_mk": C.K_ALUMINUM,
            "h_convection_w_per_m2k": C.H_CONVECTION,
            "ambient_c": C.T_AMBIENT,
            "chip_power_w": C.CHIP_POWER,
            "density_kg_per_m3": C.RHO_ALUMINUM,
            "model": "二维截面稳态热传导，格心有限体积离散 + 稀疏直接求解；"
                     "基板底面施加热流，其余表面自然对流",
        },
        "mesh": {"nx": C.NX, "ny": C.NY, "cell_mm": round(C.WIDTH / C.NX * 1e3, 3)},
        "available_methods": ["random（随机搜索基线）", "adaptive（自适应邻域搜索）"],
    }


# ---------------------------------------------------------------- 能力调用

@server.tool(description="仿真单个散热器设计，返回最高温度、热阻、质量、翅片间隙、"
                         "约束是否满足以及是否可行。输入尺寸单位为毫米。",
             annotations=_READ_ONLY)
def simulate_design(n_fins: int, fin_height_mm: float,
                    fin_thickness_mm: float, base_thickness_mm: float) -> dict:
    hs = _make_hs(n_fins, fin_height_mm, fin_thickness_mm, base_thickness_mm)
    key = (hs.n_fins, hs.fin_height, hs.fin_thickness, hs.base_thickness)

    if key in _CACHE:
        return {**_CACHE[key], "cache_hit": True}

    res = solve(hs)
    if not res.feasible:
        return {
            "geometry_feasible": False,
            "reason": res.reason,
            "design": {
                "n_fins": hs.n_fins,
                "fin_height_mm": round(hs.fin_height * 1e3, 3),
                "fin_thickness_mm": round(hs.fin_thickness * 1e3, 3),
                "base_thickness_mm": round(hs.base_thickness * 1e3, 3),
                "fin_gap_mm": round(fin_gap(hs) * 1e3, 3),
            },
            "feasible": False,
            "note": "该设计连制造可行性都不满足，未进入热仿真",
        }

    violation = constraint_violation(res.t_max, res.mass)
    out = _describe(hs, res, violation)
    out["geometry_feasible"] = True
    out["cache_hit"] = False
    _CACHE[key] = out
    return out


@server.tool(description="运行一次参数搜索，返回最优设计及其性能指标，以及收敛过程摘要。"
                         "method 取 adaptive（自适应邻域搜索，默认）或 random（随机搜索基线）；"
                         "budget 是设计评估次数上限（20–2000）。",
             annotations=_READ_ONLY)
def search_designs(method: str = "adaptive", budget: int = 200, seed: int = 0) -> dict:
    if method not in ("adaptive", "random"):
        return {"error": f"未知的搜索方法 {method!r}，可选 adaptive / random"}
    budget = max(20, min(int(budget), 2000))

    problem = Problem()
    fn = adaptive_search if method == "adaptive" else random_search
    run = fn(problem, budget, int(seed))

    best = run.best
    hist = run.rth_history
    step = max(1, len(hist) // 20)          # 收敛曲线只回传约 20 个采样点
    curve = [
        {"n_eval": int(i + 1),
         "best_r_th": None if hist[i] == float("inf") else round(float(hist[i]), 4)}
        for i in range(0, len(hist), step)
    ]

    return {
        "method": method,
        "budget": budget,
        "seed": int(seed),
        "n_solves": run.n_eval,
        "cache_hits": run.n_hit,
        "wall_time_s": round(run.wall_time, 3),
        "best_design": {
            "n_fins": best.hs.n_fins,
            "fin_height_mm": round(best.hs.fin_height * 1e3, 3),
            "fin_thickness_mm": round(best.hs.fin_thickness * 1e3, 3),
            "base_thickness_mm": round(best.hs.base_thickness * 1e3, 3),
            "fin_gap_mm": round(fin_gap(best.hs) * 1e3, 3),
        },
        "best_t_max_c": round(best.t_max, 2),
        "best_r_th_c_per_w": round(best.r_th, 4),
        "best_mass_g": round(best.mass * 1e3, 2),
        "feasible": bool(best.feasible),
        "violation": round(best.violation, 4),
        "convergence": curve,
    }


@server.tool(description="把某个设计的温度场渲染成图片，用于人工核查与演示取证。"
                         "返回图片路径。输入尺寸单位为毫米。",
             annotations=_READ_ONLY)
def render_temperature_field(n_fins: int, fin_height_mm: float,
                             fin_thickness_mm: float, base_thickness_mm: float) -> dict:
    import os

    from experiment import plot_temperature_fields

    hs = _make_hs(n_fins, fin_height_mm, fin_thickness_mm, base_thickness_mm)
    res = solve(hs)
    if not res.feasible:
        return {"error": res.reason}

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    name = (f"field_{hs.n_fins}fins_h{hs.fin_height * 1e3:.1f}"
            f"_t{hs.fin_thickness * 1e3:.1f}_b{hs.base_thickness * 1e3:.1f}.png")
    path = os.path.join(out_dir, name)

    label = f"{hs.n_fins} 片 · 翅高 {hs.fin_height * 1e3:.1f} mm"
    plot_temperature_fields([(label, hs)], path)
    return {"image_path": path, "t_max_c": round(res.t_max, 2),
            "r_th_c_per_w": round(res.r_th, 4), "mass_g": round(res.mass * 1e3, 2)}


# ---------------------------------------------------------------- 反馈与验证

@server.tool(description="独立复核一个设计：从零重算几何可行性与约束，并把网格加密 refine 倍"
                         "重新求解，检查热阻是否收敛（相对偏差不超过 tol 即认为通过）。"
                         "用于验证搜索结果是否可信。输入尺寸单位为毫米。",
             annotations=_READ_ONLY)
def verify_design(n_fins: int, fin_height_mm: float, fin_thickness_mm: float,
                  base_thickness_mm: float, refine: int = 2, tol: float = 0.05) -> dict:
    refine = max(1, min(int(refine), 4))
    hs = _make_hs(n_fins, fin_height_mm, fin_thickness_mm, base_thickness_mm)

    coarse = solve(hs)
    if not coarse.feasible:
        return {"passed": False, "reason": f"几何不可制造：{coarse.reason}",
                "geometry_feasible": False}

    fine = solve(hs, nx=C.NX * refine, ny=C.NY * refine)
    violation = constraint_violation(coarse.t_max, coarse.mass)
    rel_err = abs(fine.r_th - coarse.r_th) / coarse.r_th

    return {
        "geometry_feasible": True,
        "mass_ok": bool(coarse.mass <= C.MASS_LIMIT),
        "t_max_ok": bool(coarse.t_max <= C.T_MAX_LIMIT),
        "constraints_satisfied": bool(violation <= 0.0),
        "violation": round(violation, 4),
        "coarse": {"mesh": f"{C.NX}×{C.NY}",
                   "r_th_c_per_w": round(coarse.r_th, 4),
                   "t_max_c": round(coarse.t_max, 2)},
        "fine": {"mesh": f"{C.NX * refine}×{C.NY * refine}",
                 "r_th_c_per_w": round(fine.r_th, 4),
                 "t_max_c": round(fine.t_max, 2)},
        "relative_error": round(rel_err, 4),
        "tolerance": tol,
        "mesh_independent": bool(rel_err <= tol),
        "passed": bool(rel_err <= tol and violation <= 0.0),
        "verdict": ("复核通过：网格无关且在约束范围内"
                    if rel_err <= tol and violation <= 0.0 else
                    "复核未通过：" + ("热阻对网格敏感，需加密网格或改用更粗的离散"
                                       if rel_err > tol else "约束未满足")),
    }


@server.tool(description="在相同评估预算下对比两种搜索方法（各跑多个随机种子），"
                         "返回各自最优热阻的均值、标准差与相对改进。用于结论的最后核验。",
             annotations=_READ_ONLY)
def compare_methods(budget: int = 200, seeds: int = 3) -> dict:
    budget = max(50, min(int(budget), 2000))
    seeds = max(1, min(int(seeds), 10))

    summary = {}
    for name, fn in (("random", random_search), ("adaptive", adaptive_search)):
        rths = []
        for seed in range(seeds):
            run = fn(Problem(), budget, seed)
            if run.best.feasible:
                rths.append(run.best.r_th)
        stat = {
            "n_feasible": len(rths),
            "n_seeds": seeds,
            "mean_r_th": round(float(sum(rths) / len(rths)), 4) if rths else None,
            "min_r_th": round(min(rths), 4) if rths else None,
        }
        if rths:
            mean = sum(rths) / len(rths)
            var = sum((x - mean) ** 2 for x in rths) / len(rths)
            stat["std_r_th"] = round(var ** 0.5, 4)
        summary[name] = stat

    base = summary["random"].get("mean_r_th")
    ours = summary["adaptive"].get("mean_r_th")
    improvement = None
    if base and ours:
        improvement = round((base - ours) / base * 100, 2)

    return {
        "budget_per_seed": budget,
        "seeds": seeds,
        "methods": summary,
        "improvement_percent": improvement,
        "note": "improvement_percent 为自适应邻域搜索相对随机搜索基线的热阻降低比例",
    }


if __name__ == "__main__":
    server.run("stdio")
