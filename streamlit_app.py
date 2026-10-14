"""散热器几何参数调参台（Streamlit 云端版）。

与 webapp.py 共用同一套物理内核 sim/ 和优化层 opt/，
只把交互层从自研 HTML/FastAPI 前端换成 Streamlit，
以便部署到 Streamlit Community Cloud（免费、无需绑卡）。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from opt.problem import Problem, constraint_violation, discretize
from opt.searchers import adaptive_search, random_search
from sim import config as C
from sim.geometry import HeatSink, check_manufacturable, fin_gap
from sim.thermal import solve

st.set_page_config(page_title="散热器几何参数调参台", page_icon="🌡", layout="wide")

# 与 webapp.py 一致：(对外键, 中文名, 换算系数, 步长, 单位, 是否整数)
_FIELDS = (
    ("n_fins", "翅片数量", 1.0, 1.0, "片", True),
    ("fin_height_mm", "翅片高度", 1e3, 0.1, "mm", False),
    ("fin_thickness_mm", "翅片厚度", 1e3, 0.1, "mm", False),
    ("base_thickness_mm", "基板厚度", 1e3, 0.1, "mm", False),
)
# _FIELDS 与 C.PARAM_SPEC 顺序一一对应
_SPEC = {f[0]: (p[1], p[2], p[3]) for f, p in zip(_FIELDS, C.PARAM_SPEC)}
_DEFAULT = {"n_fins": 8, "fin_height_mm": 25.0,
            "fin_thickness_mm": 3.0, "base_thickness_mm": 4.0}


def make_hs(vals):
    """把前端的毫米输入转成内部 SI 设计，并量化到制造精度。"""
    return discretize(HeatSink(
        n_fins=int(vals["n_fins"]),
        fin_height=float(vals["fin_height_mm"]) / 1e3,
        fin_thickness=float(vals["fin_thickness_mm"]) / 1e3,
        base_thickness=float(vals["base_thickness_mm"]) / 1e3,
    ))


def design_of(hs):
    return {"n_fins": int(hs.n_fins),
            "fin_height_mm": round(hs.fin_height * 1e3, 2),
            "fin_thickness_mm": round(hs.fin_thickness * 1e3, 2),
            "base_thickness_mm": round(hs.base_thickness * 1e3, 2)}


def field_fig(res):
    """把温度场画成热力图，白色区域是空气。"""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    field = np.ma.masked_invalid(res.t_field)
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("white")
    im = ax.imshow(field, origin="lower",
                   extent=[0, C.WIDTH * 1e3, 0, C.HEIGHT * 1e3],
                   cmap=cmap, vmin=C.T_AMBIENT, vmax=res.t_max,
                   interpolation="nearest")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"T_max = {res.t_max:.1f} °C    R_th = {res.r_th:.3f} °C/W    "
                 f"m = {res.mass * 1e3:.1f} g", fontsize=9)
    fig.colorbar(im, ax=ax, label="Temperature [°C]", shrink=0.9)
    fig.tight_layout()
    return fig


# ---------------- 状态初始化 ----------------
for _k, *_rest in _FIELDS:
    if ("w_" + _k) not in st.session_state:
        st.session_state["w_" + _k] = _DEFAULT[_k]


def run_search():
    """按钮回调：跑一次闭环搜索，并把最优设计回填到滑块。"""
    method = st.session_state.get("m_method", "adaptive")
    budget = int(st.session_state.get("m_budget", 200))
    seed = int(st.session_state.get("m_seed", 0))
    fn = adaptive_search if method == "adaptive" else random_search
    run = fn(Problem(), budget, seed)

    best = run.best
    if not best.geom_feasible:
        st.session_state["search_error"] = f"搜索未找到几何可行的设计：{best.reason}"
        st.session_state.pop("search_out", None)
        return

    st.session_state["search_error"] = None
    st.session_state["search_out"] = {
        "method": method, "budget": budget, "seed": seed,
        "wall_time_s": round(run.wall_time, 2),
        "curve": [None if not np.isfinite(v) else float(v) for v in run.rth_history],
    }
    d = design_of(best.hs)
    for _k, *_rest in _FIELDS:
        st.session_state["w_" + _k] = d[_k]


st.title("散热器几何参数调参台")
st.caption("拖动左侧滑块即时查看温度场；点「开始搜索」由优化器自动闭环寻优。"
           "网页与 AGH 智能体调用的是同一套物理内核。")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("设计参数")
    for key, label, scale, step, unit, is_int in _FIELDS:
        lo, hi, _ = _SPEC[key]
        lo_s, hi_s = lo * scale, hi * scale
        if is_int:
            st.slider(f"{label}（{unit}）", int(round(lo_s)), int(round(hi_s)),
                      step=int(step), key="w_" + key)
        else:
            st.slider(f"{label}（{unit}）", float(lo_s), float(hi_s),
                      step=float(step), key="w_" + key)

    st.divider()
    st.subheader("自动搜索最优")
    st.selectbox("搜索方法", ["adaptive", "random"],
                 format_func=lambda m: "自适应邻域搜索" if m == "adaptive" else "随机搜索",
                 key="m_method")
    st.slider("评估预算", 20, 2000, 200, step=20, key="m_budget")
    st.number_input("随机种子", min_value=0, max_value=9999, value=0, step=1, key="m_seed")
    st.button("开始搜索", type="primary", use_container_width=True, on_click=run_search)

    with st.expander("约束与物理参数"):
        st.markdown(
            f"- 最高温度上限：**{C.T_MAX_LIMIT:.1f} °C**\n"
            f"- 质量上限：**{C.MASS_LIMIT * 1e3:.1f} g**\n"
            f"- 最小翅片间距：**{C.MIN_GAP * 1e3:.2f} mm**\n"
            f"- 包络尺寸：{C.WIDTH * 1e3:.0f} × {C.HEIGHT * 1e3:.0f} × {C.DEPTH * 1e3:.0f} mm\n"
            f"- 材料：铝合金，k = {C.K_ALUMINUM:.0f} W/(m·K)\n"
            f"- 对流：h = {C.H_CONVECTION:.0f} W/(m²·K)，环境 {C.T_AMBIENT:.0f} °C\n"
            f"- 芯片功率：{C.CHIP_POWER:.1f} W　网格：{C.NX} × {C.NY}"
        )

vals = {k: st.session_state["w_" + k] for k, *_rest in _FIELDS}
hs = make_hs(vals)

with right:
    st.subheader("温度场与指标")
    ok, reason = check_manufacturable(hs)
    if not ok:
        st.error(f"几何不可行：{reason}")
    else:
        res = solve(hs)
        fig = field_fig(res)
        st.pyplot(fig)
        plt.close(fig)

        violation = constraint_violation(res.t_max, res.mass)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最高温度", f"{res.t_max:.1f} °C")
        c2.metric("热阻", f"{res.r_th:.3f} °C/W")
        c3.metric("质量", f"{res.mass * 1e3:.1f} g")
        c4.metric("约束裕度", f"{violation:.3f}")
        st.caption(f"翅片间距 {fin_gap(hs) * 1e3:.2f} mm　|　"
                   f"{'✅ 满足全部约束' if violation <= 0 else '⚠️ 存在约束违反'}")

    err = st.session_state.get("search_error")
    if err:
        st.error(err)

    out = st.session_state.get("search_out")
    if out:
        st.divider()
        st.subheader("搜索过程")
        name = "自适应邻域搜索" if out["method"] == "adaptive" else "随机搜索"
        st.caption(f"{name}　预算 {out['budget']} 次　种子 {out['seed']}　"
                   f"耗时 {out['wall_time_s']} s")
        st.line_chart({"当前最优热阻 R_th [°C/W]": out["curve"]},
                      x_label="评估次数", y_label="R_th [°C/W]")
