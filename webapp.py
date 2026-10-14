# -*- coding: utf-8 -*-
"""本地网页操作台：拖动滑块改几何参数，实时看温度场与指标；一键跑自动搜索。

运行（在「项目文件」目录下）：
    python webapp.py
然后浏览器打开 http://127.0.0.1:8000

后端只有三个接口：
    GET  /api/spec      参数空间与约束（前端据此生成滑块，永远与 config.py 同步）
    POST /api/simulate  单个设计试算 → 指标 + 温度场 PNG
    POST /api/optimize  自动搜索最优 → 最优设计 + 收敛曲线

绘图不用 pyplot，直接构造 Figure + Agg canvas，避免多线程下 pyplot 全局状态打架。
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from opt.problem import Problem, constraint_violation, discretize
from opt.searchers import adaptive_search, random_search
from sim import config as C
from sim.geometry import HeatSink, check_manufacturable, fin_gap
from sim.thermal import solve

# ---------------------------------------------------------------- 对外参数定义
# (键名, 中文名, 单位换算系数, 步长, 单位, 是否整数)
_FIELDS = (
    ("n_fins", "翅片数量", 1.0, 1.0, "片", True),
    ("fin_height_mm", "翅片高度", 1e3, 0.1, "mm", False),
    ("fin_thickness_mm", "翅片厚度", 1e3, 0.1, "mm", False),
    ("base_thickness_mm", "基板厚度", 1e3, 0.1, "mm", False),
)

# 前端滑块按这个顺序生成；对外一律毫米/片。
# _FIELDS 与 C.PARAM_SPEC 顺序一一对应，据此把对外键名映射到参数范围。
_SPEC_BY_KEY = {f[0]: (p[1], p[2], p[3]) for f, p in zip(_FIELDS, C.PARAM_SPEC)}

DEFAULT = {"n_fins": 8, "fin_height_mm": 25.0,
           "fin_thickness_mm": 3.0, "base_thickness_mm": 4.0}


def _make_hs(d):
    """把前端的毫米输入转成内部 SI 单位的设计，并量化到制造精度。"""
    return discretize(HeatSink(
        n_fins=int(d["n_fins"]),
        fin_height=float(d["fin_height_mm"]) / 1e3,
        fin_thickness=float(d["fin_thickness_mm"]) / 1e3,
        base_thickness=float(d["base_thickness_mm"]) / 1e3,
    ))


def _field_png(res):
    """把温度场渲染成 PNG 的 data URI。白色区域是空气。"""
    fig = Figure(figsize=(5.6, 4.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    field = np.ma.masked_invalid(res.t_field)
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("white")

    image = ax.imshow(field, origin="lower",
                      extent=[0, C.WIDTH * 1e3, 0, C.HEIGHT * 1e3],
                      cmap=cmap, vmin=C.T_AMBIENT, vmax=res.t_max,
                      interpolation="nearest")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"T_max = {res.t_max:.1f} C     R_th = {res.r_th:.3f} C/W     "
                 f"m = {res.mass * 1e3:.1f} g", fontsize=9)
    fig.colorbar(image, ax=ax, label="Temperature [C]", shrink=0.9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _metrics(hs, res):
    """统一的结果结构，试算和搜索都用它。"""
    violation = constraint_violation(res.t_max, res.mass)
    gap = fin_gap(hs)
    return {
        "design": {
            "n_fins": hs.n_fins,
            "fin_height_mm": round(hs.fin_height * 1e3, 2),
            "fin_thickness_mm": round(hs.fin_thickness * 1e3, 2),
            "base_thickness_mm": round(hs.base_thickness * 1e3, 2),
            "fin_gap_mm": round(gap * 1e3, 3),
        },
        "t_max_c": round(res.t_max, 2),
        "r_th_c_per_w": round(res.r_th, 4),
        "mass_g": round(res.mass * 1e3, 2),
        "violation": round(violation, 4),
        "feasible": bool(violation <= 0.0),
        "t_max_ok": bool(res.t_max <= C.T_MAX_LIMIT),
        "mass_ok": bool(res.mass <= C.MASS_LIMIT),
        "limits": {"t_max_c": C.T_MAX_LIMIT, "mass_g": C.MASS_LIMIT * 1e3,
                   "min_fin_gap_mm": C.MIN_GAP * 1e3},
    }


# ---------------------------------------------------------------- 请求 / 响应

class DesignIn(BaseModel):
    n_fins: int
    fin_height_mm: float
    fin_thickness_mm: float
    base_thickness_mm: float


class OptimizeIn(BaseModel):
    method: str = "adaptive"
    budget: int = 200
    seed: int = 0


app = FastAPI(title="散热器几何参数优化操作台")


@app.get("/api/spec")
def api_spec():
    variables = []
    for key, label, scale, step, unit, is_int in _FIELDS:
        lo, hi, _ = _SPEC_BY_KEY[key]
        variables.append({"key": key, "label": label,
                          "min": round(lo * scale, 4), "max": round(hi * scale, 4),
                          "step": step, "unit": unit, "int": is_int,
                          "default": DEFAULT[key]})
    return {
        "variables": variables,
        "constraints": {"t_max_limit_c": C.T_MAX_LIMIT,
                        "mass_limit_g": C.MASS_LIMIT * 1e3,
                        "min_fin_gap_mm": C.MIN_GAP * 1e3,
                        "envelope_mm": {"width": C.WIDTH * 1e3,
                                        "height": C.HEIGHT * 1e3,
                                        "depth": C.DEPTH * 1e3}},
        "physics": {"material": "铝合金", "k_w_per_mk": C.K_ALUMINUM,
                    "h_w_per_m2k": C.H_CONVECTION, "ambient_c": C.T_AMBIENT,
                    "chip_power_w": C.CHIP_POWER},
        "mesh": {"nx": C.NX, "ny": C.NY},
    }


@app.post("/api/simulate")
def api_simulate(d: DesignIn):
    hs = _make_hs(d.model_dump())

    ok, reason = check_manufacturable(hs)
    if not ok:
        return {"geometry_feasible": False, "reason": reason,
                "design": {"n_fins": hs.n_fins,
                           "fin_height_mm": round(hs.fin_height * 1e3, 2),
                           "fin_thickness_mm": round(hs.fin_thickness * 1e3, 2),
                           "base_thickness_mm": round(hs.base_thickness * 1e3, 2),
                           "fin_gap_mm": round(fin_gap(hs) * 1e3, 3)}}

    res = solve(hs)
    return {"geometry_feasible": True, "image": _field_png(res),
            **_metrics(hs, res)}


@app.post("/api/optimize")
def api_optimize(o: OptimizeIn):
    if o.method not in ("adaptive", "random"):
        return {"error": f"未知方法 {o.method!r}"}

    budget = max(20, min(int(o.budget), 2000))
    problem = Problem()
    fn = adaptive_search if o.method == "adaptive" else random_search
    run = fn(problem, budget, int(o.seed))

    best = run.best
    if not best.geom_feasible:
        return {"error": f"搜索未找到几何可行的设计：{best.reason}"}

    res = solve(best.hs)
    hist = run.rth_history
    step = max(1, len(hist) // 60)
    curve = [{"n_eval": int(i + 1),
              "best_r_th": None if not np.isfinite(hist[i]) else round(float(hist[i]), 4)}
             for i in range(0, len(hist), step)]

    return {
        "method": o.method,
        "budget": budget,
        "seed": int(o.seed),
        "wall_time_s": round(run.wall_time, 2),
        "image": _field_png(res),
        "curve": curve,
        **_metrics(best.hs, res),
    }


# ---------------------------------------------------------------- 前端页面

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>散热器几何参数优化操作台</title>
<style>
:root{
  --bg:#f5f6f8; --panel:#ffffff; --line:#e3e6ea; --text:#1b1f24; --muted:#6b7280;
  --brand:#0b7a45; --brand-soft:#e7f7ef; --warn:#b42318; --warn-soft:#fdecea;
  --radius:10px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.55 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}
header{padding:18px 26px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{margin:0;font-size:17px;font-weight:600}
header p{margin:4px 0 0;color:var(--muted);font-size:13px}
.wrap{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:18px 26px 40px;
  align-items:start}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:16px 18px}
.panel h2{margin:0 0 12px;font-size:14px;font-weight:600}
.field{margin-bottom:14px}
.field .row{display:flex;justify-content:space-between;align-items:baseline;
  font-size:13px;margin-bottom:4px}
.field .val{font-weight:600;font-variant-numeric:tabular-nums}
.field .val span{font-weight:400;color:var(--muted);font-size:12px;margin-left:2px}
input[type=range]{width:100%;accent-color:var(--brand);margin:0}
.hint{font-size:12px;color:var(--muted);margin-top:2px}
hr{border:0;border-top:1px solid var(--line);margin:16px 0}
button{font:inherit;border-radius:8px;padding:9px 14px;cursor:pointer;border:1px solid var(--line);
  background:#fff;color:var(--text);transition:.12s}
button:hover{border-color:#c3c8cf}
button.primary{background:var(--brand);border-color:var(--brand);color:#fff;font-weight:600}
button.primary:hover{filter:brightness(1.08)}
button:disabled{opacity:.55;cursor:progress}
button.block{width:100%}
.opt-row{display:flex;gap:8px}
.opt-row label{flex:1;font-size:12px;color:var(--muted)}
select,input[type=number]{width:100%;font:inherit;padding:6px 8px;border:1px solid var(--line);
  border-radius:8px;background:#fff;color:var(--text);margin-top:3px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:var(--radius);padding:12px 14px}
.card .k{font-size:12px;color:var(--muted)}
.card .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.card .v small{font-size:12px;font-weight:400;color:var(--muted);margin-left:2px}
.bar{height:5px;border-radius:3px;background:#eceef1;margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--brand);width:0}
.bar.over i{background:var(--warn)}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.pill.ok{background:var(--brand-soft);color:var(--brand)}
.pill.bad{background:var(--warn-soft);color:var(--warn)}
.status{margin-bottom:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;
  color:var(--muted)}
#fieldWrap{margin-top:14px}
#field{width:100%;max-width:640px;display:block;margin:0 auto;border-radius:8px}
.msg{margin-top:10px;font-size:13px;color:var(--muted);min-height:18px}
.msg.bad{color:var(--warn)}
#convBox{margin-top:16px;display:none;position:relative}
#conv{width:100%;height:150px;display:block;cursor:grab}
#conv.dragging{cursor:grabbing}
#conv.boxing{cursor:crosshair}
.convhint{font-size:12px;color:var(--muted);margin-top:4px}
.convtip{position:absolute;left:0;top:0;display:none;pointer-events:none;z-index:5;
  background:#111827;color:#fff;font-size:12px;line-height:1.5;padding:4px 8px;border-radius:6px;
  white-space:nowrap;transform:translate(-50%,-135%)}
.specline{margin-top:14px;font-size:12px;color:var(--muted);line-height:1.7}
</style>
</head>
<body>
<header>
  <h1>散热器几何参数优化操作台</h1>
  <p id="subtitle">拖动滑块改参数 → 实时看温度场与指标；或一键让算法自动找最优设计</p>
</header>

<div class="wrap">
  <div>
    <div class="panel">
      <h2>设计参数</h2>
      <div id="sliders"></div>
      <div class="msg" id="simMsg"></div>
    </div>

    <div class="panel" style="margin-top:18px">
      <h2>自动搜索最优</h2>
      <div class="opt-row">
        <label>搜索方法
          <select id="method">
            <option value="adaptive">自适应邻域搜索</option>
            <option value="random">随机搜索（基线）</option>
          </select>
        </label>
      </div>
      <div class="opt-row" style="margin-top:10px">
        <label>评估预算
          <input type="number" id="budget" value="200" min="20" max="2000" step="10">
        </label>
        <label>随机种子
          <input type="number" id="seed" value="0" min="0" step="1">
        </label>
      </div>
      <button class="primary block" id="optBtn" style="margin-top:12px">自动搜索最优</button>
      <div class="msg" id="optMsg"></div>
    </div>
  </div>

  <div class="panel">
    <h2>结果</h2>
    <div class="status">
      <span id="statePill" class="pill ok">就绪</span>
      <span id="stateText">正在加载参数空间…</span>
    </div>

    <div class="cards">
      <div class="card">
        <div class="k">最高温度 T<sub>max</sub></div>
        <div class="v" id="vTmax">—</div>
        <div class="bar" id="bTmax"><i></i></div>
      </div>
      <div class="card">
        <div class="k">热阻 R<sub>th</sub></div>
        <div class="v" id="vRth">—</div>
      </div>
      <div class="card">
        <div class="k">质量</div>
        <div class="v" id="vMass">—</div>
        <div class="bar" id="bMass"><i></i></div>
      </div>
      <div class="card">
        <div class="k">翅片间隙</div>
        <div class="v" id="vGap">—</div>
      </div>
    </div>

    <div id="fieldWrap">
      <img id="field" alt="温度场">
    </div>

    <div id="convBox">
      <h2 style="margin-top:18px">搜索收敛过程</h2>
      <canvas id="conv" width="1040" height="300"></canvas>
      <div id="convTip" class="convtip"></div>
      <div class="convhint">滚轮缩放 · 拖动平移 · Shift+拖动框选放大 · 双击复位 · 悬停显示数值（Ctrl+滚轮仍是浏览器整页缩放）</div>
    </div>

    <div class="specline" id="specline"></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const sliders = {};
let spec = null;
let simTimer = null;
let busy = false;

function fmt(v, n) { return Number(v).toFixed(n); }

function buildSliders(){
  const box = $('sliders');
  box.innerHTML = '';
  for (const v of spec.variables){
    sliders[v.key] = v.default;
    const el = document.createElement('div');
    el.className = 'field';
    el.innerHTML =
      '<div class="row"><span>' + v.label + '</span>' +
      '<span class="val" id="val_' + v.key + '">' + v.default +
      '<span>' + v.unit + '</span></span></div>' +
      '<input type="range" id="sld_' + v.key + '" min="' + v.min + '" max="' + v.max +
      '" step="' + v.step + '" value="' + v.default + '">' +
      '<div class="hint">范围 ' + v.min + ' – ' + v.max + ' ' + v.unit + '</div>';
    box.appendChild(el);
    const s = el.querySelector('input');
    s.addEventListener('input', () => {
      const val = v.int ? parseInt(s.value, 10) : parseFloat(s.value);
      sliders[v.key] = val;
      $('val_' + v.key).innerHTML = val + '<span>' + v.unit + '</span>';
      scheduleSim();
    });
  }
}

function setSlider(key, val){
  const v = spec.variables.find(x => x.key === key);
  const s = $('sld_' + key);
  if (!s) return;
  s.value = val;
  sliders[key] = v.int ? Math.round(val) : val;
  $('val_' + key).innerHTML = sliders[key] + '<span>' + v.unit + '</span>';
}

function scheduleSim(){
  clearTimeout(simTimer);
  simTimer = setTimeout(simulate, 160);
}

function showState(ok, text){
  const p = $('statePill');
  p.className = 'pill ' + (ok ? 'ok' : 'bad');
  p.textContent = ok ? '可行' : '越界';
  $('stateText').textContent = text;
}

function setBar(id, ratio){
  const b = $(id);
  b.classList.toggle('over', ratio > 1);
  b.querySelector('i').style.width = Math.min(100, Math.max(0, ratio * 100)) + '%';
}

function applyMetrics(d){
  $('vTmax').innerHTML = fmt(d.t_max_c, 1) + '<small>°C</small>';
  $('vRth').innerHTML = fmt(d.r_th_c_per_w, 3) + '<small>°C/W</small>';
  $('vMass').innerHTML = fmt(d.mass_g, 1) + '<small>g</small>';
  $('vGap').innerHTML = fmt(d.design.fin_gap_mm, 2) + '<small>mm</small>';
  setBar('bTmax', d.t_max_c / d.limits.t_max_c);
  setBar('bMass', d.mass_g / d.limits.mass_g);

  let txt = 'T_max 上限 ' + d.limits.t_max_c + ' °C · 质量上限 ' +
            d.limits.mass_g + ' g · 最小翅片间隙 ' + d.limits.min_fin_gap_mm + ' mm';
  if (!d.feasible){
    const bad = [];
    if (!d.t_max_ok) bad.push('温度超标');
    if (!d.mass_ok) bad.push('质量超标');
    txt = '约束不满足：' + bad.join('、') + '　|　' + txt;
  }
  showState(d.feasible, txt);
}

async function simulate(){
  if (busy) return;
  busy = true;
  try {
    const r = await fetch('/api/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(sliders)
    });
    const d = await r.json();

    if (!d.geometry_feasible){
      $('statePill').className = 'pill bad';
      $('statePill').textContent = '不可制造';
      $('stateText').textContent = d.reason;
      $('simMsg').className = 'msg bad';
      $('simMsg').textContent = '几何不可制造：' + d.reason;
      ['vTmax','vRth','vMass','vGap'].forEach(id => $(id).innerHTML = '—');
      setBar('bTmax', 0); setBar('bMass', 0);
      return;
    }

    applyMetrics(d);
    $('field').src = d.image;
    $('simMsg').className = 'msg';
    $('simMsg').textContent = '试算完成 · ' + d.design.n_fins + ' 片 · 翅高 ' +
      d.design.fin_height_mm + ' · 翅厚 ' + d.design.fin_thickness_mm +
      ' · 基板 ' + d.design.base_thickness_mm + ' mm';
  } catch (e){
    $('simMsg').className = 'msg bad';
    $('simMsg').textContent = '请求失败：' + e;
  } finally {
    busy = false;
  }
}

// ---- 收敛曲线：滚轮缩放 + 拖动平移 + Shift 框选放大 + 悬停数值（双击复位）----
let convPts = null;    // 最近一次搜索的曲线点
let convFull = null;   // 全量数据域
let convView = null;   // 当前可视窗口
let convHover = -1;    // 悬停命中的曲线点索引（-1 表示未命中）
let convSel = null;    // 正在框选的矩形（画布后备缓冲坐标）
let convBound = false; // 交互事件只绑定一次

function curveDomain(pts){
  const xs = pts.map(p => p.n_eval), ys = pts.map(p => p.best_r_th);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 1e-6){ y1 = y0 + 0.01; y0 -= 0.01; }
  const pad = (y1 - y0) * 0.12; y0 -= pad; y1 += pad;
  return {x0: x0, x1: x1 > x0 ? x1 : x0 + 1, y0: y0, y1: y1};
}

// 把可视窗口夹在全量域内，并限制最小缩放倍率
function convClamp(v){
  const f = convFull;
  let x0 = v.x0, x1 = v.x1, y0 = v.y0, y1 = v.y1;
  const minX = (f.x1 - f.x0) * 0.02, minY = (f.y1 - f.y0) * 0.02;
  if (x1 - x0 < minX){ const m = (x0 + x1) / 2; x0 = m - minX / 2; x1 = m + minX / 2; }
  if (y1 - y0 < minY){ const m = (y0 + y1) / 2; y0 = m - minY / 2; y1 = m + minY / 2; }
  if (x1 - x0 >= f.x1 - f.x0){ x0 = f.x0; x1 = f.x1; }
  if (y1 - y0 >= f.y1 - f.y0){ y0 = f.y0; y1 = f.y1; }
  if (x0 < f.x0){ x1 += f.x0 - x0; x0 = f.x0; }
  if (x1 > f.x1){ x0 -= x1 - f.x1; x1 = f.x1; }
  if (y0 < f.y0){ y1 += f.y0 - y0; y0 = f.y0; }
  if (y1 > f.y1){ y0 -= y1 - f.y1; y1 = f.y1; }
  return {x0: x0, x1: x1, y0: y0, y1: y1};
}

// 载入一次搜索的曲线：重置视图、绑定交互、重绘
function setCurve(curve){
  convPts = (curve || []).filter(p => p && typeof p.best_r_th === 'number');
  convHover = -1; convSel = null;
  if (convPts.length >= 2){
    convFull = curveDomain(convPts);
    convView = {...convFull};
  } else {
    convFull = null; convView = null;
  }
  bindConv();
  drawCurve();
}

function drawCurve(){
  const c = $('conv');
  if (!convPts || convPts.length < 2 || !convView) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height, padL = 62, padR = 18, padT = 18, padB = 34;
  const v = convView;
  ctx.clearRect(0, 0, W, H);

  const px = (x) => padL + (x - v.x0) / (v.x1 - v.x0) * (W - padL - padR);
  const py = (y) => padT + (v.y1 - y) / (v.y1 - v.y0) * (H - padT - padB);

  ctx.strokeStyle = '#e3e6ea'; ctx.lineWidth = 1;
  ctx.fillStyle = '#6b7280'; ctx.font = '16px "Microsoft YaHei",sans-serif';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++){
    const val = v.y0 + (v.y1 - v.y0) * i / 4;
    const y = py(val);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(val.toFixed(3), padL - 8, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const narrow = (v.x1 - v.x0) < 20;
  for (let i = 0; i <= 4; i++){
    const val = v.x0 + (v.x1 - v.x0) * i / 4;
    ctx.fillText(narrow ? val.toFixed(1) : Math.round(val), px(val), H - padB + 8);
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(padL, padT, W - padL - padR, H - padT - padB); ctx.clip();
  ctx.beginPath();
  convPts.forEach((p, i) => { const X = px(p.n_eval), Y = py(p.best_r_th);
    i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
  ctx.strokeStyle = '#0b7a45'; ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round'; ctx.stroke();

  const last = convPts[convPts.length - 1];
  ctx.beginPath();
  ctx.arc(px(last.n_eval), py(last.best_r_th), 5, 0, Math.PI * 2);
  ctx.fillStyle = '#0b7a45'; ctx.fill();

  if (convHover >= 0 && convHover < convPts.length){
    const hp = convPts[convHover], hx = px(hp.n_eval), hy = py(hp.best_r_th);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#9aa3af'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, H - padB); ctx.stroke();
    ctx.restore();
    ctx.beginPath(); ctx.arc(hx, hy, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill();
    ctx.lineWidth = 2.5; ctx.strokeStyle = '#0b7a45'; ctx.stroke();
  }
  ctx.restore();

  if (convSel){
    const sx0 = Math.max(padL, Math.min(convSel.x0, convSel.x1));
    const sy0 = Math.max(padT, Math.min(convSel.y0, convSel.y1));
    const sx1 = Math.min(W - padR, Math.max(convSel.x0, convSel.x1));
    const sy1 = Math.min(H - padB, Math.max(convSel.y0, convSel.y1));
    if (sx1 > sx0 && sy1 > sy0){
      ctx.fillStyle = 'rgba(11,122,69,0.13)';
      ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      ctx.strokeStyle = 'rgba(11,122,69,0.8)'; ctx.lineWidth = 1;
      ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
    }
  }

  ctx.fillStyle = '#6b7280'; ctx.font = '15px "Microsoft YaHei",sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('R_th [°C/W]', 6, 2);
  ctx.textAlign = 'right';
  ctx.fillText('设计评估次数', W - padR, H - padB + 8);
}

// 屏幕坐标 → 画布后备缓冲坐标（CSS 尺寸与后备缓冲不一致，必须换算）
function convXY(e){
  const c = $('conv'), r = c.getBoundingClientRect();
  return [(e.clientX - r.left) * c.width / r.width,
          (e.clientY - r.top) * c.height / r.height];
}

// 悬停命中：在容差内找离光标最近的数据点（画布后备缓冲坐标）
function convHit(mx, my){
  if (!convPts || !convView) return -1;
  const c = $('conv'), W = c.width, H = c.height;
  const PL = 62, PR = 18, PT = 18, PB = 34;
  if (mx < PL || mx > W - PR || my < PT || my > H - PB) return -1;
  const v = convView;
  let best = -1, bd = 16;
  convPts.forEach((p, i) => {
    const X = PL + (p.n_eval - v.x0) / (v.x1 - v.x0) * (W - PL - PR);
    const Y = PT + (v.y1 - p.best_r_th) / (v.y1 - v.y0) * (H - PT - PB);
    const d = Math.hypot(X - mx, Y - my);
    if (d < bd){ bd = d; best = i; }
  });
  return best;
}

// 悬停提示：显示该点的评估次数与热阻；i<0 或 e 缺省则隐藏
function convTip(i, e){
  const tip = $('convTip');
  if (!tip) return;
  if (i < 0 || !convPts || !convPts[i] || !e){ tip.style.display = 'none'; return; }
  const box = $('convBox').getBoundingClientRect();
  tip.textContent = '设计评估 ' + convPts[i].n_eval + ' 次　R_th ' +
    Number(convPts[i].best_r_th).toFixed(4) + ' °C/W';
  tip.style.left = (e.clientX - box.left) + 'px';
  tip.style.top = (e.clientY - box.top) + 'px';
  tip.style.display = 'block';
}

function bindConv(){
  if (convBound) return;              // 事件只绑一次，重复载入曲线不会叠加监听
  convBound = true;
  const c = $('conv');
  const PL = 62, PR = 18, PT = 18, PB = 34;

  c.addEventListener('wheel', (e) => {
    if (!convView) return;
    if (e.ctrlKey) return;                 // 让 Ctrl+滚轮 仍走浏览器的整页缩放
    e.preventDefault();
    const xy = convXY(e), mx = xy[0], my = xy[1], v = convView;
    const dx = v.x0 + (mx - PL) / (c.width - PL - PR) * (v.x1 - v.x0);
    const dy = v.y1 - (my - PT) / (c.height - PT - PB) * (v.y1 - v.y0);
    const k = Math.exp(e.deltaY * 0.0015);  // 上滚 k<1 放大，下滚 k>1 缩小
    convView = convClamp({
      x0: dx - (dx - v.x0) * k, x1: dx + (v.x1 - dx) * k,
      y0: dy - (dy - v.y0) * k, y1: dy + (v.y1 - dy) * k});
    drawCurve();
  }, {passive: false});

  let drag = null;
  c.addEventListener('mousedown', (e) => {
    if (!convView) return;
    const xy = convXY(e);
    if (e.shiftKey){                       // Shift+拖动 = 框选放大
      drag = {mode: 'box', mx: xy[0], my: xy[1]};
      convSel = {x0: xy[0], y0: xy[1], x1: xy[0], y1: xy[1]};
      c.classList.add('boxing');
    } else {                               // 直接拖动 = 平移
      drag = {mode: 'pan', mx: xy[0], my: xy[1], base: {...convView}};
      c.classList.add('dragging');
    }
    e.preventDefault();
  });
  c.addEventListener('mousemove', (e) => {
    const xy = convXY(e), mx = xy[0], my = xy[1];
    if (drag){
      if (drag.mode === 'box'){
        convSel = {x0: drag.mx, y0: drag.my, x1: mx, y1: my};
        drawCurve();
        return;
      }
      const b = drag.base;
      const kx = (b.x1 - b.x0) / (c.width - PL - PR), ky = (b.y1 - b.y0) / (c.height - PT - PB);
      const ddx = -(mx - drag.mx) * kx, ddy = (my - drag.my) * ky;
      convView = convClamp({
        x0: b.x0 + ddx, x1: b.x1 + ddx, y0: b.y0 + ddy, y1: b.y1 + ddy});
      drawCurve();
      return;
    }
    // 未拖动 → 悬停高亮 + 数值提示
    const h = convHit(mx, my);
    if (h !== convHover){ convHover = h; drawCurve(); }
    convTip(h, e);
  });
  const endDrag = () => {
    if (drag && drag.mode === 'box'){
      const s = convSel; convSel = null;
      if (s && convView){
        const x0 = Math.min(s.x0, s.x1), x1 = Math.max(s.x0, s.x1);
        const y0 = Math.min(s.y0, s.y1), y1 = Math.max(s.y0, s.y1);
        // 忽略过小的框选（避免单击误触），并把屏幕像素矩形折算成数据窗口
        if (x1 - x0 > 8 && y1 - y0 > 8){
          const v = convView;
          const toX = (X) => v.x0 + (X - PL) / (c.width - PL - PR) * (v.x1 - v.x0);
          const toY = (Y) => v.y1 - (Y - PT) / (c.height - PT - PB) * (v.y1 - v.y0);
          convView = convClamp({x0: toX(x0), x1: toX(x1), y0: toY(y1), y1: toY(y0)});
        }
      }
      drag = null; c.classList.remove('boxing'); drawCurve(); return;
    }
    drag = null; c.classList.remove('dragging'); c.classList.remove('boxing');
  };
  c.addEventListener('mouseup', endDrag);
  c.addEventListener('mouseleave', () => {
    endDrag();
    if (convHover !== -1){ convHover = -1; drawCurve(); }
    convTip(-1);
  });
  c.addEventListener('dblclick', () => {
    if (!convFull) return;
    convView = {...convFull};
    drawCurve();
  });
}

async function optimize(){
  const btn = $('optBtn');
  btn.disabled = true;
  $('optMsg').className = 'msg';
  $('optMsg').textContent = '正在搜索…';
  try {
    const r = await fetch('/api/optimize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        method: $('method').value,
        budget: parseInt($('budget').value, 10),
        seed: parseInt($('seed').value, 10)
      })
    });
    const d = await r.json();
    if (d.error){
      $('optMsg').className = 'msg bad';
      $('optMsg').textContent = d.error;
      return;
    }

    setSlider('n_fins', d.design.n_fins);
    setSlider('fin_height_mm', d.design.fin_height_mm);
    setSlider('fin_thickness_mm', d.design.fin_thickness_mm);
    setSlider('base_thickness_mm', d.design.base_thickness_mm);

    applyMetrics(d);
    $('field').src = d.image;
    $('convBox').style.display = 'block';
    setCurve(d.curve);

    const name = d.method === 'adaptive' ? '自适应邻域搜索' : '随机搜索';
    $('optMsg').className = 'msg';
    $('optMsg').textContent = '已找到最优设计（' + name + ' · 预算 ' + d.budget +
      ' · 种子 ' + d.seed + ' · 用时 ' + d.wall_time_s + ' s），滑块已更新为该设计。';
  } catch (e){
    $('optMsg').className = 'msg bad';
    $('optMsg').textContent = '请求失败：' + e;
  } finally {
    btn.disabled = false;
  }
}

async function boot(){
  spec = await (await fetch('/api/spec')).json();
  buildSliders();
  const c = spec.constraints, p = spec.physics, m = spec.mesh;
  $('specline').innerHTML =
    '物理模型：' + p.material + ' k=' + p.k_w_per_mk + ' W/(m·K)，自然对流 h=' +
    p.h_w_per_m2k + ' W/(m²·K)，环境 ' + p.ambient_c + ' °C，芯片 ' + p.chip_power_w +
    ' W，网格 ' + m.nx + '×' + m.ny + '<br>' +
    '外形包络 ' + c.envelope_mm.width + ' × ' + c.envelope_mm.height + ' × ' +
    c.envelope_mm.depth + ' mm，温度上限 ' + c.t_max_limit_c + ' °C，质量上限 ' +
    c.mass_limit_g + ' g，最小翅片间隙 ' + c.min_fin_gap_mm + ' mm';
  $('optBtn').addEventListener('click', optimize);
  await simulate();
}

boot();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    import uvicorn

    # 本地默认 127.0.0.1:8000；云端平台（Render 等）通过环境变量注入 HOST/PORT
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
