# -*- coding: utf-8 -*-
"""端到端自检：以 MCP 客户端身份拉起 mcp_server.py，逐个调用全部工具。

作用是在接入 AGH 之前先确认工具层本身可用（协议握手、参数校验、返回值），
同时也可以当作「可复现运行说明」中的一步。

运行（在「项目文件」目录下）：
    python selfcheck.py
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))

DEMO_DESIGN = {"n_fins": 10, "fin_height_mm": 30.0,
               "fin_thickness_mm": 3.0, "base_thickness_mm": 5.0}


def _payload(result):
    """取出工具的结构化返回值；退化为文本时原样返回。"""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(HERE, "mcp_server.py")],
        cwd=HERE,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"已连接 MCP server：{session.server_info.name} "
                  f"v{session.server_info.version}")

            listed = await session.list_tools()
            print(f"\n暴露工具 {len(listed.tools)} 个：")
            for tool in listed.tools:
                first = (tool.description or "").splitlines()[0]
                print(f"  · {tool.name:<24} {first[:48]}")

            print("\n" + "=" * 70)
            print("步骤 1 · 任务规划：读取问题定义")
            print("=" * 70)
            spec = _payload(await session.call_tool("get_problem_spec", {}))
            print(f"  目标：{spec['objective']}")
            print(f"  变量：{spec['variables']}")
            print(f"  约束：{spec['constraints']}")

            print("\n" + "=" * 70)
            print("步骤 2 · 能力调用：仿真一个候选设计（并验证缓存命中）")
            print("=" * 70)
            for i in (1, 2):
                res = _payload(await session.call_tool("simulate_design", DEMO_DESIGN))
                print(f"  第 {i} 次 → {res['summary']}"
                      f"  可行={res['feasible']}  缓存命中={res['cache_hit']}")

            print("\n" + "=" * 70)
            print("步骤 3 · 能力调用：运行自适应邻域搜索")
            print("=" * 70)
            run = _payload(await session.call_tool(
                "search_designs", {"method": "adaptive", "budget": 150, "seed": 1}))
            print(f"  最优设计：{run['best_design']}")
            print(f"  T_max={run['best_t_max_c']}°C  R_th={run['best_r_th_c_per_w']}°C/W"
                  f"  m={run['best_mass_g']}g  可行={run['feasible']}")
            print(f"  求解 {run['n_solves']} 次，用时 {run['wall_time_s']} s")
            print(f"  收敛采样点 {len(run['convergence'])} 个，"
                  f"首个可行点={next((p for p in run['convergence'] if p['best_r_th']), None)}")

            best = run["best_design"]
            verify_args = {"n_fins": best["n_fins"],
                           "fin_height_mm": best["fin_height_mm"],
                           "fin_thickness_mm": best["fin_thickness_mm"],
                           "base_thickness_mm": best["base_thickness_mm"]}

            print("\n" + "=" * 70)
            print("步骤 4 · 结果验证：网格无关性复核")
            print("=" * 70)
            ver = _payload(await session.call_tool("verify_design", verify_args))
            print(f"  粗网格 {ver['coarse']}")
            print(f"  细网格 {ver['fine']}")
            print(f"  相对偏差 {ver['relative_error']}（阈值 {ver['tolerance']}）"
                  f"  通过={ver['passed']}")
            print(f"  结论：{ver['verdict']}")

            print("\n" + "=" * 70)
            print("步骤 5 · 反馈处理：两种方法在同一预算下对比")
            print("=" * 70)
            cmp_ = _payload(await session.call_tool(
                "compare_methods", {"budget": 80, "seeds": 2}))
            for name, stat in cmp_["methods"].items():
                print(f"  {name:<9} {stat}")
            print(f"  相对改进：{cmp_['improvement_percent']}%")

            print("\n" + "=" * 70)
            print("步骤 6 · 取证：渲染温度场")
            print("=" * 70)
            img = _payload(await session.call_tool(
                "render_temperature_field", verify_args))
            print(f"  图片：{img['image_path']}")

            print("\n自检完成：全部工具调用成功。")


if __name__ == "__main__":
    asyncio.run(main())
