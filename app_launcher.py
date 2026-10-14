# -*- coding: utf-8 -*-
"""打包入口：启动本地网页操作台（FastAPI + uvicorn），并自动打开浏览器。

开发环境直接运行：
    python app_launcher.py

打包（PyInstaller）后双击生成的 exe 即可，无需安装 Python 或任何依赖。
程序会在本机起一个只监听 127.0.0.1 的服务，并自动打开默认浏览器；
关闭本窗口即停止服务。

命令行参数：
    --selfcheck   只做依赖与内核自检，打印结果后退出（返回码 0=通过 / 1=失败），
                  不启动服务。用于验证「程序能否独立运行」。
"""

import os
import socket
import sys
import threading
import time
import webbrowser

# 控制台按 UTF-8 输出，避免中文在部分终端下乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

LINE = "=" * 62

# 运行所需的关键依赖：模块名 -> 用途说明
REQUIRED = [
    ("numpy", "数值计算"),
    ("scipy", "稀疏矩阵求解"),
    ("matplotlib", "温度场绘图"),
    ("matplotlib.backends.backend_agg", "绘图后端（Agg）"),
    ("fastapi", "Web 框架"),
    ("uvicorn", "ASGI 服务器"),
    ("pydantic", "请求参数校验"),
    ("opentelemetry", "FastAPI 遥测依赖"),
    ("sim", "本项目物理仿真包"),
    ("opt", "本项目优化包"),
]


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


def _app_dir() -> str:
    """程序所在目录；打包后是 exe 所在目录，开发时是源码目录。"""
    if _frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _free_port(preferred: int = 8000, tries: int = 30) -> int:
    """从 preferred 起找一个没被占用的端口；全被占用时交给系统随机分配。"""
    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # connect_ex 返回 0 说明端口已被占用
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 0


def _verify() -> list:
    """启动自检：逐个校验关键依赖能否导入，并加载一次业务内核。

    返回问题描述列表；为空表示自检通过。
    """
    problems = []
    for name, purpose in REQUIRED:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"缺少依赖 {name}（{purpose}）：{type(exc).__name__}: {exc}")
    # 真正加载一次内核，连带验证 webapp / sim / opt 能否正常构建
    try:
        import webapp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"内核加载失败（webapp / sim / opt）：{type(exc).__name__}: {exc}")
    return problems


def _fail(problems: list) -> None:
    print(LINE)
    print("  启动自检未通过，程序无法运行。")
    print(LINE)
    for p in problems:
        print("  -", p)
    print(LINE)
    print("  排查建议：")
    print("   1) 是否完整解压——exe 必须与 _internal 文件夹放在一起，不能单独拷贝 exe；")
    print("   2) 是否被杀毒软件拦截或隔离；")
    print("   3) 是否在受管控环境（部分公司电脑禁止运行未签名程序）。")
    print(LINE)


def _open_browser_later(url: str, delay: float = 1.5) -> None:
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    # 保证无论开发还是打包状态，源码目录都在 sys.path 上（sim / opt 两个包）
    root = _app_dir()
    if root not in sys.path:
        sys.path.insert(0, root)

    only_check = "--selfcheck" in sys.argv

    print(LINE)
    print("  散热器几何参数优化操作台  ·  本地版")
    print(LINE)
    print("  正在自检依赖与内核…")

    problems = _verify()
    if problems:
        _fail(problems)
        if only_check:
            sys.exit(1)
        try:
            input("  按回车键退出…")
        except Exception:
            time.sleep(15)
        sys.exit(1)

    print("  自检通过：依赖与内核均可用。")
    if only_check:
        print("  --selfcheck 完成，退出码 0。")
        sys.exit(0)

    import uvicorn
    import webapp

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT") or _free_port(8000))
    url = f"http://{host}:{port}"

    print(LINE)
    print(f"  服务地址：{url}")
    print("  浏览器将自动打开；若未弹出请手动访问上面的地址。")
    print("  关闭本窗口即停止服务。")
    print(LINE)

    threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    try:
        uvicorn.run(webapp.app, host=host, port=port, log_level="info")
    except Exception as exc:  # noqa: BLE001
        print(LINE)
        print(f"  服务启动失败：{type(exc).__name__}: {exc}")
        print(LINE)
        try:
            input("  按回车键退出…")
        except Exception:
            time.sleep(15)
        sys.exit(1)


if __name__ == "__main__":
    main()
