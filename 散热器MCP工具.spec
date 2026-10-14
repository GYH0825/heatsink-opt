# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把 MCP 工具层打成一个可独立运行的 stdio 可执行文件。

产物：
    dist/散热器MCP工具/散热器MCP工具.exe   +   _internal/ 依赖目录
由 Agnes Harness（AGH）以 stdio 传输拉起，向上暴露 6 个散热器工具；
目标机器无需安装 Python 或任何依赖。

注意：本程序用 stdout 承载与 AGH 的 JSON-RPC，因此**禁止**向 stdout 打印任何内容，
调试信息一律走日志（stderr）。这里的 console=True 只为便于观察错误输出。

构建（在「项目文件」目录下）：
    python -m PyInstaller --noconfirm 散热器MCP工具.spec
"""

from PyInstaller.utils.hooks import collect_submodules

PROJECT_DIR = SPECPATH  # 项目根目录（spec 所在目录）

# 本项目业务包 + MCP 协议栈（stdio 路径）+ 按字符串动态加载的第三方运行时。
# 不用 collect_submodules("mcp")：mcp 包内含 mcp.cli，其 __init__ 缺 typer 时会直接
# sys.exit(1)，会让子进程整体崩溃（on_error 也拦不住）。stdio 路径所需模块显式列全即可。
hiddenimports = [
    # 本项目业务包
    "sim", "opt", "experiment",
    # MCP 协议栈
    "mcp", "mcp_types",
    "mcp.server", "mcp.server.mcpserver", "mcp.server.mcpserver.server",
    "mcp.server.stdio", "mcp.server.lowlevel.server",
    # 动态加载的第三方运行时
    "anyio._backends._asyncio", "pydantic", "pydantic_core",
    # Windows stdio 依赖（mcp/os/win32）
    "pywintypes",
]
hiddenimports += collect_submodules("sim")
hiddenimports += collect_submodules("opt")

# 与网页版一致：开发机装了 torch / PyQt5 等无关大库，必须显式排除。
# 注意：opentelemetry 不排除——mcp 的遥测模块会导入它。
excludes = [
    # 深度学习 / 大型科学计算（本项目完全用不到）
    "torch", "torchvision", "torchaudio", "tensorflow", "keras",
    "sklearn", "pandas", "numba", "dask", "cv2", "mediapipe",
    "onnxruntime", "transformers", "sentence_transformers",
    # GUI 后端
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_qt5agg", "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_wx",
    "matplotlib.backends.backend_webagg", "tkinter",
    # 交互式/测试/文档环境
    "IPython", "ipykernel", "jupyter", "notebook", "sphinx", "pytest",
    # mcp 自带的命令行前端（依赖 typer，本项目用不到）
    "mcp.cli",
    # 云 SDK（无关）
    "boto3", "botocore", "pyarrow",
    # 注意：这里**不能**排除 win32com / pythoncom / pywintypes ——
    # mcp 的 Windows stdio 支持（mcp/os/win32/utilities.py，被 mcp/server/stdio.py 导入）
    # 会用 pywintypes，排掉会在拉起时直接 ModuleNotFoundError。
]


a = Analysis(
    ["mcp_server.py"],
    pathex=[PROJECT_DIR],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="散热器MCP工具",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="散热器MCP工具",
)
