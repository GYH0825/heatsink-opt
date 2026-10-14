# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把网页操作台打成一个可独立分发的绿色程序。

产物：
    dist/散热器调参平台/散热器调参平台.exe   +   _internal/ 依赖目录
双击 exe 即在本机启动服务并自动打开浏览器，目标机器无需安装 Python。

说明：绘图只用 Agg 后端（webapp.py 里已 matplotlib.use("Agg")），
因此这里显式排除 Qt / Tk 等 GUI 后端，避免把无关的大型 GUI 库打进包里。

构建（在「项目文件」目录下）：
    python -m PyInstaller --noconfirm 散热器调参平台.spec
"""

from PyInstaller.utils.hooks import collect_submodules

PROJECT_DIR = SPECPATH  # 项目根目录（spec 所在目录）

# uvicorn 通过字符串动态导入协议/事件循环实现，必须显式声明子模块
hiddenimports = collect_submodules("uvicorn")

# sim / opt 两个业务包：显式声明，确保被收集
hiddenimports += ["sim", "opt"] + collect_submodules("sim") + collect_submodules("opt")

# matplotlib 的资源（mpl-data / 字体）由 PyInstaller 内置 hook 自动收集；
# 这里排除本项目用不到的 GUI 后端与交互式环境，控制体积。
# 说明：开发机 site-packages 里装了 torch / PyQt5 等无关的大型库，
# PyInstaller 会因某些 hook 牵连到它们，必须显式排除，否则包会膨胀到数 GB。
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
    # Windows 外部组件 / 云 SDK（无关）
    # 注意：opentelemetry 不能排除——fastapi 0.142 的 fastapi.telemetry 会硬导入它
    "win32com", "pythoncom", "pywintypes",
    "boto3", "botocore", "pyarrow",
]


a = Analysis(
    ["app_launcher.py"],
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
    name="散热器调参平台",
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
    name="散热器调参平台",
)
