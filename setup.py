"""可选 C++ 扩展构建。

用法（需要 pybind11 与任意 C++14 编译器）：
    pip install pybind11
    python setup.py build_ext --inplace

**可选性是设计的一部分**（对应 DeepInfer 的多后端机制）：
未编译时 miniflow 自动回落 NumPy 后端（miniflow/tensor.py 的 BACKEND
探测），功能与正确性完全等价 —— C++ 核只影响性能与教学演示
（examples 里可对比 matmul_naive / matmul / NumPy BLAS 三层实现）。
"""

import sys

from setuptools import Extension, setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "miniflow._C",
            ["csrc/kernel.cpp"],
            cxx_std=14,
        )
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    print("=" * 60)
    print("提示：未安装 pybind11，跳过 C++ 扩展构建。")
    print("miniflow 将使用 NumPy 后端（功能不受影响）。")
    print("如需 C++ 算子后端：pip install pybind11 后重新执行本命令。")
    print("=" * 60)
    ext_modules = []
    cmdclass = {}

setup(
    name="miniflow",
    version="0.1.0",
    description="DeepInfer 教学复刻：CPU 上的生产级推理引擎最小实现",
    license="MIT",
    packages=[
        "miniflow",
        "miniflow.model",
        "miniflow.cache",
        "miniflow.entrypoints",
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=["numpy"],
)