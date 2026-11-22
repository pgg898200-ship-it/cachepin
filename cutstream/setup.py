import os

from setuptools import Extension, setup

setup(ext_modules=[
    Extension(
        "cutstream._native",
        ["src/cutstream/native.cpp"],
        include_dirs=["include"],
        depends=["include/cutstream.hpp"],
        extra_compile_args=["/std:c++17", "/O2"] if os.name == "nt" else ["-std=c++17", "-O3"],
        language="c++",
    )
])
