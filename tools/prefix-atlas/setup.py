"""Build the standalone C++ analyzer as a platform-specific wheel payload."""
import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


# Build with the host toolchain even when Python was built on another machine.
os.environ.setdefault("CC", "cc")
os.environ.setdefault("CXX", "c++")


class BuildAnalyzer(build_ext):
    def get_ext_filename(self, fullname):
        return os.path.join(*fullname.split("."))

    def build_extension(self, ext):
        target = Path(self.get_ext_fullpath(ext.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        objects = self.compiler.compile(
            ext.sources, output_dir=self.build_temp, extra_postargs=["-std=c++17", "-O3", "-g0"]
        )
        self.compiler.link_executable(objects, str(target), target_lang="c++")


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # The packaged program runs in a subprocess and never links to libpython.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    ext_modules=[Extension("prefix_atlas._curve", ["curve.cpp"], language="c++")],
    cmdclass={"build_ext": BuildAnalyzer, "bdist_wheel": PlatformWheel},
)
