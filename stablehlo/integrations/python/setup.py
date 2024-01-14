"""This setup.py builds a wheel file assuming that StableHLO is already built

Much of what is written here was largely inspired (or copied) from
https://github.com/makslevental/pristine-llvm-release/blob/main/setup.py
"""
from setuptools import find_namespace_packages, setup, Distribution
import os
import subprocess


class BinaryDistribution(Distribution):
  """Distribution which always forces a binary package with platform name"""

  def has_ext_modules(foo):
    return True


def get_version():
  # get the latest tag without the leading v
  latest_tag = subprocess.check_output(
      ["git", "describe", "--tags", "--abbrev=0"], text=True).strip('v').strip()
  latest_commit = subprocess.check_output(
      ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
  return f"{latest_tag}+{latest_commit}"


# TODO(fzakaria): The distribution (wheel) of this package is not manylinux
# conformant. Consider also running auditwheel similar to
# https://github.com/makslevental/mlir-wheels to make it a smoother installation
# experience.
setup(
    name='stablehlo',
    packages=find_namespace_packages(where=os.path.normpath("../../../build/python_packages/stablehlo")),
    package_dir={
        "": os.path.normpath("../../../build/python_packages/stablehlo")},
    package_data={'mlir': ['_mlir_libs/*.so']},
    include_package_data=True,
    distclass=BinaryDistribution,
    description='Backward compatible ML compute opset inspired by HLO/MHLO',
    url='https://github.com/openxla/stablehlo',
    # TODO(fzakaria): Figure out how to get version same as code; os.environ ?
    version=get_version()
)
