# Copyright 2024 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This setup.py builds a wheel file assuming that StableHLO is already built
"""
import os
import pathlib
import re
import subprocess
import time
import setuptools

_CURRENT_VERSION_RE = re.compile(
    r"\bgetCurrentVersion\(\)\s*\{\s*return\s+Version\(\s*"
    r"(?:/\*\s*\w+\s*=\s*\*/\s*)?(\d+)\s*,\s*"
    r"(?:/\*\s*\w+\s*=\s*\*/\s*)?(\d+)\s*,\s*"
    r"(?:/\*\s*\w+\s*=\s*\*/\s*)?(\d+)\s*\)"
)


class BinaryDistribution(setuptools.Distribution):
  """Force distribution which always forces a binary package"""

  def has_ext_modules(foo):
    return True


def get_version():
  """Returns the PEP 440 dev wheel version from Version.h and HEAD."""
  version_h = (
      pathlib.Path(__file__).resolve().parents[2] / "dialect" / "Version.h"
  )
  match = _CURRENT_VERSION_RE.search(version_h.read_text(encoding="utf-8"))
  if not match:
    raise RuntimeError(f"Could not parse getCurrentVersion() from {version_h}")
  base_version = ".".join(match.groups())
  latest_commit = subprocess.check_output(
      ["git", "rev-parse", "--short", "HEAD"], cwd=version_h.parent, text=True
  ).strip()
  # in order for the wheels to be ordered chronologically
  # include the epoch seconds as a portion of the version
  return f"{base_version}.{int(time.time())}+{latest_commit}"


# TODO(fzakaria): The distribution (wheel) of this package is not manylinux
# conformant. Consider also running auditwheel similar to
# https://github.com/makslevental/mlir-wheels to make it a smoother installation
# experience.
setuptools.setup(
    name='stablehlo',
    # TODO(fzakaria): The CMake build path is fixed here which kind of sucks.
    # Ideally it should be passed in or setup.py should do the build itself.
    packages=setuptools.find_namespace_packages(
        where=os.path.normpath("../../../build/python_packages/stablehlo")
    ),
    package_dir={
        "": os.path.normpath("../../../build/python_packages/stablehlo")},
    package_data={'mlir': ['_mlir_libs/*.so']},
    include_package_data=True,
    distclass=BinaryDistribution,
    description='Backward compatible ML compute opset inspired by HLO/MHLO',
    url='https://github.com/openxla/stablehlo',
    version=get_version()
)
