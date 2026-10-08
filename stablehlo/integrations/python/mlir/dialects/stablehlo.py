# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
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

# pylint: disable=wildcard-import,relative-beyond-top-level,g-import-not-at-top
from ._stablehlo_ops_gen import *
from .._mlir_libs._stablehlo import *

from . import _stablehlo_ops_gen

class CompositeOp(_stablehlo_ops_gen.CompositeOp):
  def __init__(self, *args, **kwargs):
    if "num_composite_regions" not in kwargs:
      kwargs["num_composite_regions"] = 0
    super().__init__(*args, **kwargs)


class DotGeneralOp(_stablehlo_ops_gen.DotGeneralOp):
  """`dot_general` builder that defaults the new `ext_operands` operand.

  `ext_operands` is the fifth positional argument and the new attributes are
  passed by keyword only, so callers of the old signature pass at most 4
  positional arguments.
  """

  def __init__(self, *args, **kwargs):
    if len(args) <= 4:
      kwargs.setdefault("ext_operands", [])
    super().__init__(*args, **kwargs)


def dot_general(*args, **kwargs):
  return DotGeneralOp(*args, **kwargs).result
