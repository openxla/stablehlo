"""Copyright 2024 The StableHLO Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

A script to generate ChloDecompositionPatternsMath.td.

See build_tools/math/README.md for usage information.
"""

import os
import warnings


def get_functional_algorithms_required_version():
  readme_md = os.path.join(os.path.dirname(__file__), "README.md")
  f = open(readme_md, "r")
  version_string = None
  for line in f.readlines():
    if line.startswith("- functional_algorithms "):
      version_string = line.split()[2]
      break
  f.close()

  if version_string is not None:
    try:
      return tuple(map(int, version_string.split(".", 4)[:3]))
    except Exception as msg:
      print(
          f"Failed to extract functiona_algorithms required version from `{version_string}`: {msg}"
      )
  else:
    print(
        f"Failed to extract functiona_algorithms required version from {readme_md}"
    )


def main(kind="CHLO"):
  try:
    import functional_algorithms as fa
  except ImportError as msg:
    print(f"Skipping: {msg}")
    return

  fa_version = tuple(map(int, fa.__version__.split(".", 4)[:3]))
  required_fa_version = get_functional_algorithms_required_version()
  if required_fa_version is None:
    print(f'Skipping.')
    return

  if fa_version < required_fa_version:
    msg = (
        f"functional_algorithm version {'.'.join(map(str, required_fa_version))}"
        f" or newer is required, got {fa.__version__}")
    warnings.warn(msg)
    return

  output_filename = dict(
      CHLO="ChloDecompositionPatternsMath.td",
      StableHLO="StablehloComplexMathExpanderPatterns.td",
  )[kind]

  output_file = os.path.relpath(
      os.path.normpath(
          os.path.join(
              os.path.dirname(__file__),
              "..",
              "..",
              "stablehlo",
              "transforms",
              output_filename,
          )
      ),
      os.getcwd(),
  )

  sources = []
  target = fa.targets.stablehlo
  for chloname, fname, args in [
      # Important: new items to this list must be added to the end,
      # otherwise, git diff may end up being unnecessarily large.
      #
      # (<op CHLO name>, <op name in fa.algorithms>, <a tuple of op arguments>)
      #
      ("CHLO_AsinAcosKernelOp", "asin_acos_kernel", ("z:complex",)),
      ("CHLO_AsinOp", "complex_asin", ("z:complex",)),
      ("CHLO_AsinOp", "real_asin", ("x:float",)),
      ("CHLO_AcosOp", "complex_acos", ("z:complex",)),
      ("CHLO_AcosOp", "real_acos", ("x:float",)),
      ("CHLO_AcoshOp", "complex_acosh", ("z:complex",)),
      ("CHLO_AcoshOp", "real_acosh", ("x:float",)),
      ("CHLO_AsinhOp", "complex_asinh", ("z:complex",)),
      ("CHLO_AsinhOp", "real_asinh", ("x:float",)),
      ("CHLO_AtanOp", "complex_atan", ("z:complex",)),
      ("CHLO_AtanhOp", "complex_atanh", ("z:complex",)),
      ("CHLO_SquareOp", "complex_square", ("z:complex",)),
      ("CHLO_SquareOp", "real_square", ("x:float",)),
      ("StableHLO_Log1pOp", "complex_log1p", ("z:complex",)),
  ]:
    if not chloname.startswith(kind):
      continue
    print(f'Generating {chloname} from {fname}{args}')
    func = getattr(fa.algorithms, fname, None)
    if func is None:
      warnings.warn(
          f"{fa.algorithms.__name__} does not define {fname}. Skipping."
      )
      continue
    ctx = fa.Context(paths=[fa.algorithms],
                     parameters=dict(rewrite_keep_integer_literals=True))
    graph = ctx.trace(func, *args).rewrite(target, fa.rewrite)
    graph.props.update(name=chloname)
    src = graph.tostring(target)
    sources.append(target.make_comment(func.__doc__)) if func.__doc__ else None
    sources[-1] += src
  source = "\n\n".join(sources) + "\n"

  if chloname.startswith("StableHLO_"):
    # an ugly hack to fix the definition of stablehlo complex math
    # functions. TODO(pearu): add the corresponding feature to
    # functional_algorithms stablehlo printer
    NameOp = chloname.split("_", 1)[1]
    source = source.replace(
        f"def : Pat<({chloname}",
        f"def {NameOp}_ComplexElementType_ComplexMathExpander :"
        f" Pat<({chloname}",
    )

  if os.path.isfile(output_file):
    f = open(output_file, "r")
    content = f.read()
    f.close()
    if content.endswith(source):
      print(f"{output_file} is up-to-date.")
      return

  f = open(output_file, "w")
  f.write("""\
/* Copyright 2024 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

""")
  f.write(
      target.make_comment(f"""\

This file is generated using functional_algorithms tool ({fa.__version__}).
See build_tools/math/README.md for more information.""") + "\n")

  if kind == "StableHLO":
    f.write("""\
include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr,
               "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

def StableHLO_ConstantLikeMaxFiniteValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeMaxFiniteValue($_builder, $_loc, $0)">;

""")
  f.write(source)
  f.close()
  print(f"Created {output_file}")


if __name__ == "__main__":
  main(kind="CHLO")
  main(kind="StableHLO")
