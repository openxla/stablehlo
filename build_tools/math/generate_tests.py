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

A script to generate test files for math functions with complex
and float inputs.

See build_tools/math/README.md for more information.
"""

import os
import re
import sys
import warnings
import mpmath
import numpy as np

to_float_dtype = {
    np.complex64: np.float32,
    np.complex128: np.float64,
    np.float32: np.float32,
    np.float64: np.float64,
}
to_complex_dtype = {
    np.float32: np.complex64,
    np.float64: np.complex128,
    np.complex128: np.complex128,
    np.complex64: np.complex64,
}

default_size = 13
default_extra_prec_multiplier = 1
default_max_ulp_difference = 1

operations = [
  # The following dictionaries may have additional keys like
  #
  #   size - defines the number of samples: size ** 2
  #
  #   max_ulp_difference - the maximal allowed ULP difference between
  #   function and reference values
  #
  #   extra_prec_multiplier - the precison multiplier for mpmath.mp
  #   that defines the precision of computing reference values:
  #   mpmath.mp.prec * extra_prec_multiplier
  #
  # When unspecifed, these parameters are retrieved from
  # functional_algorithms database of support functions.
  #
  dict(name="asin", mpmath_name="arcsin"),
  dict(name="acos", mpmath_name="arccos"),
  dict(name="atan", mpmath_name="arctan"),
  dict(name="asinh", mpmath_name="arcsinh"),
  dict(name="acosh", mpmath_name="arccosh"),
  dict(name="atanh", mpmath_name="arctanh"),
]


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


def main():
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

  target_dir = os.path.relpath(
      os.path.normpath(
          os.path.join(
              os.path.dirname(__file__),
              "..",
              "..",
              "stablehlo",
              "tests",
              "math",
          )),
      os.getcwd(),
  )

  flush_subnormals = False
  for op in operations:
    opname = op["name"]
    mpmath_opname = op.get("mpmath_name", opname)
    size_re = size_im = op.get("size", default_size)

    for dtype in [np.complex64, np.complex128, np.float32, np.float64]:
      params = fa.utils.function_validation_parameters(opname, dtype)
      max_ulp_difference = op.get(
        "max_ulp_difference",
        params.get("max_valid_ulp_count", default_max_ulp_difference))

      nmp = fa.utils.numpy_with_mpmath(
        extra_prec_multiplier = op.get(
          "extra_prec_multiplier",
          params.get("extra_prec_multiplier", default_extra_prec_multiplier)),
        flush_subnormals=flush_subnormals,
      )

      fi = np.finfo(dtype)

      float_dtype = to_float_dtype[dtype]
      finfo = np.finfo(float_dtype)

      if dtype in [np.complex64, np.complex128]:
        samples = fa.utils.complex_samples(
            size=(size_re, size_im),
            dtype=dtype,
            include_subnormal=not flush_subnormals,
        ).flatten()
      else:
        samples = fa.utils.real_samples(
            size=size_re * size_im,
            dtype=dtype,
            include_subnormal=not flush_subnormals,
        ).flatten()

      samples = np.concatenate((samples, fa.utils.extra_samples(opname, dtype)))

      expected = getattr(nmp, mpmath_opname).call(samples,
                                                  enable_progressbar=True)
      expected = np.array(expected, dtype)

      module_name = f"{opname}_{dtype.__name__}"
      m = SSA.make_module(module_name)

      samples_func = m.make_function("samples", "", mlir_type(samples))
      samples_func.assign(samples)
      samples_func.return_last()

      expected_func = m.make_function("expected", "", mlir_type(expected))
      expected_func.assign(expected)
      expected_func.return_last()

      main_func = m.make_function("main", "", "", "public")

      ref_samples = main_func.call("samples")
      actual = main_func.composite(f"chlo.{opname}", ref_samples)
      expected = main_func.call("expected")

      main_func.void_call(
          "check.expect_close",
          actual,
          expected,
          f"max_ulp_difference = {max_ulp_difference}",
          atypes=", ".join(map(main_func.get_ref_type, [actual, expected])),
      )
      main_func.void_call("func.return")
      source = str(m).rstrip() + "\n"
      fname = os.path.join(target_dir, f"{module_name}.mlir")
      if os.path.isfile(fname):
        f = open(fname, "r")
        content = f.read()
        f.close()
        if content.endswith(source):
          print(f"{fname} is up-to-date.")
          continue

      f = open(fname, "w")
      f.write("// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s |"
              " stablehlo-translate --interpret\n")
      f.write(
          "// This file is generated, see build_tools/math/README.md for more"
          " information.\n")
      f.write(source)
      f.close()
      print(f"Created {fname}")

  # Testing ULP difference
  for dtype in [np.float32, np.float64]:
    fi = np.finfo(dtype)

    max_ulp_difference = 0
    min_ulp_difference = 0

    finfo = np.finfo(dtype)
    module_name = f"ulp_difference_{dtype.__name__}"
    m = SSA.make_module(module_name)

    main_func = m.make_function("main", "", "", "public")

    def samples_generator():
      data = [
          -finfo.max,
          -1e9 - 1.2,
          -finfo.smallest_normal,
          -finfo.smallest_subnormal,
          0,
          finfo.smallest_subnormal,
          finfo.smallest_normal,
          1.2,
          1e9,
      ]
      for expected_ulp_difference in [0, 1, 5, 50]:
        if expected_ulp_difference == 0:
          actual = np.array(data + [np.inf, -np.inf, np.nan], dtype=dtype)
        else:
          actual = np.array(data, dtype=dtype)
        shifted = actual
        for i in range(expected_ulp_difference):
          shifted = np.nextafter(shifted, np.inf, dtype=dtype)
        label = str(expected_ulp_difference)
        yield actual, shifted, expected_ulp_difference, label

      actual = np.array([np.inf] * 5, dtype=dtype)
      shifted = np.array([-np.inf, np.nan, 0, 1.2, finfo.max], dtype=dtype)
      yield actual, shifted, 2**64 - 1, "nonfinite"

    for actual, shifted, expected_ulp_difference, label in samples_generator():

      actual_func = m.make_function(f"actual_{label}", "", mlir_type(actual))
      actual_func.comment(f"{list(actual)}")
      actual_func.assign(actual)
      actual_func.return_last()

      shifted_func = m.make_function(f"shifted_{label}", "", mlir_type(shifted))
      shifted_func.comment(f"{list(shifted)}")
      shifted_func.assign(shifted)
      shifted_func.return_last()

      actual_values = main_func.call(f"actual_{label}")
      shifted_values = main_func.call(f"shifted_{label}")

      main_func.void_call(
          "check.expect_close",
          actual_values,
          shifted_values,
          f"max_ulp_difference = {expected_ulp_difference}",
          f"min_ulp_difference = {expected_ulp_difference}",
          atypes=", ".join(
              map(main_func.get_ref_type, [actual_values, shifted_values])),
      )

    main_func.void_call("func.return")
    source = str(m).rstrip() + "\n"
    fname = os.path.join(target_dir, f"{module_name}.mlir")
    if os.path.isfile(fname):
      f = open(fname, "r")
      content = f.read()
      f.close()
      if content.endswith(source):
        print(f"{fname} is up-to-date.")
        continue

    f = open(fname, "w")
    f.write("// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s |"
            " stablehlo-translate --interpret\n")
    f.write("// This file is generated, see build_tools/math/README.md for more"
            " information.\n")
    f.write(source)
    f.close()
    print(f"Created {fname}")


class Block:
  """A data structure used in SSA"""

  def __init__(self, parent, prefix, suffix, start_counter=0):
    self.parent = parent
    self.prefix = prefix
    self.suffix = suffix
    self.counter = start_counter
    self.statements = {}

  def tostr(self, tab=""):
    lines = []
    lines.append(tab + self.prefix)
    for i in sorted(self.statements):
      op, expr, typ = self.statements[i]
      if op == "//":
        lines.append(f"{tab}  {op} {expr}")
      elif typ:
        lines.append(f"{tab}  {op} {expr} : {typ}")
      else:
        assert not expr, (op, expr, typ)
        lines.append(f"{tab}  {op}")
    lines.append(tab + self.suffix)
    return "\n".join(lines)

  def comment(self, message):
    # add comment to code
    self.statements[self.counter] = ("//", message, None)
    self.counter += 1

  def assign(self, expr, typ=None):
    if isinstance(expr, np.ndarray):
      assert typ is None, typ
      typ = mlir_type(expr)
      expr = shlo_constant(expr)
    elif isinstance(expr, str) and typ is not None:
      pass
    elif isinstance(expr, bool) and typ is not None:
      expr = shlo_constant(expr)
    else:
      raise NotImplementedError((expr, typ))
    target = f"%{self.counter}"
    self.statements[self.counter] = (f"{target} =", expr, typ)
    self.counter += 1
    return target

  def call(self, name, *args):
    # call function created with make_function
    sargs = ", ".join(args)
    return self.assign(f"call @{name}({sargs})",
                       typ=self.get_function_type(name))

  def composite(self, name, *args, **options):
    sargs = ", ".join(args)
    atypes = tuple(map(self.get_ref_type, args))
    rtype = options.get("rtype")
    if rtype is None:
      # assuming the first op argument defines the op type
      rtype = atypes[0]
    sargs = ", ".join(args)
    typ = f'({", ".join(atypes)}) -> {rtype}'
    return self.assign(f'"{name}"({sargs})', typ=typ)

  def void_call(self, name, *args, **options):
    # call function that has void return
    if args:
      sargs = ", ".join(args)
      atypes = options.get("atypes")
      if atypes is None:
        atypes = ", ".join(map(self.get_ref_type, args))
      self.statements[self.counter] = (name, f"{sargs}", f"{atypes}")
    else:
      self.statements[self.counter] = (name, "", "")
    self.counter += 1

  def apply(self, op, *args, **options):
    sargs = ", ".join(args)
    atypes = tuple(map(self.get_ref_type, args))
    rtype = options.get("rtype")
    if rtype is None:
      # assuming the first op argument defines the op type
      rtype = atypes[0]
    typ = f'({", ".join(atypes)}) -> {rtype}'
    return self.assign(f"{op} {sargs}", typ=typ)

  def return_last(self):
    ref = f"%{self.counter - 1}"
    self.statements[self.counter] = ("return", ref, self.get_ref_type(ref))
    self.counter += 1

  @property
  def is_function(self):
    return self.prefix.startwith("func.func")

  @property
  def function_name(self):
    if self.prefix.startswith("func.func"):
      i = self.prefix.find("@")
      j = self.prefix.find("(", i)
      assert -1 not in {i, j}, self.prefix
      return self.prefix[i + 1:j]

  @property
  def function_type(self):
    if self.prefix.startswith("func.func"):
      i = self.prefix.find("(", self.prefix.find("@"))
      j = self.prefix.find("{", i)
      assert -1 not in {i, j}, self.prefix
      return self.prefix[i:j].strip()

  def get_function_type(self, name):
    for block in self.parent.blocks:
      if block.function_name == name:
        return block.function_type

  def get_ref_type(self, ref):
    assert ref.startswith("%"), ref
    counter = int(ref[1:])
    typ = self.statements[counter][-1]
    return typ.rsplit("->", 1)[-1].strip()


class SSA:
  """A light-weight SSA form factory."""

  def __init__(self, prefix, suffix):
    self.prefix = prefix
    self.suffix = suffix
    self.blocks = []

  @classmethod
  def make_module(cls, name):
    return SSA(f"module @{name} {{", "}")

  def make_function(self, name, args, rtype, attrs="private"):
    if rtype:
      b = Block(self, f"func.func {attrs} @{name}({args}) -> {rtype} {{", "}")
    else:
      b = Block(self, f"func.func {attrs} @{name}({args}) {{", "}")
    self.blocks.append(b)
    return b

  def tostr(self, tab=""):
    lines = []
    lines.append(tab + self.prefix)
    for b in self.blocks:
      lines.extend(b.tostr(tab=tab + "  ").split("\n"))
    lines.append(tab + self.suffix)
    return "\n".join(lines)

  def __str__(self):
    return self.tostr()


def mlir_type(obj):
  if isinstance(obj, np.ndarray):
    s = "x".join(map(str, obj.shape))
    t = {
        np.bool_: "i1",
        np.float16: "f16",
        np.float32: "f32",
        np.float64: "f64",
        np.complex64: "complex<f32>",
        np.complex128: "complex<f64>",
    }[obj.dtype.type]
    return f"tensor<{s}x{t}>"
  else:
    raise NotImplementedError(type(obj))


def shlo_constant(obj):
  if isinstance(obj, bool):
    v = str(obj).lower()
    return f"stablehlo.constant dense<{v}>"
  if isinstance(obj, np.ndarray):
    if obj.dtype == np.bool_:
      h = "".join(map(lambda n: "%01x" % n, obj.view(np.uint8))).upper()
    else:
      h = "".join(map(lambda n: "%02x" % n, obj.view(np.uint8))).upper()
    return f'stablehlo.constant dense<"0x{h}">'
  else:
    raise NotImplementedError(type(obj))


if __name__ == "__main__":
  main()
