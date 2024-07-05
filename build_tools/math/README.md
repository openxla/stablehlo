# Tools for generating implementations and tests for math functions

This directory contains scripts that generate the implementations of
algorithms for various math functions that otherwise would be too
complex to develop, implement, and maintain within StableHLO.

The algorithms to various functions are defined within the software
project
[functional_algorithms](https://github.com/pearu/functional_algorithms)
as Python functions from which sources to various backends such as
StableHLO and XLA Client are generated. Also, the
`functional_algorithms` package is used to generate tests that include
sample inputs from the whole complex plane and reference values to the
corresponding functions using
[MPMath](https://github.com/mpmath/mpmath/). MPMath functions are used
as the reference functions because these are evaluated using
multi-precision arithmetics to achieve the required accuracy for the
given floating-point system. In addition, the corresponding functons
are validated to meet the special values constraints that are defined
for complex functions when assuming IEEE floating-point arithmetic
standard.

Within StableHLO, the provided scripts need to be run *only* when
adding support for more functions or when improvements to existing
function algorithms become available. The set of generated files is
considered as an immutable part of StableHLO code base which should
not be changed directly.

To run the scripts, make sure that your environment meets the
following requirements:

- Python 3.11 or newer
- mpmath 1.3 or newer
- functional_algorithms 0.7.0 or newer

that can be installed via pypi:

```sh
pip install functional_algorithms
```

or conda/mamba:

```sh
mamba install -c conda-forge functional_algorithms
```

When running the scripts, these will report which files are updated or
created. For instance:

```sh
$ python build_tools/math/generate_ChloDecompositionPatternsMath.py
stablehlo/transforms/ChloDecompositionPatternsMath.td is up-to-date.
$ python build_tools/math/generate_tests.py
Created stablehlo/tests/math/asin_complex64.mlir
Created stablehlo/tests/math/asin_complex128.mlir
Created stablehlo/tests/math/asin_float32.mlir
Created stablehlo/tests/math/asin_float64.mlir
```

To execute generated tests from a `build` directory, use:

```sh
for t in $(ls ../stablehlo/tests/math/*.mlir); \
do bin/stablehlo-opt --chlo-legalize-to-stablehlo $t \
 | bin/stablehlo-translate --interpret ; done
```

When new implementations are generated, one likely needs to update
`stablehlo/tests/chlo/chlo_legalize_to_stablehlo.mlir`. To generate
the filecheck tests, run

```sh
build/bin/stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file --verify-diagnostics \
   stablehlo/tests/chlo/chlo_legalize_to_stablehlo.mlir | python llvm-project/mlir/utils/generate-test-checks.py | less
```

and copy relevant checks to `chlo_legalize_to_stablehlo.mlir`.
