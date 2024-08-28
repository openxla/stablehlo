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
- functional_algorithms 0.10.1 or newer

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
do echo $t && ( bin/stablehlo-opt --chlo-legalize-to-stablehlo $t \
 | bin/stablehlo-translate --interpret 2>&1 | grep "^ULP difference" ) ; done
```

When new implementations are generated, one likely needs to update
`stablehlo/tests/chlo/chlo_legalize_to_stablehlo.mlir`. To generate
the filecheck tests, run

```sh
build/bin/stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file --verify-diagnostics \
   stablehlo/tests/chlo/chlo_legalize_to_stablehlo.mlir | python llvm-project/mlir/utils/generate-test-checks.py | less
```

and copy relevant checks to `chlo_legalize_to_stablehlo.mlir`.

## A procedure for adding a new algorithm to an existing operation

1. Implement a new algorithm in
   [functional_algorithms](https://github.com/pearu/functional_algorithms)
   and publish it by creating a new release of
   `functional_algorithms`.
2. Build stablehlo on top of its main branch.
3. Update the version requirement of `functional_algorithms` in this
   `README.md` and install the latest version of
   `functional_algorithms`.
4. Add a record of the operation to `generate_tests.py:operations`
   list. Use `size=1000` and `max_ulp_difference=0`.
5. Generate new tests by running `generate_tests.py`.
6. Run the generated tests (see previos section for instructions)
   which will output the ULP difference statistics of the current
   implementation to stdout; copy this information for
   comparision later.  Notice that tests failures are expected because of
   the specified `max_ulp_difference=0` in the step 4.
7. Add a record of the operation to
   `generate_ChloDecompositionPatternsMath.py`, see the for-loop in
   `main` function.
8. Generate new implementations by running
   `generate_ChloDecompositionPatternsMath.py` and remove existing
   implementations in
   `stablehlo/transforms/ChloDecompositionPatterns.td` as needed.
9. Re-build stablehlo.
10. Re-run the generated tests and compare the ULP difference statistics
    results of the implementation with the one obtained in step 6.
11. If the new implementation improves ULP difference statistics,
    prepare a PR for stablehlo. When submitting the PR, don't forget
    to apply the following steps:
    - remove the specified `max_ulp_difference=0` from
      `generate_tests.py` and re-generate tests with
      `size=default_size`,
    - update `chlo_legalize_to_stablehlo.mlir`, see previos section
      for instructions.
