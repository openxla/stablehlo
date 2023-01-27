# StableHLO Interpreter Checklist

In this document, we summarize the guidelines while implementing and reviewing
interpreter for an op. We have intentionally tied a few auxiliary action items,
related to verifier and type inference, with the idea of making progress in
those fronts alongside the interpreter implementation.

While implementing the interpreter:

1. Consult
   [hlo_evaluator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/hlo/evaluator)
   and
   [mlir_interpreter](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/mlir_hlo/tools/mlir_interpreter)
   to identify tricky implementation details and potential functionality gaps.
1. File tickets for the corresponding software components if you find any bugs
   or missing functionality.

After implementing the interpreter:

1. In [StablehloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td):
    1. Make sure that the `summary` in op's ODS follows the standard format.
       (related [ticket](https://github.com/openxla/stablehlo/issues/611))

1. In [TypeInference.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/TypeInference.cpp)
   and [StablehloOps.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.cpp):
    1. Delete comments that say things like "Verify the following properties:
       ...".

    1. Add comments that reference constraint labels from the spec in the format
       `<op_name>_Cn`, e.g. `SliceOp_C1`, to identify which parts of verifiers
       and shape functions correspond to which constraints in the specification.
        1. It is OK to have a comment with multiple constraint labels or to have
           multiple comments with the same constraint label. It all depends on
           how the constraints are implemented.
        1. In case there is a mismatch between the constraints in the
           implementation VS those in the specification, make sure there is an
           open issue reflecting that discrepancy.

1. In [interpreter tests](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests):
    1. Add file called `interpret_<op_mnemonic>.mlir`.
    1. Write tests following the [testing guidelines](https://github.com/openxla/stablehlo/blob/main/docs/reference.md#testing-guidelines).

1. In [ops_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir):
    1. Make sure that there is at least one test for each constraint in
       verifier and type inference methods. These tests will mostly be
       negative tests exercising the fact that the constraints are not met.
       Make sure to add at least one positive tests in case there are none.
       Note that it is out of scope to revisit the correctness of type
       inference as in
       [file](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir).
    1. Make sure that all the tests related to the op under test, are placed
       together.
    1. Make sure that all the tests related to the op under test, are
       prepended with `CHECK-LABEL` lit macro.
    1. Chose the function name of the tests using the format
       `op_mnemonic_Cn1_Cn2_...` for a function testing constraints `Cn1`, `Cn2`
       for op `op_mnemonic`. In cases when the proposed format does not apply,
       keep the existing name.
    1. Once the above step is complete, sort all the tests related to the op
       under test alphabetically based on the function name.
    1. Keep adding tests until the [ccov](https://github.com/openxla/stablehlo/blob/main/build_tools/github_actions/ci_build_stablehlo_code_coverage.sh)
       shows >= 90% coverage for the op.
1. In [spec.md](link):
    1. Add a link to `interpret_<op_mnemonic>.mlir` to the "Examples" section
       (e.g. [More Examples](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add)).

1. In [status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md):
    1. Update the "Interpreter" column to `yes`.
