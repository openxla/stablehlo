# StableHLO Interpreter Checklist

In this document, we summarize the guidelines while implementing and reviewing
interpreter for an op. We have intentionally tied a few auxiliary action items,
related to verifier and type inference, with the idea of making progress in
those fronts alongside the interpreter implementation.

1. While implementing the interpreter for an op, double check with related
   implementations (e.g.
   [hlo_interpreter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/hlo/evaluator)
   and
   [mlir_interpreter](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/mlir_hlo/tools/mlir_interpreter)).
   This action item will allow mutual improvements.
1. Make sure that the ODS follows standard [summary](https://github.com/openxla/stablehlo/issues/611).
1. Replace the comments, related to input/output constraints, in
   the implementations, specifically for verifiers and type inference, with
   some unique identifier containing "constraint-labels", e.g. `Cn` in
   [here](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constraints-2)
   , used in the
   [specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md).
   The unique identifier should have the format `<op_name>_Cn`.
   This is to explicitly align all the implementations with the specification:
   the single source of truth, and help to track gaps in those implementations
   back to the spec.

   **Why Unique**: The fact that the spec will be augmented with Dynamism,
   Quantization can affect the suffix number in the constraint label or might
   change it to something else. A unique identifier will enable faster updates
   to comments we are planning to add here.
   1. It is OK to have a comment with multiple constraint-labels or to have
      multiple comments with the same constraint-label. It all depends on how
      the constraint-label(s) are coded in the implementations.
   1. In case there is a mismatch between the constraints in the
      implementation VS those in the specification, make sure there is an
      open issue reflecting that discrepancy.
1. Make sure that the there are sufficient tests to cover for the
   implementations (e.g. verifiers, type inference, interpreter) of the
   specification. Specifically,
   1. Regarding interpreter tests
      1. For interpreter: Write tests following the
         [testing guidelines](https://github.com/openxla/stablehlo/blob/main/docs/reference.md#testing-guidelines).
      1. Add a link of the test file in the spec (e.g.
         [More Examples](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add)).
   1. For verifier and type inference: Affecting [this file](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir).
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
         `op_name_Cn1_Cn2_...` for a function testing constraints `Cn1`, `Cn2`
         for op `op_name`. In cases when the proposed format does not apply,
         keep the existing name.
1. Locally run the
   [ccov](https://github.com/openxla/stablehlo/blob/main/build_tools/github_actions/ci_build_stablehlo_code_coverage.sh)
   tool to make sure that the followings yield >= 90% coverage. If not, then add
   tests to achieve the goal.
   1. Code and tests introduced for implementing the interpreter.
   1. The existing code and tests for verifier and type inference.
1. Once all the items are finsihed, update the interpreter column in status.md
   to `yes`.
