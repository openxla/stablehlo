# StableHLO Interpreter Checklist

In this document, we summarize the guidelines while implementing and reviewing
interpreter for an op. We have intentionally tied a few auxiliary action items,
related to verifier and type inference, with the idea of making progress in
those fronts alongside the interpreter implementation.

1. While implementing the interpreter for an op, double check with related
   implementations (e.g.
   <!-- markdownlint-disable line-length -->
   [hlo_interpreter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/hlo/evaluator)
   <!-- markdownlint-enable line-length -->
   and
   <!-- markdownlint-disable line-length -->
   [mlir_interpreter](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/mlir_hlo/tools/mlir_interpreter)).
   <!-- markdownlint-enable line-length -->
   This action item will allow mutual improvements.
1. Replace the comments, related to input/output constraints, in
   the implementations, specifically for verifiers and type inference, with
   "constraint-labels" (e.g. `Cn` in
   <!-- markdownlint-disable line-length -->
   [here](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constraints-2) )
   <!-- markdownlint-enable line-length -->
   used in the
   <!-- markdownlint-disable line-length -->
   [specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md).
   <!-- markdownlint-enable line-length -->
   This is to explicitly align all the implementations with the specification:
   the single source of truth, and help to track gaps in those implementations
   back to the spec.
   1. It is OK to have a comment with multiple constraint-labels or to have
      multiple comments with the same constraint-label. It all depends on how
      the constraint-label(s) are coded in the implementations.
   1. In case there is a mismatch between the constraints in the
      implementation VS those in the specification, make sure there is an
      open issue reflecting that discrepancy.
1. Make sure that the there are sufficient tests to cover for the
   implementations (e.g. verifiers, type inference, interpreter) of the
   specification. Specifically,
   1. Write tests for an op interpreter following the
      <!-- markdownlint-disable line-length -->
      [testing guidelines](https://github.com/openxla/stablehlo/blob/main/docs/reference.md#testing-guidelines).
      <!-- markdownlint-enable line-length -->
   1. Double check that there is at least one test for each constraint in
      verifier and type inference methods.
1. Locally run the
   <!-- markdownlint-disable line-length -->
   [ccov](https://github.com/openxla/stablehlo/blob/main/build_tools/github_actions/ci_build_stablehlo_code_coverage.sh)
   <!-- markdownlint-enable line-length -->
   tool to make sure that the followings yield >= 90% coverage. If not, then add
   tests to achieve the goal.
   1. Code and tests introduced for implementing the interpreter.
   1. The existing code and tests for verifier and type inference.
1. Cross validate the test results of the interpreter implementation against
   hardware. TBD
