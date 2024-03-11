# StableHLO Interpreter Checklist

In this document, we summarize the guidelines for implementing and reviewing an
op for the interpreter. We have intentionally included a few auxiliary action
items related to verifier and type inference, with the idea of making progress
on those fronts alongside the interpreter implementation.

## While implementing the op

1. Provide an explicitly written testing strategy (in a PR description)
   similar to
   [this](https://github.com/openxla/stablehlo/pull/996#issue-1558631158) to use
   as a reference while reviewing the verification and type inference
   methods, and the corresponding tests. The reviewer will double check that the
   description is comprehensive.
1. Consult
   [hlo_evaluator](https://github.com/openxla/xla/blob/main/xla/hlo/evaluator)
   to identify tricky implementation details and potential functionality gaps.
1. File tickets for the corresponding software components if you find any bugs
   or missing functionality.

## After implementing the op

1. In [StablehloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td):
    1. Make sure that the `summary` in the op's ODS follows the standard format.
       (related [ticket](https://github.com/openxla/stablehlo/issues/611))
    1. Add comments referencing constraint labels (e.g. `Cn` or `In`) from the
       spec in the format `xyz_cn` or `xyz_in`, for op `XyzOp`, to identify the
       correspondence between constraints in ODS and the specification. The
       following example shows how to add the constraint labels as comments
       alongside mlir `Traits` and `TypeConstraints`. Note `xyz_c4` refers to
       constraints defined in `StableHLO_FooOp` class (e.g.
       `StableHLO_ShapedInterfaceOp`, `StableHLO_UnaryElementwiseOp`,
       `StableHLO_Op`, etc.).

       ```td
        def StableHLO_XyzOp: StableHLO_FooOp<"xyz", [Trait1,
            Trait2 /*xyz_c1, xyz_c2*/, InferTensorType /*xyz_c3*/]> { /*xyz_c4*/
             ...
          let summary = "Xyz operation";
          let arguments = (ins
             1DTensorOf<[HLO_Float]>:$a, /*xyz_c5, xyz_i1*/
             HLO_Tensor:$b, /*xyz_i2*/
             ....
          );
       );
       ```

1. In [TypeInference.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/TypeInference.cpp)
   and [StablehloOps.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.cpp):
    1. Delete comments that say things like "Verify the following properties:
       ...".
    1. Add comments referencing constraint labels (e.g. `Cn` or `In`) from the
       spec in the format `xyz_cn` or `xyz_in`, for op `XyzOp`, to identify
       which parts of verifiers and shape functions correspond to which
       constraints in the specification.
        1. It is OK to have a comment with multiple constraint labels or to have
           multiple comments with the same constraint label. It all depends on
           how the constraints are implemented. If there are consecutive constraints,
           condense them as `xyz_cn...xyz_cm, xyz_in...xyz_jn`.
        1. In case there is a mismatch between the constraints in the
           implementation VS and those in the specification, make sure there is
           an open issue reflecting that discrepancy.
1. In [interpreter tests](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret):
    1. Add a file called `<op_mnemonic>.mlir`.
    1. Write tests following the [testing guidelines](reference.md#testing-guidelines).
1. In the [testdata directory](https://github.com/openxla/stablehlo/tree/main/stablehlo/testdata):
    1. Run any disabled tests that are covered by the newly added operation.
    1. If the tests pass, enable them by converting `RUN-DISABLED` to `RUN`.
    1. If a test fails for some reason other than precision mismatches, fix the
       implementation/test.
    1. For precision mismatches, tag the test with `RUN-DISABLED(#1278)` (if
       it's not already done).
1. In [ops_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir):
    1. Make sure that there is at least one test (positive or negative) for each
       constraint in the verifier and type inference methods; constraints
       covered in ODS will not be tested. These tests will mostly be negative,
       testing that the constraints are not met or positive, testing that the
       inferred shape is correct.
    1. Make sure that all the tests related to the op under test are placed
       together.
    1. Make sure that all the tests related to the op under test are
       prepended with a `CHECK-LABEL` lit macro.
    1. Choose the function name of the tests using the format
       `xyz_cn_im_...` for a function testing constraints `Cn`, `Im`,
       etc. for op `XyzOp`. In cases when the proposed format does not
       apply, keep the existing name.
    1. Once the above step is complete, sort all the tests related to the op
       under test alphabetically based on the function name.
    1. Keep adding tests until the [ccov](https://github.com/openxla/stablehlo/blob/main/build_tools/github_actions/ci_build_cmake_code_coverage.sh)
       shows >= 90% coverage for the op.
1. In [infer_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir):
    1. Make sure all constraints related to shape inference tests are present
       in this file, following the same naming guidelines noted above.
    1. Move any shape inference tests from the [ops_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir)
       file into this file.
1. In [spec.md](spec.md):
    1. Add a link to `stablehlo/tests/interpret/<op_mnemonic>.mlir`
       to the "Examples" section
       (e.g. [More Examples](spec.md#add)).
    1. Make sure the spec only has 1 example.
    1. Make sure the spec example follows the [testing guidelines](reference.md#testing-guidelines).
    1. Make sure the spec example test is interpretable.
    1. Make sure the spec example is the same as what is in the ODS.
1. In [status.md](status.md):
    1. Update the "Interpreter" column to `yes`.
