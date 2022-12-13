# Type Inference

StableHLO has been originally bootstrapped from [the MHLO dialect](https://github.com/tensorflow/mlir-hlo#meta-hlo-dialect-mhlo), including inheriting the implementation of type inference. The implementation progress is tracked in [status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md).

To implement high-quality verifiers and shape functions for StableHLO ops, these guidelines are proposed below to follow:

# Proposal

These proposals apply to both revisiting existing implementations, and achieving new ops until a comprehensive coverage.

## (P1) Use the StableHLO spec as the source of truth.

The [spec](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md) is the source of truth for all verifiers and shape functions of the StableHLO ops. The existing verifiers and shape functions of every op need revisited to be fully aligned with the specification. Note that the specification document keeps evolving, in cases that the spec for an op is not available, the XLA implementation should be used as the source of truth instead: including [xla/service/shape\_inference.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/shape_inference.cc) and [xla/service/hlo\_verifier.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_verifier.cc). XLA implementation doesn't cover unbounded dynamism, so for unbounded dynamism we'll apply common sense until the dynamism RFC is available.


## (P2) Make the most of ODS

ODS files (like [StablehloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td)) define ops with traits and types for each operands/attributes/results and will do the verifications. Thus NO verification code needed in the verifiers or shape functions for the properties which are already guaranteed by the ODS.  Remove the verification code if duplicated with ODS, as they will never be triggered.

Do we need adding tests for the constraints from the ODS? Please see “Establish testing guidelines” below.


## (P3) Maintain verification code in verifiers and shape functions

Both
- **verifiers**: implemented by `Op::verify()`, and
- **shape functions**: implemented by `InferTypeOpInterface` like `Op::inferReturnTypes()` or `Op::inferReturnTypeComponents`

may have verification code to check operands/attributes/results. An initial split would be that: let the verifiers check the operands/attributes, then let shape functions only calculate inferred result types and check the compatibility against the real result types. However, in reality this split has a few problems:

- The shape function can be called by the auto-gen `builder()` alone, without calling the verifier first. So the related inputs must be verified in shape function as well.
- Duplicated code: for example in verifiers we do some processing on the operands then verify some intermediate results, then in shape functions these intermediate results are useful to infer the final results. These intermediate results have to be calculated twice.
- Maintenance burden: as verifications of an op are contained in two different methods.

The solution is as follows:

1. **For most ops w/o regions** (like PadOp):
Put all the verification code into the shape functions, and discard verifiers totally.

2. **For ops w/ regions** (like ReduceOp/IfOp): the `builder()` of the op can be created before all the components are in place, for [example](https://github.com/tensorflow/mlir-hlo/blob/master/lib/Dialect/mhlo/transforms/mhlo_canonicalize_reduction.cc#L222), the ReduceOp is created without regions and soon the shape functions are used.  
    1. If the regions are not needed for type inference (like [ReduceOp](https://github.com/openxla/stablehlo/pull/401)), put the region related verification logic in verifiers. Duplicate some code if it is inevitable.
    2. If the regions are needed for type inference (like IfOp/CaseOp/MapOp), the shape function must verify the regions are not empty explicitly, even though the ODS may already guarantee its existence in the Op definition.


## (P4) Establish testing guidelines

**Do we need to add/maintain tests for verifications that are covered by ODS?**

We do not. The tests should focus on the verifiers and shape functions, while changes to ODS need a revisit of this op.

But stay careful about the missing pieces: for example, if the op contains the trait `SameOperandsAndResultShape` which checks only shapes but not element type, then the verification for element types of operands/results still need tests.

**Where do we put tests for verifiers and type inference?**

[ops\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir) contains the positive cases of ops, and (at least) 1 negative test for every verification error. It is also able to check the inferred type is **compatiable** (not same!) as the real result type.

[infer\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir) verifies the existance of the shape function of an op by line with `hlo_test_infer.get_return_type_components"(%0) : ...` and the inferred type matches exactly as expected. One positive per op.

### What to do
When implement or revisit the verifier and/or shape functions of an op:
1. Put all positive tests and negative tests in [ops\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir) 
2. Add a single positive test in [infer\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir) to verify the interface 
3. (Optional) If an op is complicated and could contain a lot of tests, consider adding a separate test file named `verify_<op_name>.mlir` or` verify_<your_topic>.mlir` within the same folder.


Note: As a temparary solution, now the tests for **bounded dynamism / sparsity** are also put in [infer\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir) 
