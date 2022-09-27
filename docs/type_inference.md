# Type Inference

StableHLO is an operation set that expresses ML computations. It has been originally bootstrapped from [the MHLO dialect](https://github.com/tensorflow/mlir-hlo#meta-hlo-dialect-mhlo), including inheriting the implementation of type inference. The implementation progress is tracked in [status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md). 

To implement high-quality verifiers and shape functions for StableHLO ops, these guidelines and criterion are proposed below to follow:

# Proposal

These proposals apply to both revisiting existing implementations, and achieving new ops til a comprehensive coverage.

## (P1) Use the StableHLO spec as the source of truth. 

The [spec](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md) is the source of truth for all verifications of the StableHLO ops. The existing type inference implementation of every ops need revisited to be fully aligned with the specification. Note that the specification document keeps evolving, in cases that the spec for an op is not available, the XLA implementation should be used as the source of truth instead: including [xla/service/shape\_inference.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/shape_inference.cc) and [xla/service/hlo\_verifier.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_domain_verifier.cc). XLA implementation doesn't cover unbounded dynamism, so for unbounded dynamism we'll apply common sense until the dynamism RFC is available.


## (P2) Make the most of ODS

ODS files (like [StablehloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td)) define ops with traits and types for each operands/attributes/results and will do the verifications. Thus NO verification code needed in the verifiers or shape functions for the properties which are already guaranteed by the ODS.  Remove the verification code if duplicated with ODS, as they will never be triggered.

Do we need adding tests for the constraints from the ODS? Please see “P5 Establish testing guidelines” below.


## (P3) Merge verifiers into shape functions

Both 
- **verifiers**: implemented by `Op::verify()`, and 
- **shape functions**: implemented by `InferTypeOpInterface` like `Op::inferReturnTypes()` or `Op::inferReturnTypeComponents` 

may have verification code to check operands/attributes/results. An ideal split would be that: let the verifiers check the operands/attributes, then let shape functions to calculate inferred result types and check the compatibility against the real result types. However, in reality this split has a few problems:

1. Duplicated code: for example in verifiers we do some processing on the operands then verify some intermediate results, then in shape functions these intermediate results are useful to infer the final results. These intermediate results have to be calculated twice.
2. Maintenance burden: as verifications of an op are contained in two different method. 

Thus, in most cases the verifiers will be removed and all the verification logic will be merged into the shape functions (example: [PR#135](https://github.com/openxla/stablehlo/pull/135)). The only exceptions will be:

1. Op that cannot have a shape function: like `stablehlo.reshape`, whose result type cannot be inferred.
2. (should be rare) Op that have tons of logic in their verifier but very little logic in their shape function.


## (P4) Align error messages by following standard templates. 

Different error messages for similar types of error have inconsistent user experience and more maintenance efforts. We propose using utils to wrap each type of common errors and apply gradually to the existing code.


## (P5) Establish testing guidelines

**Do we need to add/maintain tests for verifications that are covered by ODS?**

They are expected to be optional. The tests should focus on the verifiers and shape functions, while changes to ODS need a revisit of this op.

But stay careful about the missing pieces: for example, if the op contains `SameOperandsAndResultShape` in its definition, the element types of operands/results still need to be checked with tests. 

**Where do we put tests for verifiers and type inference?**

Verifications are kept in [ops\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir) and [infer\_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir). If an op is complicated and could contain a lot of tests, consider adding a separate test file named `verify_<op_name>.mlir` or` verify_<your_topic>.mlir` within the same folder.


## (P6) Improve our infrastructure to better support dynamism

StableHLO supports dynamism, and has experimental support for bounded dynamism. In perspective of type inference, certain improvements and modifications are needed. Please refer to [bounded_dynamism.md](https://github.com/openxla/stablehlo/blob/main/stablehlo/docs/bounded_dynamism.md) for detailed plans. 
