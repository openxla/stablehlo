# StableHLO Specification Checklist

In this document, we summarize the guidelines for reviewing changes to the
specification. At the moment, these changes typically involve checking multiple
things in multiple sources, so this document summarizes them all to simplify
reviews:

  1. Check that the "Specification" column in status.md says "yes", add a row if
     adding a new op.
  1. Check if the section title matches the op's mnemonic in
     [the ODS](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td).
  1. Check if the "Semantics" section matches XLA's
     [Operation Semantics](https://www.tensorflow.org/xla/operation_semantics).
  1. Check whether the "Inputs" and "Outputs" sections:
      1. List the same items as the ODS.
      1. List the same items as [HloInstruction::CreateFromProto](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.cc).
      1. Are ordered exactly like ODS.
      1. If there are any mismatches, check that there are corresponding
         tickets.
  1. Check whether the "Constraints" section:
      1. Matches XLA's
         [shape_inference.cc](https://github.com/openxla/xla/blob/main/xla/service/shape_inference.cc).
      1. Matches XLA's
         [hlo_verifier.cc](https://github.com/openxla/xla/blob/main/xla/service/hlo_verifier.cc).
      1. Matches the ODS.
      1. Matches
         [StablehloOps.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.cpp).
      1. If there are any mismatches, check that there are corresponding
         tickets. Link all those tickets in the spec, in locations which are
         as specific as possible (e.g. if a ticket is about a constraint that
         hasn't been implemented, link the ticket right in that constraint).
      1. If the corresponding parts of the ODS and StablehloOps.cpp match the
         spec, check that the "Verification" and "Type Inference" columns in
         [status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md)
         say "yes".
  1. Check whether the "Examples" section:
      1. Only has one example. (In the future, we'll link to more examples from
         the StableHLO interpreter test suite).
      1. Uses valid MLIR syntax by running `stablehlo-opt` on code examples.
      1. Uses generic MLIR syntax which can be obtained by running
         `stablehlo-opt -mlir-print-op-generic` (we stick to generic syntax in
         the spec to avoid having to change the spec on prettyprinter changes).
  1. Check that the `description` in the op's ODS:
      1. Includes the first sentence of the spec.
      1. Then links to the corresponding section of the spec.
      1. Then uses the same example as the spec but via pretty syntax which can
         be obtaining by running `stablehlo-opt`.
  1. Check that the files related to implementing verification and type
     inference constraints follow the guidelines as mentioned below:
      1. Follow guideline [#1](https://github.com/openxla/stablehlo/blob/main/docs/reference_checklist.md#after-implementing-the-op)
         for [StablehloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td).
      1. Follow guideline [#2](https://github.com/openxla/stablehlo/blob/main/docs/reference_checklist.md#after-implementing-the-op)
         for [TypeInference.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/TypeInference.cpp)
         and [StablehloOps.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.cpp).
      1. Follow guideline [#5](https://github.com/openxla/stablehlo/blob/main/docs/reference_checklist.md#after-implementing-the-op)
         for [ops_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir).
      1. Follow guideline [#6](https://github.com/openxla/stablehlo/blob/main/docs/reference_checklist.md#after-implementing-the-op)
         for [infer_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/infer_stablehlo.mlir).
  1. Evaluate the op for [side effects and
     speculatability](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/).
      1. If the op has no side effects and is always speculatable, give it the
         `Pure` trait. This is rare, as most ops allow dynamic shapes, which may
         lead to shape mismatches at runtime, which is undefined behavior. Some
         ops can have undefined behavior in other situations as well. The vast
         majority of ops do not have side effects (they should have the
         `NoMemoryEffect` trait).
      1. Most ops fall into one of the `HLO_SpeculatableIf*` traits. If the op
         does not fit into any of those, give it the `ConditionallySpeculatable`
         trait and implement the interface methods. Add tests to
         `stablehlo/tests/ops_speculatability.mlir` to cover the speculatability
         logic.
