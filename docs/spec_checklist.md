# StableHLO Specification Checklist

In this document, we summarize the guidelines for reviewing changes to the
specification. At the moment, these changes typically involve checking multiple
things in multiple sources, so this document summarizes them all to simplify
reviews:

  1. Check if the section title matches the op's mnemonic in
     [the ODS](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td).
  1. Check if the "Semantics" section matches XLA's
     [Operation Semantics](https://www.tensorflow.org/xla/operation_semantics).
  1. Check if the "Operands" section lists items in the same order as the ODS.
  1. Check whether the "Constraints" section:
      1. Matches XLA's
         [shape_inference.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/shape_inference.cc).
      1. Matches XLA's
         [hlo_verifier.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_verifier.cc).
      1. Matches the ODS.
      1. Matches
         [StablehloOps.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.cpp).
      1. If there are any mismatches, check that there are corresponding tickets.
      1. If the corresponding parts of the ODS and StablehloOps.cpp match the
         spec, check that the "Verification" and "Type Inference" columns in
         [status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md)
         say "yes".
  1. Check whether the "Examples" section uses valid MLIR syntax by running
     `stablehlo-opt` on code examples.
  1. Check that the "Specification" column in status.md says "yes".
  1. Check that the `description` in op's ODS links to the corresponding section
     of the spec.
