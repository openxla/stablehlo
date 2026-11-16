// RUN: not stablehlo-translate --deserialize %s.bc 2>&1 | FileCheck %s

// Crafted .mlir.bc encodes TensorV1Attr whose RankedTensorV1Type has dynamic dim [-1, 2].
// CHECK: expected statically shaped tensor type in TensorV1Attr
