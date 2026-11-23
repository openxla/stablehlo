// RUN: not stablehlo-translate --deserialize %s.bc 2>&1 | FileCheck %s

// Crafted .mlir.bc encodes TensorV1Attr whose RankedTensorV1Type has unsupported UniformQuantizedV1Type element type.
// CHECK: unsupported tensor type in TensorV1Attr
