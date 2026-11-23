// RUN: not stablehlo-translate --deserialize %s.bc 2>&1 | FileCheck %s

// Crafted .mlir.bc encodes TensorV1Attr with non-shaped scalar IntegerSI32V1Type instead of RankedTensorV1Type.
// CHECK: expected RankedTensorV1Type in TensorV1Attr
