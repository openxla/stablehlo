// RUN: not stablehlo-translate --deserialize %s.bc 2>&1 | FileCheck %s

// Crafted .mlir.bc encodes TensorV1Attr whose RankedTensorV1Type shape [4611686018427387905, 4] overflows int64 element count.
// CHECK: tensor shape element count overflows in TensorV1Attr
