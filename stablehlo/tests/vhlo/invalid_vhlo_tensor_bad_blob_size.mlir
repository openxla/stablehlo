// RUN: not stablehlo-translate --deserialize %s.bc 2>&1 | FileCheck %s

// Crafted .mlir.bc encodes TensorV1Attr with tensor<576460752303423490x1xi32> and an 8-byte blob (bitwidth * count wraps mod 2^64).
// CHECK: invalid dense tensor buffer size in TensorV1Attr
