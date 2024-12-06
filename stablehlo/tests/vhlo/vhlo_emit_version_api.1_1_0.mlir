// RUN: stablehlo-translate --deserialize --print-stablehlo-version %s.bc | FileCheck %s --check-prefix=CHECK-VERSION
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize --print-stablehlo-version | FileCheck %s --check-prefix=CHECK-VERSION-LATEST
// RUN: stablehlo-translate --deserialize --print-stablehlo-version %s | FileCheck %s --check-prefix=CHECK-VERSION-NOT-BYTECODE

// This file tests the `getPortableArtifactVersion` Serialization API.
// Any breakages to this file likely indicate that the MLIR Bytecode Format
// has changed, or that the StableHLO producer string emit by
// `serializePortableArtifact` has changed.
//
// See the `getPortableArtifactVersion` doc comments for more details.

// CHECK-VERSION: // Reading portable artifact with StableHLO version: 1.1.0
// CHECK-VERSION-NOT-BYTECODE: // Failed parsing StableHLO version from portable artifact
// CHECK-VERSION-LATEST: // Reading portable artifact with StableHLO version: {{.*}}

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
  func.return %0 : tensor<f32>
}
