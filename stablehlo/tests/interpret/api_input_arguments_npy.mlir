// RUN: rm -rf %t && mkdir -p %t
// RUN: stablehlo-translate --interpret %s --probe-output-dir=%t --args="[dense<[10, 20]> : tensor<2xi32>, dense<[1, 2]> : tensor<2xi32>]" | FileCheck %s
// RUN: stablehlo-translate --interpret %s --probe-output-dir=%t --args=@%t/probe1.npy,@%t/probe2.npy | FileCheck %s

// RUN: not stablehlo-translate --interpret %s --args=@%t/missing.npy 2>&1 | FileCheck %s --check-prefixes=CHECK-ERROR-MISSING
// CHECK-ERROR-MISSING: failed to read NumPy args file

// RUN: not stablehlo-translate --interpret %s --args=@%t/other.mlir,@%t/probe1.npy 2>&1 | FileCheck %s --check-prefixes=CHECK-ERROR-MIXED
// CHECK-ERROR-MIXED: cannot mix .npy and non-.npy args files

// The probes serialize each argument to <probe-output-dir>/probe<N>.npy on
// the first run; the second run loads those files back as the arguments,
// inferring the tensor types from the NumPy headers.
func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %0 = interpreter.probe %arg0, probe_id = "arg0" : tensor<2xi32>
  %1 = interpreter.probe %arg1, probe_id = "arg1" : tensor<2xi32>
  %2 = stablehlo.add %0, %1 : tensor<2xi32>
  return %2 : tensor<2xi32>
}

// CHECK:      tensor<2xi32> {
// CHECK-NEXT:   [11, 22]
// CHECK-NEXT: }
