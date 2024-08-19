// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x2xf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5x2xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x3xf16>
    %1:2 = call @expected() : () -> (tensor<5x2xf16>, tensor<5x2xi32>)
    %values, %indices = chlo.top_k(%0, k = 2) : tensor<5x3xf16> -> (tensor<5x2xf16>, tensor<5x2xi32>)
    stablehlo.custom_call @check.expect_eq(%values, %1#0) {has_side_effect = true} : (tensor<5x2xf16>, tensor<5x2xf16>) -> ()
    stablehlo.custom_call @check.expect_eq(%indices, %1#1) {has_side_effect = true} : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    return %values, %indices : tensor<5x2xf16>, tensor<5x2xi32>
  }
  func.func private @inputs() -> (tensor<5x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.533200e+00, 2.400390e+00, 4.035640e-01], [-1.331330e-02, 8.818350e-01, 4.128910e+00], [-1.163090e+00, 3.076170e+00, 3.720700e+00], [6.723630e-01, -6.713870e-02, -1.308590e+00], [-3.121090e+00, 2.203130e+00, -1.001950e+00]]> : tensor<5x3xf16>
    return %cst : tensor<5x3xf16>
  }
  func.func private @expected() -> (tensor<5x2xf16> {mhlo.layout_mode = "default"}, tensor<5x2xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.533200e+00, 2.400390e+00], [4.128910e+00, 8.818350e-01], [3.720700e+00, 3.076170e+00], [6.723630e-01, -6.713870e-02], [2.203130e+00, -1.001950e+00]]> : tensor<5x2xf16>
    %c = stablehlo.constant dense<[[0, 1], [2, 1], [2, 1], [0, 1], [1, 2]]> : tensor<5x2xi32>
    return %cst, %c : tensor<5x2xf16>, tensor<5x2xi32>
  }
}
