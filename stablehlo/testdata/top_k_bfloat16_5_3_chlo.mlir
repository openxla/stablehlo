// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x2xbf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5x2xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x3xbf16>
    %1:2 = call @expected() : () -> (tensor<5x2xbf16>, tensor<5x2xi32>)
    %values, %indices = chlo.top_k(%0, k = 2) : tensor<5x3xbf16> -> (tensor<5x2xbf16>, tensor<5x2xi32>)
    stablehlo.custom_call @check.expect_eq(%values, %1#0) {has_side_effect = true} : (tensor<5x2xbf16>, tensor<5x2xbf16>) -> ()
    stablehlo.custom_call @check.expect_eq(%indices, %1#1) {has_side_effect = true} : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    return %values, %indices : tensor<5x2xbf16>, tensor<5x2xi32>
  }
  func.func private @inputs() -> (tensor<5x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[5.062500e+00, 2.781250e+00, -2.093750e+00], [-2.968750e+00, 7.695310e-01, 3.203130e+00], [1.679690e+00, 1.601560e-01, -3.468750e+00], [-1.359380e+00, 3.765630e+00, -9.179680e-01], [1.242190e+00, 3.484380e+00, 4.277340e-01]]> : tensor<5x3xbf16>
    return %cst : tensor<5x3xbf16>
  }
  func.func private @expected() -> (tensor<5x2xbf16> {mhlo.layout_mode = "default"}, tensor<5x2xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[5.062500e+00, 2.781250e+00], [3.203130e+00, 7.695310e-01], [1.679690e+00, 1.601560e-01], [3.765630e+00, -9.179680e-01], [3.484380e+00, 1.242190e+00]]> : tensor<5x2xbf16>
    %c = stablehlo.constant dense<[[0, 1], [2, 1], [0, 1], [1, 2], [1, 0]]> : tensor<5x2xi32>
    return %cst, %c : tensor<5x2xbf16>, tensor<5x2xi32>
  }
}
