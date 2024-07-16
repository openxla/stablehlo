// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf16>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xf16>, tensor<6xi32>)
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f16>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %cst), (%0#1 init: %c) across dimensions = [0] : (tensor<4x6xf16>, tensor<4x6xi32>, tensor<f16>, tensor<i32>) -> (tensor<6xf16>, tensor<6xi32>)
     reducer(%arg0: tensor<f16>, %arg2: tensor<f16>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.maximum %arg0, %arg2 : tensor<f16>
      %4 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f16>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_close(%2#0, %1#0) {has_side_effect = true} : (tensor<6xf16>, tensor<6xf16>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xf16>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-7.534170e-01, 3.000000e+00, -3.408200e+00, 8.002920e-01, 4.058590e+00, 5.179690e+00], [1.445310e+00, -4.501950e-01, -6.801760e-01, 1.175780e+00, -2.705080e-01, 1.590820e+00], [4.644530e+00, -1.593750e+00, 3.220700e+00, 1.577150e+00, -1.478520e+00, 1.500240e-01], [-3.230470e+00, -4.367190e+00, 3.647460e-01, 1.127930e+00, 3.892580e+00, 2.025390e+00]]> : tensor<4x6xf16>
    %c = stablehlo.constant dense<[[-2, 1, 1, -4, 2, 0], [2, 3, 0, 6, 5, -1], [3, -5, 0, 2, -3, 0], [0, -4, 2, 1, -2, 2]]> : tensor<4x6xi32>
    return %cst, %c : tensor<4x6xf16>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xf16> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[4.644530e+00, 3.000000e+00, 3.220700e+00, 3.000000e+00, 4.058590e+00, 5.179690e+00]> : tensor<6xf16>
    %c = stablehlo.constant dense<[-2, -5, 0, -4, -3, -1]> : tensor<6xi32>
    return %cst, %c : tensor<6xf16>, tensor<6xi32>
  }
}
