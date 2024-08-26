// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "stablehlo.reduce_window"(%0, %cst) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    return %2 : tensor<3x5xf32>
  }
  func.func private @inputs() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-6.80302143, 0.321375608, -1.86038101, 3.62991023, 0.598879099, -5.04817581], [-0.74092257, -0.301740915, -2.58586526, 6.663420e-01, 2.39137483, 6.18894672], [-4.40329695, 1.96387827, 2.2082417, -2.70424533, 1.05929911, -0.497613579], [3.7442131, -2.94033289, -4.3009696, -1.19836605, 1.62041414, -0.197622195]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
  func.func private @expected() -> (tensor<3x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-6.52430916, -3.42661142, 0.850006103, 8.28650569, 5.13102484], [-2.48208237, 2.28451395, -1.41552687, 2.41277075, 10.1420069], [-0.635538578, -2.0691824, -4.99533939, -0.222898126, 2.98447728]]> : tensor<3x5xf32>
    return %cst : tensor<3x5xf32>
  }
}
