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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    return %2 : tensor<3x5xf32>
  }
  func.func private @inputs() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.577798367, 0.523132384, -1.81298876, 6.48733997, 0.210517034, 1.69092274], [-4.87022209, 1.46585453, 5.6482358, -4.29839659, -0.252917409, -2.678330e+00], [-0.246699423, -6.12239218, 0.689703226, -4.10430193, -2.13349247, 1.53750014], [-2.85476828, -0.966081857, 0.410659164, -2.26567984, -1.87033808, -4.98556089]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
  func.func private @expected() -> (tensor<3x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.46585453, 5.6482358, 6.48733997, 6.48733997, 1.69092274], [1.46585453, 5.6482358, 5.6482358, 1.000000e+00, 1.53750014], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.53750014]]> : tensor<3x5xf32>
    return %cst : tensor<3x5xf32>
  }
}
