// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xf32>, tensor<6xi32>)
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %cst), (%0#1 init: %c) across dimensions = [0] : (tensor<4x6xf32>, tensor<4x6xi32>, tensor<f32>, tensor<i32>) -> (tensor<6xf32>, tensor<6xi32>)
     reducer(%arg0: tensor<f32>, %arg2: tensor<f32>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.maximum %arg0, %arg2 : tensor<f32>
      %4 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f32>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_close(%2#0, %1#0) {has_side_effect = true} : (tensor<6xf32>, tensor<6xf32>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xf32>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.310264707, -1.99879205, -1.81809437, -0.426923692, -0.390648484, 1.98554444], [-1.17653537, -5.20831203, 0.447554022, -2.48738694, -2.55485487, 4.22119665], [5.650640e+00, 4.86770153, 5.00692511, -3.77362943, 2.78308082, 0.220573112], [-3.25627398, -4.64873171, -2.7951088, -0.643593251, 1.89478385, -3.40890503]]> : tensor<4x6xf32>
    %c = stablehlo.constant dense<[[1, 5, -1, 0, -5, 1], [3, -2, -3, -4, 0, 1], [0, -5, 0, 0, 0, -3], [-2, 1, 0, 0, -4, 2]]> : tensor<4x6xi32>
    return %cst, %c : tensor<4x6xf32>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xf32> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[5.650640e+00, 4.86770153, 5.00692511, 3.000000e+00, 3.000000e+00, 4.22119665]> : tensor<6xf32>
    %c = stablehlo.constant dense<[-2, -5, -3, -4, -5, -3]> : tensor<6xi32>
    return %cst, %c : tensor<6xf32>, tensor<6xi32>
  }
}
