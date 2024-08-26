// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf64>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xf64>, tensor<6xi32>)
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %cst), (%0#1 init: %c) across dimensions = [0] : (tensor<4x6xf64>, tensor<4x6xi32>, tensor<f64>, tensor<i32>) -> (tensor<6xf64>, tensor<6xi32>)
     reducer(%arg0: tensor<f64>, %arg2: tensor<f64>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.maximum %arg0, %arg2 : tensor<f64>
      %4 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f64>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_close(%2#0, %1#0) {has_side_effect = true} : (tensor<6xf64>, tensor<6xf64>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xf64>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.7197398050369812, 0.64320452105575032, 2.2631961699769709, -2.328491537315077, -1.0646327236966586, 0.9983394391370205], [-6.7277944839589292, -2.9801497201356963, 1.1412801151040586, 5.8399824009095695, -1.2944630749197987, 4.9393520979193211], [1.1556124464376827, -3.6421368720551914, 3.6417975059106329, -3.0109402861875258, 0.65566837496731945, 4.0286355796821383], [-5.0392029602908295, 0.85577383132506623, -0.25675898760833782, 2.635548767603793, -4.7856806886501522, -0.28606486625921967]]> : tensor<4x6xf64>
    %c = stablehlo.constant dense<[[3, 0, -1, 2, 0, 1], [8, -7, -4, -2, 0, 3], [-2, 0, 1, 3, 0, 1], [0, -4, -2, -5, -3, 2]]> : tensor<4x6xi32>
    return %cst, %c : tensor<4x6xf64>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xf64> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[3.7197398050369812, 3.000000e+00, 3.6417975059106329, 5.8399824009095695, 3.000000e+00, 4.9393520979193211]> : tensor<6xf64>
    %c = stablehlo.constant dense<[-2, -7, -4, -5, -3, 0]> : tensor<6xi32>
    return %cst, %c : tensor<6xf64>, tensor<6xi32>
  }
}
