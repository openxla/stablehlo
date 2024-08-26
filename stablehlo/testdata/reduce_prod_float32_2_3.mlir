// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf32>, tensor<3xf32>) -> ()
    return %2 : tensor<3xf32>
  }
  func.func private @inputs() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.61367369, 0.980555236, -1.08861542], [1.06557202, 0.76509279, -1.83075583]]> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[3.85062957, 0.750215769, 1.99298906]> : tensor<3xf32>
    return %cst : tensor<3xf32>
  }
}
