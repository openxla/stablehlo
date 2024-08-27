// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.clamp %0#0, %0#1, %2 : tensor<2x3xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return %3 : tensor<2x3xf32>
  }
  func.func private @inputs() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<f32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.54880118, 2.385993, 3.78426337], [-5.33662415, 2.95944452, -5.00530338]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<[[0.480827123, -6.6830244, 0.745620071], [0.523653448, 1.05318642, 1.48929167]]> : tensor<2x3xf32>
    %cst_1 = stablehlo.constant dense<0.504019737> : tensor<f32>
    return %cst, %cst_0, %cst_1 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.504019737> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
}
