// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return %4 : tensor<2x3xf32>
  }
  func.func private @inputs() -> (tensor<f32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<f32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.604343534, -8.6676855, -3.67297983], [1.33244514, -0.404034257, -3.10407734]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    return %cst_0, %cst, %cst_1 : tensor<f32>, tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 1.000000e+00, 1.000000e+00], [1.33244514, 1.000000e+00, 1.000000e+00]]> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
}
