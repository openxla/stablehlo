// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xi1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xi1>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.compare  GT, %2, %0#1,  FLOAT : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    stablehlo.custom_call @check.expect_eq(%3, %1) {has_side_effect = true} : (tensor<2x3xi1>, tensor<2x3xi1>) -> ()
    return %3 : tensor<2x3xi1>
  }
  func.func private @inputs() -> (tensor<f32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.95861864, -0.114522837, -0.125803873], [1.55238354, -5.0227437, -1.20407963]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<-2.95368242> : tensor<f32>
    return %cst_0, %cst : tensor<f32>, tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<2x3xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[false, false, false], [false, true, false]]> : tensor<2x3xi1>
    return %c : tensor<2x3xi1>
  }
}
