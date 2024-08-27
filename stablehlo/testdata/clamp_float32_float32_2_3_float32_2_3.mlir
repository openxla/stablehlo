// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.clamp %2, %0#1, %0#2 : tensor<2x3xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return %3 : tensor<2x3xf32>
  }
  func.func private @inputs() -> (tensor<f32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.66605306, 1.330880e+01, 4.89149714], [1.76893127, -6.03403234, 0.186552331]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<[[-4.00151539, -1.59384251, 3.47588778], [3.14183354, 0.30163914, -1.07248604]]> : tensor<2x3xf32>
    %cst_1 = stablehlo.constant dense<-0.638752341> : tensor<f32>
    return %cst_1, %cst, %cst_0 : tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.00151539, -1.59384251, 3.47588778], [1.76893127, -0.638752341, -1.07248604]]> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
}
