// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return %2 : tensor<2x3xf32>
  }
  func.func private @inputs() -> (tensor<2x3xi1> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<2x3xi1>
    %cst = stablehlo.constant dense<[[-0.477684617, 1.12567687, -6.458900e+00], [0.738749385, -4.38916922, -4.62747955]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<[[3.92670202, 1.27711701, 4.59667206], [-3.72463632, -1.26884151, -4.21353149]]> : tensor<2x3xf32>
    return %c, %cst, %cst_0 : tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.92670202, 1.27711701, 4.59667206], [-3.72463632, -1.26884151, -4.21353149]]> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
}
