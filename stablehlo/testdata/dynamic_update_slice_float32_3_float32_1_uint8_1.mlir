// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<3xf32>, tensor<1xf32>, tensor<1xui8>)
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = stablehlo.slice %0#2 [0:1] : (tensor<1xui8>) -> tensor<1xui8>
    %3 = stablehlo.reshape %2 : (tensor<1xui8>) -> tensor<ui8>
    %4 = stablehlo.dynamic_update_slice %0#0, %0#1, %3 : (tensor<3xf32>, tensor<1xf32>, tensor<ui8>) -> tensor<3xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<3xf32>, tensor<3xf32>) -> ()
    return %4 : tensor<3xf32>
  }
  func.func private @inputs() -> (tensor<3xf32> {mhlo.layout_mode = "default"}, tensor<1xf32> {mhlo.layout_mode = "default"}, tensor<1xui8> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-3.38123536, 5.99582672, 0.870595455]> : tensor<3xf32>
    %cst_0 = stablehlo.constant dense<0.219151393> : tensor<1xf32>
    %c = stablehlo.constant dense<1> : tensor<1xui8>
    return %cst, %cst_0, %c : tensor<3xf32>, tensor<1xf32>, tensor<1xui8>
  }
  func.func private @expected() -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-3.38123536, 0.219151393, 0.870595455]> : tensor<3xf32>
    return %cst : tensor<3xf32>
  }
}
