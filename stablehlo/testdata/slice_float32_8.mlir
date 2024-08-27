// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = stablehlo.slice %0 [1:6:2] : (tensor<8xf32>) -> tensor<3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf32>, tensor<3xf32>) -> ()
    return %2 : tensor<3xf32>
  }
  func.func private @inputs() -> (tensor<8xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[3.697320e+00, 0.691851497, -4.25815248, -4.68465376, -1.92406702, 3.28017545, 0.282912105, 3.48844242]> : tensor<8xf32>
    return %cst : tensor<8xf32>
  }
  func.func private @expected() -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0.691851497, -4.68465376, 3.28017545]> : tensor<3xf32>
    return %cst : tensor<3xf32>
  }
}
