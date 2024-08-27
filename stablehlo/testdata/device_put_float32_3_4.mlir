// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xf32>
    %1 = call @expected() : () -> tensor<3x4xf32>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<3x4xf32>, tensor<3x4xf32>) -> ()
    return %0 : tensor<3x4xf32>
  }
  func.func private @inputs() -> (tensor<3x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.62003875, -1.38775826, 2.65838385, 5.53908825], [-2.93717909, -2.98530722, -5.79372835, -3.51537132], [-1.01081443, -2.62119961, 6.94263887, 2.50051951]]> : tensor<3x4xf32>
    return %cst : tensor<3x4xf32>
  }
  func.func private @expected() -> (tensor<3x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.62003875, -1.38775826, 2.65838385, 5.53908825], [-2.93717909, -2.98530722, -5.79372835, -3.51537132], [-1.01081443, -2.62119961, 6.94263887, 2.50051951]]> : tensor<3x4xf32>
    return %cst : tensor<3x4xf32>
  }
}
