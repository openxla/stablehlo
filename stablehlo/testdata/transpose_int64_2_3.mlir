// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xi64>
    %1 = call @expected() : () -> tensor<3x2xi64>
    %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi64>) -> tensor<3x2xi64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x2xi64>, tensor<3x2xi64>) -> ()
    return %2 : tensor<3x2xi64>
  }
  func.func private @inputs() -> (tensor<2x3xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, -5, 4], [0, 3, 0]]> : tensor<2x3xi64>
    return %c : tensor<2x3xi64>
  }
  func.func private @expected() -> (tensor<3x2xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, 0], [-5, 3], [4, 0]]> : tensor<3x2xi64>
    return %c : tensor<3x2xi64>
  }
}
