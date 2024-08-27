// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xi16>
    %1 = call @expected() : () -> tensor<3x2xi16>
    %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi16>) -> tensor<3x2xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x2xi16>, tensor<3x2xi16>) -> ()
    return %2 : tensor<3x2xi16>
  }
  func.func private @inputs() -> (tensor<2x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-4, -1, 0], [4, 6, -1]]> : tensor<2x3xi16>
    return %c : tensor<2x3xi16>
  }
  func.func private @expected() -> (tensor<3x2xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-4, 4], [-1, 6], [0, -1]]> : tensor<3x2xi16>
    return %c : tensor<3x2xi16>
  }
}
