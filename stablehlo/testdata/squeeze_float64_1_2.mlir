// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<1x2xf64>
    %1 = call @expected() : () -> tensor<2xf64>
    %2 = stablehlo.reshape %0 : (tensor<1x2xf64>) -> tensor<2xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xf64>, tensor<2xf64>) -> ()
    return %2 : tensor<2xf64>
  }
  func.func private @inputs() -> (tensor<1x2xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.4748586358361, 0.73655167448850767]]> : tensor<1x2xf64>
    return %cst : tensor<1x2xf64>
  }
  func.func private @expected() -> (tensor<2xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-1.4748586358361, 0.73655167448850767]> : tensor<2xf64>
    return %cst : tensor<2xf64>
  }
}
