// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xui16>
    %1 = call @expected() : () -> tensor<3xui16>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xui16>, tensor<ui16>) -> tensor<3xui16>
     reducer(%arg0: tensor<ui16>, %arg1: tensor<ui16>)  {
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xui16>, tensor<3xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xui16> {
    %0 = stablehlo.constant dense<[[2, 2, 0], [0, 1, 4]]> : tensor<2x3xui16>
    return %0 : tensor<2x3xui16>
  }
  func.func private @expected() -> tensor<3xui16> {
    %0 = stablehlo.constant dense<[2, 2, 4]> : tensor<3xui16>
    return %0 : tensor<3xui16>
  }
}
