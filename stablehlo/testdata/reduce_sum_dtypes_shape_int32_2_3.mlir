// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi32>
    %1 = call @expected() : () -> tensor<3xi32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
     reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %5 = stablehlo.add %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi32> {
    %0 = stablehlo.constant dense<[[5, -1, 3], [-3, 0, 6]]> : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
  func.func private @expected() -> tensor<3xi32> {
    %0 = stablehlo.constant dense<[2, -1, 9]> : tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}
