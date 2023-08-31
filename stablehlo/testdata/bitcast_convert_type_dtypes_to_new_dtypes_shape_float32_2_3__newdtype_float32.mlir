// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf32>
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-2.37265921, 7.448670e-02, 2.72274113], [-1.79927754, -1.90108216, 0.513932347]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-2.37265921, 7.448670e-02, 2.72274113], [-1.79927754, -1.90108216, 0.513932347]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
