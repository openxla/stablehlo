// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    %1 = stablehlo.constant dense<[[-5.72039843, -0.199320853, 2.72947836], [-2.44254518, -2.37061977, -3.022614]]> : tensor<2x3xf32>
    %2 = stablehlo.constant dense<[[-3.69347548, 4.17320395, 1.85712087], [-1.41969764, 0.223469049, -0.322650284]]> : tensor<2x3xf32>
    return %0, %1, %2 : tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-3.69347548, 4.17320395, 1.85712087], [-1.41969764, 0.223469049, -0.322650284]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
