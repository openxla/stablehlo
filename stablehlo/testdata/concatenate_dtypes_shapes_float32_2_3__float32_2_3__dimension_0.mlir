// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<4x3xf32>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[-4.2507205, -2.47488475, -1.90974963], [-2.04686093, -6.64819049, -3.30281329]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<[[0.665820598, 5.063450e-01, 0.643919349], [-2.22002125, 4.40628481, 1.91231513]]> : tensor<2x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<4x3xf32> {
    %0 = stablehlo.constant dense<[[-4.2507205, -2.47488475, -1.90974963], [-2.04686093, -6.64819049, -3.30281329], [0.665820598, 5.063450e-01, 0.643919349], [-2.22002125, 4.40628481, 1.91231513]]> : tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}
