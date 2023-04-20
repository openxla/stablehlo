// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi8>
    %1 = call @expected() : () -> tensor<2x3xui8>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xi8>) -> tensor<2x3xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xui8>, tensor<2x3xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi8> {
    %0 = stablehlo.constant dense<[[0, 2, 6], [1, -4, -5]]> : tensor<2x3xi8>
    return %0 : tensor<2x3xi8>
  }
  func.func private @expected() -> tensor<2x3xui8> {
    %0 = stablehlo.constant dense<[[0, 2, 6], [1, 252, 251]]> : tensor<2x3xui8>
    return %0 : tensor<2x3xui8>
  }
}
