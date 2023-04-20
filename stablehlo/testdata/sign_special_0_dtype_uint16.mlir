// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<2x2xui16>
    %1 = call @expected() : () -> tensor<2x2xui16>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui16>) -> tensor<2x2xui16>
    %4 = stablehlo.compare  EQ, %0, %3,  UNSIGNED : (tensor<2x2xui16>, tensor<2x2xui16>) -> tensor<2x2xi1>
    %5 = stablehlo.constant dense<0> : tensor<ui16>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui16>) -> tensor<2x2xui16>
    %7 = stablehlo.constant dense<1> : tensor<ui16>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<ui16>) -> tensor<2x2xui16>
    %9 = stablehlo.select %4, %6, %8 : tensor<2x2xi1>, tensor<2x2xui16>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<2x2xui16>, tensor<2x2xui16>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @expected() -> tensor<2x2xui16> {
    %0 = stablehlo.constant dense<0> : tensor<2x2xui16>
    return %0 : tensor<2x2xui16>
  }
}
