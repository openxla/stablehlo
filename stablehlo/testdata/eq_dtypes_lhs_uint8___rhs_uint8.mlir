// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<ui8>, tensor<ui8>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  EQ, %0#0, %0#1,  UNSIGNED : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<ui8>, tensor<ui8>) {
    %0 = stablehlo.constant dense<4> : tensor<ui8>
    %1 = stablehlo.constant dense<1> : tensor<ui8>
    return %0, %1 : tensor<ui8>, tensor<ui8>
  }
  func.func private @expected() -> tensor<i1> {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    return %0 : tensor<i1>
  }
}
