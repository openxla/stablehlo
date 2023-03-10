// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<complex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  EQ, %0#0, %0#1,  FLOAT : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<i1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<complex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<(-3.50727773,-2.87325335)> : tensor<complex<f32>>
    %1 = stablehlo.constant dense<(-1.0834347,-1.71560287)> : tensor<complex<f32>>
    return %0, %1 : tensor<complex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<i1> {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    return %0 : tensor<i1>
  }
}
