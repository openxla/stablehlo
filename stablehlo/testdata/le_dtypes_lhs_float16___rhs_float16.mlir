// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<f16>, tensor<f16>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  LE, %0#0, %0#1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f16>, tensor<f16>) {
    %0 = stablehlo.constant dense<-7.796880e+00> : tensor<f16>
    %1 = stablehlo.constant dense<5.910150e+00> : tensor<f16>
    return %0, %1 : tensor<f16>, tensor<f16>
  }
  func.func private @expected() -> tensor<i1> {
    %0 = stablehlo.constant dense<true> : tensor<i1>
    return %0 : tensor<i1>
  }
}
