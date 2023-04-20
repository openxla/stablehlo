// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<bf16>
    %1 = call @expected() : () -> tensor<bf16>
    %2 = stablehlo.reduce_precision %0, format = e8m7 : tensor<bf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<bf16> {
    %0 = stablehlo.constant dense<-1.781250e+00> : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @expected() -> tensor<bf16> {
    %0 = stablehlo.constant dense<-1.781250e+00> : tensor<bf16>
    return %0 : tensor<bf16>
  }
}
