// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<f16>, tensor<2x3xf16>, tensor<f16>)
    %1 = call @expected() : () -> tensor<2x3xf16>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f16>) -> tensor<2x3xf16>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f16>) -> tensor<2x3xf16>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f16>, tensor<2x3xf16>, tensor<f16>) {
    %0 = stablehlo.constant dense<[[-4.992190e+00, 3.960940e+00, 6.879880e-01], [-1.763670e+00, 6.917960e+00, 1.958010e+00]]> : tensor<2x3xf16>
    %1 = stablehlo.constant dense<5.437500e+00> : tensor<f16>
    %2 = stablehlo.constant dense<1.844730e+00> : tensor<f16>
    return %1, %0, %2 : tensor<f16>, tensor<2x3xf16>, tensor<f16>
  }
  func.func private @expected() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<1.844730e+00> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
}
