// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<bf16>) -> tensor<2x3xbf16>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<bf16>) -> tensor<2x3xbf16>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xbf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>) {
    %0 = stablehlo.constant dense<[[5.859380e-01, -5.468750e+00, 2.500000e+00], [-1.435550e-01, -8.359380e-01, -5.093750e+00]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<-3.031250e+00> : tensor<bf16>
    %2 = stablehlo.constant dense<-3.375000e+00> : tensor<bf16>
    return %1, %0, %2 : tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<-3.375000e+00> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}
