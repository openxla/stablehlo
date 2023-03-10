// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<3xf16>
    %2 = stablehlo.constant dense<0x7C00> : tensor<f16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xf16>, tensor<f16>) -> tensor<3xf16>
     reducer(%arg0: tensor<f16>, %arg1: tensor<f16>)  {
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xf16>, tensor<3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[1.444340e+00, 3.923830e+00, -1.068730e-01], [-1.183590e+00, 3.109380e+00, 1.297850e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<3xf16> {
    %0 = stablehlo.constant dense<[-1.183590e+00, 3.109380e+00, -1.068730e-01]> : tensor<3xf16>
    return %0 : tensor<3xf16>
  }
}
