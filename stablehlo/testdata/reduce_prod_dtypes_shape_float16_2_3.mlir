// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<3xf16>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xf16>, tensor<f16>) -> tensor<3xf16>
     reducer(%arg0: tensor<f16>, %arg1: tensor<f16>)  {
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xf16>, tensor<3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-4.978030e-01, 2.078130e+00, 1.958980e+00], [6.679690e+00, -5.292970e-01, -1.796880e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<3xf16> {
    %0 = stablehlo.constant dense<[-3.324220e+00, -1.099610e+00, -3.519530e+00]> : tensor<3xf16>
    return %0 : tensor<3xf16>
  }
}
