// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<3xbf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<3xbf16>
     reducer(%arg0: tensor<bf16>, %arg1: tensor<bf16>)  {
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xbf16>, tensor<3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[2.828130e+00, -1.890630e+00, 3.312500e+00], [-3.546880e+00, 5.000000e+00, 2.421880e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<3xbf16> {
    %0 = stablehlo.constant dense<[-7.187500e-01, 3.109380e+00, 5.750000e+00]> : tensor<3xbf16>
    return %0 : tensor<3xbf16>
  }
}
