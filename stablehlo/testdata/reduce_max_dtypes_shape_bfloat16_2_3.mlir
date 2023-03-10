// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<3xbf16>
    %2 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<3xbf16>
     reducer(%arg0: tensor<bf16>, %arg1: tensor<bf16>)  {
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xbf16>, tensor<3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[-3.218750e+00, 4.218750e+00, 3.796880e+00], [3.828130e+00, 4.875000e+00, 1.914060e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<3xbf16> {
    %0 = stablehlo.constant dense<[3.828130e+00, 4.875000e+00, 3.796880e+00]> : tensor<3xbf16>
    return %0 : tensor<3xbf16>
  }
}
