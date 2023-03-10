// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<3xf16>
    %2 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xf16>, tensor<f16>) -> tensor<3xf16>
     reducer(%arg0: tensor<f16>, %arg1: tensor<f16>)  {
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xf16>, tensor<3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-5.434570e-01, -4.773440e+00, -2.001950e+00], [2.824220e+00, -5.264280e-02, 3.634770e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<3xf16> {
    %0 = stablehlo.constant dense<[2.824220e+00, -5.264280e-02, 3.634770e+00]> : tensor<3xf16>
    return %0 : tensor<3xf16>
  }
}
