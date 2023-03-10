// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[-5.500000e+00, 1.820310e+00, -2.500000e+00, 1.929690e+00, -1.437500e+00, -2.796880e+00], [-1.218750e+00, 2.781250e+00, -1.632810e+00, -7.906250e+00, -4.121090e-01, -4.500000e+00], [7.734380e-01, 9.921870e-01, -8.046880e-01, -1.679690e+00, -3.734380e+00, 7.539060e-01], [-6.328130e-01, 3.078130e+00, 9.882810e-01, 3.564450e-02, 7.968750e+00, 1.750000e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[2.781250e+00, 2.781250e+00, 1.929690e+00, 1.929690e+00, 1.000000e+00], [2.781250e+00, 2.781250e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [3.078130e+00, 3.078130e+00, 1.000000e+00, 7.968750e+00, 7.968750e+00]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

