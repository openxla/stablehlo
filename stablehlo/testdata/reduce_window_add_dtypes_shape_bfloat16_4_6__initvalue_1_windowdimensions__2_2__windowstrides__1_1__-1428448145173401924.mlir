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
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[1.148440e+00, -2.921880e+00, 7.304680e-01, 1.234380e+00, -1.552730e-01, 2.937500e+00], [-4.125000e+00, 1.406250e+00, -3.359380e+00, 3.218750e+00, 7.968750e-01, -8.750000e-01], [-1.117190e+00, 1.218750e+00, 6.445310e-01, -3.671880e-01, -1.312500e+00, 4.843750e+00], [3.296880e+00, 1.203130e+00, 3.140630e+00, 5.712890e-02, -2.046880e+00, -2.281250e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[-3.468750e+00, -3.140630e+00, 2.828130e+00, 6.125000e+00, 3.687500e+00], [-1.625000e+00, 9.101560e-01, 1.132810e+00, 3.312500e+00, 4.437500e+00], [5.625000e+00, 7.187500e+00, 4.468750e+00, -2.671880e+00, 2.031250e-01]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

