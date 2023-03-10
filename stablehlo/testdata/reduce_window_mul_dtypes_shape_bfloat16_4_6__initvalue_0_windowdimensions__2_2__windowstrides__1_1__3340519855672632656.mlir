// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[-1.960940e+00, -5.031250e+00, -2.703130e+00, -5.593750e+00, 1.445310e+00, 1.593750e+00], [-3.281250e+00, 3.765630e+00, -3.609380e+00, 6.132810e-01, -3.000000e+00, -4.531250e+00], [4.812500e+00, -2.500000e+00, 1.546880e+00, -1.921880e+00, -5.593750e+00, 3.140630e+00], [-1.437500e+00, 1.265630e+00, 1.367190e+00, -3.640630e+00, 7.421880e-01, 4.281250e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[-0.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -0.000000e+00, -0.000000e+00], [0.000000e+00, -0.000000e+00, 0.000000e+00, -0.000000e+00, -0.000000e+00]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

