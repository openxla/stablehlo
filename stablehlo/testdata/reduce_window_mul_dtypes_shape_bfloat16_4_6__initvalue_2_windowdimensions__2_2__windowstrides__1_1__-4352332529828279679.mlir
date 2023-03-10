// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[-6.875000e-01, -4.781250e+00, -3.203130e+00, 2.937500e+00, -2.656250e+00, -2.640630e+00], [-4.824220e-01, 3.390630e+00, -1.201170e-01, -3.750000e+00, -3.859380e+00, 8.710930e-01], [2.046880e+00, -6.156250e+00, -2.828130e+00, 3.750000e+00, 1.523440e+00, 9.960930e-01], [2.656250e+00, 2.515630e+00, -4.375000e+00, -5.031250e+00, 2.093750e+00, 5.093750e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[-1.075000e+01, -1.250000e+01, -8.500000e+00, -2.260000e+02, -4.700000e+01], [4.125000e+01, -1.425000e+01, -9.562500e+00, 1.660000e+02, -1.018750e+01], [-1.690000e+02, -3.820000e+02, -4.680000e+02, -1.205000e+02, 3.225000e+01]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

