// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[-5.550780e+00, 2.511720e+00, 2.566410e+00, -9.316400e-01, -4.375000e-01, 3.510740e-01], [-3.529300e+00, 9.062500e-01, -2.654300e+00, 4.321290e-01, -3.615230e+00, 8.835930e+00], [-1.925780e+00, -2.626950e+00, 1.622070e+00, -5.074220e+00, -5.519530e+00, 1.708010e+00], [-2.337890e+00, 3.185550e+00, -1.107420e+00, 2.347660e+00, -2.587890e+00, 2.861330e+00]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[0.000000e+00, -0.000000e+00, 0.000000e+00, -0.000000e+00, 0.000000e+00], [-0.000000e+00, 0.000000e+00, 0.000000e+00, -0.000000e+00, 0.000000e+00], [-0.000000e+00, 0.000000e+00, 0.000000e+00, -0.000000e+00, 0.000000e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

