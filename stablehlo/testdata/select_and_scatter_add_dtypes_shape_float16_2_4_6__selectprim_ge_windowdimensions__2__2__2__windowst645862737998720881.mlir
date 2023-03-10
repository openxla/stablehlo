// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xf16>, tensor<2x4x6xf16>)
    %1 = call @expected() : () -> tensor<2x4x6xf16>
    %2 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %3 = stablehlo.pad %0#1, %2, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %8 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %8 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %8 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<3xi64>} : (tensor<2x4x6xf16>, tensor<1x3x5xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xf16>) -> tensor<2x4x6xf16>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf16>, tensor<2x4x6xf16>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x5xf16>, tensor<2x4x6xf16>) {
    %0 = stablehlo.constant dense<[[[-6.035150e-01, 5.503900e+00, 2.365230e+00, -2.863770e-01, 2.042970e+00], [-3.175780e+00, 1.456050e+00, 2.150390e+00, -2.011720e+00, -4.417970e+00], [-1.253910e+00, 1.260740e+00, 3.824220e+00, -1.892090e-01, -3.078130e+00]]]> : tensor<1x3x5xf16>
    %1 = stablehlo.constant dense<[[[-2.566410e+00, 9.664060e+00, 1.090820e+00, -3.889160e-01, 2.419920e+00, -4.296880e+00], [1.486330e+00, -3.148440e+00, -3.123050e+00, -9.765620e-01, -2.857420e+00, 4.804690e+00], [2.195310e+00, 3.851560e+00, 1.888670e+00, 4.781250e+00, 1.798830e+00, -3.408200e-01], [2.542970e+00, -1.731450e+00, 4.484860e-01, -1.126950e+00, -3.127440e-01, -2.544920e+00]], [[-1.700200e+00, 2.589840e+00, 8.183590e-01, 1.253660e-01, 5.007810e+00, -7.382810e-01], [-2.193360e+00, -1.587890e+00, -1.166990e+00, -2.828130e+00, -6.589840e+00, -2.560550e+00], [-1.237300e+00, 6.722650e+00, -1.347660e+00, -2.244140e+00, 6.292960e+00, -1.539060e+00], [-3.539060e+00, -3.609380e+00, 1.350100e-01, -1.654300e+00, -2.394530e+00, -2.148440e+00]]]> : tensor<2x4x6xf16>
    return %0, %1 : tensor<1x3x5xf16>, tensor<2x4x6xf16>
  }
  func.func private @expected() -> tensor<2x4x6xf16> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 4.898440e+00, 2.365230e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.976560e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.756840e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -1.711910e+00, 0.000000e+00, 0.000000e+00, -9.695310e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf16>
    return %0 : tensor<2x4x6xf16>
  }
}

