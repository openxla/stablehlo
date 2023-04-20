// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xf16>, tensor<2x4x6xf16>)
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
    }) {window_dimensions = dense<[1, 3, 1]> : tensor<3xi64>, window_strides = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<2x4x6xf16>, tensor<2x1x6xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xf16>) -> tensor<2x4x6xf16>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf16>, tensor<2x4x6xf16>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x1x6xf16>, tensor<2x4x6xf16>) {
    %0 = stablehlo.constant dense<[[[4.433590e+00, -4.148440e+00, 3.183590e+00, 4.082030e-01, -4.722660e+00, -5.335940e+00]], [[2.078130e+00, -1.750980e+00, -5.053710e-01, -1.489260e-01, -2.300780e+00, -2.984380e+00]]]> : tensor<2x1x6xf16>
    %1 = stablehlo.constant dense<[[[-4.785160e+00, -3.021480e+00, 4.015630e+00, -1.683590e+00, 7.275390e-01, -3.066410e-01], [4.507810e+00, -2.763670e+00, -1.994140e+00, -4.035160e+00, -4.648440e+00, 6.863280e+00], [3.226560e+00, -2.892580e+00, -1.348630e+00, -1.852540e+00, 1.824950e-01, 6.882810e+00], [-2.000000e+00, 3.203130e+00, -1.163090e+00, -1.171880e+00, 5.304690e+00, 3.058590e+00]], [[-3.671880e+00, 4.953610e-01, 1.513670e+00, -2.128910e+00, -2.724610e+00, -3.582030e+00], [3.476560e+00, 1.780270e+00, 4.398440e+00, 2.298830e+00, 2.398440e+00, -2.898440e+00], [7.417960e+00, -6.049800e-01, 5.576170e-01, 1.250000e+00, 4.078130e+00, -4.370120e-01], [-2.181640e+00, -2.853520e+00, 1.332030e+00, -2.136720e+00, 7.417960e+00, -2.441410e+00]]]> : tensor<2x4x6xf16>
    return %0, %1 : tensor<2x1x6xf16>, tensor<2x4x6xf16>
  }
  func.func private @expected() -> tensor<2x4x6xf16> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 3.183590e+00, 4.082030e-01, -4.722660e+00, 0.000000e+00], [4.433590e+00, -4.148440e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -5.335940e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -1.750980e+00, -5.053710e-01, -1.489260e-01, 0.000000e+00, 0.000000e+00], [2.078130e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -2.300780e+00, -2.984380e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf16>
    return %0 : tensor<2x4x6xf16>
  }
}

