// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xbf16>, tensor<2x4x6xbf16>)
    %1 = call @expected() : () -> tensor<2x4x6xbf16>
    %2 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3 = stablehlo.pad %0#1, %2, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %8 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %8 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %8 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<3xi64>} : (tensor<2x4x6xbf16>, tensor<1x3x5xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xbf16>) -> tensor<2x4x6xbf16>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xbf16>, tensor<2x4x6xbf16>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x5xbf16>, tensor<2x4x6xbf16>) {
    %0 = stablehlo.constant dense<[[[1.031250e+00, 6.484380e-01, -3.171880e+00, -6.625000e+00, 2.875000e+00], [7.218750e+00, -3.625000e+00, 5.906250e+00, -3.890630e+00, -2.250000e+00], [-3.078130e+00, -2.285160e-01, -2.281250e+00, 2.062500e+00, 8.320310e-01]]]> : tensor<1x3x5xbf16>
    %1 = stablehlo.constant dense<[[[-3.640630e+00, -2.796880e+00, 2.703130e+00, -4.062500e+00, -3.417970e-01, -2.250000e+00], [-5.742190e-01, 1.773440e+00, -2.402340e-01, 4.562500e+00, -3.453130e+00, -4.937500e+00], [-1.416020e-01, -7.562500e+00, -6.812500e+00, -2.843750e+00, -2.226560e-01, -2.453130e+00], [6.406250e-01, 3.930660e-02, -3.843750e+00, 2.328130e+00, -3.312500e+00, 5.062500e+00]], [[1.054690e+00, 7.750000e+00, 3.093750e+00, -3.421880e+00, 2.949220e-01, -3.281250e+00], [3.640630e+00, -1.992190e-01, 3.390630e+00, -2.687500e+00, 1.570310e+00, -5.812500e+00], [1.343750e+00, -5.343750e+00, 2.312500e+00, 3.609380e+00, 2.812500e-01, -2.937500e+00], [4.062500e+00, 3.625000e+00, -9.218750e-01, -1.453130e+00, 4.125000e+00, -3.609380e+00]]]> : tensor<2x4x6xbf16>
    return %0, %1 : tensor<1x3x5xbf16>, tensor<2x4x6xbf16>
  }
  func.func private @expected() -> tensor<2x4x6xbf16> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -7.812500e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 8.320310e-01]], [[0.000000e+00, 1.679690e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [7.218750e+00, 0.000000e+00, -3.625000e+00, 0.000000e+00, 6.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -2.281250e+00, 0.000000e+00, 0.000000e+00], [-3.078130e+00, -2.285160e-01, 0.000000e+00, 0.000000e+00, 2.062500e+00, 0.000000e+00]]]> : tensor<2x4x6xbf16>
    return %0 : tensor<2x4x6xbf16>
  }
}

