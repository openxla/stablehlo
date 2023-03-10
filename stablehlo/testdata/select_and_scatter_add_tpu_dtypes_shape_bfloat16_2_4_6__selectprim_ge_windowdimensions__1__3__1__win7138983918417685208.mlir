// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xbf16>, tensor<2x4x6xbf16>)
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
    }) {window_dimensions = dense<[1, 3, 1]> : tensor<3xi64>, window_strides = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<2x4x6xbf16>, tensor<2x1x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xbf16>) -> tensor<2x4x6xbf16>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xbf16>, tensor<2x4x6xbf16>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x1x6xbf16>, tensor<2x4x6xbf16>) {
    %0 = stablehlo.constant dense<[[[-4.031250e+00, -2.703130e+00, -2.187500e+00, -3.703130e+00, -4.707030e-01, -1.554690e+00]], [[-3.281250e+00, -4.531250e+00, 1.007810e+00, 7.500000e-01, 4.355470e-01, -1.921880e+00]]]> : tensor<2x1x6xbf16>
    %1 = stablehlo.constant dense<[[[-8.593750e-01, 4.343750e+00, 4.406250e+00, 7.929680e-01, -2.281250e+00, 2.921880e+00], [1.671880e+00, 2.597660e-01, 5.906250e+00, 1.585940e+00, 2.949220e-01, -2.984380e+00], [-7.187500e-01, -1.195310e+00, 5.875000e+00, 5.195310e-01, -1.867190e+00, 8.828120e-01], [-3.812500e+00, 3.710940e-02, 3.484380e+00, -1.335940e+00, 2.812500e+00, 2.078130e+00]], [[4.062500e+00, 2.640630e+00, -2.984380e+00, -1.664060e+00, 2.093750e+00, -2.093750e+00], [1.796880e-01, 4.187500e+00, -2.265630e+00, -9.960930e-02, 5.062500e+00, 5.937500e-01], [-2.546880e+00, 2.119140e-01, 2.531250e+00, -9.882810e-01, -4.562500e+00, -1.546880e+00], [-1.890630e+00, -2.640630e+00, 4.531250e+00, -2.093750e+00, -7.578130e-01, 5.031250e+00]]]> : tensor<2x4x6xbf16>
    return %0, %1 : tensor<2x1x6xbf16>, tensor<2x4x6xbf16>
  }
  func.func private @expected() -> tensor<2x4x6xbf16> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, -2.703130e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.554690e+00], [-4.031250e+00, 0.000000e+00, -2.187500e+00, -3.703130e+00, -4.707030e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[-3.281250e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -4.531250e+00, 0.000000e+00, 7.500000e-01, 4.355470e-01, -1.921880e+00], [0.000000e+00, 0.000000e+00, 1.007810e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xbf16>
    return %0 : tensor<2x4x6xbf16>
  }
}

