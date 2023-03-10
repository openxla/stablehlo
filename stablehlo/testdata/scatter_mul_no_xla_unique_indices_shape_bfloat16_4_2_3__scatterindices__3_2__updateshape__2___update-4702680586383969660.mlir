// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xbf16>, tensor<2xbf16>)
    %2 = call @expected() : () -> tensor<4x2x3xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xbf16>, tensor<2xi32>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[[[-7.070310e-01, -7.656250e-01, 2.500000e-01], [4.199220e-02, 2.011720e-01, -1.179690e+00]], [[-1.734380e+00, -1.781250e+00, 9.765620e-01], [2.265630e+00, -2.140630e+00, -3.546880e+00]], [[6.953130e-01, 3.218750e+00, -2.937500e+00], [-9.875000e+00, -1.421880e+00, -8.750000e-01]], [[1.664060e+00, 4.687500e+00, 3.687500e+00], [1.125000e+00, -2.171880e+00, 4.125000e+00]]]> : tensor<4x2x3xbf16>
    %1 = stablehlo.constant dense<[-4.656250e+00, 3.500000e+00]> : tensor<2xbf16>
    return %0, %1 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<4x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[-7.070310e-01, -7.656250e-01, 2.500000e-01], [4.199220e-02, 2.011720e-01, -1.179690e+00]], [[-1.734380e+00, -1.781250e+00, 9.765620e-01], [2.265630e+00, -2.140630e+00, -3.546880e+00]], [[6.953130e-01, 3.218750e+00, -2.937500e+00], [-9.875000e+00, -1.421880e+00, -8.750000e-01]], [[1.664060e+00, 4.687500e+00, -1.712500e+01], [1.125000e+00, -2.171880e+00, 1.443750e+01]]]> : tensor<4x2x3xbf16>
    return %0 : tensor<4x2x3xbf16>
  }
}

