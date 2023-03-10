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
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xbf16>, tensor<2xi32>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[[[-1.523440e+00, -2.265630e+00, -6.593750e+00], [5.195310e-01, -1.000000e+00, -4.187500e+00]], [[-3.750000e+00, 2.265630e+00, 2.312500e+00], [-1.093750e+00, 4.500000e+00, 1.640630e+00]], [[1.265630e+00, 3.062500e+00, -3.515630e-01], [-1.289060e+00, 2.125000e+00, 3.828130e+00]], [[-2.796880e+00, 3.554690e-01, -9.765620e-02], [7.695310e-01, -2.359380e+00, 2.515630e+00]]]> : tensor<4x2x3xbf16>
    %1 = stablehlo.constant dense<[-4.843750e+00, 1.843750e+00]> : tensor<2xbf16>
    return %0, %1 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<4x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[-1.523440e+00, -2.265630e+00, -6.593750e+00], [5.195310e-01, -1.000000e+00, -4.187500e+00]], [[-3.750000e+00, 2.265630e+00, 2.312500e+00], [-1.093750e+00, 4.500000e+00, 1.640630e+00]], [[1.265630e+00, 3.062500e+00, -3.515630e-01], [-1.289060e+00, 2.125000e+00, 3.828130e+00]], [[-2.796880e+00, 3.554690e-01, -4.843750e+00], [7.695310e-01, -2.359380e+00, 1.843750e+00]]]> : tensor<4x2x3xbf16>
    return %0 : tensor<4x2x3xbf16>
  }
}

