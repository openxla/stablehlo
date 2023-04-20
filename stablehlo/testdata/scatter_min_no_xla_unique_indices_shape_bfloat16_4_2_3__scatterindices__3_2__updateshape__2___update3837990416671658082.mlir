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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xbf16>, tensor<2xi32>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[[[3.921880e+00, 1.210940e+00, 3.359380e+00], [-1.484380e+00, 1.585940e+00, -1.414060e+00]], [[3.718750e+00, 9.492180e-01, 2.500000e+00], [-2.988280e-01, -2.575680e-02, 7.531250e+00]], [[2.015630e+00, 3.296880e+00, 4.082030e-01], [1.554690e+00, -3.453130e+00, 5.875000e+00]], [[5.546880e-01, 3.281250e+00, -1.123050e-01], [1.023440e+00, -1.945310e+00, -2.171880e+00]]]> : tensor<4x2x3xbf16>
    %1 = stablehlo.constant dense<[-2.890630e+00, 1.621090e-01]> : tensor<2xbf16>
    return %0, %1 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<4x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[3.921880e+00, 1.210940e+00, 3.359380e+00], [-1.484380e+00, 1.585940e+00, -1.414060e+00]], [[3.718750e+00, 9.492180e-01, 2.500000e+00], [-2.988280e-01, -2.575680e-02, 7.531250e+00]], [[2.015630e+00, 3.296880e+00, 4.082030e-01], [1.554690e+00, -3.453130e+00, 5.875000e+00]], [[5.546880e-01, 3.281250e+00, -2.890630e+00], [1.023440e+00, -1.945310e+00, -2.171880e+00]]]> : tensor<4x2x3xbf16>
    return %0 : tensor<4x2x3xbf16>
  }
}

