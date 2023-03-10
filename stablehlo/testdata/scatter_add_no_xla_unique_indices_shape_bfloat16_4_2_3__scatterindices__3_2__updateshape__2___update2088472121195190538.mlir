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
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xbf16>, tensor<2xi32>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[[[1.738280e-01, -6.125000e+00, -2.125000e+00], [2.171880e+00, -5.375000e+00, -7.382810e-01]], [[4.468750e+00, 7.437500e+00, 1.242190e+00], [1.250000e+00, -5.375000e+00, -1.078130e+00]], [[1.789060e+00, 5.786130e-02, -6.679690e-01], [2.062500e+00, -6.718750e+00, 1.921880e+00]], [[9.414060e-01, 5.078130e-01, -3.125000e+00], [-6.210940e-01, 3.125000e-01, -2.312500e+00]]]> : tensor<4x2x3xbf16>
    %1 = stablehlo.constant dense<[4.312500e+00, -4.550780e-01]> : tensor<2xbf16>
    return %0, %1 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<4x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[1.738280e-01, -6.125000e+00, -2.125000e+00], [2.171880e+00, -5.375000e+00, -7.382810e-01]], [[4.468750e+00, 7.437500e+00, 1.242190e+00], [1.250000e+00, -5.375000e+00, -1.078130e+00]], [[1.789060e+00, 5.786130e-02, -6.679690e-01], [2.062500e+00, -6.718750e+00, 1.921880e+00]], [[9.414060e-01, 5.078130e-01, 1.187500e+00], [-6.210940e-01, 3.125000e-01, -2.765630e+00]]]> : tensor<4x2x3xbf16>
    return %0 : tensor<4x2x3xbf16>
  }
}

