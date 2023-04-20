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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xbf16>, tensor<2xi32>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[[[-1.750000e+00, -1.904300e-01, 2.968750e+00], [3.906250e+00, 4.687500e+00, -9.765620e-01]], [[-1.765630e+00, 6.015630e-01, 2.953130e+00], [-1.937500e+00, 2.177730e-01, -3.218750e+00]], [[1.117190e+00, -1.945310e+00, -1.398440e+00], [2.671880e+00, -3.921880e+00, -1.671880e+00]], [[2.031250e+00, -6.445310e-01, 2.968750e-01], [-3.062500e+00, 7.861330e-02, -2.781250e+00]]]> : tensor<4x2x3xbf16>
    %1 = stablehlo.constant dense<[-2.656250e+00, -2.275390e-01]> : tensor<2xbf16>
    return %0, %1 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<4x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[-1.750000e+00, -1.904300e-01, 2.968750e+00], [3.906250e+00, 4.687500e+00, -9.765620e-01]], [[-1.765630e+00, 6.015630e-01, 2.953130e+00], [-1.937500e+00, 2.177730e-01, -3.218750e+00]], [[1.117190e+00, -1.945310e+00, -1.398440e+00], [2.671880e+00, -3.921880e+00, -1.671880e+00]], [[2.031250e+00, -6.445310e-01, 2.968750e-01], [-3.062500e+00, 7.861330e-02, -2.275390e-01]]]> : tensor<4x2x3xbf16>
    return %0 : tensor<4x2x3xbf16>
  }
}

