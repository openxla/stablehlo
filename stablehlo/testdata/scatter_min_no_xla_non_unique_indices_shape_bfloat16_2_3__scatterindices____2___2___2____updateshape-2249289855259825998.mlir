// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<2> : tensor<1x3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<2x1x3xbf16>)
    %2 = call @expected() : () -> tensor<2x3xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>} : (tensor<2x3xbf16>, tensor<1x3x1xi32>, tensor<2x1x3xbf16>) -> tensor<2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xbf16>, tensor<2x1x3xbf16>) {
    %0 = stablehlo.constant dense<[[3.265630e+00, 9.277340e-02, 1.585940e+00], [2.234380e+00, -3.906250e+00, -8.281250e-01]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<[[[-2.089840e-01, 6.953130e-01, -5.468750e-01]], [[2.890630e+00, 4.062500e-01, 4.531250e+00]]]> : tensor<2x1x3xbf16>
    return %0, %1 : tensor<2x3xbf16>, tensor<2x1x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[3.265630e+00, 9.277340e-02, -5.468750e-01], [2.234380e+00, -3.906250e+00, -8.281250e-01]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}

