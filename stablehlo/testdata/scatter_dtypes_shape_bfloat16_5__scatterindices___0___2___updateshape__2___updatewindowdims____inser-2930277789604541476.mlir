// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5xbf16>, tensor<2xbf16>)
    %2 = call @expected() : () -> tensor<5xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xbf16>, tensor<2x1xi32>, tensor<2xbf16>) -> tensor<5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xbf16>, tensor<5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xbf16>, tensor<2xbf16>) {
    %0 = stablehlo.constant dense<[-8.085930e-01, 9.765620e-01, -2.015630e+00, -1.483150e-02, -1.546880e+00]> : tensor<5xbf16>
    %1 = stablehlo.constant dense<[-1.312500e+00, -3.171880e+00]> : tensor<2xbf16>
    return %0, %1 : tensor<5xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<5xbf16> {
    %0 = stablehlo.constant dense<[-1.312500e+00, 9.765620e-01, -3.171880e+00, -1.483150e-02, -1.546880e+00]> : tensor<5xbf16>
    return %0 : tensor<5xbf16>
  }
}

