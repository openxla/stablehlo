// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3xbf16>, tensor<3xbf16>)
    %2 = call @expected() : () -> tensor<3xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<3xbf16>, tensor<3x1xi32>, tensor<3xbf16>) -> tensor<3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3xbf16>, tensor<3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xbf16>, tensor<3xbf16>) {
    %0 = stablehlo.constant dense<[3.515630e+00, -5.343750e+00, 8.828120e-01]> : tensor<3xbf16>
    %1 = stablehlo.constant dense<[1.562500e-02, -1.414060e+00, 2.937500e+00]> : tensor<3xbf16>
    return %0, %1 : tensor<3xbf16>, tensor<3xbf16>
  }
  func.func private @expected() -> tensor<3xbf16> {
    %0 = stablehlo.constant dense<[2.093750e+00, -2.375000e+00, 8.828120e-01]> : tensor<3xbf16>
    return %0 : tensor<3xbf16>
  }
}

