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
    %0 = stablehlo.constant dense<[[1.101560e+00, 1.108400e-01, 1.164060e+00], [1.218750e+00, 1.320310e+00, -2.402340e-01]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<[[[3.515630e+00, -3.250000e+00, -3.000000e+00]], [[-2.406250e+00, 3.343750e+00, -1.867190e+00]]]> : tensor<2x1x3xbf16>
    return %0, %1 : tensor<2x3xbf16>, tensor<2x1x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[1.101560e+00, 1.108400e-01, -3.250000e+00], [1.218750e+00, 1.320310e+00, -2.406250e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}

