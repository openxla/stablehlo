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
    %0 = stablehlo.constant dense<[[[2.703130e+00, 1.554690e+00, 5.281250e+00], [-2.921880e+00, -1.664060e+00, 1.781250e+00]], [[7.070310e-01, 6.757810e-01, -1.960940e+00], [-3.437500e+00, 2.281250e+00, -1.148440e+00]], [[4.562500e+00, -4.781250e+00, 1.710940e+00], [2.546880e+00, -7.312500e+00, -2.265630e+00]], [[1.851560e+00, 1.914060e+00, 2.578130e+00], [-2.125000e+00, 2.203130e+00, -2.062500e+00]]]> : tensor<4x2x3xbf16>
    %1 = stablehlo.constant dense<[4.625000e+00, -7.851560e-01]> : tensor<2xbf16>
    return %0, %1 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> tensor<4x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[2.703130e+00, 1.554690e+00, 5.281250e+00], [-2.921880e+00, -1.664060e+00, 1.781250e+00]], [[7.070310e-01, 6.757810e-01, -1.960940e+00], [-3.437500e+00, 2.281250e+00, -1.148440e+00]], [[4.562500e+00, -4.781250e+00, 1.710940e+00], [2.546880e+00, -7.312500e+00, -2.265630e+00]], [[1.851560e+00, 1.914060e+00, 4.625000e+00], [-2.125000e+00, 2.203130e+00, -7.851560e-01]]]> : tensor<4x2x3xbf16>
    return %0 : tensor<4x2x3xbf16>
  }
}

