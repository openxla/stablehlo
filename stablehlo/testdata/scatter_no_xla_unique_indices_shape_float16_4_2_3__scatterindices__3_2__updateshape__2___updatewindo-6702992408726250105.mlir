// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %2 = call @expected() : () -> tensor<4x2x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      stablehlo.return %arg1 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[1.075200e+00, -7.968750e+00, -1.567380e+00], [-3.710940e+00, -2.392580e+00, 1.809570e+00]], [[4.187010e-01, -8.031250e+00, -5.355470e+00], [6.445310e-01, 6.101560e+00, 5.722650e+00]], [[3.785160e+00, -4.230470e+00, 8.583980e-01], [-3.406250e+00, -1.212890e+00, 6.455080e-01]], [[-9.824210e-01, -2.968750e+00, 1.233400e+00], [-4.711910e-01, -1.875980e+00, -4.640630e+00]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[6.176760e-01, -1.138670e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[1.075200e+00, -7.968750e+00, -1.567380e+00], [-3.710940e+00, -2.392580e+00, 1.809570e+00]], [[4.187010e-01, -8.031250e+00, -5.355470e+00], [6.445310e-01, 6.101560e+00, 5.722650e+00]], [[3.785160e+00, -4.230470e+00, 8.583980e-01], [-3.406250e+00, -1.212890e+00, 6.455080e-01]], [[-9.824210e-01, -2.968750e+00, 6.176760e-01], [-4.711910e-01, -1.875980e+00, -1.138670e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

