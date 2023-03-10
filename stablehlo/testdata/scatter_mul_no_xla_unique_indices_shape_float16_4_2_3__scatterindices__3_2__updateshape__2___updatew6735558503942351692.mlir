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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[-2.636720e+00, 1.851560e+00, 4.506840e-01], [1.213870e+00, -1.149410e+00, 2.992190e+00]], [[2.323910e-02, -2.742190e+00, -1.965820e+00], [-6.589840e+00, 1.131840e+00, -4.687500e+00]], [[4.121090e+00, 6.031250e+00, 8.437500e-01], [-3.566410e+00, 2.734380e+00, -8.234380e+00]], [[-5.169680e-02, 5.671880e+00, -3.560550e+00], [-7.524410e-01, 1.118160e+00, -5.317380e-01]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[8.750000e-01, 4.000000e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-2.636720e+00, 1.851560e+00, 4.506840e-01], [1.213870e+00, -1.149410e+00, 2.992190e+00]], [[2.323910e-02, -2.742190e+00, -1.965820e+00], [-6.589840e+00, 1.131840e+00, -4.687500e+00]], [[4.121090e+00, 6.031250e+00, 8.437500e-01], [-3.566410e+00, 2.734380e+00, -8.234380e+00]], [[-5.169680e-02, 5.671880e+00, -3.115230e+00], [-7.524410e-01, 1.118160e+00, -2.126950e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

