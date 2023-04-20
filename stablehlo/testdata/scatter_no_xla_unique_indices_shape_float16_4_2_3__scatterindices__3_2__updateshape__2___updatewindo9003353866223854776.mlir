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
    %0 = stablehlo.constant dense<[[[-3.330080e+00, -6.845700e-01, -4.191410e+00], [-2.863280e+00, -2.236330e+00, -3.150390e+00]], [[1.557620e+00, 1.542970e+00, 2.111330e+00], [6.027340e+00, 3.139650e-01, -5.024410e-01]], [[-6.757810e+00, 3.941410e+00, 1.936040e-01], [-3.923830e+00, -5.261720e+00, 2.644530e+00]], [[3.050780e+00, 2.229000e-01, -1.560550e+00], [-4.984380e+00, 2.437740e-01, -5.791020e-01]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[5.578130e+00, 2.455080e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-3.330080e+00, -6.845700e-01, -4.191410e+00], [-2.863280e+00, -2.236330e+00, -3.150390e+00]], [[1.557620e+00, 1.542970e+00, 2.111330e+00], [6.027340e+00, 3.139650e-01, -5.024410e-01]], [[-6.757810e+00, 3.941410e+00, 1.936040e-01], [-3.923830e+00, -5.261720e+00, 2.644530e+00]], [[3.050780e+00, 2.229000e-01, 5.578130e+00], [-4.984380e+00, 2.437740e-01, 2.455080e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

