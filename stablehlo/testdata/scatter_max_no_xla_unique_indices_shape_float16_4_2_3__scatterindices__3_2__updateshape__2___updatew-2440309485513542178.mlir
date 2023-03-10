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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[-1.689450e+00, 9.550780e-01, 8.489990e-02], [-1.218750e+00, 1.265630e+00, -7.036130e-01]], [[-2.753910e-01, 2.123050e+00, -4.069820e-01], [4.072270e-01, -7.441400e-01, 2.595210e-01]], [[1.573240e+00, 1.642580e+00, 1.099610e+00], [2.337890e+00, 1.299800e+00, 3.863280e+00]], [[1.646480e+00, -1.030270e+00, 2.882810e+00], [-2.919920e+00, -2.435550e+00, 3.197270e+00]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[5.957030e+00, -8.208000e-01]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-1.689450e+00, 9.550780e-01, 8.489990e-02], [-1.218750e+00, 1.265630e+00, -7.036130e-01]], [[-2.753910e-01, 2.123050e+00, -4.069820e-01], [4.072270e-01, -7.441400e-01, 2.595210e-01]], [[1.573240e+00, 1.642580e+00, 1.099610e+00], [2.337890e+00, 1.299800e+00, 3.863280e+00]], [[1.646480e+00, -1.030270e+00, 5.957030e+00], [-2.919920e+00, -2.435550e+00, 3.197270e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

