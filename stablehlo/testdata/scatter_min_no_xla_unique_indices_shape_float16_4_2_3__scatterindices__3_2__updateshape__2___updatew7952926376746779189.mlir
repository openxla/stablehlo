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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[-1.679690e-01, 5.195310e-01, 3.025390e+00], [6.207030e+00, -3.513670e+00, 4.906250e+00]], [[1.395510e+00, -1.781250e+00, 4.565430e-01], [1.784670e-01, 4.982910e-01, 2.779300e+00]], [[1.895510e+00, -1.304690e+00, 2.193360e+00], [-1.830080e+00, 3.792970e+00, -1.536130e+00]], [[-1.888670e+00, 3.424070e-02, 4.210940e+00], [1.915040e+00, 3.140630e+00, 1.005470e+01]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[1.023440e+00, 1.615230e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-1.679690e-01, 5.195310e-01, 3.025390e+00], [6.207030e+00, -3.513670e+00, 4.906250e+00]], [[1.395510e+00, -1.781250e+00, 4.565430e-01], [1.784670e-01, 4.982910e-01, 2.779300e+00]], [[1.895510e+00, -1.304690e+00, 2.193360e+00], [-1.830080e+00, 3.792970e+00, -1.536130e+00]], [[-1.888670e+00, 3.424070e-02, 1.023440e+00], [1.915040e+00, 3.140630e+00, 1.615230e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

