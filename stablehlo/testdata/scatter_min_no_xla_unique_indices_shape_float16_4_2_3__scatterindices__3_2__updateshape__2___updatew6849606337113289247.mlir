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
    %0 = stablehlo.constant dense<[[[6.230460e-01, -5.019530e+00, -1.706050e+00], [-1.333010e+00, 2.773440e+00, -2.900390e+00]], [[-3.095700e-01, 6.977530e-01, 3.316410e+00], [-1.112300e+00, 2.322270e+00, -5.141600e-01]], [[9.071350e-03, -5.746090e+00, -1.179690e+00], [2.566410e+00, 2.867190e+00, -1.001950e+00]], [[-1.091800e+00, 1.614260e+00, -9.228510e-01], [-1.058590e+00, -5.328130e+00, 3.005860e+00]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[-3.734380e+00, 2.867190e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[6.230460e-01, -5.019530e+00, -1.706050e+00], [-1.333010e+00, 2.773440e+00, -2.900390e+00]], [[-3.095700e-01, 6.977530e-01, 3.316410e+00], [-1.112300e+00, 2.322270e+00, -5.141600e-01]], [[9.071350e-03, -5.746090e+00, -1.179690e+00], [2.566410e+00, 2.867190e+00, -1.001950e+00]], [[-1.091800e+00, 1.614260e+00, -3.734380e+00], [-1.058590e+00, -5.328130e+00, 2.867190e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

