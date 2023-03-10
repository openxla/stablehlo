// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %2 = call @expected() : () -> tensor<3x5x4xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xbf16>, tensor<2x1xi32>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) {
    %0 = stablehlo.constant dense<[[[4.062500e-01, -4.437500e+00, 1.386720e-01, 2.695310e-01], [1.162110e-01, 4.062500e-01, 1.062500e+00, 4.687500e+00], [-2.250000e+00, -2.265630e+00, -4.343750e+00, -8.710930e-01], [5.375000e+00, 5.039060e-01, -3.718750e+00, -8.789060e-01], [-4.156250e+00, 3.765630e+00, 6.679690e-01, -2.343750e-01]], [[1.218750e+00, -5.281250e+00, -6.906250e+00, 6.757810e-01], [4.921880e-01, -3.406250e+00, -1.777340e-01, 1.375000e+00], [-3.890630e+00, -2.468750e+00, -2.539060e-01, -5.312500e+00], [7.187500e+00, -1.671880e+00, -4.281250e+00, 2.187500e+00], [3.406250e+00, -4.277340e-01, 1.117190e+00, -2.906250e+00]], [[1.562500e+00, -3.343750e+00, -1.734380e+00, -4.218750e+00], [4.824220e-01, 5.062500e+00, -1.460940e+00, -5.781250e-01], [-1.226560e+00, 1.046880e+00, 1.382810e+00, -1.000000e+00], [8.007810e-01, 2.125000e+00, -9.218750e-01, -1.304690e+00], [-5.375000e+00, -1.312500e+00, -3.656250e+00, 3.066410e-01]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[1.572270e-01, -6.000000e+00, 2.781250e+00, 1.312500e+00], [-5.093750e+00, 9.140620e-01, -1.726560e+00, 4.906250e+00]], [[1.539060e+00, 7.968750e-01, 3.859380e+00, 9.882810e-01], [-4.656250e+00, 1.515630e+00, 5.546880e-01, -5.718750e+00]], [[5.750000e+00, -3.843750e+00, -7.500000e+00, 3.750000e+00], [1.890630e+00, 5.937500e-01, -1.273440e+00, -4.500000e+00]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[4.062500e-01, -4.437500e+00, 1.386720e-01, 2.695310e-01], [-5.093750e+00, -6.000000e+00, -1.726560e+00, 1.312500e+00], [-2.250000e+00, -2.265630e+00, -4.343750e+00, -8.710930e-01], [5.375000e+00, 5.039060e-01, -3.718750e+00, -8.789060e-01], [-4.156250e+00, 3.765630e+00, 6.679690e-01, -2.343750e-01]], [[1.218750e+00, -5.281250e+00, -6.906250e+00, 6.757810e-01], [-4.656250e+00, -3.406250e+00, -1.777340e-01, -5.718750e+00], [-3.890630e+00, -2.468750e+00, -2.539060e-01, -5.312500e+00], [7.187500e+00, -1.671880e+00, -4.281250e+00, 2.187500e+00], [3.406250e+00, -4.277340e-01, 1.117190e+00, -2.906250e+00]], [[1.562500e+00, -3.343750e+00, -1.734380e+00, -4.218750e+00], [4.824220e-01, -3.843750e+00, -7.500000e+00, -4.500000e+00], [-1.226560e+00, 1.046880e+00, 1.382810e+00, -1.000000e+00], [8.007810e-01, 2.125000e+00, -9.218750e-01, -1.304690e+00], [-5.375000e+00, -1.312500e+00, -3.656250e+00, 3.066410e-01]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

