// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>)
    %1 = call @expected() : () -> tensor<3x5x4xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<3x5x4xf16>, tensor<2x1xi64>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> ()
    return %2 : tensor<3x5x4xf16>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16> {mhlo.layout_mode = "default"}, tensor<3x2x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[1.814450e+00, -4.394530e+00, 3.552730e+00, -5.230470e+00], [3.003910e+00, -1.508790e+00, 4.528810e-01, -1.768550e+00], [-2.255860e+00, -1.878910e+00, 5.519870e-03, 2.285160e+00], [-4.859380e+00, 3.271480e+00, -1.022460e+00, -7.109380e+00], [1.234130e-01, -8.857720e-03, 3.556640e+00, 2.943360e+00]], [[-2.412110e+00, -1.712890e+00, -3.357420e+00, -1.675780e+00], [2.113280e+00, -3.833010e-01, 6.245120e-01, -3.158200e+00], [-7.509770e-01, -2.113280e+00, 4.160160e+00, -7.329100e-01], [3.343750e+00, 2.753910e+00, 1.393550e+00, 2.923830e+00], [-2.919920e+00, 1.632810e+00, 4.025880e-01, -2.230470e+00]], [[1.650390e+00, -3.080080e+00, 3.156740e-01, 7.895500e-01], [-2.546880e+00, -4.660640e-01, -6.850590e-01, -3.339840e-01], [4.681400e-02, 6.499020e-01, 1.334960e+00, -5.921880e+00], [4.289060e+00, 1.113280e+00, 2.347660e+00, -8.627920e-01], [-3.923830e+00, -1.303710e+00, 1.770510e+00, 5.864260e-01]]]> : tensor<3x5x4xf16>
    %cst_0 = stablehlo.constant dense<[[[-1.854490e+00, -4.174800e-01, -4.191410e+00, -2.595700e+00], [1.275390e+00, -2.315670e-01, -4.593750e+00, 3.601070e-01]], [[-2.817380e-01, -1.020510e+00, 5.324220e+00, -3.419920e+00], [-6.226560e+00, 5.573270e-03, 5.230470e+00, -6.953130e+00]], [[3.427120e-02, 3.291020e+00, 3.251950e+00, 7.187500e-01], [7.187500e+00, -4.897460e-01, -1.257810e+00, 3.455080e+00]]]> : tensor<3x2x4xf16>
    return %cst, %cst_0 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> (tensor<3x5x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[1.814450e+00, -4.394530e+00, 3.552730e+00, -5.230470e+00], [3.003910e+00, -2.315670e-01, 4.528810e-01, 3.601070e-01], [-2.255860e+00, -1.878910e+00, 5.519870e-03, 2.285160e+00], [-4.859380e+00, 3.271480e+00, -1.022460e+00, -7.109380e+00], [1.234130e-01, -8.857720e-03, 3.556640e+00, 2.943360e+00]], [[-2.412110e+00, -1.712890e+00, -3.357420e+00, -1.675780e+00], [2.113280e+00, 5.573270e-03, 5.324220e+00, -3.158200e+00], [-7.509770e-01, -2.113280e+00, 4.160160e+00, -7.329100e-01], [3.343750e+00, 2.753910e+00, 1.393550e+00, 2.923830e+00], [-2.919920e+00, 1.632810e+00, 4.025880e-01, -2.230470e+00]], [[1.650390e+00, -3.080080e+00, 3.156740e-01, 7.895500e-01], [7.187500e+00, 3.291020e+00, 3.251950e+00, 3.455080e+00], [4.681400e-02, 6.499020e-01, 1.334960e+00, -5.921880e+00], [4.289060e+00, 1.113280e+00, 2.347660e+00, -8.627920e-01], [-3.923830e+00, -1.303710e+00, 1.770510e+00, 5.864260e-01]]]> : tensor<3x5x4xf16>
    return %cst : tensor<3x5x4xf16>
  }
}
