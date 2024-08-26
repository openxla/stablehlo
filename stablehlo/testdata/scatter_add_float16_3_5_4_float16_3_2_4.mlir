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
      %3 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<3x5x4xf16>, tensor<2x1xi64>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> ()
    return %2 : tensor<3x5x4xf16>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16> {mhlo.layout_mode = "default"}, tensor<3x2x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-6.650390e-01, 6.187500e+00, -1.487300e+00, 9.291990e-01], [-1.287110e+00, -7.578130e-01, 4.222660e+00, 1.351560e+00], [-1.206050e+00, -1.457030e+00, -8.882810e+00, 8.093750e+00], [-1.044920e+00, -7.290030e-01, 5.649410e-01, 2.294920e+00], [1.787110e+00, 3.574220e+00, -5.128910e+00, -1.977540e+00]], [[-1.047850e+00, -1.062500e+00, 4.597660e+00, -1.060550e+00], [5.074220e+00, 2.143860e-03, 1.145510e+00, -1.787110e+00], [7.065420e-01, -1.348630e+00, -1.007810e+00, 1.192380e+00], [-3.842160e-02, -6.949210e+00, -3.056640e+00, -1.616210e-01], [2.755860e+00, -1.373290e-01, 4.013670e-01, -1.238280e+00]], [[2.437500e+00, -7.978520e-01, 2.845700e+00, 9.003900e-01], [3.097530e-02, -1.885740e+00, -3.080080e+00, 1.555660e+00], [-4.636720e+00, -2.197270e+00, -2.281250e+00, 6.054690e+00], [5.519530e+00, 5.824210e+00, -6.605460e+00, -4.226560e+00], [1.500000e+00, 3.175780e+00, -3.695310e+00, 1.579100e+00]]]> : tensor<3x5x4xf16>
    %cst_0 = stablehlo.constant dense<[[[-2.964840e+00, 2.830080e+00, 4.758300e-01, 1.995120e+00], [-2.656250e-01, -7.378900e+00, 2.363280e+00, 1.236330e+00]], [[3.980470e+00, -1.183590e+00, 2.007810e+00, 3.072270e+00], [3.203130e+00, 4.781250e+00, -1.264650e+00, 3.152340e+00]], [[2.927250e-01, -2.117190e+00, 3.244140e+00, -1.189450e+00], [-2.822270e+00, 4.113280e+00, -1.642580e+00, -5.039060e+00]]]> : tensor<3x2x4xf16>
    return %cst, %cst_0 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> (tensor<3x5x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-6.650390e-01, 6.187500e+00, -1.487300e+00, 9.291990e-01], [-4.515630e+00, -5.304690e+00, 7.062500e+00, 4.585940e+00], [-1.206050e+00, -1.457030e+00, -8.882810e+00, 8.093750e+00], [-1.044920e+00, -7.290030e-01, 5.649410e-01, 2.294920e+00], [1.787110e+00, 3.574220e+00, -5.128910e+00, -1.977540e+00]], [[-1.047850e+00, -1.062500e+00, 4.597660e+00, -1.060550e+00], [1.225780e+01, 3.599610e+00, 1.887700e+00, 4.437500e+00], [7.065420e-01, -1.348630e+00, -1.007810e+00, 1.192380e+00], [-3.842160e-02, -6.949210e+00, -3.056640e+00, -1.616210e-01], [2.755860e+00, -1.373290e-01, 4.013670e-01, -1.238280e+00]], [[2.437500e+00, -7.978520e-01, 2.845700e+00, 9.003900e-01], [-2.498050e+00, 1.093750e-01, -1.478520e+00, -4.671880e+00], [-4.636720e+00, -2.197270e+00, -2.281250e+00, 6.054690e+00], [5.519530e+00, 5.824210e+00, -6.605460e+00, -4.226560e+00], [1.500000e+00, 3.175780e+00, -3.695310e+00, 1.579100e+00]]]> : tensor<3x5x4xf16>
    return %cst : tensor<3x5x4xf16>
  }
}
