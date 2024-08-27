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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<3x5x4xf16>, tensor<2x1xi64>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> ()
    return %2 : tensor<3x5x4xf16>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16> {mhlo.layout_mode = "default"}, tensor<3x2x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[5.742190e+00, -4.301760e-01, -2.829590e-01, -2.575680e-01], [4.062500e+00, 2.166020e+00, -9.550780e-01, -1.287110e+00], [2.322270e+00, 4.153440e-02, -1.708980e+00, -6.804680e+00], [2.767580e+00, -1.573240e+00, -1.152340e+00, -3.825680e-01], [5.050780e+00, 3.703130e+00, -1.816410e+00, 2.763670e-01]], [[2.595700e+00, -3.869630e-01, -4.824220e+00, -1.029300e+00], [-2.945310e+00, 5.151370e-01, 4.042970e+00, 3.167970e+00], [2.341800e+00, 1.926760e+00, -1.823240e+00, 1.345700e+00], [-4.800780e+00, -9.086910e-01, 5.798330e-03, 3.886720e+00], [2.666020e+00, -3.982420e+00, -1.229490e+00, 1.343750e+00]], [[7.543940e-01, 2.605470e+00, -1.845700e+00, 4.781250e+00], [1.710940e+00, -5.488280e+00, -2.404300e+00, -1.883790e+00], [1.851560e+00, 1.458980e+00, -1.187500e+00, -2.048340e-01], [2.005860e+00, -9.672850e-01, -3.324220e+00, 9.472650e-01], [1.335940e+00, 2.773440e-01, 3.771480e+00, 5.898440e-01]]]> : tensor<3x5x4xf16>
    %cst_0 = stablehlo.constant dense<[[[5.473630e-01, 3.199220e+00, 1.132810e+00, -2.246090e+00], [3.794920e+00, -2.173830e+00, -1.941410e+00, 2.656250e+00]], [[-1.496090e+00, 2.617190e+00, 1.331050e+00, 2.292970e+00], [-3.482420e+00, -1.144530e+00, -2.740230e+00, -1.338870e+00]], [[1.649410e+00, -1.147460e+00, 4.345700e-01, 1.711910e+00], [9.203120e+00, -4.550780e+00, -5.824210e+00, -6.335940e+00]]]> : tensor<3x2x4xf16>
    return %cst, %cst_0 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> (tensor<3x5x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[5.742190e+00, -4.301760e-01, -2.829590e-01, -2.575680e-01], [5.473630e-01, -2.173830e+00, -1.941410e+00, -2.246090e+00], [2.322270e+00, 4.153440e-02, -1.708980e+00, -6.804680e+00], [2.767580e+00, -1.573240e+00, -1.152340e+00, -3.825680e-01], [5.050780e+00, 3.703130e+00, -1.816410e+00, 2.763670e-01]], [[2.595700e+00, -3.869630e-01, -4.824220e+00, -1.029300e+00], [-3.482420e+00, -1.144530e+00, -2.740230e+00, -1.338870e+00], [2.341800e+00, 1.926760e+00, -1.823240e+00, 1.345700e+00], [-4.800780e+00, -9.086910e-01, 5.798330e-03, 3.886720e+00], [2.666020e+00, -3.982420e+00, -1.229490e+00, 1.343750e+00]], [[7.543940e-01, 2.605470e+00, -1.845700e+00, 4.781250e+00], [1.649410e+00, -5.488280e+00, -5.824210e+00, -6.335940e+00], [1.851560e+00, 1.458980e+00, -1.187500e+00, -2.048340e-01], [2.005860e+00, -9.672850e-01, -3.324220e+00, 9.472650e-01], [1.335940e+00, 2.773440e-01, 3.771480e+00, 5.898440e-01]]]> : tensor<3x5x4xf16>
    return %cst : tensor<3x5x4xf16>
  }
}
