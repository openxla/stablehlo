// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %1 = call @expected() : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    return %2 : tensor<4x2x3xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}, tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.472660e+00, -2.609380e+00, -4.694820e-01], [-3.058590e+00, -6.464840e-01, -2.281250e+00]], [[-1.517580e+00, 1.289060e+00, 2.335940e+00], [-5.253910e+00, 3.238280e+00, -7.416990e-01]], [[-3.402340e+00, -5.883790e-01, -2.954100e-01], [2.277340e+00, 1.465820e+00, 4.523440e+00]], [[-2.695310e-01, 2.501950e+00, 2.963870e-01], [-4.585940e+00, 2.514650e-01, -1.791020e+00]]]> : tensor<4x2x3xf16>
    %cst_0 = stablehlo.constant dense<[-2.777340e+00, -1.203130e+00]> : tensor<2xf16>
    return %cst, %cst_0 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.472660e+00, -2.609380e+00, -4.694820e-01], [-3.058590e+00, -6.464840e-01, -2.281250e+00]], [[-1.517580e+00, 1.289060e+00, 2.335940e+00], [-5.253910e+00, 3.238280e+00, -7.416990e-01]], [[-3.402340e+00, -5.883790e-01, -2.954100e-01], [2.277340e+00, 1.465820e+00, 4.523440e+00]], [[-2.695310e-01, 2.501950e+00, -8.232420e-01], [-4.585940e+00, 2.514650e-01, 2.154300e+00]]]> : tensor<4x2x3xf16>
    return %cst : tensor<4x2x3xf16>
  }
}
