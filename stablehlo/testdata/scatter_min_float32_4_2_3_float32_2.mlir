// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %1 = call @expected() : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    return %2 : tensor<4x2x3xf32>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}, tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.93096745, -2.901610e+00, -0.136921942], [-1.80305481, 0.274286896, -2.30252504]], [[-4.93505478, 1.80857766, 1.2502718], [7.76369047, -5.67987299, -2.081790e+00]], [[-2.27660036, 1.0958333, 0.296646416], [3.69325113, -4.21838379, 2.43013287]], [[-3.66318417, 0.186973795, -1.25945151], [-1.06467712, 3.4913063, 1.16593659]]]> : tensor<4x2x3xf32>
    %cst_0 = stablehlo.constant dense<[-2.92589569, 2.0510335]> : tensor<2xf32>
    return %cst, %cst_0 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.93096745, -2.901610e+00, -0.136921942], [-1.80305481, 0.274286896, -2.30252504]], [[-4.93505478, 1.80857766, 1.2502718], [7.76369047, -5.67987299, -2.081790e+00]], [[-2.27660036, 1.0958333, 0.296646416], [3.69325113, -4.21838379, 2.43013287]], [[-3.66318417, 0.186973795, -2.92589569], [-1.06467712, 3.4913063, 1.16593659]]]> : tensor<4x2x3xf32>
    return %cst : tensor<4x2x3xf32>
  }
}
