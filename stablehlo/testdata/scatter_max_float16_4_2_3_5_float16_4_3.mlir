// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x2x3x5xf16>, tensor<2xi64>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> ()
    return %2 : tensor<4x2x3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16> {mhlo.layout_mode = "default"}, tensor<4x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x4DC1813C2BC05DBF7D4342C0743603BB6ABF03B961C300C40DBDC3A632419DC2C3C3C7BFC3B715BD80C10EB9EAB5DDC22AC54B406C4076BC0EB9DE3DB3BD05BD8D4067C6BF3876BDD83918BF5EC502398EBD3240C63FD5B85BC1FE4192BE9EC163B49340FFB4C4BD2341D9B2E93E7AC1A73F2B4476BF7FC280B82BC14B3D23B8C4BC12C3023C2F29C3C0A241B83C90C6883B193FF73E714533B43EC0D0C0A53F772E52477340C1C459C10943CCB2514015B64CBBCBC180C590BF343BD044A1BCDE3730C09B37CB389E3E9039B339DCBD943DF53F2AC6D9B7FC41AD4218C8EAC590C17FC4973F244415C645429C3DDA38"> : tensor<4x2x3x5xf16>
    %cst_0 = stablehlo.constant dense<[[-1.026610e-01, 4.308590e+00, 1.567380e+00], [1.117190e+00, 4.054690e+00, 3.425780e+00], [-4.503910e+00, 1.511720e+00, 3.412110e+00], [1.402340e+00, -2.863280e+00, -1.077150e+00]]> : tensor<4x3xf16>
    return %cst, %cst_0 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x4DC1813C2BC05DBF7D4342C0743603BB6ABF4F4461C300C40DBDC3A632419DC2C3C3C7BFC3B715BD80C10EB9EAB5DDC22AC54B406C4076BC0EB9DE3DB3BD05BD8D4067C6783C76BDD83918BF5EC50E448EBD3240C63FD5B8DA42FE4192BE9EC163B49340FFB4C4BD2341D9B2E93E7AC1A73F2B4476BF7FC280B82BC14B3D23B8C4BC12C3023C2F29C3C0A241B83C90C6883B193FD342714533B43EC0D0C0A53F772E52477340C1C459C10943CCB2514015B64CBBCBC180C590BF343BD044A1BCDE3730C09B37CB389E3E9039B339DCBD943DF53F2AC6D9B7FC41AD4218C8EAC590C17FC4973F244415C645429C3DDA38"> : tensor<4x2x3x5xf16>
    return %cst : tensor<4x2x3x5xf16>
  }
}
