// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %1 = call @expected() : () -> tensor<3x5x4xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<3x5x4xbf16>, tensor<2x1xi64>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> ()
    return %2 : tensor<3x5x4xbf16>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16> {mhlo.layout_mode = "default"}, tensor<3x2x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[2.281250e+00, -6.500000e+00, -4.296880e-02, -1.156250e+00], [3.500000e+00, 2.875000e+00, -2.203130e+00, -1.226560e+00], [-1.710940e+00, 1.281250e+00, 1.218750e+00, -4.312500e+00], [-9.414060e-01, 4.902340e-01, 3.109380e+00, 4.812500e+00], [2.734380e+00, -2.871090e-01, -6.093750e-01, -2.484380e+00]], [[-2.203130e+00, 2.353520e-01, 2.921880e+00, 3.390630e+00], [-3.609380e+00, 1.695310e+00, -8.671870e-01, 1.656250e+00], [2.578130e+00, 4.031250e+00, -1.867190e+00, 4.593750e+00], [-2.531250e+00, 5.351560e-01, -2.609380e+00, -1.062500e+00], [-2.328130e+00, 2.437500e+00, 9.218750e-01, -1.789060e+00]], [[7.148430e-01, -1.468750e+00, -2.671880e+00, 2.687500e+00], [1.546880e+00, -3.046880e+00, 5.820310e-01, -8.476560e-01], [1.187500e+00, 2.218750e+00, -7.187500e-01, -4.687500e-01], [-8.515620e-01, -3.875000e+00, -1.953130e+00, 4.375000e+00], [-2.875000e+00, 1.648440e+00, 3.093750e+00, -1.406250e+00]]]> : tensor<3x5x4xbf16>
    %cst_0 = stablehlo.constant dense<[[[3.218750e+00, 1.718750e+00, 1.031250e+00, -1.578130e+00], [-3.796880e+00, -1.851560e+00, -1.648440e+00, -2.500000e+00]], [[2.609380e+00, 4.257810e-01, 1.953130e+00, -3.609380e+00], [8.500000e+00, -4.468750e+00, 1.429690e+00, 3.015630e+00]], [[-2.625000e+00, 2.281250e+00, -2.015630e+00, -6.289060e-01], [-4.406250e+00, 3.218750e+00, 8.632810e-01, 6.062500e+00]]]> : tensor<3x2x4xbf16>
    return %cst, %cst_0 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> (tensor<3x5x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[2.281250e+00, -6.500000e+00, -4.296880e-02, -1.156250e+00], [2.921880e+00, 2.750000e+00, -2.812500e+00, -5.312500e+00], [-1.710940e+00, 1.281250e+00, 1.218750e+00, -4.312500e+00], [-9.414060e-01, 4.902340e-01, 3.109380e+00, 4.812500e+00], [2.734380e+00, -2.871090e-01, -6.093750e-01, -2.484380e+00]], [[-2.203130e+00, 2.353520e-01, 2.921880e+00, 3.390630e+00], [7.500000e+00, -2.343750e+00, 2.515630e+00, 1.062500e+00], [2.578130e+00, 4.031250e+00, -1.867190e+00, 4.593750e+00], [-2.531250e+00, 5.351560e-01, -2.609380e+00, -1.062500e+00], [-2.328130e+00, 2.437500e+00, 9.218750e-01, -1.789060e+00]], [[7.148430e-01, -1.468750e+00, -2.671880e+00, 2.687500e+00], [-5.500000e+00, 2.453130e+00, -5.742190e-01, 4.593750e+00], [1.187500e+00, 2.218750e+00, -7.187500e-01, -4.687500e-01], [-8.515620e-01, -3.875000e+00, -1.953130e+00, 4.375000e+00], [-2.875000e+00, 1.648440e+00, 3.093750e+00, -1.406250e+00]]]> : tensor<3x5x4xbf16>
    return %cst : tensor<3x5x4xbf16>
  }
}
