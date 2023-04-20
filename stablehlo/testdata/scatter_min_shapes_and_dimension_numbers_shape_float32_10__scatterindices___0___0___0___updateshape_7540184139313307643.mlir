// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<10xf32>, tensor<3x2xf32>)
    %2 = call @expected() : () -> tensor<10xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<10xf32>, tensor<3x1xi32>, tensor<3x2xf32>) -> tensor<10xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<10xf32>, tensor<3x2xf32>) {
    %0 = stablehlo.constant dense<[4.88633156, 3.21240902, -1.79143333, -2.8212409, 0.31472373, -3.6573689, -2.43788576, 5.20532751, -1.75892818, 0.826414227]> : tensor<10xf32>
    %1 = stablehlo.constant dense<[[0.143054083, 0.009603668], [-1.54706168, -6.87445354], [-1.93186462, 1.26245236]]> : tensor<3x2xf32>
    return %0, %1 : tensor<10xf32>, tensor<3x2xf32>
  }
  func.func private @expected() -> tensor<10xf32> {
    %0 = stablehlo.constant dense<[-1.93186462, -6.87445354, -1.79143333, -2.8212409, 0.31472373, -3.6573689, -2.43788576, 5.20532751, -1.75892818, 0.826414227]> : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

