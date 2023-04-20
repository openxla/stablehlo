// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<1x?x16xf32> {mhlo.sharding = ""}) -> tensor<1x?x1xf32> {
    %0 = stablehlo.constant dense<16> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.constant dense<1> : tensor<1xi32>
    %6 = stablehlo.concatenate %2, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.dynamic_reshape %arg1, %6 : (tensor<1x?x16xf32>, tensor<3xi32>) -> tensor<1x?x1xf32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<f32>
    %10 = "stablehlo.reduce_window"(%7, %9) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %11 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) {padding = dense<[[0, 0], [1, 1], [0, 0]]> : tensor<3x2xi64>, window_dimensions = dense<[1, 4, 1]> : tensor<3xi64>, window_strides = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<1x?x1xf32>, tensor<f32>) -> tensor<1x?x1xf32>
    return %10 : tensor<1x?x1xf32>
  }
}

