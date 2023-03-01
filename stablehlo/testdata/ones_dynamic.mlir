// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x4xf32> {mhlo.sharding = ""}) -> tensor<?x2x4xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.constant dense<4> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %0, %5, dims = [] {known_expanding_dimensions = dense<> : tensor<0xi64>, known_nonexpanding_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xi32>) -> tensor<?x2x4xf32>
    %7 = stablehlo.add %6, %arg1 : tensor<?x2x4xf32>
    return %7 : tensor<?x2x4xf32>
  }
}

