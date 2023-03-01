// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3xf32> {mhlo.sharding = ""}) -> tensor<?x3xf32> {
    %0 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<3> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %0, %4, dims = [] {known_expanding_dimensions = dense<> : tensor<0xi64>, known_nonexpanding_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xi32>) -> tensor<?x3xf32>
    %6 = stablehlo.compare  GT, %arg1, %5,  FLOAT : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
    %7 = stablehlo.select %6, %arg1, %arg1 : tensor<?x3xi1>, tensor<?x3xf32>
    return %7 : tensor<?x3xf32>
  }
}

