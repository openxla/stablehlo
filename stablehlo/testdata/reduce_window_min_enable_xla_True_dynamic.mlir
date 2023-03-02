// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8xf32> {mhlo.sharding = ""}) -> tensor<?x7xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "stablehlo.reduce_window"(%arg1, %0) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.minimum %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<f32>) -> tensor<?x7xf32>
    return %1 : tensor<?x7xf32>
  }
}

