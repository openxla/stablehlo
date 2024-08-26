// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<2x2xf32>
    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2:2 = "stablehlo.reduce_window"(%0#1, %0#0, %cst, %cst_0) <{window_dimensions = array<i64: 2, 2>, window_strides = array<i64: 2, 3>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %3 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.select %3, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %5 = stablehlo.select %3, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %4, %5 : tensor<f32>, tensor<f32>
    }) : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)
    stablehlo.custom_call @check.expect_close(%2#1, %1) {has_side_effect = true} : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    return %2#1 : tensor<2x2xf32>
  }
  func.func private @inputs() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}, tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.13653708, 3.850970e+00, 1.41188276, 5.03700876, 1.48468447, 0.925753116], [-0.248426422, -4.28072739, -4.20720768, -1.19028091, 0.54057169, -6.95183658], [-1.31342614, 3.07995915, 5.3993783, -1.28211379, 1.39402068, 2.28532982], [-2.4763267, -2.75141668, 0.0943910331, 4.40151119, 1.8391434, -4.97099495]]> : tensor<4x6xf32>
    %cst_0 = stablehlo.constant dense<[[-0.614279747, 0.590600848, 3.51997685, -2.04704905, -0.412673563, 1.09668195], [-0.0427753292, 2.34471607, 1.80304348, -5.56420183, 0.834002256, -3.3200314], [-0.92481637, -1.27236295, -2.63647914, -0.438251436, -1.3858273, -1.73599696], [3.89021254, -0.658202826, -5.43405294, -2.19236541, 6.30242633, -3.99410748]]> : tensor<4x6xf32>
    return %cst, %cst_0 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> (tensor<2x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.13653708, -1.19028091], [3.07995915, 4.40151119]]> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
  }
}
