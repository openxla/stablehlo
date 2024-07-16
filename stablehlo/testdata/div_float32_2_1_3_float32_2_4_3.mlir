// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.divide %2, %0#1 : tensor<2x4x3xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> ()
    return %3 : tensor<2x4x3xf32>
  }
  func.func private @inputs() -> (tensor<2x1x3xf32> {mhlo.layout_mode = "default"}, tensor<2x4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.31914806, 1.75971186, 1.72489071]], [[-1.93688345, 1.02332175, -0.309879154]]]> : tensor<2x1x3xf32>
    %cst_0 = stablehlo.constant dense<[[[-1.14735425, 3.3193202, -1.03531241], [-2.8952837, 3.5428803, -3.58815479], [-3.1280818, -4.30367327, -2.70969319], [-2.74512243, -1.34495366, -2.91428113]], [[0.329057246, -0.310782731, 1.88425231], [-4.17743444, 3.25827026, -9.39599895], [0.432958812, -0.670874357, 0.816599965], [-0.894271969, -6.00379705, -1.6435746]]]> : tensor<2x4x3xf32>
    return %cst, %cst_0 : tensor<2x1x3xf32>, tensor<2x4x3xf32>
  }
  func.func private @expected() -> (tensor<2x4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.76444173, 0.530142248, -1.66605818], [-1.49178755, 0.496689618, -0.480718046], [-1.38076568, -0.408886015, -0.636563122], [-1.573390e+00, -1.30838108, -0.591875196]], [[-5.88615942, -3.29272389, -0.164457351], [0.463653833, 0.314069033, 0.0329799056], [-4.47359753, -1.52535534, -0.379474849], [2.16587734, -0.170445755, 0.188539758]]]> : tensor<2x4x3xf32>
    return %cst : tensor<2x4x3xf32>
  }
}
