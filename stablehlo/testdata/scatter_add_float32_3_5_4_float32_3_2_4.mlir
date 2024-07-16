// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>)
    %1 = call @expected() : () -> tensor<3x5x4xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3x5x4xf32>, tensor<2x1xi64>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> ()
    return %2 : tensor<3x5x4xf32>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}, tensor<3x2x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[5.54262066, 5.57175779, 3.81925797, -2.95259094], [-1.56661248, -0.948714852, 1.52475572, 3.814680e+00], [2.97826672, -2.81383252, -0.79894644, 1.24333346], [-0.525452614, -1.315804, 3.13520741, -3.366440e+00], [3.28200102, -0.866681277, 0.886234522, 2.35566807]], [[-2.19363308, 2.13371205, 0.862892448, -3.08019686], [0.299612582, -1.05456924, -6.026940e+00, -1.03918779], [-2.50264311, -2.4207828, 0.0245697517, 0.169797048], [2.12421846, -0.126787126, 0.0282802042, -4.93500757], [-3.34236455, -1.74407804, 6.21869373, -1.12497795]], [[-6.36245489, -0.889751911, -2.84010291, 1.94205558], [2.10301113, 0.129371941, 0.49645108, -1.45919967], [1.22990572, 2.63327575, 4.146610e+00, -0.249940678], [0.471881032, 1.70934308, 3.58652592, 0.559464157], [0.230565473, -6.87079954, -9.19039916, 7.12022161]]]> : tensor<3x5x4xf32>
    %cst_0 = stablehlo.constant dense<[[[3.41905904, 0.962474763, 0.343635887, -6.476120e-01], [0.636109292, -4.86534309, 0.243750781, 1.5103277]], [[-4.35427237, 4.65195227, 3.95961356, 1.22606468], [-0.296483964, -2.80553436, -1.91541898, -1.00489783]], [[2.56223631, 6.62017584, 2.77548909, -0.516119123], [2.16615558, -2.3243134, -0.698968887, 4.72885323]]]> : tensor<3x2x4xf32>
    return %cst, %cst_0 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[5.54262066, 5.57175779, 3.81925797, -2.95259094], [2.48855591, -4.851583, 2.11214232, 4.67739582], [2.97826672, -2.81383252, -0.79894644, 1.24333346], [-0.525452614, -1.315804, 3.13520741, -3.366440e+00], [3.28200102, -0.866681277, 0.886234522, 2.35566807]], [[-2.19363308, 2.13371205, 0.862892448, -3.08019686], [-4.35114384, 0.791848659, -3.98274517, -0.818020939], [-2.50264311, -2.4207828, 0.0245697517, 0.169797048], [2.12421846, -0.126787126, 0.0282802042, -4.93500757], [-3.34236455, -1.74407804, 6.21869373, -1.12497795]], [[-6.36245489, -0.889751911, -2.84010291, 1.94205558], [6.83140277, 4.42523479, 2.57297134, 2.75353432], [1.22990572, 2.63327575, 4.146610e+00, -0.249940678], [0.471881032, 1.70934308, 3.58652592, 0.559464157], [0.230565473, -6.87079954, -9.19039916, 7.12022161]]]> : tensor<3x5x4xf32>
    return %cst : tensor<3x5x4xf32>
  }
}
