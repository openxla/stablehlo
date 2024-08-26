// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2x4x6xf32>, tensor<2x1x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    return %4 : tensor<2x4x6xf32>
  }
  func.func private @inputs() -> (tensor<2x1x6xf32> {mhlo.layout_mode = "default"}, tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[1.19308376, 0.574337304, 0.771823287, -0.544335902, -3.51993728, 2.39790201]], [[-1.84060347, 2.64280605, 1.53013074, -2.40419912, 3.80637503, 1.55404234]]]> : tensor<2x1x6xf32>
    %cst_0 = stablehlo.constant dense<[[[3.8620379, 2.20822835, 3.35131288, -1.51076293, 1.85093927, -3.03109074], [1.26185918, 6.36994696, 1.05354285, -2.17369485, 3.45685482, -6.95882701], [6.67202854, 0.383125722, 0.0620134398, 1.42351806, -2.77465606, 2.41579247], [2.30130601, 3.26142049, 0.825048804, 1.56609392, -2.18073368, 2.59537792]], [[1.84301186, -6.02817058, 1.54950094, 4.0299511, -0.491929442, -8.70993995], [3.2347393, 3.72320461, -2.91740704, 1.03011239, 5.61139584, -3.50814581], [3.689810e-01, -0.252711564, 0.726274729, 0.11600668, -0.334616214, -4.00004578], [-2.13016582, -0.317696303, 1.65582919, 6.62267398, 1.85173309, -0.621818244]]]> : tensor<2x4x6xf32>
    return %cst, %cst_0 : tensor<2x1x6xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> (tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.771823287, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.574337304, 0.000000e+00, 0.000000e+00, -3.51993728, 0.000000e+00], [1.19308376, 0.000000e+00, 0.000000e+00, -0.544335902, 0.000000e+00, 2.39790201], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 1.53013074, -2.40419912, 0.000000e+00, 0.000000e+00], [-1.84060347, 2.64280605, 0.000000e+00, 0.000000e+00, 3.80637503, 1.55404234], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %cst : tensor<2x4x6xf32>
  }
}
