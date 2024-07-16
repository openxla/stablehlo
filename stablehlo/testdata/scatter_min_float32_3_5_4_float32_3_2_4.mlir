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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3x5x4xf32>, tensor<2x1xi64>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> ()
    return %2 : tensor<3x5x4xf32>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}, tensor<3x2x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.334703326, -1.00370944, 7.2926755, -0.974267601], [-2.76496267, 3.35438061, 2.4857378, 1.26531649], [-2.46908736, 1.25269723, 0.777086198, -0.336898446], [-0.545637071, 0.525594175, -2.59066415, 4.89420938], [-6.93882942, 2.40634537, -3.6558907, -0.376302212]], [[3.63124752, -5.428339, 0.352188379, -4.45552874], [-1.26497698, -1.07686889, 6.10807562, 1.39777231], [-1.45547986, 4.52103758, -0.751950085, 0.792478561], [3.71486878, 1.60447514, 1.4117856, -5.10069084], [-1.45618606, 2.21383023, 0.828865826, 1.50428724]], [[-4.02513266, -3.78741717, 2.10727525, -3.10417604], [-0.887514173, 0.870662451, -1.05202508, -7.0422616], [0.525637329, 8.83676719, 1.95870376, 4.2857461], [-2.12659931, -1.38829195, 2.52645493, -0.905206084], [-4.27453852, 5.08933496, -5.86508703, 3.34458423]]]> : tensor<3x5x4xf32>
    %cst_0 = stablehlo.constant dense<[[[-0.179849893, -6.49150753, -2.16031384, -1.11623347], [-1.18611979, -4.10119295, -1.14754248, -5.07246447]], [[-3.9286809, 5.23708057, -1.02071917, 1.22224128], [2.69600606, 1.43522382, 1.05542088, 0.57545495]], [[-4.27722692, 4.07409143, 2.34377432, -3.54027915], [0.437135667, 1.5164839, 1.57926643, 1.48136163]]]> : tensor<3x2x4xf32>
    return %cst, %cst_0 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.334703326, -1.00370944, 7.2926755, -0.974267601], [-2.76496267, -6.49150753, -2.16031384, -5.07246447], [-2.46908736, 1.25269723, 0.777086198, -0.336898446], [-0.545637071, 0.525594175, -2.59066415, 4.89420938], [-6.93882942, 2.40634537, -3.6558907, -0.376302212]], [[3.63124752, -5.428339, 0.352188379, -4.45552874], [-3.9286809, -1.07686889, -1.02071917, 0.57545495], [-1.45547986, 4.52103758, -0.751950085, 0.792478561], [3.71486878, 1.60447514, 1.4117856, -5.10069084], [-1.45618606, 2.21383023, 0.828865826, 1.50428724]], [[-4.02513266, -3.78741717, 2.10727525, -3.10417604], [-4.27722692, 0.870662451, -1.05202508, -7.0422616], [0.525637329, 8.83676719, 1.95870376, 4.2857461], [-2.12659931, -1.38829195, 2.52645493, -0.905206084], [-4.27453852, 5.08933496, -5.86508703, 3.34458423]]]> : tensor<3x5x4xf32>
    return %cst : tensor<3x5x4xf32>
  }
}
