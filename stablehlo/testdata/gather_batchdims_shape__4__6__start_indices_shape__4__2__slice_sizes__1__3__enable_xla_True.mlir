// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x2xi32>)
    %1 = call @expected() : () -> tensor<4x3xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x2xi32>) -> tensor<4x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x2xi32>) {
    %0 = stablehlo.constant dense<[[-2.82557797, 2.39072633, 1.59782159, 5.14471102, -0.118122488, 1.23312056], [-1.81219053, -2.04905701, 2.10215306, -1.29667866, -0.0825303718, 1.88295043], [2.51706767, 0.0771943628, 2.18911791, -0.366536409, -2.39656186, 0.698230087], [2.96748114, 0.137859881, 1.44472873, -1.30095637, 1.24915195, -2.93037224]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 2]]> : tensor<4x2xi32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x2xi32>
  }
  func.func private @expected() -> tensor<4x3xf32> {
    %0 = stablehlo.constant dense<[[2.39072633, 1.59782159, 5.14471102], [2.10215306, -1.29667866, -0.0825303718], [-0.366536409, -2.39656186, 0.698230087], [1.44472873, -1.30095637, 1.24915195]]> : tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

