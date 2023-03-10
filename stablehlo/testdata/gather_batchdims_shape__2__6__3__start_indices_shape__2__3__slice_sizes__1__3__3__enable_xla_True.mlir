// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x6x3xf32>, tensor<2x3xi32>)
    %1 = call @expected() : () -> tensor<2x3x3xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = dense<[1, 3, 3]> : tensor<3xi64>} : (tensor<2x6x3xf32>, tensor<2x3xi32>) -> tensor<2x3x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3x3xf32>, tensor<2x3x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x6x3xf32>, tensor<2x3xi32>) {
    %0 = stablehlo.constant dense<[[[3.82548904, 0.791181862, -2.22872925], [1.47356987, 3.81562257, -4.14531422], [-2.14515972, 1.42112124, 4.571450e+00], [-1.68962431, 3.14189243, 5.90857506], [3.88763309, 3.85987115, 0.356856197], [0.954816877, 1.0329355, -0.830992698]], [[-2.23060846, 0.0469221957, -0.450263053], [-6.04691744, 6.02806186, -2.51375771], [0.53378284, 3.34858298, 1.84060633], [3.92621756, -1.48187923, 3.34925771], [-2.8641305, 0.439090401, 4.06969261], [-7.4090166, 3.41720462, -4.32454443]]]> : tensor<2x6x3xf32>
    %1 = stablehlo.constant dense<[[0, 1, 0], [1, 2, 0]]> : tensor<2x3xi32>
    return %0, %1 : tensor<2x6x3xf32>, tensor<2x3xi32>
  }
  func.func private @expected() -> tensor<2x3x3xf32> {
    %0 = stablehlo.constant dense<[[[1.47356987, 3.81562257, -4.14531422], [-2.14515972, 1.42112124, 4.571450e+00], [-1.68962431, 3.14189243, 5.90857506]], [[0.53378284, 3.34858298, 1.84060633], [3.92621756, -1.48187923, 3.34925771], [-2.8641305, 0.439090401, 4.06969261]]]> : tensor<2x3x3xf32>
    return %0 : tensor<2x3x3xf32>
  }
}

