// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<10xf32>, tensor<3x1xi32>)
    %1 = call @expected() : () -> tensor<3x2xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 2>} : (tensor<10xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<10xf32>, tensor<3x1xi32>) {
    %0 = stablehlo.constant dense<[-0.317652494, -0.498273045, -1.63233531, -0.124743178, 2.18847871, 1.92351472, 1.37014866, -3.42049432, -2.30765843, 2.53218222]> : tensor<10xf32>
    %1 = stablehlo.constant dense<0> : tensor<3x1xi32>
    return %0, %1 : tensor<10xf32>, tensor<3x1xi32>
  }
  func.func private @expected() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<[[-0.317652494, -0.498273045], [-0.317652494, -0.498273045], [-0.317652494, -0.498273045]]> : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}

