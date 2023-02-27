module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x4xf32>
    %1 = call @expected() : () -> tensor<4x4xf32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<4> : tensor<i32>
    %5 = stablehlo.custom_call @lapack_spotrf(%2, %3, %4, %0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<i32>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<4x4xf32>, tensor<i32>>) -> tensor<4x4xf32>
    %7 = stablehlo.get_tuple_element %5[1] : (tuple<tensor<4x4xf32>, tensor<i32>>) -> tensor<i32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<i32>
    %10 = stablehlo.compare  EQ, %7, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %12 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1>
    %15 = stablehlo.select %14, %6, %13 : tensor<4x4xi1>, tensor<4x4xf32>
    %16 = stablehlo.custom_call @check.eq(%15, %1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<i1>
    return %16 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x4xf32> {
    %0 = stablehlo.constant dense<[[24.9063759, 36.3029785, 4.37652588, -16.0666733], [36.3029785, 81.0648345, -13.9889526, -37.3652344], [4.37652588, -13.9889526, 19.0756989, -4.50418091], [-16.0666733, -37.3652344, -4.50418091, 69.3703155]]> : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
  func.func private @expected() -> tensor<4x4xf32> {
    %0 = stablehlo.constant dense<[[4.99062872, 36.3029785, 4.37652588, -16.0666733], [7.27422953, 5.30569696, -13.9889526, -37.3652344], [0.876948833, -3.83890748, 1.88929868, -4.50418091], [-3.2193687, -2.62864757, -6.23093367, 3.6430285]]> : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
