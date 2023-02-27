module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<4x1xf32>)
    %1 = call @expected() : () -> tensor<4x1xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_strsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<4x1xf32>) {
    %0 = stablehlo.constant dense<[[-2.21177268, 3.50446677, 2.98239207, 3.1239171], [-1.17310357, -2.35611105, 5.30595875, -0.549132824], [-1.77899659, -0.786007821, -4.29225397, -0.98729664], [-2.68095684, 1.48484135, -2.49199128, -4.529320e+00]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[[-0.0613860302], [0.519516766], [-3.78435087], [0.663207471]]> : tensor<4x1xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4x1xf32>
  }
  func.func private @expected() -> tensor<4x1xf32> {
    %0 = stablehlo.constant dense<[[0.0277542211], [-0.234316349], [0.913075209], [-0.742034912]]> : tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
