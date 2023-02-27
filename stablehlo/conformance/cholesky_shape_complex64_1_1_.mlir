module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<1x1xcomplex<f32>>
    %1 = call @expected() : () -> tensor<1x1xcomplex<f32>>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.custom_call @lapack_cpotrf(%2, %3, %4, %0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1xcomplex<f32>>) -> tuple<tensor<1x1xcomplex<f32>>, tensor<i32>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<1x1xcomplex<f32>>, tensor<i32>>) -> tensor<1x1xcomplex<f32>>
    %7 = stablehlo.get_tuple_element %5[1] : (tuple<tensor<1x1xcomplex<f32>>, tensor<i32>>) -> tensor<i32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<i32>
    %10 = stablehlo.compare  EQ, %7, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %12 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<complex<f32>>) -> tensor<1x1xcomplex<f32>>
    %14 = stablehlo.select %11, %6, %13 : tensor<1x1xi1>, tensor<1x1xcomplex<f32>>
    %15 = stablehlo.custom_call @check.eq(%14, %1) : (tensor<1x1xcomplex<f32>>, tensor<1x1xcomplex<f32>>) -> tensor<i1>
    return %15 : tensor<i1>
  }
  func.func private @inputs() -> tensor<1x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<(35.5453262,0.000000e+00)> : tensor<1x1xcomplex<f32>>
    return %0 : tensor<1x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<1x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<(5.96199036,0.000000e+00)> : tensor<1x1xcomplex<f32>>
    return %0 : tensor<1x1xcomplex<f32>>
  }
}
