module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x5x5xf32>
    %1 = call @expected() : () -> tensor<2x5x5xf32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.constant dense<5> : tensor<i32>
    %5 = stablehlo.custom_call @lapack_spotrf(%2, %3, %4, %0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x5x5xf32>) -> tuple<tensor<2x5x5xf32>, tensor<2xi32>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<2x5x5xf32>, tensor<2xi32>>) -> tensor<2x5x5xf32>
    %7 = stablehlo.get_tuple_element %5[1] : (tuple<tensor<2x5x5xf32>, tensor<2xi32>>) -> tensor<2xi32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %10 = stablehlo.compare  EQ, %7, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1>
    %12 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<2x5x5xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x5x5xi1>
    %15 = stablehlo.select %14, %6, %13 : tensor<2x5x5xi1>, tensor<2x5x5xf32>
    %16 = stablehlo.custom_call @check.eq(%15, %1) : (tensor<2x5x5xf32>, tensor<2x5x5xf32>) -> tensor<i1>
    return %16 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x5x5xf32> {
    %0 = stablehlo.constant dense<[[[21.4709015, 3.8059082, -24.6269531, 1.12866211, 4.78710938], [3.8059082, 31.4267731, -22.8348389, 9.32849121, -30.4909592], [-24.6269531, -22.8348389, 62.130188, -0.340942383, -7.6810913], [1.12866211, 9.32849121, -0.340942383, 49.4057617, -21.1902466], [4.78710938, -30.4909592, -7.6810913, -21.1902466, 53.8049965]], [[43.291214, 12.7822571, 7.94232177, 39.1208496, 16.9005737], [12.7822571, 13.7719116, -2.62670898, 15.5733032, -2.17785645], [7.94232177, -2.62670898, 29.2323151, 2.42077637, -6.40606689], [39.1208496, 15.5733032, 2.42077637, 40.4729614, 5.57177734], [16.9005737, -2.17785645, -6.40606689, 5.57177734, 64.6977539]]]> : tensor<2x5x5xf32>
    return %0 : tensor<2x5x5xf32>
  }
  func.func private @expected() -> tensor<2x5x5xf32> {
    %0 = stablehlo.constant dense<[[[4.63367033, 3.8059082, -24.6269531, 1.12866211, 4.78710938], [0.821359276, 5.54546118, -22.8348389, 9.32849121, -30.4909592], [-5.3147831, -3.33056021, 4.77395391, -0.340942383, -7.6810913], [0.243578419, 1.64610755, 1.34816647, 6.69471502, -21.1902466], [1.03311396, -5.65138149, -4.40150499, -0.926872432, 0.753140688]], [[6.57960605, 12.7822571, 7.94232177, 39.1208496, 16.9005737], [1.94270849, 3.16192913, -2.62670898, 15.5733032, -2.17785645], [1.20711207, -1.57238698, 5.03018856, 2.42077637, -6.40606689], [5.9457736, 1.27213418, -5.479220e-01, 1.78946817, 5.57177734], [2.568630e+00, -2.26695657, -2.59855556, -4.60508251, 5.00015259]]]> : tensor<2x5x5xf32>
    return %0 : tensor<2x5x5xf32>
  }
}
