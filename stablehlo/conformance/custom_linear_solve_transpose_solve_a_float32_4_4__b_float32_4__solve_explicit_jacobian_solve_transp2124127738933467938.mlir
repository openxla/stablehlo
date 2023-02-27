module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<4xf32>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %3 = stablehlo.iota dim = 0 : tensor<4x4xi32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<4x4xi32>
    %6 = stablehlo.add %3, %5 : tensor<4x4xi32>
    %7 = stablehlo.iota dim = 1 : tensor<4x4xi32>
    %8 = stablehlo.compare  EQ, %6, %7,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %9 = stablehlo.convert %8 : (tensor<4x4xi1>) -> tensor<4x4xf32>
    %10 = "stablehlo.slice"(%9) {limit_indices = dense<4> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %11 = "stablehlo.dot_general"(%10, %0#0) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %12 = "stablehlo.slice"(%11) {limit_indices = dense<4> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %13 = call @solve(%12, %0#1) : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %14 = stablehlo.custom_call @check.eq(%13, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %14 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<4xf32>) {
    %0 = stablehlo.constant dense<[[1.97273397, 0.916630148, 0.354775578, -1.3009268], [0.317744642, -1.40722716, -1.75071871, 1.31074286], [2.69010639, 1.81482482, 1.41035283, -1.61985612], [3.68680429, -1.91120124, -3.26228023, -1.71144366]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[2.41641307, -3.8662827, -1.33561039, -2.44155431]> : tensor<4xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4xf32>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[-6.32361364, 18.6027279, -15.8938589, -2.67359757]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @solve(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.constant dense<4> : tensor<i32>
    %2 = stablehlo.constant dense<4> : tensor<i32>
    %3 = stablehlo.custom_call @lapack_sgetrf(%0, %1, %2, %arg0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>
    %4 = stablehlo.get_tuple_element %3[0] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<4x4xf32>
    %5 = stablehlo.get_tuple_element %3[1] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<4xi32>
    %6 = stablehlo.get_tuple_element %3[2] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<i32>
    %7 = stablehlo.constant dense<1> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %9 = stablehlo.subtract %5, %8 : tensor<4xi32>
    %10 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<i32>
    %12 = stablehlo.compare  GE, %6, %11,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %14 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
    %16 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1>
    %17 = stablehlo.select %16, %4, %15 : tensor<4x4xi1>, tensor<4x4xf32>
    %18 = stablehlo.iota dim = 0 : tensor<4xi32>
    %19 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.constant dense<0> : tensor<i32>
    %21:4 = stablehlo.while(%iterArg = %20, %iterArg_0 = %19, %iterArg_1 = %18, %iterArg_2 = %9) : tensor<i32>, tensor<i32>, tensor<4xi32>, tensor<4xi32>
     cond {
      %23 = stablehlo.constant dense<4> : tensor<i32>
      %24 = stablehlo.compare  LT, %iterArg, %23,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %24 : tensor<i1>
    } do {
      %23 = stablehlo.constant dense<1> : tensor<i32>
      %24 = stablehlo.add %iterArg_0, %23 : tensor<i32>
      %25 = stablehlo.constant dense<0> : tensor<i32>
      %26 = stablehlo.compare  LT, %iterArg_0, %25,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %27 = stablehlo.constant dense<4> : tensor<i32>
      %28 = stablehlo.add %iterArg_0, %27 : tensor<i32>
      %29 = stablehlo.select %26, %28, %iterArg_0 : tensor<i1>, tensor<i32>
      %30 = stablehlo.convert %29 : tensor<i32>
      %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %32 = "stablehlo.gather"(%iterArg_2, %31) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<4xi32>, tensor<1xi32>) -> tensor<i32>
      %33 = stablehlo.constant dense<0> : tensor<i32>
      %34 = stablehlo.compare  LT, %iterArg_0, %33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %35 = stablehlo.constant dense<4> : tensor<i32>
      %36 = stablehlo.add %iterArg_0, %35 : tensor<i32>
      %37 = stablehlo.select %34, %36, %iterArg_0 : tensor<i1>, tensor<i32>
      %38 = stablehlo.convert %37 : tensor<i32>
      %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %40 = "stablehlo.gather"(%iterArg_1, %39) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<4xi32>, tensor<1xi32>) -> tensor<i32>
      %41 = stablehlo.constant dense<0> : tensor<i32>
      %42 = stablehlo.compare  LT, %32, %41,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %43 = stablehlo.constant dense<4> : tensor<i32>
      %44 = stablehlo.add %32, %43 : tensor<i32>
      %45 = stablehlo.select %42, %44, %32 : tensor<i1>, tensor<i32>
      %46 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %47 = "stablehlo.gather"(%iterArg_1, %46) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<4xi32>, tensor<1xi32>) -> tensor<i32>
      %48 = stablehlo.constant dense<0> : tensor<i32>
      %49 = stablehlo.compare  LT, %iterArg_0, %48,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %50 = stablehlo.constant dense<4> : tensor<i32>
      %51 = stablehlo.add %iterArg_0, %50 : tensor<i32>
      %52 = stablehlo.select %49, %51, %iterArg_0 : tensor<i1>, tensor<i32>
      %53 = stablehlo.convert %52 : tensor<i32>
      %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %55 = "stablehlo.scatter"(%iterArg_1, %54, %47) ({
      ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
        stablehlo.return %arg3 : tensor<i32>
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
      %56 = stablehlo.constant dense<0> : tensor<i32>
      %57 = stablehlo.compare  LT, %32, %56,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %58 = stablehlo.constant dense<4> : tensor<i32>
      %59 = stablehlo.add %32, %58 : tensor<i32>
      %60 = stablehlo.select %57, %59, %32 : tensor<i1>, tensor<i32>
      %61 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %62 = "stablehlo.scatter"(%55, %61, %40) ({
      ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
        stablehlo.return %arg3 : tensor<i32>
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
      %63 = stablehlo.constant dense<1> : tensor<i32>
      %64 = stablehlo.add %iterArg, %63 : tensor<i32>
      stablehlo.return %64, %24, %62, %iterArg_2 : tensor<i32>, tensor<i32>, tensor<4xi32>, tensor<4xi32>
    }
    %22 = call @_lu_solve(%17, %21#2, %arg1) : (tensor<4x4xf32>, tensor<4xi32>, tensor<4xf32>) -> tensor<4xf32>
    return %22 : tensor<4xf32>
  }
  func.func private @_lu_solve(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<4xf32>) -> tensor<4x1xf32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %3 = stablehlo.compare  LT, %arg1, %2,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %4 = stablehlo.constant dense<4> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %6 = stablehlo.add %arg1, %5 : tensor<4xi32>
    %7 = stablehlo.select %3, %6, %arg1 : tensor<4xi1>, tensor<4xi32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %9 = "stablehlo.gather"(%0, %8) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<4x1xf32>, tensor<4x1xi32>) -> tensor<4x1xf32>
    %10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.constant dense<1> : tensor<i32>
    %15 = stablehlo.constant dense<4> : tensor<i32>
    %16 = stablehlo.constant dense<1> : tensor<i32>
    %17 = stablehlo.constant dense<1> : tensor<i32>
    %18 = stablehlo.custom_call @blas_strsm(%11, %12, %13, %14, %15, %16, %17, %10, %arg0, %9) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = stablehlo.constant dense<1> : tensor<i32>
    %21 = stablehlo.constant dense<0> : tensor<i32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24 = stablehlo.constant dense<4> : tensor<i32>
    %25 = stablehlo.constant dense<1> : tensor<i32>
    %26 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.custom_call @blas_strsm(%20, %21, %22, %23, %24, %25, %26, %19, %arg0, %18) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %28 = stablehlo.constant dense<0> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %30 = "stablehlo.gather"(%27, %29) {dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = true, slice_sizes = dense<[4, 1]> : tensor<2xi64>} : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
    return %30 : tensor<4xf32>
  }
}
