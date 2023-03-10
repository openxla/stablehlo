// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<1xui32>
    %3 = stablehlo.iota dim = 0 : tensor<1xui32>
    %4 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %7 = stablehlo.reshape %6 : (tensor<1xui32>) -> tensor<ui32>
    %8 = stablehlo.concatenate %3, %2, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = "stablehlo.slice"(%8) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %11 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %12 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %13 = stablehlo.xor %5, %7 : tensor<ui32>
    %14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %15 = stablehlo.xor %13, %14 : tensor<ui32>
    %16 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %17 = stablehlo.add %9, %16 : tensor<1xui32>
    %18 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %19 = stablehlo.add %10, %18 : tensor<1xui32>
    %20 = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.constant dense<0> : tensor<i32>
    %22:9 = stablehlo.while(%iterArg = %21, %iterArg_0 = %20, %iterArg_1 = %17, %iterArg_2 = %19, %iterArg_3 = %7, %iterArg_4 = %15, %iterArg_5 = %5, %iterArg_6 = %11, %iterArg_7 = %12) : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %45 = stablehlo.constant dense<5> : tensor<i32>
      %46 = stablehlo.compare  LT, %iterArg, %45,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %46 : tensor<i1>
    } do {
      %45 = stablehlo.constant dense<1> : tensor<i32>
      %46 = stablehlo.add %iterArg_0, %45 : tensor<i32>
      %47 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %48 = stablehlo.reshape %47 : (tensor<1xui32>) -> tensor<ui32>
      %49 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1xui32>
      %50 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %51 = stablehlo.shift_left %iterArg_2, %50 : tensor<1xui32>
      %52 = stablehlo.constant dense<32> : tensor<ui32>
      %53 = stablehlo.subtract %52, %48 : tensor<ui32>
      %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %55 = stablehlo.shift_right_logical %iterArg_2, %54 : tensor<1xui32>
      %56 = stablehlo.or %51, %55 : tensor<1xui32>
      %57 = stablehlo.xor %49, %56 : tensor<1xui32>
      %58 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %59 = stablehlo.reshape %58 : (tensor<1xui32>) -> tensor<ui32>
      %60 = stablehlo.add %49, %57 : tensor<1xui32>
      %61 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %62 = stablehlo.shift_left %57, %61 : tensor<1xui32>
      %63 = stablehlo.constant dense<32> : tensor<ui32>
      %64 = stablehlo.subtract %63, %59 : tensor<ui32>
      %65 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %66 = stablehlo.shift_right_logical %57, %65 : tensor<1xui32>
      %67 = stablehlo.or %62, %66 : tensor<1xui32>
      %68 = stablehlo.xor %60, %67 : tensor<1xui32>
      %69 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %70 = stablehlo.reshape %69 : (tensor<1xui32>) -> tensor<ui32>
      %71 = stablehlo.add %60, %68 : tensor<1xui32>
      %72 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %73 = stablehlo.shift_left %68, %72 : tensor<1xui32>
      %74 = stablehlo.constant dense<32> : tensor<ui32>
      %75 = stablehlo.subtract %74, %70 : tensor<ui32>
      %76 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %77 = stablehlo.shift_right_logical %68, %76 : tensor<1xui32>
      %78 = stablehlo.or %73, %77 : tensor<1xui32>
      %79 = stablehlo.xor %71, %78 : tensor<1xui32>
      %80 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %81 = stablehlo.reshape %80 : (tensor<1xui32>) -> tensor<ui32>
      %82 = stablehlo.add %71, %79 : tensor<1xui32>
      %83 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %84 = stablehlo.shift_left %79, %83 : tensor<1xui32>
      %85 = stablehlo.constant dense<32> : tensor<ui32>
      %86 = stablehlo.subtract %85, %81 : tensor<ui32>
      %87 = stablehlo.broadcast_in_dim %86, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %88 = stablehlo.shift_right_logical %79, %87 : tensor<1xui32>
      %89 = stablehlo.or %84, %88 : tensor<1xui32>
      %90 = stablehlo.xor %82, %89 : tensor<1xui32>
      %91 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %92 = stablehlo.add %82, %91 : tensor<1xui32>
      %93 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %94 = stablehlo.add %90, %93 : tensor<1xui32>
      %95 = stablehlo.constant dense<1> : tensor<i32>
      %96 = stablehlo.add %iterArg_0, %95 : tensor<i32>
      %97 = stablehlo.convert %96 : (tensor<i32>) -> tensor<ui32>
      %98 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %99 = stablehlo.add %94, %98 : tensor<1xui32>
      %100 = stablehlo.constant dense<1> : tensor<i32>
      %101 = stablehlo.add %iterArg, %100 : tensor<i32>
      stablehlo.return %101, %46, %92, %99, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %23 = stablehlo.concatenate %22#2, %22#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %24 = stablehlo.constant dense<0> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = "stablehlo.gather"(%23, %25) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
    %27 = stablehlo.reshape %26 : (tensor<1xui32>) -> tensor<ui32>
    %28 = stablehlo.constant dense<9> : tensor<ui32>
    %29 = stablehlo.shift_right_logical %27, %28 : tensor<ui32>
    %30 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %31 = stablehlo.or %29, %30 : tensor<ui32>
    %32 = stablehlo.bitcast_convert %31 : (tensor<ui32>) -> tensor<f32>
    %33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.subtract %32, %33 : tensor<f32>
    %35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.subtract %35, %36 : tensor<f32>
    %38 = stablehlo.multiply %34, %37 : tensor<f32>
    %39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %40 = stablehlo.add %38, %39 : tensor<f32>
    %41 = stablehlo.reshape %40 : (tensor<f32>) -> tensor<f32>
    %42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.maximum %42, %41 : tensor<f32>
    %44 = stablehlo.custom_call @check.eq(%43, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %44 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.733732224> : tensor<f32>
    return %0 : tensor<f32>
  }
}
