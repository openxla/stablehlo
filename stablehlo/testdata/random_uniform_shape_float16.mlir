// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<f16>
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
      %60 = stablehlo.constant dense<5> : tensor<i32>
      %61 = stablehlo.compare  LT, %iterArg, %60,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %61 : tensor<i1>
    } do {
      %60 = stablehlo.constant dense<1> : tensor<i32>
      %61 = stablehlo.add %iterArg_0, %60 : tensor<i32>
      %62 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %63 = stablehlo.reshape %62 : (tensor<1xui32>) -> tensor<ui32>
      %64 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<1xui32>
      %65 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %66 = stablehlo.shift_left %iterArg_2, %65 : tensor<1xui32>
      %67 = stablehlo.constant dense<32> : tensor<ui32>
      %68 = stablehlo.subtract %67, %63 : tensor<ui32>
      %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %70 = stablehlo.shift_right_logical %iterArg_2, %69 : tensor<1xui32>
      %71 = stablehlo.or %66, %70 : tensor<1xui32>
      %72 = stablehlo.xor %64, %71 : tensor<1xui32>
      %73 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %74 = stablehlo.reshape %73 : (tensor<1xui32>) -> tensor<ui32>
      %75 = stablehlo.add %64, %72 : tensor<1xui32>
      %76 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %77 = stablehlo.shift_left %72, %76 : tensor<1xui32>
      %78 = stablehlo.constant dense<32> : tensor<ui32>
      %79 = stablehlo.subtract %78, %74 : tensor<ui32>
      %80 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %81 = stablehlo.shift_right_logical %72, %80 : tensor<1xui32>
      %82 = stablehlo.or %77, %81 : tensor<1xui32>
      %83 = stablehlo.xor %75, %82 : tensor<1xui32>
      %84 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %85 = stablehlo.reshape %84 : (tensor<1xui32>) -> tensor<ui32>
      %86 = stablehlo.add %75, %83 : tensor<1xui32>
      %87 = stablehlo.broadcast_in_dim %85, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %88 = stablehlo.shift_left %83, %87 : tensor<1xui32>
      %89 = stablehlo.constant dense<32> : tensor<ui32>
      %90 = stablehlo.subtract %89, %85 : tensor<ui32>
      %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %92 = stablehlo.shift_right_logical %83, %91 : tensor<1xui32>
      %93 = stablehlo.or %88, %92 : tensor<1xui32>
      %94 = stablehlo.xor %86, %93 : tensor<1xui32>
      %95 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %96 = stablehlo.reshape %95 : (tensor<1xui32>) -> tensor<ui32>
      %97 = stablehlo.add %86, %94 : tensor<1xui32>
      %98 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %99 = stablehlo.shift_left %94, %98 : tensor<1xui32>
      %100 = stablehlo.constant dense<32> : tensor<ui32>
      %101 = stablehlo.subtract %100, %96 : tensor<ui32>
      %102 = stablehlo.broadcast_in_dim %101, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %103 = stablehlo.shift_right_logical %94, %102 : tensor<1xui32>
      %104 = stablehlo.or %99, %103 : tensor<1xui32>
      %105 = stablehlo.xor %97, %104 : tensor<1xui32>
      %106 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %107 = stablehlo.add %97, %106 : tensor<1xui32>
      %108 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %109 = stablehlo.add %105, %108 : tensor<1xui32>
      %110 = stablehlo.constant dense<1> : tensor<i32>
      %111 = stablehlo.add %iterArg_0, %110 : tensor<i32>
      %112 = stablehlo.convert %111 : (tensor<i32>) -> tensor<ui32>
      %113 = stablehlo.broadcast_in_dim %112, dims = [] : (tensor<ui32>) -> tensor<1xui32>
      %114 = stablehlo.add %109, %113 : tensor<1xui32>
      %115 = stablehlo.constant dense<1> : tensor<i32>
      %116 = stablehlo.add %iterArg, %115 : tensor<i32>
      stablehlo.return %116, %61, %107, %114, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %23 = stablehlo.concatenate %22#2, %22#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %24 = stablehlo.constant dense<0> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = "stablehlo.gather"(%23, %25) {dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xui32>, tensor<1xi32>) -> tensor<1xui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [1] : (tensor<1xui32>) -> tensor<1x1xui32>
    %28 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %29 = stablehlo.constant dense<16> : tensor<ui32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %31 = stablehlo.multiply %30, %28 : tensor<2x1xui32>
    %32 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<1x1xui32>) -> tensor<2x1xui32>
    %33 = stablehlo.shift_right_logical %32, %31 : tensor<2x1xui32>
    %34 = stablehlo.constant dense<65535> : tensor<ui32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %36 = stablehlo.and %35, %33 : tensor<2x1xui32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<2x1xui32>) -> tensor<1x2xui32>
    %38 = stablehlo.reshape %37 : (tensor<1x2xui32>) -> tensor<2xui32>
    %39 = stablehlo.convert %38 : (tensor<2xui32>) -> tensor<2xui16>
    %40 = stablehlo.constant dense<0> : tensor<i32>
    %41 = stablehlo.dynamic_slice %39, %40, sizes = [1] : (tensor<2xui16>, tensor<i32>) -> tensor<1xui16>
    %42 = stablehlo.reshape %41 : (tensor<1xui16>) -> tensor<ui16>
    %43 = stablehlo.constant dense<6> : tensor<ui16>
    %44 = stablehlo.shift_right_logical %42, %43 : tensor<ui16>
    %45 = stablehlo.constant dense<15360> : tensor<ui16>
    %46 = stablehlo.or %44, %45 : tensor<ui16>
    %47 = stablehlo.bitcast_convert %46 : (tensor<ui16>) -> tensor<f16>
    %48 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %49 = stablehlo.subtract %47, %48 : tensor<f16>
    %50 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %51 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %52 = stablehlo.subtract %50, %51 : tensor<f16>
    %53 = stablehlo.multiply %49, %52 : tensor<f16>
    %54 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %55 = stablehlo.add %53, %54 : tensor<f16>
    %56 = stablehlo.reshape %55 : (tensor<f16>) -> tensor<f16>
    %57 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %58 = stablehlo.maximum %57, %56 : tensor<f16>
    %59 = stablehlo.custom_call @check.eq(%58, %1) : (tensor<f16>, tensor<f16>) -> tensor<i1>
    return %59 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<f16> {
    %0 = stablehlo.constant dense<8.808590e-01> : tensor<f16>
    return %0 : tensor<f16>
  }
}
