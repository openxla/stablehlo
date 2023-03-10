// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<5x4xbf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<1x1xbf16>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<bf16>) -> tensor<1x1xbf16>
    %6 = stablehlo.iota dim = 0 : tensor<10xui32>
    %7 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<10xui32>) -> tensor<5xui32>
    %12 = "stablehlo.slice"(%6) {limit_indices = dense<10> : tensor<1xi64>, start_indices = dense<5> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<10xui32>) -> tensor<5xui32>
    %13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %15 = stablehlo.xor %8, %10 : tensor<ui32>
    %16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %17 = stablehlo.xor %15, %16 : tensor<ui32>
    %18 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<5xui32>
    %19 = stablehlo.add %11, %18 : tensor<5xui32>
    %20 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<5xui32>
    %21 = stablehlo.add %12, %20 : tensor<5xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %10, %iterArg_4 = %17, %iterArg_5 = %8, %iterArg_6 = %13, %iterArg_7 = %14) : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %59 = stablehlo.constant dense<5> : tensor<i32>
      %60 = stablehlo.compare  LT, %iterArg, %59,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %60 : tensor<i1>
    } do {
      %59 = stablehlo.constant dense<1> : tensor<i32>
      %60 = stablehlo.add %iterArg_0, %59 : tensor<i32>
      %61 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %62 = stablehlo.reshape %61 : (tensor<1xui32>) -> tensor<ui32>
      %63 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<5xui32>
      %64 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %65 = stablehlo.shift_left %iterArg_2, %64 : tensor<5xui32>
      %66 = stablehlo.constant dense<32> : tensor<ui32>
      %67 = stablehlo.subtract %66, %62 : tensor<ui32>
      %68 = stablehlo.broadcast_in_dim %67, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %69 = stablehlo.shift_right_logical %iterArg_2, %68 : tensor<5xui32>
      %70 = stablehlo.or %65, %69 : tensor<5xui32>
      %71 = stablehlo.xor %63, %70 : tensor<5xui32>
      %72 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %73 = stablehlo.reshape %72 : (tensor<1xui32>) -> tensor<ui32>
      %74 = stablehlo.add %63, %71 : tensor<5xui32>
      %75 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %76 = stablehlo.shift_left %71, %75 : tensor<5xui32>
      %77 = stablehlo.constant dense<32> : tensor<ui32>
      %78 = stablehlo.subtract %77, %73 : tensor<ui32>
      %79 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %80 = stablehlo.shift_right_logical %71, %79 : tensor<5xui32>
      %81 = stablehlo.or %76, %80 : tensor<5xui32>
      %82 = stablehlo.xor %74, %81 : tensor<5xui32>
      %83 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %84 = stablehlo.reshape %83 : (tensor<1xui32>) -> tensor<ui32>
      %85 = stablehlo.add %74, %82 : tensor<5xui32>
      %86 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %87 = stablehlo.shift_left %82, %86 : tensor<5xui32>
      %88 = stablehlo.constant dense<32> : tensor<ui32>
      %89 = stablehlo.subtract %88, %84 : tensor<ui32>
      %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %91 = stablehlo.shift_right_logical %82, %90 : tensor<5xui32>
      %92 = stablehlo.or %87, %91 : tensor<5xui32>
      %93 = stablehlo.xor %85, %92 : tensor<5xui32>
      %94 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %95 = stablehlo.reshape %94 : (tensor<1xui32>) -> tensor<ui32>
      %96 = stablehlo.add %85, %93 : tensor<5xui32>
      %97 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %98 = stablehlo.shift_left %93, %97 : tensor<5xui32>
      %99 = stablehlo.constant dense<32> : tensor<ui32>
      %100 = stablehlo.subtract %99, %95 : tensor<ui32>
      %101 = stablehlo.broadcast_in_dim %100, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %102 = stablehlo.shift_right_logical %93, %101 : tensor<5xui32>
      %103 = stablehlo.or %98, %102 : tensor<5xui32>
      %104 = stablehlo.xor %96, %103 : tensor<5xui32>
      %105 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %106 = stablehlo.add %96, %105 : tensor<5xui32>
      %107 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %108 = stablehlo.add %104, %107 : tensor<5xui32>
      %109 = stablehlo.constant dense<1> : tensor<i32>
      %110 = stablehlo.add %iterArg_0, %109 : tensor<i32>
      %111 = stablehlo.convert %110 : (tensor<i32>) -> tensor<ui32>
      %112 = stablehlo.broadcast_in_dim %111, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %113 = stablehlo.add %108, %112 : tensor<5xui32>
      %114 = stablehlo.constant dense<1> : tensor<i32>
      %115 = stablehlo.add %iterArg, %114 : tensor<i32>
      stablehlo.return %115, %60, %106, %113, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 0 : (tensor<5xui32>, tensor<5xui32>) -> tensor<10xui32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<10xui32>) -> tensor<1x10xui32>
    %27 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %28 = stablehlo.constant dense<16> : tensor<ui32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %30 = stablehlo.multiply %29, %27 : tensor<2x1xui32>
    %31 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x10xui32>) -> tensor<2x10xui32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x10xui32>
    %33 = stablehlo.shift_right_logical %31, %32 : tensor<2x10xui32>
    %34 = stablehlo.constant dense<65535> : tensor<ui32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<ui32>) -> tensor<2x10xui32>
    %36 = stablehlo.and %35, %33 : tensor<2x10xui32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<2x10xui32>) -> tensor<10x2xui32>
    %38 = stablehlo.reshape %37 : (tensor<10x2xui32>) -> tensor<20xui32>
    %39 = stablehlo.convert %38 : (tensor<20xui32>) -> tensor<20xui16>
    %40 = stablehlo.reshape %39 : (tensor<20xui16>) -> tensor<5x4xui16>
    %41 = stablehlo.constant dense<9> : tensor<ui16>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<ui16>) -> tensor<5x4xui16>
    %43 = stablehlo.shift_right_logical %40, %42 : tensor<5x4xui16>
    %44 = stablehlo.constant dense<16256> : tensor<ui16>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<ui16>) -> tensor<5x4xui16>
    %46 = stablehlo.or %43, %45 : tensor<5x4xui16>
    %47 = stablehlo.bitcast_convert %46 : (tensor<5x4xui16>) -> tensor<5x4xbf16>
    %48 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %49 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<bf16>) -> tensor<5x4xbf16>
    %50 = stablehlo.subtract %47, %49 : tensor<5x4xbf16>
    %51 = stablehlo.subtract %5, %3 : tensor<1x1xbf16>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<1x1xbf16>) -> tensor<5x4xbf16>
    %53 = stablehlo.multiply %50, %52 : tensor<5x4xbf16>
    %54 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xbf16>) -> tensor<5x4xbf16>
    %55 = stablehlo.add %53, %54 : tensor<5x4xbf16>
    %56 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xbf16>) -> tensor<5x4xbf16>
    %57 = stablehlo.maximum %56, %55 : tensor<5x4xbf16>
    %58 = stablehlo.custom_call @check.eq(%57, %1) : (tensor<5x4xbf16>, tensor<5x4xbf16>) -> tensor<i1>
    return %58 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<5x4xbf16> {
    %0 = stablehlo.constant dense<[[5.312500e-01, 7.890630e-01, 5.859380e-01, 7.109380e-01], [3.203130e-01, 5.937500e-01, 8.984370e-01, 4.453130e-01], [3.125000e-02, 1.484380e-01, 8.906250e-01, 2.265630e-01], [5.937500e-01, 5.546880e-01, 8.281250e-01, 7.187500e-01], [9.531250e-01, 4.687500e-01, 2.812500e-01, 1.015630e-01]]> : tensor<5x4xbf16>
    return %0 : tensor<5x4xbf16>
  }
}
