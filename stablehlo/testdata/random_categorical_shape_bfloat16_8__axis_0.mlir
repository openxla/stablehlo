// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<8xbf16>)
    %1 = call @expected() : () -> tensor<i32>
    %2 = stablehlo.constant dense<1.175490e-38> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %6 = stablehlo.iota dim = 0 : tensor<4xui32>
    %7 = "stablehlo.slice"(%0#0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0#0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %12 = "stablehlo.slice"(%6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %15 = stablehlo.xor %8, %10 : tensor<ui32>
    %16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %17 = stablehlo.xor %15, %16 : tensor<ui32>
    %18 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %19 = stablehlo.add %11, %18 : tensor<2xui32>
    %20 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %21 = stablehlo.add %12, %20 : tensor<2xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %10, %iterArg_4 = %17, %iterArg_5 = %8, %iterArg_6 = %13, %iterArg_7 = %14) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %64 = stablehlo.constant dense<5> : tensor<i32>
      %65 = stablehlo.compare  LT, %iterArg, %64,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %65 : tensor<i1>
    } do {
      %64 = stablehlo.constant dense<1> : tensor<i32>
      %65 = stablehlo.add %iterArg_0, %64 : tensor<i32>
      %66 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %67 = stablehlo.reshape %66 : (tensor<1xui32>) -> tensor<ui32>
      %68 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %69 = stablehlo.broadcast_in_dim %67, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %70 = stablehlo.shift_left %iterArg_2, %69 : tensor<2xui32>
      %71 = stablehlo.constant dense<32> : tensor<ui32>
      %72 = stablehlo.subtract %71, %67 : tensor<ui32>
      %73 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %74 = stablehlo.shift_right_logical %iterArg_2, %73 : tensor<2xui32>
      %75 = stablehlo.or %70, %74 : tensor<2xui32>
      %76 = stablehlo.xor %68, %75 : tensor<2xui32>
      %77 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %78 = stablehlo.reshape %77 : (tensor<1xui32>) -> tensor<ui32>
      %79 = stablehlo.add %68, %76 : tensor<2xui32>
      %80 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %81 = stablehlo.shift_left %76, %80 : tensor<2xui32>
      %82 = stablehlo.constant dense<32> : tensor<ui32>
      %83 = stablehlo.subtract %82, %78 : tensor<ui32>
      %84 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %85 = stablehlo.shift_right_logical %76, %84 : tensor<2xui32>
      %86 = stablehlo.or %81, %85 : tensor<2xui32>
      %87 = stablehlo.xor %79, %86 : tensor<2xui32>
      %88 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %89 = stablehlo.reshape %88 : (tensor<1xui32>) -> tensor<ui32>
      %90 = stablehlo.add %79, %87 : tensor<2xui32>
      %91 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %92 = stablehlo.shift_left %87, %91 : tensor<2xui32>
      %93 = stablehlo.constant dense<32> : tensor<ui32>
      %94 = stablehlo.subtract %93, %89 : tensor<ui32>
      %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %96 = stablehlo.shift_right_logical %87, %95 : tensor<2xui32>
      %97 = stablehlo.or %92, %96 : tensor<2xui32>
      %98 = stablehlo.xor %90, %97 : tensor<2xui32>
      %99 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %100 = stablehlo.reshape %99 : (tensor<1xui32>) -> tensor<ui32>
      %101 = stablehlo.add %90, %98 : tensor<2xui32>
      %102 = stablehlo.broadcast_in_dim %100, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %103 = stablehlo.shift_left %98, %102 : tensor<2xui32>
      %104 = stablehlo.constant dense<32> : tensor<ui32>
      %105 = stablehlo.subtract %104, %100 : tensor<ui32>
      %106 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %107 = stablehlo.shift_right_logical %98, %106 : tensor<2xui32>
      %108 = stablehlo.or %103, %107 : tensor<2xui32>
      %109 = stablehlo.xor %101, %108 : tensor<2xui32>
      %110 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %111 = stablehlo.add %101, %110 : tensor<2xui32>
      %112 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %113 = stablehlo.add %109, %112 : tensor<2xui32>
      %114 = stablehlo.constant dense<1> : tensor<i32>
      %115 = stablehlo.add %iterArg_0, %114 : tensor<i32>
      %116 = stablehlo.convert %115 : (tensor<i32>) -> tensor<ui32>
      %117 = stablehlo.broadcast_in_dim %116, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %118 = stablehlo.add %113, %117 : tensor<2xui32>
      %119 = stablehlo.constant dense<1> : tensor<i32>
      %120 = stablehlo.add %iterArg, %119 : tensor<i32>
      stablehlo.return %120, %65, %111, %118, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<4xui32>) -> tensor<1x4xui32>
    %27 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %28 = stablehlo.constant dense<16> : tensor<ui32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %30 = stablehlo.multiply %29, %27 : tensor<2x1xui32>
    %31 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x4xui32>) -> tensor<2x4xui32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x4xui32>
    %33 = stablehlo.shift_right_logical %31, %32 : tensor<2x4xui32>
    %34 = stablehlo.constant dense<65535> : tensor<ui32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<ui32>) -> tensor<2x4xui32>
    %36 = stablehlo.and %35, %33 : tensor<2x4xui32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<2x4xui32>) -> tensor<4x2xui32>
    %38 = stablehlo.reshape %37 : (tensor<4x2xui32>) -> tensor<8xui32>
    %39 = stablehlo.convert %38 : (tensor<8xui32>) -> tensor<8xui16>
    %40 = stablehlo.constant dense<9> : tensor<ui16>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<ui16>) -> tensor<8xui16>
    %42 = stablehlo.shift_right_logical %39, %41 : tensor<8xui16>
    %43 = stablehlo.constant dense<16256> : tensor<ui16>
    %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<ui16>) -> tensor<8xui16>
    %45 = stablehlo.or %42, %44 : tensor<8xui16>
    %46 = stablehlo.bitcast_convert %45 : (tensor<8xui16>) -> tensor<8xbf16>
    %47 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<bf16>) -> tensor<8xbf16>
    %49 = stablehlo.subtract %46, %48 : tensor<8xbf16>
    %50 = stablehlo.subtract %5, %3 : tensor<1xbf16>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0] : (tensor<1xbf16>) -> tensor<8xbf16>
    %52 = stablehlo.multiply %49, %51 : tensor<8xbf16>
    %53 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xbf16>) -> tensor<8xbf16>
    %54 = stablehlo.add %52, %53 : tensor<8xbf16>
    %55 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xbf16>) -> tensor<8xbf16>
    %56 = stablehlo.maximum %55, %54 : tensor<8xbf16>
    %57 = stablehlo.log %56 : tensor<8xbf16>
    %58 = stablehlo.negate %57 : tensor<8xbf16>
    %59 = stablehlo.log %58 : tensor<8xbf16>
    %60 = stablehlo.negate %59 : tensor<8xbf16>
    %61 = stablehlo.add %60, %0#1 : tensor<8xbf16>
    %62 = call @argmax(%61) : (tensor<8xbf16>) -> tensor<i32>
    %63 = stablehlo.custom_call @check.eq(%62, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %63 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<8xbf16>) {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[-7.851560e-01, -5.093750e+00, 6.937500e+00, 4.312500e+00, -1.851560e+00, -6.343750e+00, -3.554690e-01, -5.531250e+00]> : tensor<8xbf16>
    return %0, %1 : tensor<2xui32>, tensor<8xbf16>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @argmax(%arg0: tensor<8xbf16>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %1 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<8xbf16>, tensor<8xi32>, tensor<bf16>, tensor<i32>) -> (tensor<bf16>, tensor<i32>)
     reducer(%arg1: tensor<bf16>, %arg3: tensor<bf16>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<bf16>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<bf16>, tensor<i32>
    }
    return %3#1 : tensor<i32>
  }
}
