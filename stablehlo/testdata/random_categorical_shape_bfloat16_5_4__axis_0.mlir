// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<5x4xbf16>)
    %1 = call @expected() : () -> tensor<4xi32>
    %2 = stablehlo.constant dense<1.175490e-38> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<1x1xbf16>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<bf16>) -> tensor<1x1xbf16>
    %6 = stablehlo.iota dim = 0 : tensor<10xui32>
    %7 = "stablehlo.slice"(%0#0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0#0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
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
      %65 = stablehlo.constant dense<5> : tensor<i32>
      %66 = stablehlo.compare  LT, %iterArg, %65,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %66 : tensor<i1>
    } do {
      %65 = stablehlo.constant dense<1> : tensor<i32>
      %66 = stablehlo.add %iterArg_0, %65 : tensor<i32>
      %67 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %68 = stablehlo.reshape %67 : (tensor<1xui32>) -> tensor<ui32>
      %69 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<5xui32>
      %70 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %71 = stablehlo.shift_left %iterArg_2, %70 : tensor<5xui32>
      %72 = stablehlo.constant dense<32> : tensor<ui32>
      %73 = stablehlo.subtract %72, %68 : tensor<ui32>
      %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %75 = stablehlo.shift_right_logical %iterArg_2, %74 : tensor<5xui32>
      %76 = stablehlo.or %71, %75 : tensor<5xui32>
      %77 = stablehlo.xor %69, %76 : tensor<5xui32>
      %78 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %79 = stablehlo.reshape %78 : (tensor<1xui32>) -> tensor<ui32>
      %80 = stablehlo.add %69, %77 : tensor<5xui32>
      %81 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %82 = stablehlo.shift_left %77, %81 : tensor<5xui32>
      %83 = stablehlo.constant dense<32> : tensor<ui32>
      %84 = stablehlo.subtract %83, %79 : tensor<ui32>
      %85 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %86 = stablehlo.shift_right_logical %77, %85 : tensor<5xui32>
      %87 = stablehlo.or %82, %86 : tensor<5xui32>
      %88 = stablehlo.xor %80, %87 : tensor<5xui32>
      %89 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %90 = stablehlo.reshape %89 : (tensor<1xui32>) -> tensor<ui32>
      %91 = stablehlo.add %80, %88 : tensor<5xui32>
      %92 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %93 = stablehlo.shift_left %88, %92 : tensor<5xui32>
      %94 = stablehlo.constant dense<32> : tensor<ui32>
      %95 = stablehlo.subtract %94, %90 : tensor<ui32>
      %96 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %97 = stablehlo.shift_right_logical %88, %96 : tensor<5xui32>
      %98 = stablehlo.or %93, %97 : tensor<5xui32>
      %99 = stablehlo.xor %91, %98 : tensor<5xui32>
      %100 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %101 = stablehlo.reshape %100 : (tensor<1xui32>) -> tensor<ui32>
      %102 = stablehlo.add %91, %99 : tensor<5xui32>
      %103 = stablehlo.broadcast_in_dim %101, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %104 = stablehlo.shift_left %99, %103 : tensor<5xui32>
      %105 = stablehlo.constant dense<32> : tensor<ui32>
      %106 = stablehlo.subtract %105, %101 : tensor<ui32>
      %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %108 = stablehlo.shift_right_logical %99, %107 : tensor<5xui32>
      %109 = stablehlo.or %104, %108 : tensor<5xui32>
      %110 = stablehlo.xor %102, %109 : tensor<5xui32>
      %111 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %112 = stablehlo.add %102, %111 : tensor<5xui32>
      %113 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %114 = stablehlo.add %110, %113 : tensor<5xui32>
      %115 = stablehlo.constant dense<1> : tensor<i32>
      %116 = stablehlo.add %iterArg_0, %115 : tensor<i32>
      %117 = stablehlo.convert %116 : (tensor<i32>) -> tensor<ui32>
      %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<ui32>) -> tensor<5xui32>
      %119 = stablehlo.add %114, %118 : tensor<5xui32>
      %120 = stablehlo.constant dense<1> : tensor<i32>
      %121 = stablehlo.add %iterArg, %120 : tensor<i32>
      stablehlo.return %121, %66, %112, %119, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<5xui32>, tensor<5xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
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
    %58 = stablehlo.log %57 : tensor<5x4xbf16>
    %59 = stablehlo.negate %58 : tensor<5x4xbf16>
    %60 = stablehlo.log %59 : tensor<5x4xbf16>
    %61 = stablehlo.negate %60 : tensor<5x4xbf16>
    %62 = stablehlo.add %61, %0#1 : tensor<5x4xbf16>
    %63 = call @argmax(%62) : (tensor<5x4xbf16>) -> tensor<4xi32>
    %64 = stablehlo.custom_call @check.eq(%63, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<i1>
    return %64 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<5x4xbf16>) {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[[6.187500e+00, -4.750000e+00, -4.218750e+00, -1.937500e+00], [5.976560e-01, 3.515630e+00, 4.023440e-01, 3.750000e+00], [-3.265630e+00, -3.671880e-01, 5.859380e-01, 3.375000e+00], [-9.765620e-01, -2.695310e-01, -6.843750e+00, 5.968750e+00], [1.070310e+00, -1.054690e+00, 4.156250e+00, 3.562500e+00]]> : tensor<5x4xbf16>
    return %0, %1 : tensor<2xui32>, tensor<5x4xbf16>
  }
  func.func private @expected() -> tensor<4xi32> {
    %0 = stablehlo.constant dense<[0, 1, 4, 3]> : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
  func.func private @argmax(%arg0: tensor<5x4xbf16>) -> tensor<4xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<5x4xi32>
    %1 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<5x4xbf16>, tensor<5x4xi32>, tensor<bf16>, tensor<i32>) -> (tensor<4xbf16>, tensor<4xi32>)
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
    return %3#1 : tensor<4xi32>
  }
}
