// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<32xf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f16>) -> tensor<1xf16>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f16>) -> tensor<1xf16>
    %6 = stablehlo.iota dim = 0 : tensor<16xui32>
    %7 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<16xui32>) -> tensor<8xui32>
    %12 = "stablehlo.slice"(%6) {limit_indices = dense<16> : tensor<1xi64>, start_indices = dense<8> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<16xui32>) -> tensor<8xui32>
    %13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %15 = stablehlo.xor %8, %10 : tensor<ui32>
    %16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %17 = stablehlo.xor %15, %16 : tensor<ui32>
    %18 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %19 = stablehlo.add %11, %18 : tensor<8xui32>
    %20 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %21 = stablehlo.add %12, %20 : tensor<8xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %10, %iterArg_4 = %17, %iterArg_5 = %8, %iterArg_6 = %13, %iterArg_7 = %14) : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %58 = stablehlo.constant dense<5> : tensor<i32>
      %59 = stablehlo.compare  LT, %iterArg, %58,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %59 : tensor<i1>
    } do {
      %58 = stablehlo.constant dense<1> : tensor<i32>
      %59 = stablehlo.add %iterArg_0, %58 : tensor<i32>
      %60 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %61 = stablehlo.reshape %60 : (tensor<1xui32>) -> tensor<ui32>
      %62 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<8xui32>
      %63 = stablehlo.broadcast_in_dim %61, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %64 = stablehlo.shift_left %iterArg_2, %63 : tensor<8xui32>
      %65 = stablehlo.constant dense<32> : tensor<ui32>
      %66 = stablehlo.subtract %65, %61 : tensor<ui32>
      %67 = stablehlo.broadcast_in_dim %66, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %68 = stablehlo.shift_right_logical %iterArg_2, %67 : tensor<8xui32>
      %69 = stablehlo.or %64, %68 : tensor<8xui32>
      %70 = stablehlo.xor %62, %69 : tensor<8xui32>
      %71 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %72 = stablehlo.reshape %71 : (tensor<1xui32>) -> tensor<ui32>
      %73 = stablehlo.add %62, %70 : tensor<8xui32>
      %74 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %75 = stablehlo.shift_left %70, %74 : tensor<8xui32>
      %76 = stablehlo.constant dense<32> : tensor<ui32>
      %77 = stablehlo.subtract %76, %72 : tensor<ui32>
      %78 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %79 = stablehlo.shift_right_logical %70, %78 : tensor<8xui32>
      %80 = stablehlo.or %75, %79 : tensor<8xui32>
      %81 = stablehlo.xor %73, %80 : tensor<8xui32>
      %82 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %83 = stablehlo.reshape %82 : (tensor<1xui32>) -> tensor<ui32>
      %84 = stablehlo.add %73, %81 : tensor<8xui32>
      %85 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %86 = stablehlo.shift_left %81, %85 : tensor<8xui32>
      %87 = stablehlo.constant dense<32> : tensor<ui32>
      %88 = stablehlo.subtract %87, %83 : tensor<ui32>
      %89 = stablehlo.broadcast_in_dim %88, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %90 = stablehlo.shift_right_logical %81, %89 : tensor<8xui32>
      %91 = stablehlo.or %86, %90 : tensor<8xui32>
      %92 = stablehlo.xor %84, %91 : tensor<8xui32>
      %93 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %94 = stablehlo.reshape %93 : (tensor<1xui32>) -> tensor<ui32>
      %95 = stablehlo.add %84, %92 : tensor<8xui32>
      %96 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %97 = stablehlo.shift_left %92, %96 : tensor<8xui32>
      %98 = stablehlo.constant dense<32> : tensor<ui32>
      %99 = stablehlo.subtract %98, %94 : tensor<ui32>
      %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %101 = stablehlo.shift_right_logical %92, %100 : tensor<8xui32>
      %102 = stablehlo.or %97, %101 : tensor<8xui32>
      %103 = stablehlo.xor %95, %102 : tensor<8xui32>
      %104 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %105 = stablehlo.add %95, %104 : tensor<8xui32>
      %106 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %107 = stablehlo.add %103, %106 : tensor<8xui32>
      %108 = stablehlo.constant dense<1> : tensor<i32>
      %109 = stablehlo.add %iterArg_0, %108 : tensor<i32>
      %110 = stablehlo.convert %109 : (tensor<i32>) -> tensor<ui32>
      %111 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<ui32>) -> tensor<8xui32>
      %112 = stablehlo.add %107, %111 : tensor<8xui32>
      %113 = stablehlo.constant dense<1> : tensor<i32>
      %114 = stablehlo.add %iterArg, %113 : tensor<i32>
      stablehlo.return %114, %59, %105, %112, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 0 : (tensor<8xui32>, tensor<8xui32>) -> tensor<16xui32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<16xui32>) -> tensor<1x16xui32>
    %27 = stablehlo.iota dim = 0 : tensor<2x1xui32>
    %28 = stablehlo.constant dense<16> : tensor<ui32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<ui32>) -> tensor<2x1xui32>
    %30 = stablehlo.multiply %29, %27 : tensor<2x1xui32>
    %31 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x16xui32>) -> tensor<2x16xui32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<2x1xui32>) -> tensor<2x16xui32>
    %33 = stablehlo.shift_right_logical %31, %32 : tensor<2x16xui32>
    %34 = stablehlo.constant dense<65535> : tensor<ui32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<ui32>) -> tensor<2x16xui32>
    %36 = stablehlo.and %35, %33 : tensor<2x16xui32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<2x16xui32>) -> tensor<16x2xui32>
    %38 = stablehlo.reshape %37 : (tensor<16x2xui32>) -> tensor<32xui32>
    %39 = stablehlo.convert %38 : (tensor<32xui32>) -> tensor<32xui16>
    %40 = stablehlo.constant dense<6> : tensor<ui16>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<ui16>) -> tensor<32xui16>
    %42 = stablehlo.shift_right_logical %39, %41 : tensor<32xui16>
    %43 = stablehlo.constant dense<15360> : tensor<ui16>
    %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<ui16>) -> tensor<32xui16>
    %45 = stablehlo.or %42, %44 : tensor<32xui16>
    %46 = stablehlo.bitcast_convert %45 : (tensor<32xui16>) -> tensor<32xf16>
    %47 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f16>) -> tensor<32xf16>
    %49 = stablehlo.subtract %46, %48 : tensor<32xf16>
    %50 = stablehlo.subtract %5, %3 : tensor<1xf16>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0] : (tensor<1xf16>) -> tensor<32xf16>
    %52 = stablehlo.multiply %49, %51 : tensor<32xf16>
    %53 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf16>) -> tensor<32xf16>
    %54 = stablehlo.add %52, %53 : tensor<32xf16>
    %55 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf16>) -> tensor<32xf16>
    %56 = stablehlo.maximum %55, %54 : tensor<32xf16>
    %57 = stablehlo.custom_call @check.eq(%56, %1) : (tensor<32xf16>, tensor<32xf16>) -> tensor<i1>
    return %57 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<32xf16> {
    %0 = stablehlo.constant dense<[6.582030e-01, 4.775390e-01, 3.701170e-01, 2.236330e-01, 9.375000e-02, 8.046880e-01, 1.757810e-01, 2.617190e-01, 8.593750e-02, 7.421880e-02, 4.853520e-01, 2.324220e-01, 3.525390e-01, 1.494140e-01, 1.220700e-01, 4.082030e-01, 5.283200e-01, 9.892570e-01, 8.281250e-01, 6.298830e-01, 5.302730e-01, 3.955080e-01, 9.775390e-01, 6.015630e-01, 3.427730e-01, 2.373050e-01, 1.044920e-01, 4.628910e-01, 6.513670e-01, 7.832030e-01, 1.767580e-01, 8.652340e-01]> : tensor<32xf16>
    return %0 : tensor<32xf16>
  }
}
