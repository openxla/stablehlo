// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<32xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.iota dim = 0 : tensor<32xui32>
    %7 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<16> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<32xui32>) -> tensor<16xui32>
    %12 = "stablehlo.slice"(%6) {limit_indices = dense<32> : tensor<1xi64>, start_indices = dense<16> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<32xui32>) -> tensor<16xui32>
    %13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %15 = stablehlo.xor %8, %10 : tensor<ui32>
    %16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %17 = stablehlo.xor %15, %16 : tensor<ui32>
    %18 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<16xui32>
    %19 = stablehlo.add %11, %18 : tensor<16xui32>
    %20 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<16xui32>
    %21 = stablehlo.add %12, %20 : tensor<16xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %10, %iterArg_4 = %17, %iterArg_5 = %8, %iterArg_6 = %13, %iterArg_7 = %14) : tensor<i32>, tensor<i32>, tensor<16xui32>, tensor<16xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %44 = stablehlo.constant dense<5> : tensor<i32>
      %45 = stablehlo.compare  LT, %iterArg, %44,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %44 = stablehlo.constant dense<1> : tensor<i32>
      %45 = stablehlo.add %iterArg_0, %44 : tensor<i32>
      %46 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %47 = stablehlo.reshape %46 : (tensor<1xui32>) -> tensor<ui32>
      %48 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<16xui32>
      %49 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %50 = stablehlo.shift_left %iterArg_2, %49 : tensor<16xui32>
      %51 = stablehlo.constant dense<32> : tensor<ui32>
      %52 = stablehlo.subtract %51, %47 : tensor<ui32>
      %53 = stablehlo.broadcast_in_dim %52, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %54 = stablehlo.shift_right_logical %iterArg_2, %53 : tensor<16xui32>
      %55 = stablehlo.or %50, %54 : tensor<16xui32>
      %56 = stablehlo.xor %48, %55 : tensor<16xui32>
      %57 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %58 = stablehlo.reshape %57 : (tensor<1xui32>) -> tensor<ui32>
      %59 = stablehlo.add %48, %56 : tensor<16xui32>
      %60 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %61 = stablehlo.shift_left %56, %60 : tensor<16xui32>
      %62 = stablehlo.constant dense<32> : tensor<ui32>
      %63 = stablehlo.subtract %62, %58 : tensor<ui32>
      %64 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %65 = stablehlo.shift_right_logical %56, %64 : tensor<16xui32>
      %66 = stablehlo.or %61, %65 : tensor<16xui32>
      %67 = stablehlo.xor %59, %66 : tensor<16xui32>
      %68 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %69 = stablehlo.reshape %68 : (tensor<1xui32>) -> tensor<ui32>
      %70 = stablehlo.add %59, %67 : tensor<16xui32>
      %71 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %72 = stablehlo.shift_left %67, %71 : tensor<16xui32>
      %73 = stablehlo.constant dense<32> : tensor<ui32>
      %74 = stablehlo.subtract %73, %69 : tensor<ui32>
      %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %76 = stablehlo.shift_right_logical %67, %75 : tensor<16xui32>
      %77 = stablehlo.or %72, %76 : tensor<16xui32>
      %78 = stablehlo.xor %70, %77 : tensor<16xui32>
      %79 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %80 = stablehlo.reshape %79 : (tensor<1xui32>) -> tensor<ui32>
      %81 = stablehlo.add %70, %78 : tensor<16xui32>
      %82 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %83 = stablehlo.shift_left %78, %82 : tensor<16xui32>
      %84 = stablehlo.constant dense<32> : tensor<ui32>
      %85 = stablehlo.subtract %84, %80 : tensor<ui32>
      %86 = stablehlo.broadcast_in_dim %85, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %87 = stablehlo.shift_right_logical %78, %86 : tensor<16xui32>
      %88 = stablehlo.or %83, %87 : tensor<16xui32>
      %89 = stablehlo.xor %81, %88 : tensor<16xui32>
      %90 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %91 = stablehlo.add %81, %90 : tensor<16xui32>
      %92 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %93 = stablehlo.add %89, %92 : tensor<16xui32>
      %94 = stablehlo.constant dense<1> : tensor<i32>
      %95 = stablehlo.add %iterArg_0, %94 : tensor<i32>
      %96 = stablehlo.convert %95 : (tensor<i32>) -> tensor<ui32>
      %97 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<ui32>) -> tensor<16xui32>
      %98 = stablehlo.add %93, %97 : tensor<16xui32>
      %99 = stablehlo.constant dense<1> : tensor<i32>
      %100 = stablehlo.add %iterArg, %99 : tensor<i32>
      stablehlo.return %100, %45, %91, %98, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<16xui32>, tensor<16xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 0 : (tensor<16xui32>, tensor<16xui32>) -> tensor<32xui32>
    %26 = stablehlo.constant dense<9> : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<32xui32>
    %28 = stablehlo.shift_right_logical %25, %27 : tensor<32xui32>
    %29 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<ui32>) -> tensor<32xui32>
    %31 = stablehlo.or %28, %30 : tensor<32xui32>
    %32 = stablehlo.bitcast_convert %31 : (tensor<32xui32>) -> tensor<32xf32>
    %33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %35 = stablehlo.subtract %32, %34 : tensor<32xf32>
    %36 = stablehlo.subtract %5, %3 : tensor<1xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0] : (tensor<1xf32>) -> tensor<32xf32>
    %38 = stablehlo.multiply %35, %37 : tensor<32xf32>
    %39 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf32>) -> tensor<32xf32>
    %40 = stablehlo.add %38, %39 : tensor<32xf32>
    %41 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf32>) -> tensor<32xf32>
    %42 = stablehlo.maximum %41, %40 : tensor<32xf32>
    %43 = stablehlo.custom_call @check.eq(%42, %1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<i1>
    return %43 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<32xf32> {
    %0 = stablehlo.constant dense<[2.823540e-01, 0.299847364, 0.244365454, 0.256954193, 0.0448132753, 0.991990923, 0.773812651, 0.798774958, 0.720505834, 8.353770e-01, 0.607188225, 0.692148685, 0.0396916866, 0.230736732, 0.488599062, 0.885657191, 0.702889085, 0.875597953, 0.837782382, 3.822130e-01, 0.930974722, 0.162423134, 0.411574602, 0.193137169, 0.896075964, 0.963304162, 0.0231051445, 0.538873792, 0.694255828, 0.779361486, 0.724679947, 0.399741054]> : tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}
