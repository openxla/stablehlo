// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<5x4xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %6 = stablehlo.iota dim = 0 : tensor<20xui32>
    %7 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<10> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<20xui32>) -> tensor<10xui32>
    %12 = "stablehlo.slice"(%6) {limit_indices = dense<20> : tensor<1xi64>, start_indices = dense<10> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<20xui32>) -> tensor<10xui32>
    %13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %15 = stablehlo.xor %8, %10 : tensor<ui32>
    %16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %17 = stablehlo.xor %15, %16 : tensor<ui32>
    %18 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %19 = stablehlo.add %11, %18 : tensor<10xui32>
    %20 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %21 = stablehlo.add %12, %20 : tensor<10xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %10, %iterArg_4 = %17, %iterArg_5 = %8, %iterArg_6 = %13, %iterArg_7 = %14) : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %45 = stablehlo.constant dense<5> : tensor<i32>
      %46 = stablehlo.compare  LT, %iterArg, %45,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %46 : tensor<i1>
    } do {
      %45 = stablehlo.constant dense<1> : tensor<i32>
      %46 = stablehlo.add %iterArg_0, %45 : tensor<i32>
      %47 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %48 = stablehlo.reshape %47 : (tensor<1xui32>) -> tensor<ui32>
      %49 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<10xui32>
      %50 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %51 = stablehlo.shift_left %iterArg_2, %50 : tensor<10xui32>
      %52 = stablehlo.constant dense<32> : tensor<ui32>
      %53 = stablehlo.subtract %52, %48 : tensor<ui32>
      %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %55 = stablehlo.shift_right_logical %iterArg_2, %54 : tensor<10xui32>
      %56 = stablehlo.or %51, %55 : tensor<10xui32>
      %57 = stablehlo.xor %49, %56 : tensor<10xui32>
      %58 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %59 = stablehlo.reshape %58 : (tensor<1xui32>) -> tensor<ui32>
      %60 = stablehlo.add %49, %57 : tensor<10xui32>
      %61 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %62 = stablehlo.shift_left %57, %61 : tensor<10xui32>
      %63 = stablehlo.constant dense<32> : tensor<ui32>
      %64 = stablehlo.subtract %63, %59 : tensor<ui32>
      %65 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %66 = stablehlo.shift_right_logical %57, %65 : tensor<10xui32>
      %67 = stablehlo.or %62, %66 : tensor<10xui32>
      %68 = stablehlo.xor %60, %67 : tensor<10xui32>
      %69 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %70 = stablehlo.reshape %69 : (tensor<1xui32>) -> tensor<ui32>
      %71 = stablehlo.add %60, %68 : tensor<10xui32>
      %72 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %73 = stablehlo.shift_left %68, %72 : tensor<10xui32>
      %74 = stablehlo.constant dense<32> : tensor<ui32>
      %75 = stablehlo.subtract %74, %70 : tensor<ui32>
      %76 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %77 = stablehlo.shift_right_logical %68, %76 : tensor<10xui32>
      %78 = stablehlo.or %73, %77 : tensor<10xui32>
      %79 = stablehlo.xor %71, %78 : tensor<10xui32>
      %80 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %81 = stablehlo.reshape %80 : (tensor<1xui32>) -> tensor<ui32>
      %82 = stablehlo.add %71, %79 : tensor<10xui32>
      %83 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %84 = stablehlo.shift_left %79, %83 : tensor<10xui32>
      %85 = stablehlo.constant dense<32> : tensor<ui32>
      %86 = stablehlo.subtract %85, %81 : tensor<ui32>
      %87 = stablehlo.broadcast_in_dim %86, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %88 = stablehlo.shift_right_logical %79, %87 : tensor<10xui32>
      %89 = stablehlo.or %84, %88 : tensor<10xui32>
      %90 = stablehlo.xor %82, %89 : tensor<10xui32>
      %91 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %92 = stablehlo.add %82, %91 : tensor<10xui32>
      %93 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %94 = stablehlo.add %90, %93 : tensor<10xui32>
      %95 = stablehlo.constant dense<1> : tensor<i32>
      %96 = stablehlo.add %iterArg_0, %95 : tensor<i32>
      %97 = stablehlo.convert %96 : (tensor<i32>) -> tensor<ui32>
      %98 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %99 = stablehlo.add %94, %98 : tensor<10xui32>
      %100 = stablehlo.constant dense<1> : tensor<i32>
      %101 = stablehlo.add %iterArg, %100 : tensor<i32>
      stablehlo.return %101, %46, %92, %99, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 0 : (tensor<10xui32>, tensor<10xui32>) -> tensor<20xui32>
    %26 = stablehlo.reshape %25 : (tensor<20xui32>) -> tensor<5x4xui32>
    %27 = stablehlo.constant dense<9> : tensor<ui32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<ui32>) -> tensor<5x4xui32>
    %29 = stablehlo.shift_right_logical %26, %28 : tensor<5x4xui32>
    %30 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<ui32>) -> tensor<5x4xui32>
    %32 = stablehlo.or %29, %31 : tensor<5x4xui32>
    %33 = stablehlo.bitcast_convert %32 : (tensor<5x4xui32>) -> tensor<5x4xf32>
    %34 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<5x4xf32>
    %36 = stablehlo.subtract %33, %35 : tensor<5x4xf32>
    %37 = stablehlo.subtract %5, %3 : tensor<1x1xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<5x4xf32>
    %39 = stablehlo.multiply %36, %38 : tensor<5x4xf32>
    %40 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<5x4xf32>
    %41 = stablehlo.add %39, %40 : tensor<5x4xf32>
    %42 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<5x4xf32>
    %43 = stablehlo.maximum %42, %41 : tensor<5x4xf32>
    %44 = stablehlo.custom_call @check.eq(%43, %1) : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<i1>
    return %44 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<5x4xf32> {
    %0 = stablehlo.constant dense<[[0.848995924, 0.163511038, 0.54098928, 0.144006252], [0.0662380457, 0.724974871, 0.793085575, 0.132232547], [0.777481675, 0.0385639668, 0.425514221, 0.399323106], [0.588744521, 0.44528532, 0.921235442, 0.373255253], [0.0618017912, 0.160264134, 0.264244676, 0.533137918]]> : tensor<5x4xf32>
    return %0 : tensor<5x4xf32>
  }
}
