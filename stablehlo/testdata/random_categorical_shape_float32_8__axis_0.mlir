// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<8xf32>)
    %1 = call @expected() : () -> tensor<i32>
    %2 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.iota dim = 0 : tensor<8xui32>
    %7 = "stablehlo.slice"(%0#0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0#0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.reshape %9 : (tensor<1xui32>) -> tensor<ui32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xui32>) -> tensor<4xui32>
    %12 = "stablehlo.slice"(%6) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<4> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xui32>) -> tensor<4xui32>
    %13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %15 = stablehlo.xor %8, %10 : tensor<ui32>
    %16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %17 = stablehlo.xor %15, %16 : tensor<ui32>
    %18 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %19 = stablehlo.add %11, %18 : tensor<4xui32>
    %20 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %21 = stablehlo.add %12, %20 : tensor<4xui32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<0> : tensor<i32>
    %24:9 = stablehlo.while(%iterArg = %23, %iterArg_0 = %22, %iterArg_1 = %19, %iterArg_2 = %21, %iterArg_3 = %10, %iterArg_4 = %17, %iterArg_5 = %8, %iterArg_6 = %13, %iterArg_7 = %14) : tensor<i32>, tensor<i32>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %50 = stablehlo.constant dense<5> : tensor<i32>
      %51 = stablehlo.compare  LT, %iterArg, %50,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %51 : tensor<i1>
    } do {
      %50 = stablehlo.constant dense<1> : tensor<i32>
      %51 = stablehlo.add %iterArg_0, %50 : tensor<i32>
      %52 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %53 = stablehlo.reshape %52 : (tensor<1xui32>) -> tensor<ui32>
      %54 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<4xui32>
      %55 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %56 = stablehlo.shift_left %iterArg_2, %55 : tensor<4xui32>
      %57 = stablehlo.constant dense<32> : tensor<ui32>
      %58 = stablehlo.subtract %57, %53 : tensor<ui32>
      %59 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %60 = stablehlo.shift_right_logical %iterArg_2, %59 : tensor<4xui32>
      %61 = stablehlo.or %56, %60 : tensor<4xui32>
      %62 = stablehlo.xor %54, %61 : tensor<4xui32>
      %63 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %64 = stablehlo.reshape %63 : (tensor<1xui32>) -> tensor<ui32>
      %65 = stablehlo.add %54, %62 : tensor<4xui32>
      %66 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %67 = stablehlo.shift_left %62, %66 : tensor<4xui32>
      %68 = stablehlo.constant dense<32> : tensor<ui32>
      %69 = stablehlo.subtract %68, %64 : tensor<ui32>
      %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %71 = stablehlo.shift_right_logical %62, %70 : tensor<4xui32>
      %72 = stablehlo.or %67, %71 : tensor<4xui32>
      %73 = stablehlo.xor %65, %72 : tensor<4xui32>
      %74 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %75 = stablehlo.reshape %74 : (tensor<1xui32>) -> tensor<ui32>
      %76 = stablehlo.add %65, %73 : tensor<4xui32>
      %77 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %78 = stablehlo.shift_left %73, %77 : tensor<4xui32>
      %79 = stablehlo.constant dense<32> : tensor<ui32>
      %80 = stablehlo.subtract %79, %75 : tensor<ui32>
      %81 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %82 = stablehlo.shift_right_logical %73, %81 : tensor<4xui32>
      %83 = stablehlo.or %78, %82 : tensor<4xui32>
      %84 = stablehlo.xor %76, %83 : tensor<4xui32>
      %85 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %86 = stablehlo.reshape %85 : (tensor<1xui32>) -> tensor<ui32>
      %87 = stablehlo.add %76, %84 : tensor<4xui32>
      %88 = stablehlo.broadcast_in_dim %86, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %89 = stablehlo.shift_left %84, %88 : tensor<4xui32>
      %90 = stablehlo.constant dense<32> : tensor<ui32>
      %91 = stablehlo.subtract %90, %86 : tensor<ui32>
      %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %93 = stablehlo.shift_right_logical %84, %92 : tensor<4xui32>
      %94 = stablehlo.or %89, %93 : tensor<4xui32>
      %95 = stablehlo.xor %87, %94 : tensor<4xui32>
      %96 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %97 = stablehlo.add %87, %96 : tensor<4xui32>
      %98 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %99 = stablehlo.add %95, %98 : tensor<4xui32>
      %100 = stablehlo.constant dense<1> : tensor<i32>
      %101 = stablehlo.add %iterArg_0, %100 : tensor<i32>
      %102 = stablehlo.convert %101 : (tensor<i32>) -> tensor<ui32>
      %103 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<ui32>) -> tensor<4xui32>
      %104 = stablehlo.add %99, %103 : tensor<4xui32>
      %105 = stablehlo.constant dense<1> : tensor<i32>
      %106 = stablehlo.add %iterArg, %105 : tensor<i32>
      stablehlo.return %106, %51, %97, %104, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %25 = stablehlo.concatenate %24#2, %24#3, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %26 = stablehlo.constant dense<9> : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %28 = stablehlo.shift_right_logical %25, %27 : tensor<8xui32>
    %29 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<ui32>) -> tensor<8xui32>
    %31 = stablehlo.or %28, %30 : tensor<8xui32>
    %32 = stablehlo.bitcast_convert %31 : (tensor<8xui32>) -> tensor<8xf32>
    %33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %35 = stablehlo.subtract %32, %34 : tensor<8xf32>
    %36 = stablehlo.subtract %5, %3 : tensor<1xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0] : (tensor<1xf32>) -> tensor<8xf32>
    %38 = stablehlo.multiply %35, %37 : tensor<8xf32>
    %39 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf32>) -> tensor<8xf32>
    %40 = stablehlo.add %38, %39 : tensor<8xf32>
    %41 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf32>) -> tensor<8xf32>
    %42 = stablehlo.maximum %41, %40 : tensor<8xf32>
    %43 = stablehlo.log %42 : tensor<8xf32>
    %44 = stablehlo.negate %43 : tensor<8xf32>
    %45 = stablehlo.log %44 : tensor<8xf32>
    %46 = stablehlo.negate %45 : tensor<8xf32>
    %47 = stablehlo.add %46, %0#1 : tensor<8xf32>
    %48 = call @argmax(%47) : (tensor<8xf32>) -> tensor<i32>
    %49 = stablehlo.custom_call @check.eq(%48, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %49 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<8xf32>) {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[1.29666257, -4.96239424, -3.63283134, -1.66793478, -0.427061021, 3.17029357, -3.75322676, -1.72779012]> : tensor<8xf32>
    return %0, %1 : tensor<2xui32>, tensor<8xf32>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<5> : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @argmax(%arg0: tensor<8xf32>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<8xf32>, tensor<8xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<f32>, tensor<i32>
    }
    return %3#1 : tensor<i32>
  }
}
