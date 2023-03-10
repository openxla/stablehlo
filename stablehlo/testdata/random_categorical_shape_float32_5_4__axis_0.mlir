// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui32>, tensor<5x4xf32>)
    %1 = call @expected() : () -> tensor<4xi32>
    %2 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %6 = stablehlo.iota dim = 0 : tensor<20xui32>
    %7 = "stablehlo.slice"(%0#0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%0#0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
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
      %51 = stablehlo.constant dense<5> : tensor<i32>
      %52 = stablehlo.compare  LT, %iterArg, %51,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %52 : tensor<i1>
    } do {
      %51 = stablehlo.constant dense<1> : tensor<i32>
      %52 = stablehlo.add %iterArg_0, %51 : tensor<i32>
      %53 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %54 = stablehlo.reshape %53 : (tensor<1xui32>) -> tensor<ui32>
      %55 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<10xui32>
      %56 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %57 = stablehlo.shift_left %iterArg_2, %56 : tensor<10xui32>
      %58 = stablehlo.constant dense<32> : tensor<ui32>
      %59 = stablehlo.subtract %58, %54 : tensor<ui32>
      %60 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %61 = stablehlo.shift_right_logical %iterArg_2, %60 : tensor<10xui32>
      %62 = stablehlo.or %57, %61 : tensor<10xui32>
      %63 = stablehlo.xor %55, %62 : tensor<10xui32>
      %64 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %65 = stablehlo.reshape %64 : (tensor<1xui32>) -> tensor<ui32>
      %66 = stablehlo.add %55, %63 : tensor<10xui32>
      %67 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %68 = stablehlo.shift_left %63, %67 : tensor<10xui32>
      %69 = stablehlo.constant dense<32> : tensor<ui32>
      %70 = stablehlo.subtract %69, %65 : tensor<ui32>
      %71 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %72 = stablehlo.shift_right_logical %63, %71 : tensor<10xui32>
      %73 = stablehlo.or %68, %72 : tensor<10xui32>
      %74 = stablehlo.xor %66, %73 : tensor<10xui32>
      %75 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %76 = stablehlo.reshape %75 : (tensor<1xui32>) -> tensor<ui32>
      %77 = stablehlo.add %66, %74 : tensor<10xui32>
      %78 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %79 = stablehlo.shift_left %74, %78 : tensor<10xui32>
      %80 = stablehlo.constant dense<32> : tensor<ui32>
      %81 = stablehlo.subtract %80, %76 : tensor<ui32>
      %82 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %83 = stablehlo.shift_right_logical %74, %82 : tensor<10xui32>
      %84 = stablehlo.or %79, %83 : tensor<10xui32>
      %85 = stablehlo.xor %77, %84 : tensor<10xui32>
      %86 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %87 = stablehlo.reshape %86 : (tensor<1xui32>) -> tensor<ui32>
      %88 = stablehlo.add %77, %85 : tensor<10xui32>
      %89 = stablehlo.broadcast_in_dim %87, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %90 = stablehlo.shift_left %85, %89 : tensor<10xui32>
      %91 = stablehlo.constant dense<32> : tensor<ui32>
      %92 = stablehlo.subtract %91, %87 : tensor<ui32>
      %93 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %94 = stablehlo.shift_right_logical %85, %93 : tensor<10xui32>
      %95 = stablehlo.or %90, %94 : tensor<10xui32>
      %96 = stablehlo.xor %88, %95 : tensor<10xui32>
      %97 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %98 = stablehlo.add %88, %97 : tensor<10xui32>
      %99 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %100 = stablehlo.add %96, %99 : tensor<10xui32>
      %101 = stablehlo.constant dense<1> : tensor<i32>
      %102 = stablehlo.add %iterArg_0, %101 : tensor<i32>
      %103 = stablehlo.convert %102 : (tensor<i32>) -> tensor<ui32>
      %104 = stablehlo.broadcast_in_dim %103, dims = [] : (tensor<ui32>) -> tensor<10xui32>
      %105 = stablehlo.add %100, %104 : tensor<10xui32>
      %106 = stablehlo.constant dense<1> : tensor<i32>
      %107 = stablehlo.add %iterArg, %106 : tensor<i32>
      stablehlo.return %107, %52, %98, %105, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<10xui32>, tensor<10xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
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
    %44 = stablehlo.log %43 : tensor<5x4xf32>
    %45 = stablehlo.negate %44 : tensor<5x4xf32>
    %46 = stablehlo.log %45 : tensor<5x4xf32>
    %47 = stablehlo.negate %46 : tensor<5x4xf32>
    %48 = stablehlo.add %47, %0#1 : tensor<5x4xf32>
    %49 = call @argmax(%48) : (tensor<5x4xf32>) -> tensor<4xi32>
    %50 = stablehlo.custom_call @check.eq(%49, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<i1>
    return %50 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui32>, tensor<5x4xf32>) {
    %0 = stablehlo.constant dense<[42, 43]> : tensor<2xui32>
    %1 = stablehlo.constant dense<[[1.96323013, -0.390677154, 0.503751457, 0.881054341], [5.00193501, -4.74096632, 0.0176270455, -3.80246925], [-0.257672608, 1.18176281, -0.455962569, 2.99298453], [1.1658287, 0.760262966, 2.20987248, 1.36728334], [2.02152896, 2.88497472, 0.670826613, 4.52177095]]> : tensor<5x4xf32>
    return %0, %1 : tensor<2xui32>, tensor<5x4xf32>
  }
  func.func private @expected() -> tensor<4xi32> {
    %0 = stablehlo.constant dense<[1, 4, 3, 4]> : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
  func.func private @argmax(%arg0: tensor<5x4xf32>) -> tensor<4xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<5x4xi32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<5x4xf32>, tensor<5x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
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
    return %3#1 : tensor<4xi32>
  }
}
