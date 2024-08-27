// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x4xf64>
    %1 = call @expected() : () -> tensor<5xi64>
    %c = stablehlo.constant dense<42> : tensor<i64>
    %c_0 = stablehlo.constant dense<32> : tensor<i64>
    %2 = stablehlo.shift_right_logical %c, %c_0 : tensor<i64>
    %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<ui32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %c_1 = stablehlo.constant dense<4294967295> : tensor<ui32>
    %5 = stablehlo.convert %c_1 : (tensor<ui32>) -> tensor<i64>
    %6 = stablehlo.and %c, %5 : tensor<i64>
    %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<ui32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %9 = stablehlo.concatenate %4, %8, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %10 = call @_gumbel(%9) : (tensor<2xui32>) -> tensor<5x4xf64>
    %11 = stablehlo.add %10, %0 : tensor<5x4xf64>
    %12 = call @argmax(%11) : (tensor<5x4xf64>) -> tensor<5xi64>
    stablehlo.custom_call @check.expect_eq(%12, %1) {has_side_effect = true} : (tensor<5xi64>, tensor<5xi64>) -> ()
    return %12 : tensor<5xi64>
  }
  func.func private @inputs() -> (tensor<5x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.6044160679950643, 2.9321362344077575, -3.2103477133467098, -1.8187160634972799], [2.1577879203212329, 0.94549340118604897, -1.2300833153827209, 6.037781617043855], [2.6481661417030908, 0.76780148062618547, 0.25212015749028915, -2.0297729471187482], [-8.2722376135920346, 2.0099702230233962, 3.0598081711609653, -1.2354732207389678], [-3.2511804161164988, 1.7202350343151254, 3.8548454509404677, -0.67754275745650916]]> : tensor<5x4xf64>
    return %cst : tensor<5x4xf64>
  }
  func.func private @expected() -> (tensor<5xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[1, 3, 0, 2, 2]> : tensor<5xi64>
    return %c : tensor<5xi64>
  }
  func.func private @_gumbel(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<5x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<2.2250738585072014E-308> : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = call @_uniform(%arg0, %cst, %cst_0) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<5x4xf64>
    %1 = stablehlo.log %0 : tensor<5x4xf64>
    %2 = stablehlo.negate %1 : tensor<5x4xf64>
    %3 = stablehlo.log %2 : tensor<5x4xf64>
    %4 = stablehlo.negate %3 : tensor<5x4xf64>
    return %4 : tensor<5x4xf64>
  }
  func.func private @_uniform(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<f64> {mhlo.layout_mode = "default"}, %arg2: tensor<f64> {mhlo.layout_mode = "default"}) -> (tensor<5x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg2 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %3 = stablehlo.iota dim = 0 : tensor<40xui32>
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %7 = stablehlo.reshape %6 : (tensor<1xui32>) -> tensor<ui32>
    %8 = stablehlo.slice %3 [0:20] : (tensor<40xui32>) -> tensor<20xui32>
    %9 = stablehlo.slice %3 [20:40] : (tensor<40xui32>) -> tensor<20xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %10 = stablehlo.xor %5, %7 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %11 = stablehlo.xor %10, %c_1 : tensor<ui32>
    %12 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %13 = stablehlo.add %8, %12 : tensor<20xui32>
    %14 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %15 = stablehlo.add %9, %14 : tensor<20xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %16:9 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %c_2, %iterArg_8 = %13, %iterArg_9 = %15, %iterArg_10 = %7, %iterArg_11 = %11, %iterArg_12 = %5, %iterArg_13 = %c, %iterArg_14 = %c_0) : tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_15 = stablehlo.constant dense<5> : tensor<i64>
      %40 = stablehlo.compare  LT, %iterArg, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %40 : tensor<i1>
    } do {
      %40:8 = func.call @None(%iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14) : (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_15 = stablehlo.constant dense<1> : tensor<i64>
      %41 = stablehlo.add %iterArg, %c_15 : tensor<i64>
      stablehlo.return %41, %40#0, %40#1, %40#2, %40#3, %40#4, %40#5, %40#6, %40#7 : tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %17 = stablehlo.concatenate %16#2, %16#3, dim = 0 : (tensor<20xui32>, tensor<20xui32>) -> tensor<40xui32>
    %18 = stablehlo.slice %17 [0:20] : (tensor<40xui32>) -> tensor<20xui32>
    %19 = stablehlo.slice %17 [20:40] : (tensor<40xui32>) -> tensor<20xui32>
    %20 = stablehlo.convert %18 : (tensor<20xui32>) -> tensor<20xui64>
    %21 = stablehlo.convert %19 : (tensor<20xui32>) -> tensor<20xui64>
    %c_4 = stablehlo.constant dense<32> : tensor<ui64>
    %22 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<20xui64>
    %23 = stablehlo.shift_left %20, %22 : tensor<20xui64>
    %24 = stablehlo.or %23, %21 : tensor<20xui64>
    %25 = stablehlo.reshape %24 : (tensor<20xui64>) -> tensor<5x4xui64>
    %c_5 = stablehlo.constant dense<12> : tensor<ui64>
    %26 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui64>) -> tensor<5x4xui64>
    %27 = stablehlo.shift_right_logical %25, %26 : tensor<5x4xui64>
    %c_6 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %28 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui64>) -> tensor<5x4xui64>
    %29 = stablehlo.or %27, %28 : tensor<5x4xui64>
    %30 = stablehlo.bitcast_convert %29 : (tensor<5x4xui64>) -> tensor<5x4xf64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %31 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<5x4xf64>
    %32 = stablehlo.subtract %30, %31 : tensor<5x4xf64>
    %33 = stablehlo.subtract %2, %1 : tensor<1x1xf64>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %35 = stablehlo.multiply %32, %34 : tensor<5x4xf64>
    %36 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %37 = stablehlo.add %35, %36 : tensor<5x4xf64>
    %38 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %39 = stablehlo.maximum %38, %37 : tensor<5x4xf64>
    return %39 : tensor<5x4xf64>
  }
  func.func private @None(%arg0: tensor<i64>, %arg1: tensor<20xui32>, %arg2: tensor<20xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<20xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<20xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<20xui32>
    %9 = stablehlo.or %5, %8 : tensor<20xui32>
    %10 = stablehlo.xor %3, %9 : tensor<20xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<20xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<20xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<20xui32>
    %19 = stablehlo.or %15, %18 : tensor<20xui32>
    %20 = stablehlo.xor %13, %19 : tensor<20xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<20xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<20xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<20xui32>
    %29 = stablehlo.or %25, %28 : tensor<20xui32>
    %30 = stablehlo.xor %23, %29 : tensor<20xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<20xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<20xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<20xui32>
    %39 = stablehlo.or %35, %38 : tensor<20xui32>
    %40 = stablehlo.xor %33, %39 : tensor<20xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %42 = stablehlo.add %33, %41 : tensor<20xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %44 = stablehlo.add %40, %43 : tensor<20xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %48 = stablehlo.add %44, %47 : tensor<20xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
  }
  func.func private @argmax(%arg0: tensor<5x4xf64>) -> tensor<5xi64> {
    %0 = stablehlo.iota dim = 1 : tensor<5x4xi64>
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<5x4xf64>, tensor<5x4xi64>, tensor<f64>, tensor<i64>) -> (tensor<5xf64>, tensor<5xi64>)
     reducer(%arg1: tensor<f64>, %arg3: tensor<f64>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f64>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i64>
      stablehlo.return %9, %10 : tensor<f64>, tensor<i64>
    }
    return %1#1 : tensor<5xi64>
  }
}
