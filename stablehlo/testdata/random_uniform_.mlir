// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x4xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @expected() : () -> tensor<5x4xf64>
    %c = stablehlo.constant dense<42> : tensor<i64>
    %c_0 = stablehlo.constant dense<32> : tensor<i64>
    %1 = stablehlo.shift_right_logical %c, %c_0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %c_1 = stablehlo.constant dense<4294967295> : tensor<ui32>
    %4 = stablehlo.convert %c_1 : (tensor<ui32>) -> tensor<i64>
    %5 = stablehlo.and %c, %4 : tensor<i64>
    %6 = stablehlo.convert %5 : (tensor<i64>) -> tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %8 = stablehlo.concatenate %3, %7, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %9 = call @_uniform(%8, %cst, %cst_2) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<5x4xf64>
    stablehlo.custom_call @check.expect_close(%9, %0) {has_side_effect = true} : (tensor<5x4xf64>, tensor<5x4xf64>) -> ()
    return %9 : tensor<5x4xf64>
  }
  func.func private @expected() -> (tensor<5x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.52286734638827848, 0.63809277992222024, 0.48479882418789, 0.76266020446279748], [0.67996636822643519, 0.44532535364606862, 0.75625579280848321, 0.76073724858951675], [0.32504364334015712, 0.58233053152090486, 0.88008197627653684, 0.56040213468002809], [0.96747217212282344, 0.49304563867921836, 0.72374622890728268, 0.95975933869077212], [0.55588321000681051, 0.049615688944020686, 0.48405548065598669, 0.79875184812853339]]> : tensor<5x4xf64>
    return %cst : tensor<5x4xf64>
  }
  func.func private @_uniform(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<f64> {mhlo.layout_mode = "default"}, %arg2: tensor<f64> {mhlo.layout_mode = "default"}) -> (tensor<5x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.convert %arg2 : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %3 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %4 = stablehlo.iota dim = 0 : tensor<40xui32>
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = stablehlo.slice %4 [0:20] : (tensor<40xui32>) -> tensor<20xui32>
    %10 = stablehlo.slice %4 [20:40] : (tensor<40xui32>) -> tensor<20xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %11 = stablehlo.xor %6, %8 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %12 = stablehlo.xor %11, %c_1 : tensor<ui32>
    %13 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %14 = stablehlo.add %9, %13 : tensor<20xui32>
    %15 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<20xui32>
    %16 = stablehlo.add %10, %15 : tensor<20xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %17:9 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %c_2, %iterArg_8 = %14, %iterArg_9 = %16, %iterArg_10 = %8, %iterArg_11 = %12, %iterArg_12 = %6, %iterArg_13 = %c, %iterArg_14 = %c_0) : tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_15 = stablehlo.constant dense<5> : tensor<i64>
      %41 = stablehlo.compare  LT, %iterArg, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %41 : tensor<i1>
    } do {
      %41:8 = func.call @None(%iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14) : (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_15 = stablehlo.constant dense<1> : tensor<i64>
      %42 = stablehlo.add %iterArg, %c_15 : tensor<i64>
      stablehlo.return %42, %41#0, %41#1, %41#2, %41#3, %41#4, %41#5, %41#6, %41#7 : tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %18 = stablehlo.concatenate %17#2, %17#3, dim = 0 : (tensor<20xui32>, tensor<20xui32>) -> tensor<40xui32>
    %19 = stablehlo.slice %18 [0:20] : (tensor<40xui32>) -> tensor<20xui32>
    %20 = stablehlo.slice %18 [20:40] : (tensor<40xui32>) -> tensor<20xui32>
    %21 = stablehlo.convert %19 : (tensor<20xui32>) -> tensor<20xui64>
    %22 = stablehlo.convert %20 : (tensor<20xui32>) -> tensor<20xui64>
    %c_4 = stablehlo.constant dense<32> : tensor<ui64>
    %23 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<20xui64>
    %24 = stablehlo.shift_left %21, %23 : tensor<20xui64>
    %25 = stablehlo.or %24, %22 : tensor<20xui64>
    %26 = stablehlo.reshape %25 : (tensor<20xui64>) -> tensor<5x4xui64>
    %c_5 = stablehlo.constant dense<12> : tensor<ui64>
    %27 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui64>) -> tensor<5x4xui64>
    %28 = stablehlo.shift_right_logical %26, %27 : tensor<5x4xui64>
    %c_6 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %29 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui64>) -> tensor<5x4xui64>
    %30 = stablehlo.or %28, %29 : tensor<5x4xui64>
    %31 = stablehlo.bitcast_convert %30 : (tensor<5x4xui64>) -> tensor<5x4xf64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %32 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<5x4xf64>
    %33 = stablehlo.subtract %31, %32 : tensor<5x4xf64>
    %34 = stablehlo.subtract %3, %2 : tensor<1x1xf64>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %36 = stablehlo.multiply %33, %35 : tensor<5x4xf64>
    %37 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %38 = stablehlo.add %36, %37 : tensor<5x4xf64>
    %39 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %40 = stablehlo.maximum %39, %38 : tensor<5x4xf64>
    return %40 : tensor<5x4xf64>
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
}
