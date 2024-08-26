// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<f32>
    %1 = call @expected() : () -> tensor<f64>
    %2 = call @"<lambda>"(%0) : (tensor<f32>) -> tensor<f64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<f64>, tensor<f64>) -> ()
    return %2 : tensor<f64>
  }
  func.func private @inputs() -> (tensor<f32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-2.72118402> : tensor<f32>
    return %cst : tensor<f32>
  }
  func.func private @expected() -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0xFFF8000000000000> : tensor<f64>
    return %cst : tensor<f64>
  }
  func.func private @"<lambda>"(%arg0: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<42> : tensor<i64>
    %c_0 = stablehlo.constant dense<32> : tensor<i64>
    %0 = stablehlo.shift_right_logical %c, %c_0 : tensor<i64>
    %1 = stablehlo.convert %0 : (tensor<i64>) -> tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %c_1 = stablehlo.constant dense<4294967295> : tensor<ui32>
    %3 = stablehlo.convert %c_1 : (tensor<ui32>) -> tensor<i64>
    %4 = stablehlo.and %c, %3 : tensor<i64>
    %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<ui32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %7 = stablehlo.concatenate %2, %6, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %8 = call @_gamma(%7, %arg0) : (tensor<2xui32>, tensor<f32>) -> tensor<f64>
    return %8 : tensor<f64>
  }
  func.func private @_gamma(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<f64>
    %1 = stablehlo.reshape %arg0 : (tensor<2xui32>) -> tensor<1x2xui32>
    %2 = call @_threefry_split(%1) : (tensor<1x2xui32>) -> tensor<1x1x2xui32>
    %3 = stablehlo.reshape %2 : (tensor<1x1x2xui32>) -> tensor<1x2xui32>
    %4 = stablehlo.reshape %0 : (tensor<f64>) -> tensor<1xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %6:4 = stablehlo.while(%iterArg = %3, %iterArg_0 = %4, %iterArg_1 = %c, %iterArg_2 = %5) : tensor<1x2xui32>, tensor<1xf64>, tensor<i64>, tensor<1xf64>
     cond {
      %c_3 = stablehlo.constant dense<1> : tensor<i64>
      %8 = stablehlo.compare  LT, %iterArg_1, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    } do {
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %8 = stablehlo.compare  LT, %iterArg_1, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = stablehlo.convert %iterArg_1 : tensor<i64>
      %c_4 = stablehlo.constant dense<1> : tensor<i64>
      %10 = stablehlo.add %9, %c_4 : tensor<i64>
      %11 = stablehlo.select %8, %10, %iterArg_1 : tensor<i1>, tensor<i64>
      %c_5 = stablehlo.constant dense<0> : tensor<i64>
      %12 = stablehlo.dynamic_slice %iterArg, %11, %c_5, sizes = [1, 2] : (tensor<1x2xui32>, tensor<i64>, tensor<i64>) -> tensor<1x2xui32>
      %13 = stablehlo.reshape %12 : (tensor<1x2xui32>) -> tensor<2xui32>
      %c_6 = stablehlo.constant dense<0> : tensor<i64>
      %14 = stablehlo.compare  LT, %iterArg_1, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %15 = stablehlo.convert %iterArg_1 : tensor<i64>
      %c_7 = stablehlo.constant dense<1> : tensor<i64>
      %16 = stablehlo.add %15, %c_7 : tensor<i64>
      %17 = stablehlo.select %14, %16, %iterArg_1 : tensor<i1>, tensor<i64>
      %18 = stablehlo.dynamic_slice %iterArg_0, %17, sizes = [1] : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
      %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
      %20 = func.call @None_0(%13, %19) : (tensor<2xui32>, tensor<f64>) -> tensor<f64>
      %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %c_8 = stablehlo.constant dense<0> : tensor<i64>
      %22 = stablehlo.compare  LT, %iterArg_1, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %23 = stablehlo.convert %iterArg_1 : tensor<i64>
      %c_9 = stablehlo.constant dense<1> : tensor<i64>
      %24 = stablehlo.add %23, %c_9 : tensor<i64>
      %25 = stablehlo.select %22, %24, %iterArg_1 : tensor<i1>, tensor<i64>
      %26 = stablehlo.dynamic_update_slice %iterArg_2, %21, %25 : (tensor<1xf64>, tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
      %c_10 = stablehlo.constant dense<1> : tensor<i64>
      %27 = stablehlo.add %iterArg_1, %c_10 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_0, %27, %26 : tensor<1x2xui32>, tensor<1xf64>, tensor<i64>, tensor<1xf64>
    }
    %7 = stablehlo.reshape %6#3 : (tensor<1xf64>) -> tensor<f64>
    return %7 : tensor<f64>
  }
  func.func private @_threefry_split(%arg0: tensor<1x2xui32> {mhlo.layout_mode = "default"}) -> (tensor<1x1x2xui32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<2xui32>
    %1 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<1x2xui32>) -> tensor<1x1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1x1xui32>) -> tensor<1xui32>
    %3 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<1x2xui32>) -> tensor<1x1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1x1xui32>) -> tensor<1xui32>
    %5 = stablehlo.slice %0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.slice %0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %7 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<1xui32>) -> tensor<1x1xui32>
    %8 = stablehlo.broadcast_in_dim %6, dims = [1] : (tensor<1xui32>) -> tensor<1x1xui32>
    %9 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %10 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %11 = stablehlo.xor %9, %10 : tensor<1x1xui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %12 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %13 = stablehlo.xor %11, %12 : tensor<1x1xui32>
    %14 = stablehlo.add %7, %9 : tensor<1x1xui32>
    %15 = stablehlo.add %8, %10 : tensor<1x1xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %16:9 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c_2, %iterArg_5 = %14, %iterArg_6 = %15, %iterArg_7 = %10, %iterArg_8 = %13, %iterArg_9 = %9, %iterArg_10 = %c, %iterArg_11 = %c_0) : tensor<i64>, tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_12 = stablehlo.constant dense<5> : tensor<i64>
      %19 = stablehlo.compare  LT, %iterArg, %c_12,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %19 : tensor<i1>
    } do {
      %19:8 = func.call @None(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %20 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %20, %19#0, %19#1, %19#2, %19#3, %19#4, %19#5, %19#6, %19#7 : tensor<i64>, tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>
    }
    %17 = stablehlo.concatenate %16#2, %16#3, dim = 1 : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x2xui32>
    %18 = stablehlo.reshape %17 : (tensor<1x2xui32>) -> tensor<1x1x2xui32>
    return %18 : tensor<1x1x2xui32>
  }
  func.func private @None(%arg0: tensor<i64>, %arg1: tensor<1x1xui32>, %arg2: tensor<1x1xui32>, %arg3: tensor<1x1xui32>, %arg4: tensor<1x1xui32>, %arg5: tensor<1x1xui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<1x1xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<1x1xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<1x1xui32>
    %9 = stablehlo.or %5, %8 : tensor<1x1xui32>
    %10 = stablehlo.xor %3, %9 : tensor<1x1xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<1x1xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<1x1xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<1x1xui32>
    %19 = stablehlo.or %15, %18 : tensor<1x1xui32>
    %20 = stablehlo.xor %13, %19 : tensor<1x1xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<1x1xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<1x1xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<1x1xui32>
    %29 = stablehlo.or %25, %28 : tensor<1x1xui32>
    %30 = stablehlo.xor %23, %29 : tensor<1x1xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<1x1xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<1x1xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<1x1xui32>
    %39 = stablehlo.or %35, %38 : tensor<1x1xui32>
    %40 = stablehlo.xor %33, %39 : tensor<1x1xui32>
    %41 = stablehlo.add %33, %arg3 : tensor<1x1xui32>
    %42 = stablehlo.add %40, %arg4 : tensor<1x1xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %43 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %44 = stablehlo.convert %43 : (tensor<i64>) -> tensor<ui32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<ui32>) -> tensor<1x1xui32>
    %46 = stablehlo.add %42, %45 : tensor<1x1xui32>
    return %0, %41, %46, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>
  }
  func.func private @None_0(%arg0: tensor<2xui32>, %arg1: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.compare  GE, %arg1, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.add %arg1, %cst_0 : tensor<f64>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<i1>, tensor<f64>
    %cst_1 = stablehlo.constant dense<0.33333333333333331> : tensor<f64>
    %3 = stablehlo.subtract %2, %cst_1 : tensor<f64>
    %4 = stablehlo.sqrt %3 : tensor<f64>
    %cst_2 = stablehlo.constant dense<0.33333333333333331> : tensor<f64>
    %5 = stablehlo.divide %cst_2, %4 : tensor<f64>
    %6 = call @_threefry_split_1(%arg0) : (tensor<2xui32>) -> tensor<2x2xui32>
    %7 = stablehlo.slice %6 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %8 = stablehlo.reshape %7 : (tensor<1x2xui32>) -> tensor<2xui32>
    %9 = stablehlo.slice %6 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %10 = stablehlo.reshape %9 : (tensor<1x2xui32>) -> tensor<2xui32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %11:6 = stablehlo.while(%iterArg = %3, %iterArg_11 = %5, %iterArg_12 = %8, %iterArg_13 = %cst_3, %iterArg_14 = %cst_4, %iterArg_15 = %cst_5) : tensor<f64>, tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>, tensor<f64>
     cond {
      %19 = stablehlo.multiply %iterArg_13, %iterArg_13 : tensor<f64>
      %cst_16 = stablehlo.constant dense<3.310000e-02> : tensor<f64>
      %20 = stablehlo.multiply %cst_16, %19 : tensor<f64>
      %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %21 = stablehlo.subtract %cst_17, %20 : tensor<f64>
      %22 = stablehlo.compare  GE, %iterArg_15, %21,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %23 = stablehlo.log %iterArg_15 : tensor<f64>
      %cst_18 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
      %24 = stablehlo.multiply %iterArg_13, %cst_18 : tensor<f64>
      %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %25 = stablehlo.subtract %cst_19, %iterArg_14 : tensor<f64>
      %26 = stablehlo.log %iterArg_14 : tensor<f64>
      %27 = stablehlo.add %25, %26 : tensor<f64>
      %28 = stablehlo.multiply %iterArg, %27 : tensor<f64>
      %29 = stablehlo.add %24, %28 : tensor<f64>
      %30 = stablehlo.compare  GE, %23, %29,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %31 = stablehlo.and %22, %30 : tensor<i1>
      stablehlo.return %31 : tensor<i1>
    } do {
      %19 = func.call @_threefry_split_3(%iterArg_12) : (tensor<2xui32>) -> tensor<3x2xui32>
      %20 = stablehlo.slice %19 [0:1, 0:2] : (tensor<3x2xui32>) -> tensor<1x2xui32>
      %21 = stablehlo.reshape %20 : (tensor<1x2xui32>) -> tensor<2xui32>
      %22 = stablehlo.slice %19 [1:2, 0:2] : (tensor<3x2xui32>) -> tensor<1x2xui32>
      %23 = stablehlo.reshape %22 : (tensor<1x2xui32>) -> tensor<2xui32>
      %24 = stablehlo.slice %19 [2:3, 0:2] : (tensor<3x2xui32>) -> tensor<1x2xui32>
      %25 = stablehlo.reshape %24 : (tensor<1x2xui32>) -> tensor<2xui32>
      %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %cst_17 = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
      %26:4 = stablehlo.while(%iterArg_20 = %iterArg_11, %iterArg_21 = %23, %iterArg_22 = %cst_16, %iterArg_23 = %cst_17) : tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>
       cond {
        %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %31 = stablehlo.compare  LE, %iterArg_23, %cst_24,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        stablehlo.return %31 : tensor<i1>
      } do {
        %31 = func.call @_threefry_split_1(%iterArg_21) : (tensor<2xui32>) -> tensor<2x2xui32>
        %32 = stablehlo.slice %31 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
        %33 = stablehlo.reshape %32 : (tensor<1x2xui32>) -> tensor<2xui32>
        %34 = stablehlo.slice %31 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
        %35 = stablehlo.reshape %34 : (tensor<1x2xui32>) -> tensor<2xui32>
        %36 = func.call @_normal(%35) : (tensor<2xui32>) -> tensor<f64>
        %37 = stablehlo.multiply %36, %iterArg_20 : tensor<f64>
        %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %38 = stablehlo.add %cst_24, %37 : tensor<f64>
        stablehlo.return %iterArg_20, %33, %36, %38 : tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>
      }
      %27 = stablehlo.multiply %26#2, %26#2 : tensor<f64>
      %28 = stablehlo.multiply %26#3, %26#3 : tensor<f64>
      %29 = stablehlo.multiply %28, %26#3 : tensor<f64>
      %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %30 = func.call @_uniform_6(%25, %cst_18, %cst_19) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>
      stablehlo.return %iterArg, %iterArg_11, %21, %27, %29, %30 : tensor<f64>, tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>, tensor<f64>
    }
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %12 = call @_uniform_6(%10, %cst_6, %cst_7) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %13 = stablehlo.subtract %cst_8, %12 : tensor<f64>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %14 = stablehlo.divide %cst_9, %arg1 : tensor<f64>
    %15 = stablehlo.power %13, %14 : tensor<f64>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %16 = stablehlo.select %0, %cst_10, %15 : tensor<i1>, tensor<f64>
    %17 = stablehlo.multiply %3, %11#4 : tensor<f64>
    %18 = stablehlo.multiply %17, %16 : tensor<f64>
    return %18 : tensor<f64>
  }
  func.func private @_threefry_split_1(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<2x2xui32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<4xui32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %0 [0:2] : (tensor<4xui32>) -> tensor<2xui32>
    %6 = stablehlo.slice %0 [2:4] : (tensor<4xui32>) -> tensor<2xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %7 = stablehlo.xor %2, %4 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %8 = stablehlo.xor %7, %c_1 : tensor<ui32>
    %9 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %10 = stablehlo.add %5, %9 : tensor<2xui32>
    %11 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %12 = stablehlo.add %6, %11 : tensor<2xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %13:9 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c_2, %iterArg_5 = %10, %iterArg_6 = %12, %iterArg_7 = %4, %iterArg_8 = %8, %iterArg_9 = %2, %iterArg_10 = %c, %iterArg_11 = %c_0) : tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_12 = stablehlo.constant dense<5> : tensor<i64>
      %16 = stablehlo.compare  LT, %iterArg, %c_12,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %16 : tensor<i1>
    } do {
      %16:8 = func.call @None_2(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %17 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %17, %16#0, %16#1, %16#2, %16#3, %16#4, %16#5, %16#6, %16#7 : tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %14 = stablehlo.concatenate %13#2, %13#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %15 = stablehlo.reshape %14 : (tensor<4xui32>) -> tensor<2x2xui32>
    return %15 : tensor<2x2xui32>
  }
  func.func private @None_2(%arg0: tensor<i64>, %arg1: tensor<2xui32>, %arg2: tensor<2xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<2xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<2xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<2xui32>
    %9 = stablehlo.or %5, %8 : tensor<2xui32>
    %10 = stablehlo.xor %3, %9 : tensor<2xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<2xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<2xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<2xui32>
    %19 = stablehlo.or %15, %18 : tensor<2xui32>
    %20 = stablehlo.xor %13, %19 : tensor<2xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<2xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<2xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<2xui32>
    %29 = stablehlo.or %25, %28 : tensor<2xui32>
    %30 = stablehlo.xor %23, %29 : tensor<2xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<2xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<2xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<2xui32>
    %39 = stablehlo.or %35, %38 : tensor<2xui32>
    %40 = stablehlo.xor %33, %39 : tensor<2xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %42 = stablehlo.add %33, %41 : tensor<2xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %44 = stablehlo.add %40, %43 : tensor<2xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %48 = stablehlo.add %44, %47 : tensor<2xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
  }
  func.func private @_threefry_split_3(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<3x2xui32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<6xui32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %0 [0:3] : (tensor<6xui32>) -> tensor<3xui32>
    %6 = stablehlo.slice %0 [3:6] : (tensor<6xui32>) -> tensor<3xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %7 = stablehlo.xor %2, %4 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %8 = stablehlo.xor %7, %c_1 : tensor<ui32>
    %9 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %10 = stablehlo.add %5, %9 : tensor<3xui32>
    %11 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %12 = stablehlo.add %6, %11 : tensor<3xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %13:9 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c_2, %iterArg_5 = %10, %iterArg_6 = %12, %iterArg_7 = %4, %iterArg_8 = %8, %iterArg_9 = %2, %iterArg_10 = %c, %iterArg_11 = %c_0) : tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_12 = stablehlo.constant dense<5> : tensor<i64>
      %16 = stablehlo.compare  LT, %iterArg, %c_12,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %16 : tensor<i1>
    } do {
      %16:8 = func.call @None_4(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %17 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %17, %16#0, %16#1, %16#2, %16#3, %16#4, %16#5, %16#6, %16#7 : tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %14 = stablehlo.concatenate %13#2, %13#3, dim = 0 : (tensor<3xui32>, tensor<3xui32>) -> tensor<6xui32>
    %15 = stablehlo.reshape %14 : (tensor<6xui32>) -> tensor<3x2xui32>
    return %15 : tensor<3x2xui32>
  }
  func.func private @None_4(%arg0: tensor<i64>, %arg1: tensor<3xui32>, %arg2: tensor<3xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<3xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<3xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<3xui32>
    %9 = stablehlo.or %5, %8 : tensor<3xui32>
    %10 = stablehlo.xor %3, %9 : tensor<3xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<3xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<3xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<3xui32>
    %19 = stablehlo.or %15, %18 : tensor<3xui32>
    %20 = stablehlo.xor %13, %19 : tensor<3xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<3xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<3xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<3xui32>
    %29 = stablehlo.or %25, %28 : tensor<3xui32>
    %30 = stablehlo.xor %23, %29 : tensor<3xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<3xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<3xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<3xui32>
    %39 = stablehlo.or %35, %38 : tensor<3xui32>
    %40 = stablehlo.xor %33, %39 : tensor<3xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %42 = stablehlo.add %33, %41 : tensor<3xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %44 = stablehlo.add %40, %43 : tensor<3xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %48 = stablehlo.add %44, %47 : tensor<3xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
  }
  func.func private @_normal(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %0 = call @_normal_real(%arg0) : (tensor<2xui32>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @_normal_real(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-0.99999999999999988> : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = call @_uniform(%arg0, %cst, %cst_0) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %1 = chlo.erf_inv %0 : tensor<f64> -> tensor<f64>
    %cst_1 = stablehlo.constant dense<1.4142135623730951> : tensor<f64>
    %2 = stablehlo.multiply %cst_1, %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @_uniform(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<f64> {mhlo.layout_mode = "default"}, %arg2: tensor<f64> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<2xui32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.slice %0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %7 = stablehlo.xor %2, %4 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %8 = stablehlo.xor %7, %c_1 : tensor<ui32>
    %9 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %10 = stablehlo.add %5, %9 : tensor<1xui32>
    %11 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %12 = stablehlo.add %6, %11 : tensor<1xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %13:9 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %c_2, %iterArg_8 = %10, %iterArg_9 = %12, %iterArg_10 = %4, %iterArg_11 = %8, %iterArg_12 = %2, %iterArg_13 = %c, %iterArg_14 = %c_0) : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_15 = stablehlo.constant dense<5> : tensor<i64>
      %32 = stablehlo.compare  LT, %iterArg, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %32 : tensor<i1>
    } do {
      %32:8 = func.call @None_5(%iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14) : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_15 = stablehlo.constant dense<1> : tensor<i64>
      %33 = stablehlo.add %iterArg, %c_15 : tensor<i64>
      stablehlo.return %33, %32#0, %32#1, %32#2, %32#3, %32#4, %32#5, %32#6, %32#7 : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %14 = stablehlo.concatenate %13#2, %13#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %15 = stablehlo.slice %14 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %16 = stablehlo.slice %14 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %17 = stablehlo.convert %15 : (tensor<1xui32>) -> tensor<1xui64>
    %18 = stablehlo.convert %16 : (tensor<1xui32>) -> tensor<1xui64>
    %c_4 = stablehlo.constant dense<32> : tensor<ui64>
    %19 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<1xui64>
    %20 = stablehlo.shift_left %17, %19 : tensor<1xui64>
    %21 = stablehlo.or %20, %18 : tensor<1xui64>
    %22 = stablehlo.reshape %21 : (tensor<1xui64>) -> tensor<ui64>
    %c_5 = stablehlo.constant dense<12> : tensor<ui64>
    %23 = stablehlo.shift_right_logical %22, %c_5 : tensor<ui64>
    %c_6 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %24 = stablehlo.or %23, %c_6 : tensor<ui64>
    %25 = stablehlo.bitcast_convert %24 : (tensor<ui64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %26 = stablehlo.subtract %25, %cst : tensor<f64>
    %27 = stablehlo.subtract %arg2, %arg1 : tensor<f64>
    %28 = stablehlo.multiply %26, %27 : tensor<f64>
    %29 = stablehlo.add %28, %arg1 : tensor<f64>
    %30 = stablehlo.reshape %29 : (tensor<f64>) -> tensor<f64>
    %31 = stablehlo.maximum %arg1, %30 : tensor<f64>
    return %31 : tensor<f64>
  }
  func.func private @None_5(%arg0: tensor<i64>, %arg1: tensor<1xui32>, %arg2: tensor<1xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<1xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<1xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<1xui32>
    %9 = stablehlo.or %5, %8 : tensor<1xui32>
    %10 = stablehlo.xor %3, %9 : tensor<1xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<1xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<1xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<1xui32>
    %19 = stablehlo.or %15, %18 : tensor<1xui32>
    %20 = stablehlo.xor %13, %19 : tensor<1xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<1xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<1xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<1xui32>
    %29 = stablehlo.or %25, %28 : tensor<1xui32>
    %30 = stablehlo.xor %23, %29 : tensor<1xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<1xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<1xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<1xui32>
    %39 = stablehlo.or %35, %38 : tensor<1xui32>
    %40 = stablehlo.xor %33, %39 : tensor<1xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %42 = stablehlo.add %33, %41 : tensor<1xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %44 = stablehlo.add %40, %43 : tensor<1xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %48 = stablehlo.add %44, %47 : tensor<1xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
  }
  func.func private @_uniform_6(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<f64> {mhlo.layout_mode = "default"}, %arg2: tensor<f64> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.convert %arg2 : tensor<f64>
    %2 = stablehlo.iota dim = 0 : tensor<2xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.slice %2 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %9 = stablehlo.xor %4, %6 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %10 = stablehlo.xor %9, %c_1 : tensor<ui32>
    %11 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %12 = stablehlo.add %7, %11 : tensor<1xui32>
    %13 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %14 = stablehlo.add %8, %13 : tensor<1xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %15:9 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %c_2, %iterArg_8 = %12, %iterArg_9 = %14, %iterArg_10 = %6, %iterArg_11 = %10, %iterArg_12 = %4, %iterArg_13 = %c, %iterArg_14 = %c_0) : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_15 = stablehlo.constant dense<5> : tensor<i64>
      %34 = stablehlo.compare  LT, %iterArg, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %34 : tensor<i1>
    } do {
      %34:8 = func.call @None_5(%iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14) : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_15 = stablehlo.constant dense<1> : tensor<i64>
      %35 = stablehlo.add %iterArg, %c_15 : tensor<i64>
      stablehlo.return %35, %34#0, %34#1, %34#2, %34#3, %34#4, %34#5, %34#6, %34#7 : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %16 = stablehlo.concatenate %15#2, %15#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %17 = stablehlo.slice %16 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %18 = stablehlo.slice %16 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %19 = stablehlo.convert %17 : (tensor<1xui32>) -> tensor<1xui64>
    %20 = stablehlo.convert %18 : (tensor<1xui32>) -> tensor<1xui64>
    %c_4 = stablehlo.constant dense<32> : tensor<ui64>
    %21 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui64>) -> tensor<1xui64>
    %22 = stablehlo.shift_left %19, %21 : tensor<1xui64>
    %23 = stablehlo.or %22, %20 : tensor<1xui64>
    %24 = stablehlo.reshape %23 : (tensor<1xui64>) -> tensor<ui64>
    %c_5 = stablehlo.constant dense<12> : tensor<ui64>
    %25 = stablehlo.shift_right_logical %24, %c_5 : tensor<ui64>
    %c_6 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %26 = stablehlo.or %25, %c_6 : tensor<ui64>
    %27 = stablehlo.bitcast_convert %26 : (tensor<ui64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %28 = stablehlo.subtract %27, %cst : tensor<f64>
    %29 = stablehlo.subtract %1, %0 : tensor<f64>
    %30 = stablehlo.multiply %28, %29 : tensor<f64>
    %31 = stablehlo.add %30, %0 : tensor<f64>
    %32 = stablehlo.reshape %31 : (tensor<f64>) -> tensor<f64>
    %33 = stablehlo.maximum %0, %32 : tensor<f64>
    return %33 : tensor<f64>
  }
}
