// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8xbf16>
    %1 = call @expected() : () -> tensor<i64>
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
    %10 = call @_gumbel(%9) : (tensor<2xui32>) -> tensor<8xbf16>
    %11 = stablehlo.add %10, %0 : tensor<8xbf16>
    %12 = call @argmax(%11) : (tensor<8xbf16>) -> tensor<i64>
    stablehlo.custom_call @check.expect_eq(%12, %1) {has_side_effect = true} : (tensor<i64>, tensor<i64>) -> ()
    return %12 : tensor<i64>
  }
  func.func private @inputs() -> (tensor<8xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[3.093750e+00, -1.039060e+00, 5.468750e-01, 1.375000e+00, 2.890630e+00, 1.789060e+00, -5.390630e-01, -3.984380e+00]> : tensor<8xbf16>
    return %cst : tensor<8xbf16>
  }
  func.func private @expected() -> (tensor<i64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<4> : tensor<i64>
    return %c : tensor<i64>
  }
  func.func private @_gumbel(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<8xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<1.175490e-38> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = call @_uniform(%arg0, %cst, %cst_0) : (tensor<2xui32>, tensor<bf16>, tensor<f64>) -> tensor<8xbf16>
    %1 = stablehlo.log %0 : tensor<8xbf16>
    %2 = stablehlo.negate %1 : tensor<8xbf16>
    %3 = stablehlo.log %2 : tensor<8xbf16>
    %4 = stablehlo.negate %3 : tensor<8xbf16>
    return %4 : tensor<8xbf16>
  }
  func.func private @_uniform(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<bf16> {mhlo.layout_mode = "default"}, %arg2: tensor<f64> {mhlo.layout_mode = "default"}) -> (tensor<8xbf16> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<1xbf16>
    %3 = stablehlo.iota dim = 0 : tensor<2xui32>
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %7 = stablehlo.reshape %6 : (tensor<1xui32>) -> tensor<ui32>
    %8 = stablehlo.slice %3 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %9 = stablehlo.slice %3 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %10 = stablehlo.xor %5, %7 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %11 = stablehlo.xor %10, %c_1 : tensor<ui32>
    %12 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %13 = stablehlo.add %8, %12 : tensor<1xui32>
    %14 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %15 = stablehlo.add %9, %14 : tensor<1xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %16:9 = stablehlo.while(%iterArg = %c_3, %iterArg_8 = %c_2, %iterArg_9 = %13, %iterArg_10 = %15, %iterArg_11 = %7, %iterArg_12 = %11, %iterArg_13 = %5, %iterArg_14 = %c, %iterArg_15 = %c_0) : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_16 = stablehlo.constant dense<5> : tensor<i64>
      %45 = stablehlo.compare  LT, %iterArg, %c_16,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %45:8 = func.call @None(%iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15) : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_16 = stablehlo.constant dense<1> : tensor<i64>
      %46 = stablehlo.add %iterArg, %c_16 : tensor<i64>
      stablehlo.return %46, %45#0, %45#1, %45#2, %45#3, %45#4, %45#5, %45#6, %45#7 : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %17 = stablehlo.concatenate %16#2, %16#3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xui32>) -> tensor<1x2xui32>
    %19 = stablehlo.iota dim = 0 : tensor<4x1xui32>
    %c_4 = stablehlo.constant dense<8> : tensor<ui32>
    %20 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<4x1xui32>
    %21 = stablehlo.multiply %20, %19 : tensor<4x1xui32>
    %22 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x2xui32>) -> tensor<4x2xui32>
    %23 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<4x1xui32>) -> tensor<4x2xui32>
    %24 = stablehlo.shift_right_logical %22, %23 : tensor<4x2xui32>
    %c_5 = stablehlo.constant dense<255> : tensor<ui32>
    %25 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<4x2xui32>
    %26 = stablehlo.and %25, %24 : tensor<4x2xui32>
    %27 = stablehlo.transpose %26, dims = [1, 0] : (tensor<4x2xui32>) -> tensor<2x4xui32>
    %28 = stablehlo.reshape %27 : (tensor<2x4xui32>) -> tensor<8xui32>
    %29 = stablehlo.convert %28 : (tensor<8xui32>) -> tensor<8xui8>
    %30 = stablehlo.convert %29 : (tensor<8xui8>) -> tensor<8xui16>
    %c_6 = stablehlo.constant dense<1> : tensor<ui16>
    %31 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui16>) -> tensor<8xui16>
    %32 = stablehlo.shift_right_logical %30, %31 : tensor<8xui16>
    %c_7 = stablehlo.constant dense<16256> : tensor<ui16>
    %33 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui16>) -> tensor<8xui16>
    %34 = stablehlo.or %32, %33 : tensor<8xui16>
    %35 = stablehlo.bitcast_convert %34 : (tensor<8xui16>) -> tensor<8xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %36 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8xbf16>
    %37 = stablehlo.subtract %35, %36 : tensor<8xbf16>
    %38 = stablehlo.subtract %2, %1 : tensor<1xbf16>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0] : (tensor<1xbf16>) -> tensor<8xbf16>
    %40 = stablehlo.multiply %37, %39 : tensor<8xbf16>
    %41 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<1xbf16>) -> tensor<8xbf16>
    %42 = stablehlo.add %40, %41 : tensor<8xbf16>
    %43 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<1xbf16>) -> tensor<8xbf16>
    %44 = stablehlo.maximum %43, %42 : tensor<8xbf16>
    return %44 : tensor<8xbf16>
  }
  func.func private @None(%arg0: tensor<i64>, %arg1: tensor<1xui32>, %arg2: tensor<1xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
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
  func.func private @argmax(%arg0: tensor<8xbf16>) -> tensor<i64> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi64>
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [0] : (tensor<8xbf16>, tensor<8xi64>, tensor<bf16>, tensor<i64>) -> (tensor<bf16>, tensor<i64>)
     reducer(%arg1: tensor<bf16>, %arg3: tensor<bf16>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<bf16>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i64>
      stablehlo.return %9, %10 : tensor<bf16>, tensor<i64>
    }
    return %1#1 : tensor<i64>
  }
}
