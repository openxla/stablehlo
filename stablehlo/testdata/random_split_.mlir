// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x2xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @expected() : () -> tensor<2x2xui32>
    %1 = call @wrap_and_split() : () -> tensor<2x2xui32>
    stablehlo.custom_call @check.expect_eq(%1, %0) {has_side_effect = true} : (tensor<2x2xui32>, tensor<2x2xui32>) -> ()
    return %1 : tensor<2x2xui32>
  }
  func.func private @expected() -> (tensor<2x2xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2465931498, 3679230171], [255383827, 267815257]]> : tensor<2x2xui32>
    return %c : tensor<2x2xui32>
  }
  func.func private @wrap_and_split() -> (tensor<2x2xui32> {mhlo.layout_mode = "default"}) {
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
    %8 = call @_threefry_split(%7) : (tensor<2xui32>) -> tensor<2x2xui32>
    return %8 : tensor<2x2xui32>
  }
  func.func private @_threefry_split(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}) -> (tensor<2x2xui32> {mhlo.layout_mode = "default"}) {
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
      %16:8 = func.call @None(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %17 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %17, %16#0, %16#1, %16#2, %16#3, %16#4, %16#5, %16#6, %16#7 : tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %14 = stablehlo.concatenate %13#2, %13#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %15 = stablehlo.reshape %14 : (tensor<4xui32>) -> tensor<2x2xui32>
    return %15 : tensor<2x2xui32>
  }
  func.func private @None(%arg0: tensor<i64>, %arg1: tensor<2xui32>, %arg2: tensor<2xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
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
}
