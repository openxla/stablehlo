// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<32xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @expected() : () -> tensor<32xi8>
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
    %c_2 = stablehlo.constant dense<-5> : tensor<i64>
    %c_3 = stablehlo.constant dense<5> : tensor<i64>
    %9 = call @_randint(%8, %c_2, %c_3) : (tensor<2xui32>, tensor<i64>, tensor<i64>) -> tensor<32xi8>
    stablehlo.custom_call @check.expect_eq(%9, %0) {has_side_effect = true} : (tensor<32xi8>, tensor<32xi8>) -> ()
    return %9 : tensor<32xi8>
  }
  func.func private @expected() -> (tensor<32xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[2, 2, 2, -1, 1, 0, 2, -3, -1, 1, 3, -2, -4, 1, -1, -1, 4, -5, 3, 3, 2, 3, 3, -5, 2, 4, 2, -5, 4, -1, -4, -2]> : tensor<32xi8>
    return %c : tensor<32xi8>
  }
  func.func private @_randint(%arg0: tensor<2xui32> {mhlo.layout_mode = "default"}, %arg1: tensor<i64> {mhlo.layout_mode = "default"}, %arg2: tensor<i64> {mhlo.layout_mode = "default"}) -> (tensor<32xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<127> : tensor<i8>
    %c_0 = stablehlo.constant dense<-128> : tensor<i8>
    %c_1 = stablehlo.constant dense<127> : tensor<i8>
    %0 = call @clip(%c, %c_0, %c_1) : (tensor<i8>, tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = stablehlo.convert %0 : (tensor<i8>) -> tensor<i64>
    %2 = stablehlo.compare  GT, %arg2, %1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<-128> : tensor<i64>
    %c_3 = stablehlo.constant dense<127> : tensor<i64>
    %3 = call @clip_0(%arg1, %c_2, %c_3) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i8>
    %c_4 = stablehlo.constant dense<-128> : tensor<i64>
    %c_5 = stablehlo.constant dense<127> : tensor<i64>
    %5 = call @clip_0(%arg2, %c_4, %c_5) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %6 = stablehlo.convert %5 : (tensor<i64>) -> tensor<i8>
    %7 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i8>) -> tensor<1xi8>
    %8 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i8>) -> tensor<1xi8>
    %9 = call @_threefry_split(%arg0) : (tensor<2xui32>) -> tensor<2x2xui32>
    %10 = stablehlo.slice %9 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %11 = stablehlo.reshape %10 : (tensor<1x2xui32>) -> tensor<2xui32>
    %12 = stablehlo.slice %9 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %13 = stablehlo.reshape %12 : (tensor<1x2xui32>) -> tensor<2xui32>
    %14 = stablehlo.iota dim = 0 : tensor<8xui32>
    %15 = stablehlo.slice %11 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %16 = stablehlo.reshape %15 : (tensor<1xui32>) -> tensor<ui32>
    %17 = stablehlo.slice %11 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %18 = stablehlo.reshape %17 : (tensor<1xui32>) -> tensor<ui32>
    %19 = stablehlo.slice %14 [0:4] : (tensor<8xui32>) -> tensor<4xui32>
    %20 = stablehlo.slice %14 [4:8] : (tensor<8xui32>) -> tensor<4xui32>
    %c_6 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_7 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %21 = stablehlo.xor %16, %18 : tensor<ui32>
    %c_8 = stablehlo.constant dense<466688986> : tensor<ui32>
    %22 = stablehlo.xor %21, %c_8 : tensor<ui32>
    %23 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %24 = stablehlo.add %19, %23 : tensor<4xui32>
    %25 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %26 = stablehlo.add %20, %25 : tensor<4xui32>
    %c_9 = stablehlo.constant dense<0> : tensor<i64>
    %c_10 = stablehlo.constant dense<0> : tensor<i64>
    %27:9 = stablehlo.while(%iterArg = %c_10, %iterArg_23 = %c_9, %iterArg_24 = %24, %iterArg_25 = %26, %iterArg_26 = %18, %iterArg_27 = %22, %iterArg_28 = %16, %iterArg_29 = %c_6, %iterArg_30 = %c_7) : tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_31 = stablehlo.constant dense<5> : tensor<i64>
      %95 = stablehlo.compare  LT, %iterArg, %c_31,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %95 : tensor<i1>
    } do {
      %95:8 = func.call @None_1(%iterArg_23, %iterArg_24, %iterArg_25, %iterArg_26, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30) : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_31 = stablehlo.constant dense<1> : tensor<i64>
      %96 = stablehlo.add %iterArg, %c_31 : tensor<i64>
      stablehlo.return %96, %95#0, %95#1, %95#2, %95#3, %95#4, %95#5, %95#6, %95#7 : tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %28 = stablehlo.concatenate %27#2, %27#3, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [1] : (tensor<8xui32>) -> tensor<1x8xui32>
    %30 = stablehlo.iota dim = 0 : tensor<4x1xui32>
    %c_11 = stablehlo.constant dense<8> : tensor<ui32>
    %31 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<4x1xui32>
    %32 = stablehlo.multiply %31, %30 : tensor<4x1xui32>
    %33 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1x8xui32>) -> tensor<4x8xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [0, 1] : (tensor<4x1xui32>) -> tensor<4x8xui32>
    %35 = stablehlo.shift_right_logical %33, %34 : tensor<4x8xui32>
    %c_12 = stablehlo.constant dense<255> : tensor<ui32>
    %36 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<4x8xui32>
    %37 = stablehlo.and %36, %35 : tensor<4x8xui32>
    %38 = stablehlo.transpose %37, dims = [1, 0] : (tensor<4x8xui32>) -> tensor<8x4xui32>
    %39 = stablehlo.reshape %38 : (tensor<8x4xui32>) -> tensor<32xui32>
    %40 = stablehlo.convert %39 : (tensor<32xui32>) -> tensor<32xui8>
    %41 = stablehlo.iota dim = 0 : tensor<8xui32>
    %42 = stablehlo.slice %13 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %43 = stablehlo.reshape %42 : (tensor<1xui32>) -> tensor<ui32>
    %44 = stablehlo.slice %13 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %45 = stablehlo.reshape %44 : (tensor<1xui32>) -> tensor<ui32>
    %46 = stablehlo.slice %41 [0:4] : (tensor<8xui32>) -> tensor<4xui32>
    %47 = stablehlo.slice %41 [4:8] : (tensor<8xui32>) -> tensor<4xui32>
    %c_13 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_14 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %48 = stablehlo.xor %43, %45 : tensor<ui32>
    %c_15 = stablehlo.constant dense<466688986> : tensor<ui32>
    %49 = stablehlo.xor %48, %c_15 : tensor<ui32>
    %50 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %51 = stablehlo.add %46, %50 : tensor<4xui32>
    %52 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %53 = stablehlo.add %47, %52 : tensor<4xui32>
    %c_16 = stablehlo.constant dense<0> : tensor<i64>
    %c_17 = stablehlo.constant dense<0> : tensor<i64>
    %54:9 = stablehlo.while(%iterArg = %c_17, %iterArg_23 = %c_16, %iterArg_24 = %51, %iterArg_25 = %53, %iterArg_26 = %45, %iterArg_27 = %49, %iterArg_28 = %43, %iterArg_29 = %c_13, %iterArg_30 = %c_14) : tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %c_31 = stablehlo.constant dense<5> : tensor<i64>
      %95 = stablehlo.compare  LT, %iterArg, %c_31,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %95 : tensor<i1>
    } do {
      %95:8 = func.call @None_1(%iterArg_23, %iterArg_24, %iterArg_25, %iterArg_26, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30) : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_31 = stablehlo.constant dense<1> : tensor<i64>
      %96 = stablehlo.add %iterArg, %c_31 : tensor<i64>
      stablehlo.return %96, %95#0, %95#1, %95#2, %95#3, %95#4, %95#5, %95#6, %95#7 : tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %55 = stablehlo.concatenate %54#2, %54#3, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [1] : (tensor<8xui32>) -> tensor<1x8xui32>
    %57 = stablehlo.iota dim = 0 : tensor<4x1xui32>
    %c_18 = stablehlo.constant dense<8> : tensor<ui32>
    %58 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<ui32>) -> tensor<4x1xui32>
    %59 = stablehlo.multiply %58, %57 : tensor<4x1xui32>
    %60 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<1x8xui32>) -> tensor<4x8xui32>
    %61 = stablehlo.broadcast_in_dim %59, dims = [0, 1] : (tensor<4x1xui32>) -> tensor<4x8xui32>
    %62 = stablehlo.shift_right_logical %60, %61 : tensor<4x8xui32>
    %c_19 = stablehlo.constant dense<255> : tensor<ui32>
    %63 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<ui32>) -> tensor<4x8xui32>
    %64 = stablehlo.and %63, %62 : tensor<4x8xui32>
    %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<4x8xui32>) -> tensor<8x4xui32>
    %66 = stablehlo.reshape %65 : (tensor<8x4xui32>) -> tensor<32xui32>
    %67 = stablehlo.convert %66 : (tensor<32xui32>) -> tensor<32xui8>
    %68 = stablehlo.subtract %8, %7 : tensor<1xi8>
    %69 = stablehlo.convert %68 : (tensor<1xi8>) -> tensor<1xui8>
    %70 = stablehlo.compare  LE, %8, %7,  SIGNED : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %c_20 = stablehlo.constant dense<1> : tensor<ui8>
    %71 = stablehlo.broadcast_in_dim %c_20, dims = [] : (tensor<ui8>) -> tensor<1xui8>
    %72 = stablehlo.select %70, %71, %69 : tensor<1xi1>, tensor<1xui8>
    %73 = stablehlo.compare  GT, %8, %7,  SIGNED : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %74 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %75 = stablehlo.and %74, %73 : tensor<1xi1>
    %c_21 = stablehlo.constant dense<1> : tensor<ui8>
    %76 = stablehlo.broadcast_in_dim %c_21, dims = [] : (tensor<ui8>) -> tensor<1xui8>
    %77 = stablehlo.add %72, %76 : tensor<1xui8>
    %78 = stablehlo.select %75, %77, %72 : tensor<1xi1>, tensor<1xui8>
    %c_22 = stablehlo.constant dense<16> : tensor<ui8>
    %79 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<ui8>) -> tensor<1xui8>
    %80 = stablehlo.remainder %79, %78 : tensor<1xui8>
    %81 = stablehlo.multiply %80, %80 : tensor<1xui8>
    %82 = stablehlo.remainder %81, %78 : tensor<1xui8>
    %83 = stablehlo.broadcast_in_dim %78, dims = [0] : (tensor<1xui8>) -> tensor<32xui8>
    %84 = stablehlo.remainder %40, %83 : tensor<32xui8>
    %85 = stablehlo.broadcast_in_dim %82, dims = [0] : (tensor<1xui8>) -> tensor<32xui8>
    %86 = stablehlo.multiply %84, %85 : tensor<32xui8>
    %87 = stablehlo.broadcast_in_dim %78, dims = [0] : (tensor<1xui8>) -> tensor<32xui8>
    %88 = stablehlo.remainder %67, %87 : tensor<32xui8>
    %89 = stablehlo.add %86, %88 : tensor<32xui8>
    %90 = stablehlo.broadcast_in_dim %78, dims = [0] : (tensor<1xui8>) -> tensor<32xui8>
    %91 = stablehlo.remainder %89, %90 : tensor<32xui8>
    %92 = stablehlo.convert %91 : (tensor<32xui8>) -> tensor<32xi8>
    %93 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<1xi8>) -> tensor<32xi8>
    %94 = stablehlo.add %93, %92 : tensor<32xi8>
    return %94 : tensor<32xi8>
  }
  func.func private @clip(%arg0: tensor<i8> {mhlo.layout_mode = "default"}, %arg1: tensor<i8> {mhlo.layout_mode = "default"}, %arg2: tensor<i8> {mhlo.layout_mode = "default"}) -> (tensor<i8> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i8>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i8>
    return %1 : tensor<i8>
  }
  func.func private @clip_0(%arg0: tensor<i64> {mhlo.layout_mode = "default"}, %arg1: tensor<i64> {mhlo.layout_mode = "default"}, %arg2: tensor<i64> {mhlo.layout_mode = "default"}) -> (tensor<i64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i64>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i64>
    return %1 : tensor<i64>
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
  func.func private @None_1(%arg0: tensor<i64>, %arg1: tensor<4xui32>, %arg2: tensor<4xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<4xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<4xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<4xui32>
    %9 = stablehlo.or %5, %8 : tensor<4xui32>
    %10 = stablehlo.xor %3, %9 : tensor<4xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<4xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<4xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<4xui32>
    %19 = stablehlo.or %15, %18 : tensor<4xui32>
    %20 = stablehlo.xor %13, %19 : tensor<4xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<4xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<4xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<4xui32>
    %29 = stablehlo.or %25, %28 : tensor<4xui32>
    %30 = stablehlo.xor %23, %29 : tensor<4xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<4xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<4xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<4xui32>
    %39 = stablehlo.or %35, %38 : tensor<4xui32>
    %40 = stablehlo.xor %33, %39 : tensor<4xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %42 = stablehlo.add %33, %41 : tensor<4xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %44 = stablehlo.add %40, %43 : tensor<4xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %48 = stablehlo.add %44, %47 : tensor<4xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
  }
}
