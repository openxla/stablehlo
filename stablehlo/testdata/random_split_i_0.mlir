// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xui32>
    %1 = call @expected() : () -> tensor<2x2xui32>
    %2 = call @"<lambda>"(%0) : (tensor<2xui32>) -> tensor<2x2xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xui32> {
    %0 = stablehlo.constant dense<0> : tensor<2xui32>
    return %0 : tensor<2xui32>
  }
  func.func private @expected() -> tensor<2x2xui32> {
    %0 = stablehlo.constant dense<[[4146024105, 967050713], [2718843009, 1272950319]]> : tensor<2x2xui32>
    return %0 : tensor<2x2xui32>
  }
  func.func private @"<lambda>"(%arg0: tensor<2xui32>) -> tensor<2x2xui32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xui32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = "stablehlo.slice"(%arg0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %6 = "stablehlo.slice"(%0) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<2xui32>
    %7 = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %8 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %9 = stablehlo.xor %2, %4 : tensor<ui32>
    %10 = stablehlo.constant dense<466688986> : tensor<ui32>
    %11 = stablehlo.xor %9, %10 : tensor<ui32>
    %12 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %13 = stablehlo.add %5, %12 : tensor<2xui32>
    %14 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %15 = stablehlo.add %6, %14 : tensor<2xui32>
    %16 = stablehlo.constant dense<0> : tensor<i32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18:9 = stablehlo.while(%iterArg = %17, %iterArg_0 = %16, %iterArg_1 = %13, %iterArg_2 = %15, %iterArg_3 = %4, %iterArg_4 = %11, %iterArg_5 = %2, %iterArg_6 = %7, %iterArg_7 = %8) : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
     cond {
      %21 = stablehlo.constant dense<5> : tensor<i32>
      %22 = stablehlo.compare  LT, %iterArg, %21,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %22 : tensor<i1>
    } do {
      %21 = stablehlo.constant dense<1> : tensor<i32>
      %22 = stablehlo.add %iterArg_0, %21 : tensor<i32>
      %23 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %24 = stablehlo.reshape %23 : (tensor<1xui32>) -> tensor<ui32>
      %25 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<2xui32>
      %26 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %27 = stablehlo.shift_left %iterArg_2, %26 : tensor<2xui32>
      %28 = stablehlo.constant dense<32> : tensor<ui32>
      %29 = stablehlo.subtract %28, %24 : tensor<ui32>
      %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %31 = stablehlo.shift_right_logical %iterArg_2, %30 : tensor<2xui32>
      %32 = stablehlo.or %27, %31 : tensor<2xui32>
      %33 = stablehlo.xor %25, %32 : tensor<2xui32>
      %34 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %35 = stablehlo.reshape %34 : (tensor<1xui32>) -> tensor<ui32>
      %36 = stablehlo.add %25, %33 : tensor<2xui32>
      %37 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %38 = stablehlo.shift_left %33, %37 : tensor<2xui32>
      %39 = stablehlo.constant dense<32> : tensor<ui32>
      %40 = stablehlo.subtract %39, %35 : tensor<ui32>
      %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %42 = stablehlo.shift_right_logical %33, %41 : tensor<2xui32>
      %43 = stablehlo.or %38, %42 : tensor<2xui32>
      %44 = stablehlo.xor %36, %43 : tensor<2xui32>
      %45 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %46 = stablehlo.reshape %45 : (tensor<1xui32>) -> tensor<ui32>
      %47 = stablehlo.add %36, %44 : tensor<2xui32>
      %48 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %49 = stablehlo.shift_left %44, %48 : tensor<2xui32>
      %50 = stablehlo.constant dense<32> : tensor<ui32>
      %51 = stablehlo.subtract %50, %46 : tensor<ui32>
      %52 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %53 = stablehlo.shift_right_logical %44, %52 : tensor<2xui32>
      %54 = stablehlo.or %49, %53 : tensor<2xui32>
      %55 = stablehlo.xor %47, %54 : tensor<2xui32>
      %56 = "stablehlo.slice"(%iterArg_6) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xui32>) -> tensor<1xui32>
      %57 = stablehlo.reshape %56 : (tensor<1xui32>) -> tensor<ui32>
      %58 = stablehlo.add %47, %55 : tensor<2xui32>
      %59 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %60 = stablehlo.shift_left %55, %59 : tensor<2xui32>
      %61 = stablehlo.constant dense<32> : tensor<ui32>
      %62 = stablehlo.subtract %61, %57 : tensor<ui32>
      %63 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %64 = stablehlo.shift_right_logical %55, %63 : tensor<2xui32>
      %65 = stablehlo.or %60, %64 : tensor<2xui32>
      %66 = stablehlo.xor %58, %65 : tensor<2xui32>
      %67 = stablehlo.broadcast_in_dim %iterArg_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %68 = stablehlo.add %58, %67 : tensor<2xui32>
      %69 = stablehlo.broadcast_in_dim %iterArg_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %70 = stablehlo.add %66, %69 : tensor<2xui32>
      %71 = stablehlo.constant dense<1> : tensor<i32>
      %72 = stablehlo.add %iterArg_0, %71 : tensor<i32>
      %73 = stablehlo.convert %72 : (tensor<i32>) -> tensor<ui32>
      %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<ui32>) -> tensor<2xui32>
      %75 = stablehlo.add %70, %74 : tensor<2xui32>
      %76 = stablehlo.constant dense<1> : tensor<i32>
      %77 = stablehlo.add %iterArg, %76 : tensor<i32>
      stablehlo.return %77, %22, %68, %75, %iterArg_4, %iterArg_5, %iterArg_3, %iterArg_7, %iterArg_6 : tensor<i32>, tensor<i32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    %19 = stablehlo.concatenate %18#2, %18#3, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %20 = stablehlo.reshape %19 : (tensor<4xui32>) -> tensor<2x2xui32>
    return %20 : tensor<2x2xui32>
  }
}
