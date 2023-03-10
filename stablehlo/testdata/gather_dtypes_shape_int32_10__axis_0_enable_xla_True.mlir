// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<10xi32>, tensor<i32>)
    %1 = call @expected() : () -> tensor<i32>
    %2 = call @_take(%0#0, %0#1) : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<10xi32>, tensor<i32>) {
    %0 = stablehlo.constant dense<[1, 2, 4, 0, -2, 0, -1, 0, 1, 0]> : tensor<10xi32>
    %1 = stablehlo.constant dense<2> : tensor<i32>
    return %0, %1 : tensor<10xi32>, tensor<i32>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<4> : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_take(%arg0: tensor<10xi32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.constant dense<10> : tensor<i32>
    %3 = stablehlo.add %arg1, %2 : tensor<i32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<0> : tensor<1xi32>
    %7 = stablehlo.constant dense<0> : tensor<1xi32>
    %8 = stablehlo.constant dense<10> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.compare  LT, %6, %11,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %13 = stablehlo.constant dense<1> : tensor<i32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %15 = stablehlo.add %6, %14 : tensor<1xi32>
    %16 = stablehlo.select %12, %15, %6 : tensor<1xi1>, tensor<1xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %18 = "stablehlo.gather"(%9, %17) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %19 = stablehlo.constant dense<1> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %21 = stablehlo.constant dense<0> : tensor<i32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.compare  LT, %7, %22,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %24 = stablehlo.constant dense<1> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.add %7, %25 : tensor<1xi32>
    %27 = stablehlo.select %23, %26, %7 : tensor<1xi1>, tensor<1xi32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %29 = "stablehlo.gather"(%20, %28) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %30 = stablehlo.subtract %18, %29 : tensor<1xi32>
    %31 = stablehlo.constant dense<0> : tensor<i32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %33 = stablehlo.compare  GE, %5, %32,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %34 = stablehlo.compare  LE, %5, %30,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %35 = stablehlo.and %33, %34 : tensor<1xi1>
    %36 = stablehlo.constant dense<true> : tensor<i1>
    %37 = stablehlo.reduce(%35 init: %36) across dimensions = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
      %41 = stablehlo.and %arg2, %arg3 : tensor<i1>
      stablehlo.return %41 : tensor<i1>
    }
    %38 = "stablehlo.gather"(%arg0, %5) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<10xi32>, tensor<1xi32>) -> tensor<i32>
    %39 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %40 = stablehlo.select %37, %38, %39 : tensor<i1>, tensor<i32>
    return %40 : tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}

