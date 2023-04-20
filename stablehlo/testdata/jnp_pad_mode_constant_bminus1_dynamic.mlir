// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5xf32> {mhlo.sharding = ""}) -> tensor<?x11xf32> {
    %0 = stablehlo.constant dense<1> : tensor<i64>
    %1 = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.constant dense<-1> : tensor<i64>
    %3 = stablehlo.add %arg0, %2 : tensor<i64>
    %4 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.concatenate %10, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %14 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
    %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
    %16 = stablehlo.constant dense<5> : tensor<1xi32>
    %17 = stablehlo.concatenate %15, %16, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %18 = stablehlo.add %13, %17 : tensor<2xi32>
    %19 = stablehlo.constant dense<1> : tensor<1xi32>
    %20 = stablehlo.constant dense<1> : tensor<1xi32>
    %21 = stablehlo.concatenate %19, %20, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %22 = stablehlo.real_dynamic_slice %arg1, %8, %18, %21 : (tensor<?x5xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x5xf32>
    %23 = stablehlo.constant dense<0> : tensor<i64>
    %24 = call @_pad(%arg0, %22, %23) : (tensor<i64>, tensor<?x5xf32>, tensor<i64>) -> tensor<?x11xf32>
    return %24 : tensor<?x11xf32>
  }
  func.func private @_pad(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<i64>) -> tensor<?x11xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<2x2xi64>
    %1 = stablehlo.convert %0 : (tensor<2x2xi64>) -> tensor<2x2xf32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = "stablehlo.gather"(%1, %6) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<0> : tensor<1xi32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = stablehlo.constant dense<0> : tensor<1xi32>
    %13 = stablehlo.constant dense<0> : tensor<1xi32>
    %14 = stablehlo.concatenate %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.constant dense<0> : tensor<1xi32>
    %16 = stablehlo.constant dense<0> : tensor<1xi32>
    %17 = stablehlo.concatenate %15, %16, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %18 = stablehlo.dynamic_pad %arg1, %7, %11, %14, %17 : (tensor<?x5xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x5xf32>
    %19 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %21 = stablehlo.constant dense<1> : tensor<i32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.concatenate %20, %22, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %24 = "stablehlo.gather"(%1, %23) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %25 = stablehlo.pad %18, %24, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?x5xf32>
    %26 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.constant dense<0> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.concatenate %27, %29, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %31 = "stablehlo.gather"(%1, %30) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %32 = stablehlo.pad %25, %31, low = [0, 5], high = [0, 0], interior = [0, 0] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?x10xf32>
    %33 = stablehlo.constant dense<1> : tensor<i32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.constant dense<1> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %37 = stablehlo.concatenate %34, %36, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %38 = "stablehlo.gather"(%1, %37) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %39 = stablehlo.pad %32, %38, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<?x10xf32>, tensor<f32>) -> tensor<?x11xf32>
    return %39 : tensor<?x11xf32>
  }
}

