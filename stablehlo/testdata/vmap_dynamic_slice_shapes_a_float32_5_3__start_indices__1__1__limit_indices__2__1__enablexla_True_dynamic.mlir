// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5x3xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2xi64> {mhlo.sharding = ""}) -> tensor<?x1x0xf32> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1 = stablehlo.constant dense<0> : tensor<1xi32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.constant dense<1> : tensor<1xi32>
    %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.constant dense<1> : tensor<1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1xi32>
    %9 = stablehlo.concatenate %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = stablehlo.real_dynamic_slice %arg2, %2, %6, %9 : (tensor<?x2xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
    %11 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.dynamic_reshape %10, %12 : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64>
    %14 = stablehlo.constant dense<0> : tensor<1xi32>
    %15 = stablehlo.constant dense<1> : tensor<1xi32>
    %16 = stablehlo.concatenate %14, %15, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %17 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %18 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
    %19 = stablehlo.constant dense<2> : tensor<1xi32>
    %20 = stablehlo.concatenate %18, %19, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %21 = stablehlo.constant dense<1> : tensor<1xi32>
    %22 = stablehlo.constant dense<1> : tensor<1xi32>
    %23 = stablehlo.concatenate %21, %22, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %24 = stablehlo.real_dynamic_slice %arg2, %16, %20, %23 : (tensor<?x2xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
    %25 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.dynamic_reshape %24, %26 : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64>
    %28 = stablehlo.constant dense<0> : tensor<i64>
    %29 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %30 = stablehlo.reshape %29 : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.dynamic_broadcast_in_dim %28, %30, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %32 = stablehlo.compare  LT, %13, %31,  SIGNED : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
    %33 = stablehlo.constant dense<5> : tensor<i64>
    %34 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %35 = stablehlo.reshape %34 : (tensor<i32>) -> tensor<1xi32>
    %36 = stablehlo.dynamic_broadcast_in_dim %33, %35, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %37 = stablehlo.add %13, %36 : tensor<?xi64>
    %38 = stablehlo.select %32, %37, %13 : tensor<?xi1>, tensor<?xi64>
    %39 = stablehlo.constant dense<0> : tensor<i64>
    %40 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %41 = stablehlo.reshape %40 : (tensor<i32>) -> tensor<1xi32>
    %42 = stablehlo.dynamic_broadcast_in_dim %39, %41, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %43 = stablehlo.compare  LT, %27, %42,  SIGNED : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
    %44 = stablehlo.constant dense<3> : tensor<i64>
    %45 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %46 = stablehlo.reshape %45 : (tensor<i32>) -> tensor<1xi32>
    %47 = stablehlo.dynamic_broadcast_in_dim %44, %46, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %48 = stablehlo.add %27, %47 : tensor<?xi64>
    %49 = stablehlo.select %43, %48, %27 : tensor<?xi1>, tensor<?xi64>
    %50 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %51 = stablehlo.reshape %50 : (tensor<i32>) -> tensor<1xi32>
    %52 = stablehlo.constant dense<1> : tensor<1xi32>
    %53 = stablehlo.concatenate %51, %52, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %54 = stablehlo.dynamic_broadcast_in_dim %38, %53, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %55 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %56 = stablehlo.reshape %55 : (tensor<i32>) -> tensor<1xi32>
    %57 = stablehlo.constant dense<1> : tensor<1xi32>
    %58 = stablehlo.concatenate %56, %57, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %59 = stablehlo.dynamic_broadcast_in_dim %49, %58, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %60 = stablehlo.concatenate %54, %59, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %61 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %62 = stablehlo.reshape %61 : (tensor<i32>) -> tensor<1xi32>
    %63 = stablehlo.constant dense<1> : tensor<1xi32>
    %64 = stablehlo.concatenate %62, %63, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %65 = stablehlo.dynamic_iota %64, dim = 0 : (tensor<2xi32>) -> tensor<?x1xi64>
    %66 = stablehlo.concatenate %65, %60, dim = 1 : (tensor<?x1xi64>, tensor<?x2xi64>) -> tensor<?x3xi64>
    %67 = "stablehlo.gather"(%arg1, %66) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = dense<[1, 1, 0]> : tensor<3xi64>} : (tensor<?x5x3xf32>, tensor<?x3xi64>) -> tensor<?x1x0xf32>
    return %67 : tensor<?x1x0xf32>
  }
}

