// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x7x5x3xf32> {mhlo.sharding = ""}, %arg2: tensor<?x3xi64> {mhlo.sharding = ""}) -> tensor<?x3x1x2xf32> {
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
    %10 = stablehlo.real_dynamic_slice %arg2, %2, %6, %9 : (tensor<?x3xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
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
    %24 = stablehlo.real_dynamic_slice %arg2, %16, %20, %23 : (tensor<?x3xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
    %25 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.dynamic_reshape %24, %26 : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64>
    %28 = stablehlo.constant dense<0> : tensor<1xi32>
    %29 = stablehlo.constant dense<2> : tensor<1xi32>
    %30 = stablehlo.concatenate %28, %29, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %31 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %32 = stablehlo.reshape %31 : (tensor<i32>) -> tensor<1xi32>
    %33 = stablehlo.constant dense<3> : tensor<1xi32>
    %34 = stablehlo.concatenate %32, %33, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %35 = stablehlo.constant dense<1> : tensor<1xi32>
    %36 = stablehlo.constant dense<1> : tensor<1xi32>
    %37 = stablehlo.concatenate %35, %36, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %38 = stablehlo.real_dynamic_slice %arg2, %30, %34, %37 : (tensor<?x3xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
    %39 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %40 = stablehlo.reshape %39 : (tensor<i32>) -> tensor<1xi32>
    %41 = stablehlo.dynamic_reshape %38, %40 : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64>
    %42 = stablehlo.constant dense<0> : tensor<i64>
    %43 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %44 = stablehlo.reshape %43 : (tensor<i32>) -> tensor<1xi32>
    %45 = stablehlo.dynamic_broadcast_in_dim %42, %44, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %46 = stablehlo.compare  LT, %13, %45,  SIGNED : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
    %47 = stablehlo.constant dense<7> : tensor<i64>
    %48 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %49 = stablehlo.reshape %48 : (tensor<i32>) -> tensor<1xi32>
    %50 = stablehlo.dynamic_broadcast_in_dim %47, %49, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %51 = stablehlo.add %13, %50 : tensor<?xi64>
    %52 = stablehlo.select %46, %51, %13 : tensor<?xi1>, tensor<?xi64>
    %53 = stablehlo.constant dense<0> : tensor<i64>
    %54 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %55 = stablehlo.reshape %54 : (tensor<i32>) -> tensor<1xi32>
    %56 = stablehlo.dynamic_broadcast_in_dim %53, %55, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %57 = stablehlo.compare  LT, %27, %56,  SIGNED : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
    %58 = stablehlo.constant dense<5> : tensor<i64>
    %59 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %60 = stablehlo.reshape %59 : (tensor<i32>) -> tensor<1xi32>
    %61 = stablehlo.dynamic_broadcast_in_dim %58, %60, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %62 = stablehlo.add %27, %61 : tensor<?xi64>
    %63 = stablehlo.select %57, %62, %27 : tensor<?xi1>, tensor<?xi64>
    %64 = stablehlo.constant dense<0> : tensor<i64>
    %65 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %66 = stablehlo.reshape %65 : (tensor<i32>) -> tensor<1xi32>
    %67 = stablehlo.dynamic_broadcast_in_dim %64, %66, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %68 = stablehlo.compare  LT, %41, %67,  SIGNED : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
    %69 = stablehlo.constant dense<3> : tensor<i64>
    %70 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %71 = stablehlo.reshape %70 : (tensor<i32>) -> tensor<1xi32>
    %72 = stablehlo.dynamic_broadcast_in_dim %69, %71, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %73 = stablehlo.add %41, %72 : tensor<?xi64>
    %74 = stablehlo.select %68, %73, %41 : tensor<?xi1>, tensor<?xi64>
    %75 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %76 = stablehlo.reshape %75 : (tensor<i32>) -> tensor<1xi32>
    %77 = stablehlo.constant dense<1> : tensor<1xi32>
    %78 = stablehlo.concatenate %76, %77, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %79 = stablehlo.dynamic_broadcast_in_dim %52, %78, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %80 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %81 = stablehlo.reshape %80 : (tensor<i32>) -> tensor<1xi32>
    %82 = stablehlo.constant dense<1> : tensor<1xi32>
    %83 = stablehlo.concatenate %81, %82, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %84 = stablehlo.dynamic_broadcast_in_dim %63, %83, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %85 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %86 = stablehlo.reshape %85 : (tensor<i32>) -> tensor<1xi32>
    %87 = stablehlo.constant dense<1> : tensor<1xi32>
    %88 = stablehlo.concatenate %86, %87, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %89 = stablehlo.dynamic_broadcast_in_dim %74, %88, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %90 = stablehlo.concatenate %79, %84, %89, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x3xi64>
    %91 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %92 = stablehlo.reshape %91 : (tensor<i32>) -> tensor<1xi32>
    %93 = stablehlo.constant dense<1> : tensor<1xi32>
    %94 = stablehlo.concatenate %92, %93, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %95 = stablehlo.dynamic_iota %94, dim = 0 : (tensor<2xi32>) -> tensor<?x1xi64>
    %96 = stablehlo.concatenate %95, %90, dim = 1 : (tensor<?x1xi64>, tensor<?x3xi64>) -> tensor<?x4xi64>
    %97 = "stablehlo.gather"(%arg1, %96) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3], collapsed_slice_dims = [0], start_index_map = [0, 1, 2, 3], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = dense<[1, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x7x5x3xf32>, tensor<?x4xi64>) -> tensor<?x3x1x2xf32>
    return %97 : tensor<?x3x1x2xf32>
  }
}

