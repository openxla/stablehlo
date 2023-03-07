// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<3x?x?x3xf32> {mhlo.sharding = ""}) -> tensor<3x?x?x3xf32> {
    %0 = call @_resize(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<i64>, tensor<3x?x?x3xf32>) -> tensor<3x?x?x3xf32>
    return %0 : tensor<3x?x?x3xf32>
  }
  func.func private @_resize(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<3x?x?x3xf32>) -> tensor<3x?x?x3xf32> {
    %0 = stablehlo.constant dense<2> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_iota %3, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
    %5 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %6 = stablehlo.constant dense<2> : tensor<i64>
    %7 = stablehlo.multiply %arg0, %6 : tensor<i64>
    %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.dynamic_broadcast_in_dim %5, %9, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %11 = stablehlo.add %4, %10 : tensor<?xf32>
    %12 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f32>
    %13 = stablehlo.constant dense<2> : tensor<i64>
    %14 = stablehlo.multiply %arg0, %13 : tensor<i64>
    %15 = stablehlo.convert %14 : (tensor<i64>) -> tensor<i32>
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.dynamic_broadcast_in_dim %12, %16, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %18 = stablehlo.multiply %11, %17 : tensor<?xf32>
    %19 = stablehlo.constant dense<2> : tensor<i64>
    %20 = stablehlo.multiply %arg0, %19 : tensor<i64>
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<f32>
    %22 = stablehlo.constant dense<2> : tensor<i64>
    %23 = stablehlo.multiply %arg0, %22 : tensor<i64>
    %24 = stablehlo.convert %23 : (tensor<i64>) -> tensor<i32>
    %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.dynamic_broadcast_in_dim %21, %25, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %27 = stablehlo.divide %18, %26 : tensor<?xf32>
    %28 = stablehlo.floor %27 : tensor<?xf32>
    %29 = stablehlo.convert %28 : (tensor<?xf32>) -> tensor<?xi32>
    %30 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %31 = stablehlo.constant dense<0> : tensor<i32>
    %32 = stablehlo.constant dense<2> : tensor<i64>
    %33 = stablehlo.multiply %arg0, %32 : tensor<i64>
    %34 = stablehlo.convert %33 : (tensor<i64>) -> tensor<i32>
    %35 = stablehlo.reshape %34 : (tensor<i32>) -> tensor<1xi32>
    %36 = stablehlo.dynamic_broadcast_in_dim %31, %35, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %37 = stablehlo.compare  LT, %29, %36,  SIGNED : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %38 = stablehlo.constant dense<2> : tensor<i64>
    %39 = stablehlo.multiply %arg0, %38 : tensor<i64>
    %40 = stablehlo.convert %39 : (tensor<i64>) -> tensor<i32>
    %41 = stablehlo.reshape %40 : (tensor<i32>) -> tensor<1xi32>
    %42 = stablehlo.dynamic_broadcast_in_dim %30, %41, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %43 = stablehlo.add %29, %42 : tensor<?xi32>
    %44 = stablehlo.select %37, %43, %29 : tensor<?xi1>, tensor<?xi32>
    %45 = stablehlo.convert %44 : (tensor<?xi32>) -> tensor<?xi64>
    %46 = stablehlo.constant dense<2> : tensor<i64>
    %47 = stablehlo.multiply %arg0, %46 : tensor<i64>
    %48 = stablehlo.convert %47 : (tensor<i64>) -> tensor<i32>
    %49 = stablehlo.reshape %48 : (tensor<i32>) -> tensor<1xi32>
    %50 = stablehlo.constant dense<1> : tensor<1xi32>
    %51 = stablehlo.concatenate %49, %50, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %52 = stablehlo.dynamic_broadcast_in_dim %45, %51, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %53 = stablehlo.constant dense<3> : tensor<1xi32>
    %54 = stablehlo.constant dense<1> : tensor<1xi32>
    %55 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %56 = stablehlo.reshape %55 : (tensor<i32>) -> tensor<1xi32>
    %57 = stablehlo.constant dense<3> : tensor<1xi32>
    %58 = stablehlo.concatenate %53, %54, %56, %57, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %59 = "stablehlo.dynamic_gather"(%arg2, %52, %58) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>} : (tensor<3x?x?x3xf32>, tensor<?x1xi64>, tensor<4xi32>) -> tensor<3x?x?x3xf32>
    %60 = stablehlo.constant dense<2> : tensor<i64>
    %61 = stablehlo.multiply %arg1, %60 : tensor<i64>
    %62 = stablehlo.convert %61 : (tensor<i64>) -> tensor<i32>
    %63 = stablehlo.reshape %62 : (tensor<i32>) -> tensor<1xi32>
    %64 = stablehlo.dynamic_iota %63, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
    %65 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %66 = stablehlo.constant dense<2> : tensor<i64>
    %67 = stablehlo.multiply %arg1, %66 : tensor<i64>
    %68 = stablehlo.convert %67 : (tensor<i64>) -> tensor<i32>
    %69 = stablehlo.reshape %68 : (tensor<i32>) -> tensor<1xi32>
    %70 = stablehlo.dynamic_broadcast_in_dim %65, %69, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %71 = stablehlo.add %64, %70 : tensor<?xf32>
    %72 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f32>
    %73 = stablehlo.constant dense<2> : tensor<i64>
    %74 = stablehlo.multiply %arg1, %73 : tensor<i64>
    %75 = stablehlo.convert %74 : (tensor<i64>) -> tensor<i32>
    %76 = stablehlo.reshape %75 : (tensor<i32>) -> tensor<1xi32>
    %77 = stablehlo.dynamic_broadcast_in_dim %72, %76, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %78 = stablehlo.multiply %71, %77 : tensor<?xf32>
    %79 = stablehlo.constant dense<2> : tensor<i64>
    %80 = stablehlo.multiply %arg1, %79 : tensor<i64>
    %81 = stablehlo.convert %80 : (tensor<i64>) -> tensor<f32>
    %82 = stablehlo.constant dense<2> : tensor<i64>
    %83 = stablehlo.multiply %arg1, %82 : tensor<i64>
    %84 = stablehlo.convert %83 : (tensor<i64>) -> tensor<i32>
    %85 = stablehlo.reshape %84 : (tensor<i32>) -> tensor<1xi32>
    %86 = stablehlo.dynamic_broadcast_in_dim %81, %85, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %87 = stablehlo.divide %78, %86 : tensor<?xf32>
    %88 = stablehlo.floor %87 : tensor<?xf32>
    %89 = stablehlo.convert %88 : (tensor<?xf32>) -> tensor<?xi32>
    %90 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %91 = stablehlo.constant dense<0> : tensor<i32>
    %92 = stablehlo.constant dense<2> : tensor<i64>
    %93 = stablehlo.multiply %arg1, %92 : tensor<i64>
    %94 = stablehlo.convert %93 : (tensor<i64>) -> tensor<i32>
    %95 = stablehlo.reshape %94 : (tensor<i32>) -> tensor<1xi32>
    %96 = stablehlo.dynamic_broadcast_in_dim %91, %95, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %97 = stablehlo.compare  LT, %89, %96,  SIGNED : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %98 = stablehlo.constant dense<2> : tensor<i64>
    %99 = stablehlo.multiply %arg1, %98 : tensor<i64>
    %100 = stablehlo.convert %99 : (tensor<i64>) -> tensor<i32>
    %101 = stablehlo.reshape %100 : (tensor<i32>) -> tensor<1xi32>
    %102 = stablehlo.dynamic_broadcast_in_dim %90, %101, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %103 = stablehlo.add %89, %102 : tensor<?xi32>
    %104 = stablehlo.select %97, %103, %89 : tensor<?xi1>, tensor<?xi32>
    %105 = stablehlo.convert %104 : (tensor<?xi32>) -> tensor<?xi64>
    %106 = stablehlo.constant dense<2> : tensor<i64>
    %107 = stablehlo.multiply %arg1, %106 : tensor<i64>
    %108 = stablehlo.convert %107 : (tensor<i64>) -> tensor<i32>
    %109 = stablehlo.reshape %108 : (tensor<i32>) -> tensor<1xi32>
    %110 = stablehlo.constant dense<1> : tensor<1xi32>
    %111 = stablehlo.concatenate %109, %110, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %112 = stablehlo.dynamic_broadcast_in_dim %105, %111, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %113 = stablehlo.constant dense<2> : tensor<i64>
    %114 = stablehlo.multiply %arg0, %113 : tensor<i64>
    %115 = stablehlo.constant dense<3> : tensor<1xi32>
    %116 = stablehlo.convert %114 : (tensor<i64>) -> tensor<i32>
    %117 = stablehlo.reshape %116 : (tensor<i32>) -> tensor<1xi32>
    %118 = stablehlo.constant dense<1> : tensor<1xi32>
    %119 = stablehlo.constant dense<3> : tensor<1xi32>
    %120 = stablehlo.concatenate %115, %117, %118, %119, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %121 = "stablehlo.dynamic_gather"(%59, %112, %120) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 3], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>} : (tensor<3x?x?x3xf32>, tensor<?x1xi64>, tensor<4xi32>) -> tensor<3x?x?x3xf32>
    return %121 : tensor<3x?x?x3xf32>
  }
}

