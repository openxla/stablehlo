// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32> {mhlo.sharding = ""}, %arg2: tensor<?xi32> {mhlo.sharding = ""}) -> tensor<?x10x10xf32> {
    %0 = call @_take(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>
    return %0 : tensor<?x10x10xf32>
  }
  func.func private @_take(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?xi32>) -> tensor<?x10x10xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_broadcast_in_dim %0, %2, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %4 = stablehlo.compare  LT, %arg2, %3,  SIGNED : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %5 = stablehlo.constant dense<10> : tensor<i32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.dynamic_broadcast_in_dim %5, %7, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %9 = stablehlo.add %arg2, %8 : tensor<?xi32>
    %10 = call @_where(%arg0, %4, %9, %arg2) : (tensor<i64>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %11 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.constant dense<1> : tensor<1xi32>
    %14 = stablehlo.concatenate %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.dynamic_broadcast_in_dim %10, %14, dims = [0] : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.constant dense<1> : tensor<1xi32>
    %19 = stablehlo.concatenate %17, %18, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %20 = stablehlo.dynamic_iota %19, dim = 0 : (tensor<2xi32>) -> tensor<?x1xi32>
    %21 = stablehlo.concatenate %20, %15, dim = 1 : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x2xi32>
    %22 = stablehlo.constant dense<[0, 3]> : tensor<2xi64>
    %23 = stablehlo.constant dense<[0, 3]> : tensor<2xi64>
    %24 = stablehlo.convert %arg0 : tensor<i64>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %26 = stablehlo.constant dense<10> : tensor<i64>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %28 = stablehlo.constant dense<10> : tensor<i64>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %30 = stablehlo.constant dense<10> : tensor<i64>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %32 = stablehlo.concatenate %25, %27, %29, %31, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %33 = stablehlo.convert %21 : (tensor<?x2xi32>) -> tensor<?x2xi64>
    %34 = stablehlo.constant dense<0> : tensor<i64>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %36 = stablehlo.compare  LT, %22, %35,  SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %37 = stablehlo.constant dense<4> : tensor<i64>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %39 = stablehlo.add %22, %38 : tensor<2xi64>
    %40 = stablehlo.select %36, %39, %22 : tensor<2xi1>, tensor<2xi64>
    %41 = stablehlo.convert %40 : (tensor<2xi64>) -> tensor<2xi32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %43 = "stablehlo.gather"(%32, %42) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<4xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %44 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %46 = stablehlo.constant dense<10> : tensor<i64>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %48 = stablehlo.constant dense<10> : tensor<i64>
    %49 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %50 = stablehlo.constant dense<1> : tensor<i64>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %52 = stablehlo.concatenate %45, %47, %49, %51, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %53 = stablehlo.constant dense<0> : tensor<i64>
    %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %55 = stablehlo.compare  LT, %23, %54,  SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %56 = stablehlo.constant dense<4> : tensor<i64>
    %57 = stablehlo.broadcast_in_dim %56, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %58 = stablehlo.add %23, %57 : tensor<2xi64>
    %59 = stablehlo.select %55, %58, %23 : tensor<2xi1>, tensor<2xi64>
    %60 = stablehlo.convert %59 : (tensor<2xi64>) -> tensor<2xi32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %62 = "stablehlo.gather"(%52, %61) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<4xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %63 = stablehlo.subtract %43, %62 : tensor<2xi64>
    %64 = stablehlo.constant dense<0> : tensor<i64>
    %65 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %66 = stablehlo.reshape %65 : (tensor<i32>) -> tensor<1xi32>
    %67 = stablehlo.constant dense<2> : tensor<1xi32>
    %68 = stablehlo.concatenate %66, %67, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %69 = stablehlo.dynamic_broadcast_in_dim %64, %68, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x2xi64>
    %70 = stablehlo.compare  GE, %33, %69,  SIGNED : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %71 = stablehlo.broadcast_in_dim %63, dims = [1] : (tensor<2xi64>) -> tensor<1x2xi64>
    %72 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %73 = stablehlo.reshape %72 : (tensor<i32>) -> tensor<1xi32>
    %74 = stablehlo.constant dense<2> : tensor<1xi32>
    %75 = stablehlo.concatenate %73, %74, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %76 = stablehlo.dynamic_broadcast_in_dim %71, %75, dims = [0, 1] : (tensor<1x2xi64>, tensor<2xi32>) -> tensor<?x2xi64>
    %77 = stablehlo.compare  LE, %33, %76,  SIGNED : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %78 = stablehlo.and %70, %77 : tensor<?x2xi1>
    %79 = stablehlo.constant dense<true> : tensor<i1>
    %80 = stablehlo.reduce(%78 init: %79) across dimensions = [1] : (tensor<?x2xi1>, tensor<i1>) -> tensor<?xi1>
     reducer(%arg3: tensor<i1>, %arg4: tensor<i1>)  {
      %96 = stablehlo.and %arg3, %arg4 : tensor<i1>
      stablehlo.return %96 : tensor<i1>
    }
    %81 = "stablehlo.gather"(%arg1, %33) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0, 3], start_index_map = [0, 3], index_vector_dim = 1>, slice_sizes = dense<[1, 10, 10, 1]> : tensor<4xi64>} : (tensor<?x10x10x10xf32>, tensor<?x2xi64>) -> tensor<?x10x10xf32>
    %82 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %83 = stablehlo.reshape %82 : (tensor<i32>) -> tensor<1xi32>
    %84 = stablehlo.constant dense<10> : tensor<1xi32>
    %85 = stablehlo.constant dense<10> : tensor<1xi32>
    %86 = stablehlo.concatenate %83, %84, %85, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %87 = stablehlo.dynamic_broadcast_in_dim %80, %86, dims = [0] : (tensor<?xi1>, tensor<3xi32>) -> tensor<?x10x10xi1>
    %88 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %89 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %90 = stablehlo.reshape %89 : (tensor<i32>) -> tensor<1xi32>
    %91 = stablehlo.constant dense<10> : tensor<1xi32>
    %92 = stablehlo.constant dense<10> : tensor<1xi32>
    %93 = stablehlo.concatenate %90, %91, %92, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %94 = stablehlo.dynamic_broadcast_in_dim %88, %93, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x10x10xf32>
    %95 = stablehlo.select %87, %81, %94 : tensor<?x10x10xi1>, tensor<?x10x10xf32>
    return %95 : tensor<?x10x10xf32>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<?xi1>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.select %arg1, %arg2, %arg3 : tensor<?xi1>, tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

