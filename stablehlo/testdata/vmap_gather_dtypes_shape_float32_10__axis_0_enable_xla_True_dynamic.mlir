// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x10xf32> {mhlo.sharding = ""}, %arg2: tensor<?xi32> {mhlo.sharding = ""}) -> tensor<?xf32> {
    %0 = call @_take(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x10xf32>, tensor<?xi32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @_take(%arg0: tensor<i64>, %arg1: tensor<?x10xf32>, %arg2: tensor<?xi32>) -> tensor<?xf32> {
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
    %22 = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
    %23 = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
    %24 = stablehlo.convert %arg0 : tensor<i64>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %26 = stablehlo.constant dense<10> : tensor<i64>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %29 = stablehlo.convert %21 : (tensor<?x2xi32>) -> tensor<?x2xi64>
    %30 = stablehlo.constant dense<0> : tensor<i64>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %32 = stablehlo.compare  LT, %22, %31,  SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %33 = stablehlo.constant dense<2> : tensor<i64>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %35 = stablehlo.add %22, %34 : tensor<2xi64>
    %36 = stablehlo.select %32, %35, %22 : tensor<2xi1>, tensor<2xi64>
    %37 = stablehlo.convert %36 : (tensor<2xi64>) -> tensor<2xi32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %39 = "stablehlo.gather"(%28, %38) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %40 = stablehlo.constant dense<1> : tensor<i64>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %42 = stablehlo.constant dense<1> : tensor<i64>
    %43 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %44 = stablehlo.concatenate %41, %43, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %45 = stablehlo.constant dense<0> : tensor<i64>
    %46 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %47 = stablehlo.compare  LT, %23, %46,  SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %48 = stablehlo.constant dense<2> : tensor<i64>
    %49 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %50 = stablehlo.add %23, %49 : tensor<2xi64>
    %51 = stablehlo.select %47, %50, %23 : tensor<2xi1>, tensor<2xi64>
    %52 = stablehlo.convert %51 : (tensor<2xi64>) -> tensor<2xi32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %54 = "stablehlo.gather"(%44, %53) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %55 = stablehlo.subtract %39, %54 : tensor<2xi64>
    %56 = stablehlo.constant dense<0> : tensor<i64>
    %57 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %58 = stablehlo.reshape %57 : (tensor<i32>) -> tensor<1xi32>
    %59 = stablehlo.constant dense<2> : tensor<1xi32>
    %60 = stablehlo.concatenate %58, %59, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %61 = stablehlo.dynamic_broadcast_in_dim %56, %60, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x2xi64>
    %62 = stablehlo.compare  GE, %29, %61,  SIGNED : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %63 = stablehlo.broadcast_in_dim %55, dims = [1] : (tensor<2xi64>) -> tensor<1x2xi64>
    %64 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %65 = stablehlo.reshape %64 : (tensor<i32>) -> tensor<1xi32>
    %66 = stablehlo.constant dense<2> : tensor<1xi32>
    %67 = stablehlo.concatenate %65, %66, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %68 = stablehlo.dynamic_broadcast_in_dim %63, %67, dims = [0, 1] : (tensor<1x2xi64>, tensor<2xi32>) -> tensor<?x2xi64>
    %69 = stablehlo.compare  LE, %29, %68,  SIGNED : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %70 = stablehlo.and %62, %69 : tensor<?x2xi1>
    %71 = stablehlo.constant dense<true> : tensor<i1>
    %72 = stablehlo.reduce(%70 init: %71) across dimensions = [1] : (tensor<?x2xi1>, tensor<i1>) -> tensor<?xi1>
     reducer(%arg3: tensor<i1>, %arg4: tensor<i1>)  {
      %79 = stablehlo.and %arg3, %arg4 : tensor<i1>
      stablehlo.return %79 : tensor<i1>
    }
    %73 = "stablehlo.gather"(%arg1, %29) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<?x10xf32>, tensor<?x2xi64>) -> tensor<?xf32>
    %74 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %75 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %76 = stablehlo.reshape %75 : (tensor<i32>) -> tensor<1xi32>
    %77 = stablehlo.dynamic_broadcast_in_dim %74, %76, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %78 = stablehlo.select %72, %73, %77 : tensor<?xi1>, tensor<?xf32>
    return %78 : tensor<?xf32>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<?xi1>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.select %arg1, %arg2, %arg3 : tensor<?xi1>, tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

