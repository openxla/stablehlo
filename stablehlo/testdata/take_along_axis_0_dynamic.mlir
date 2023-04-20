// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}, %arg2: tensor<?x1xi32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = call @take_along_axis(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x2xf32>, tensor<?x1xi32>) -> tensor<?x2xf32>
    return %0 : tensor<?x2xf32>
  }
  func.func private @take_along_axis(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>, %arg2: tensor<?x1xi32>) -> tensor<?x2xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<?x1xi32>) -> tensor<?x1xi64>
    %1 = stablehlo.convert %arg0 : tensor<i64>
    %2 = stablehlo.constant dense<0> : tensor<i64>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.constant dense<1> : tensor<1xi32>
    %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.dynamic_broadcast_in_dim %2, %6, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %8 = stablehlo.compare  LT, %0, %7,  SIGNED : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi1>
    %9 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %13 = stablehlo.dynamic_broadcast_in_dim %1, %12, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %14 = stablehlo.add %0, %13 : tensor<?x1xi64>
    %15 = stablehlo.select %8, %14, %0 : tensor<?x1xi1>, tensor<?x1xi64>
    %16 = stablehlo.constant dense<0> : tensor<1xi64>
    %17 = stablehlo.constant dense<0> : tensor<1xi64>
    %18 = stablehlo.convert %arg0 : tensor<i64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %20 = stablehlo.constant dense<2> : tensor<i64>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %22 = stablehlo.concatenate %19, %21, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %23 = stablehlo.constant dense<0> : tensor<i64>
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %25 = stablehlo.compare  LT, %16, %24,  SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %26 = stablehlo.constant dense<2> : tensor<i64>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %28 = stablehlo.add %16, %27 : tensor<1xi64>
    %29 = stablehlo.select %25, %28, %16 : tensor<1xi1>, tensor<1xi64>
    %30 = stablehlo.convert %29 : (tensor<1xi64>) -> tensor<1xi32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %32 = "stablehlo.gather"(%22, %31) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xi64>, tensor<1x1xi32>) -> tensor<1xi64>
    %33 = stablehlo.constant dense<1> : tensor<i64>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %35 = stablehlo.constant dense<2> : tensor<i64>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %37 = stablehlo.concatenate %34, %36, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %38 = stablehlo.constant dense<0> : tensor<i64>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %40 = stablehlo.compare  LT, %17, %39,  SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %41 = stablehlo.constant dense<2> : tensor<i64>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %43 = stablehlo.add %17, %42 : tensor<1xi64>
    %44 = stablehlo.select %40, %43, %17 : tensor<1xi1>, tensor<1xi64>
    %45 = stablehlo.convert %44 : (tensor<1xi64>) -> tensor<1xi32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %47 = "stablehlo.gather"(%37, %46) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xi64>, tensor<1x1xi32>) -> tensor<1xi64>
    %48 = stablehlo.subtract %32, %47 : tensor<1xi64>
    %49 = stablehlo.constant dense<0> : tensor<i64>
    %50 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %51 = stablehlo.reshape %50 : (tensor<i32>) -> tensor<1xi32>
    %52 = stablehlo.constant dense<1> : tensor<1xi32>
    %53 = stablehlo.concatenate %51, %52, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %54 = stablehlo.dynamic_broadcast_in_dim %49, %53, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %55 = stablehlo.compare  GE, %15, %54,  SIGNED : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi1>
    %56 = stablehlo.broadcast_in_dim %48, dims = [1] : (tensor<1xi64>) -> tensor<1x1xi64>
    %57 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %58 = stablehlo.reshape %57 : (tensor<i32>) -> tensor<1xi32>
    %59 = stablehlo.constant dense<1> : tensor<1xi32>
    %60 = stablehlo.concatenate %58, %59, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %61 = stablehlo.dynamic_broadcast_in_dim %56, %60, dims = [0, 1] : (tensor<1x1xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %62 = stablehlo.compare  LE, %15, %61,  SIGNED : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi1>
    %63 = stablehlo.and %55, %62 : tensor<?x1xi1>
    %64 = stablehlo.constant dense<true> : tensor<i1>
    %65 = stablehlo.reduce(%63 init: %64) across dimensions = [1] : (tensor<?x1xi1>, tensor<i1>) -> tensor<?xi1>
     reducer(%arg3: tensor<i1>, %arg4: tensor<i1>)  {
      %79 = stablehlo.and %arg3, %arg4 : tensor<i1>
      stablehlo.return %79 : tensor<i1>
    }
    %66 = "stablehlo.gather"(%arg1, %15) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x2xf32>, tensor<?x1xi64>) -> tensor<?x2xf32>
    %67 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %68 = stablehlo.reshape %67 : (tensor<i32>) -> tensor<1xi32>
    %69 = stablehlo.constant dense<2> : tensor<1xi32>
    %70 = stablehlo.concatenate %68, %69, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %71 = stablehlo.dynamic_broadcast_in_dim %65, %70, dims = [0] : (tensor<?xi1>, tensor<2xi32>) -> tensor<?x2xi1>
    %72 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %73 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %74 = stablehlo.reshape %73 : (tensor<i32>) -> tensor<1xi32>
    %75 = stablehlo.constant dense<2> : tensor<1xi32>
    %76 = stablehlo.concatenate %74, %75, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %77 = stablehlo.dynamic_broadcast_in_dim %72, %76, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x2xf32>
    %78 = stablehlo.select %71, %66, %77 : tensor<?x2xi1>, tensor<?x2xf32>
    return %78 : tensor<?x2xf32>
  }
}

