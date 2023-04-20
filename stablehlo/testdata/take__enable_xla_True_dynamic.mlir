// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x5xf32> {mhlo.sharding = ""}, %arg2: tensor<2xi32> {mhlo.sharding = ""}) -> tensor<?x2x5xf32> {
    %0 = call @_take(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x4x5xf32>, tensor<2xi32>) -> tensor<?x2x5xf32>
    return %0 : tensor<?x2x5xf32>
  }
  func.func private @_take(%arg0: tensor<i64>, %arg1: tensor<?x4x5xf32>, %arg2: tensor<2xi32>) -> tensor<?x2x5xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2 = stablehlo.compare  LT, %arg2, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %3 = stablehlo.constant dense<4> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %5 = stablehlo.add %arg2, %4 : tensor<2xi32>
    %6 = call @_where(%arg0, %2, %5, %arg2) : (tensor<i64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1xi64>
    %9 = stablehlo.constant dense<1> : tensor<1xi64>
    %10 = stablehlo.convert %arg0 : tensor<i64>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %12 = stablehlo.constant dense<4> : tensor<i64>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %14 = stablehlo.constant dense<5> : tensor<i64>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %16 = stablehlo.concatenate %11, %13, %15, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %17 = stablehlo.convert %7 : (tensor<2x1xi32>) -> tensor<2x1xi64>
    %18 = stablehlo.constant dense<0> : tensor<i64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %20 = stablehlo.compare  LT, %8, %19,  SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %21 = stablehlo.constant dense<3> : tensor<i64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %23 = stablehlo.add %8, %22 : tensor<1xi64>
    %24 = stablehlo.select %20, %23, %8 : tensor<1xi1>, tensor<1xi64>
    %25 = stablehlo.convert %24 : (tensor<1xi64>) -> tensor<1xi32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %27 = "stablehlo.gather"(%16, %26) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi64>, tensor<1x1xi32>) -> tensor<1xi64>
    %28 = stablehlo.convert %arg0 : tensor<i64>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %30 = stablehlo.constant dense<1> : tensor<i64>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %32 = stablehlo.constant dense<5> : tensor<i64>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %34 = stablehlo.concatenate %29, %31, %33, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %35 = stablehlo.constant dense<0> : tensor<i64>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %37 = stablehlo.compare  LT, %9, %36,  SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %38 = stablehlo.constant dense<3> : tensor<i64>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %40 = stablehlo.add %9, %39 : tensor<1xi64>
    %41 = stablehlo.select %37, %40, %9 : tensor<1xi1>, tensor<1xi64>
    %42 = stablehlo.convert %41 : (tensor<1xi64>) -> tensor<1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %44 = "stablehlo.gather"(%34, %43) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi64>, tensor<1x1xi32>) -> tensor<1xi64>
    %45 = stablehlo.subtract %27, %44 : tensor<1xi64>
    %46 = stablehlo.constant dense<0> : tensor<i64>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<i64>) -> tensor<2x1xi64>
    %48 = stablehlo.compare  GE, %17, %47,  SIGNED : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi1>
    %49 = stablehlo.broadcast_in_dim %45, dims = [1] : (tensor<1xi64>) -> tensor<1x1xi64>
    %50 = stablehlo.broadcast_in_dim %49, dims = [0, 1] : (tensor<1x1xi64>) -> tensor<2x1xi64>
    %51 = stablehlo.compare  LE, %17, %50,  SIGNED : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi1>
    %52 = stablehlo.and %48, %51 : tensor<2x1xi1>
    %53 = stablehlo.constant dense<true> : tensor<i1>
    %54 = stablehlo.reduce(%52 init: %53) across dimensions = [1] : (tensor<2x1xi1>, tensor<i1>) -> tensor<2xi1>
     reducer(%arg3: tensor<i1>, %arg4: tensor<i1>)  {
      %75 = stablehlo.and %arg3, %arg4 : tensor<i1>
      stablehlo.return %75 : tensor<i1>
    }
    %55 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %56 = stablehlo.reshape %55 : (tensor<i32>) -> tensor<1xi32>
    %57 = stablehlo.constant dense<1> : tensor<1xi32>
    %58 = stablehlo.constant dense<5> : tensor<1xi32>
    %59 = stablehlo.concatenate %56, %57, %58, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %60 = "stablehlo.dynamic_gather"(%arg1, %17, %59) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>} : (tensor<?x4x5xf32>, tensor<2x1xi64>, tensor<3xi32>) -> tensor<?x2x5xf32>
    %61 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %62 = stablehlo.reshape %61 : (tensor<i32>) -> tensor<1xi32>
    %63 = stablehlo.constant dense<2> : tensor<1xi32>
    %64 = stablehlo.constant dense<5> : tensor<1xi32>
    %65 = stablehlo.concatenate %62, %63, %64, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %66 = stablehlo.dynamic_broadcast_in_dim %54, %65, dims = [1] : (tensor<2xi1>, tensor<3xi32>) -> tensor<?x2x5xi1>
    %67 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %68 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %69 = stablehlo.reshape %68 : (tensor<i32>) -> tensor<1xi32>
    %70 = stablehlo.constant dense<2> : tensor<1xi32>
    %71 = stablehlo.constant dense<5> : tensor<1xi32>
    %72 = stablehlo.concatenate %69, %70, %71, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %73 = stablehlo.dynamic_broadcast_in_dim %67, %72, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x2x5xf32>
    %74 = stablehlo.select %66, %60, %73 : tensor<?x2x5xi1>, tensor<?x2x5xf32>
    return %74 : tensor<?x2x5xf32>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<2xi1>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>) -> tensor<2xi32> {
    %0 = stablehlo.select %arg1, %arg2, %arg3 : tensor<2xi1>, tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

