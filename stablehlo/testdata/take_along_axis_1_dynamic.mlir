// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}, %arg2: tensor<?x1xi32> {mhlo.sharding = ""}) -> tensor<?x1xf32> {
    %0 = call @take_along_axis(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x2xf32>, tensor<?x1xi32>) -> tensor<?x1xf32>
    return %0 : tensor<?x1xf32>
  }
  func.func private @take_along_axis(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>, %arg2: tensor<?x1xi32>) -> tensor<?x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<?x1xi32>) -> tensor<?x1xi64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<1> : tensor<1xi32>
    %4 = stablehlo.constant dense<1> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_iota %5, dim = 0 : (tensor<3xi32>) -> tensor<?x1x1xi64>
    %7 = stablehlo.constant dense<0> : tensor<i64>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<1> : tensor<1xi32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = stablehlo.dynamic_broadcast_in_dim %7, %11, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %13 = stablehlo.compare  LT, %0, %12,  SIGNED : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi1>
    %14 = stablehlo.constant dense<2> : tensor<i64>
    %15 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.constant dense<1> : tensor<1xi32>
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = stablehlo.dynamic_broadcast_in_dim %14, %18, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %20 = stablehlo.add %0, %19 : tensor<?x1xi64>
    %21 = stablehlo.select %13, %20, %0 : tensor<?x1xi1>, tensor<?x1xi64>
    %22 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %23 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
    %24 = stablehlo.constant dense<1> : tensor<1xi32>
    %25 = stablehlo.constant dense<1> : tensor<1xi32>
    %26 = stablehlo.concatenate %23, %24, %25, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %27 = stablehlo.dynamic_reshape %21, %26 : (tensor<?x1xi64>, tensor<3xi32>) -> tensor<?x1x1xi64>
    %28 = stablehlo.concatenate %6, %27, dim = 2 : (tensor<?x1x1xi64>, tensor<?x1x1xi64>) -> tensor<?x1x2xi64>
    %29 = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
    %30 = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
    %31 = stablehlo.convert %arg0 : tensor<i64>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %33 = stablehlo.constant dense<2> : tensor<i64>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %35 = stablehlo.concatenate %32, %34, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %36 = stablehlo.constant dense<0> : tensor<i64>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %38 = stablehlo.compare  LT, %29, %37,  SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %39 = stablehlo.constant dense<2> : tensor<i64>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %41 = stablehlo.add %29, %40 : tensor<2xi64>
    %42 = stablehlo.select %38, %41, %29 : tensor<2xi1>, tensor<2xi64>
    %43 = stablehlo.convert %42 : (tensor<2xi64>) -> tensor<2xi32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %45 = "stablehlo.gather"(%35, %44) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %46 = stablehlo.constant dense<1> : tensor<i64>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %48 = stablehlo.constant dense<1> : tensor<i64>
    %49 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %50 = stablehlo.concatenate %47, %49, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %51 = stablehlo.constant dense<0> : tensor<i64>
    %52 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %53 = stablehlo.compare  LT, %30, %52,  SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %54 = stablehlo.constant dense<2> : tensor<i64>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %56 = stablehlo.add %30, %55 : tensor<2xi64>
    %57 = stablehlo.select %53, %56, %30 : tensor<2xi1>, tensor<2xi64>
    %58 = stablehlo.convert %57 : (tensor<2xi64>) -> tensor<2xi32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %60 = "stablehlo.gather"(%50, %59) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %61 = stablehlo.subtract %45, %60 : tensor<2xi64>
    %62 = stablehlo.constant dense<0> : tensor<i64>
    %63 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %64 = stablehlo.reshape %63 : (tensor<i32>) -> tensor<1xi32>
    %65 = stablehlo.constant dense<1> : tensor<1xi32>
    %66 = stablehlo.constant dense<2> : tensor<1xi32>
    %67 = stablehlo.concatenate %64, %65, %66, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %68 = stablehlo.dynamic_broadcast_in_dim %62, %67, dims = [] : (tensor<i64>, tensor<3xi32>) -> tensor<?x1x2xi64>
    %69 = stablehlo.compare  GE, %28, %68,  SIGNED : (tensor<?x1x2xi64>, tensor<?x1x2xi64>) -> tensor<?x1x2xi1>
    %70 = stablehlo.broadcast_in_dim %61, dims = [2] : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %71 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %72 = stablehlo.reshape %71 : (tensor<i32>) -> tensor<1xi32>
    %73 = stablehlo.constant dense<1> : tensor<1xi32>
    %74 = stablehlo.constant dense<2> : tensor<1xi32>
    %75 = stablehlo.concatenate %72, %73, %74, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %76 = stablehlo.dynamic_broadcast_in_dim %70, %75, dims = [0, 1, 2] : (tensor<1x1x2xi64>, tensor<3xi32>) -> tensor<?x1x2xi64>
    %77 = stablehlo.compare  LE, %28, %76,  SIGNED : (tensor<?x1x2xi64>, tensor<?x1x2xi64>) -> tensor<?x1x2xi1>
    %78 = stablehlo.and %69, %77 : tensor<?x1x2xi1>
    %79 = stablehlo.constant dense<true> : tensor<i1>
    %80 = stablehlo.reduce(%78 init: %79) across dimensions = [2] : (tensor<?x1x2xi1>, tensor<i1>) -> tensor<?x1xi1>
     reducer(%arg3: tensor<i1>, %arg4: tensor<i1>)  {
      %89 = stablehlo.and %arg3, %arg4 : tensor<i1>
      stablehlo.return %89 : tensor<i1>
    }
    %81 = "stablehlo.gather"(%arg1, %28) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<?x2xf32>, tensor<?x1x2xi64>) -> tensor<?x1xf32>
    %82 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %83 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %84 = stablehlo.reshape %83 : (tensor<i32>) -> tensor<1xi32>
    %85 = stablehlo.constant dense<1> : tensor<1xi32>
    %86 = stablehlo.concatenate %84, %85, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %87 = stablehlo.dynamic_broadcast_in_dim %82, %86, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x1xf32>
    %88 = stablehlo.select %80, %81, %87 : tensor<?x1xi1>, tensor<?x1xf32>
    return %88 : tensor<?x1xf32>
  }
}

