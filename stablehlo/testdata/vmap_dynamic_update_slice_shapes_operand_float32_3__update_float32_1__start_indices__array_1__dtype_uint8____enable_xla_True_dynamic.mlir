// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3xf32> {mhlo.sharding = ""}, %arg2: tensor<?x1xf32> {mhlo.sharding = ""}, %arg3: tensor<?x1xui8> {mhlo.sharding = ""}) -> tensor<?x3xf32> {
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
    %10 = stablehlo.real_dynamic_slice %arg3, %2, %6, %9 : (tensor<?x1xui8>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xui8>
    %11 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.dynamic_reshape %10, %12 : (tensor<?x1xui8>, tensor<1xi32>) -> tensor<?xui8>
    %14 = stablehlo.constant dense<0> : tensor<ui8>
    %15 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.dynamic_broadcast_in_dim %14, %16, dims = [] : (tensor<ui8>, tensor<1xi32>) -> tensor<?xui8>
    %18 = stablehlo.compare  LT, %13, %17,  UNSIGNED : (tensor<?xui8>, tensor<?xui8>) -> tensor<?xi1>
    %19 = stablehlo.constant dense<3> : tensor<ui8>
    %20 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %21 = stablehlo.reshape %20 : (tensor<i32>) -> tensor<1xi32>
    %22 = stablehlo.dynamic_broadcast_in_dim %19, %21, dims = [] : (tensor<ui8>, tensor<1xi32>) -> tensor<?xui8>
    %23 = stablehlo.add %13, %22 : tensor<?xui8>
    %24 = stablehlo.select %18, %23, %13 : tensor<?xi1>, tensor<?xui8>
    %25 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.constant dense<1> : tensor<1xi32>
    %28 = stablehlo.concatenate %26, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = stablehlo.dynamic_broadcast_in_dim %24, %28, dims = [0] : (tensor<?xui8>, tensor<2xi32>) -> tensor<?x1xui8>
    %30 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %31 = stablehlo.reshape %30 : (tensor<i32>) -> tensor<1xi32>
    %32 = stablehlo.constant dense<1> : tensor<1xi32>
    %33 = stablehlo.concatenate %31, %32, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %34 = stablehlo.dynamic_iota %33, dim = 0 : (tensor<2xi32>) -> tensor<?x1xui8>
    %35 = stablehlo.concatenate %34, %29, dim = 1 : (tensor<?x1xui8>, tensor<?x1xui8>) -> tensor<?x2xui8>
    %36 = stablehlo.constant dense<-1> : tensor<i64>
    %37 = stablehlo.add %arg0, %36 : tensor<i64>
    %38 = stablehlo.convert %37 : tensor<i64>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %40 = stablehlo.constant dense<2> : tensor<i64>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %42 = stablehlo.concatenate %39, %41, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %43 = stablehlo.constant dense<255> : tensor<ui64>
    %44 = stablehlo.convert %43 : (tensor<ui64>) -> tensor<i64>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %46 = stablehlo.minimum %42, %45 : tensor<2xi64>
    %47 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %48 = stablehlo.reshape %47 : (tensor<i32>) -> tensor<1xi32>
    %49 = stablehlo.constant dense<2> : tensor<1xi32>
    %50 = stablehlo.concatenate %48, %49, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %51 = stablehlo.dynamic_broadcast_in_dim %46, %50, dims = [1] : (tensor<2xi64>, tensor<2xi32>) -> tensor<?x2xi64>
    %52 = stablehlo.convert %35 : (tensor<?x2xui8>) -> tensor<?x2xi64>
    %53 = stablehlo.constant dense<0> : tensor<i64>
    %54 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %55 = stablehlo.reshape %54 : (tensor<i32>) -> tensor<1xi32>
    %56 = stablehlo.constant dense<2> : tensor<1xi32>
    %57 = stablehlo.concatenate %55, %56, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %58 = stablehlo.dynamic_broadcast_in_dim %53, %57, dims = [] : (tensor<i64>, tensor<2xi32>) -> tensor<?x2xi64>
    %59 = stablehlo.clamp %58, %52, %51 : tensor<?x2xi64>
    %60 = "stablehlo.scatter"(%arg1, %59, %arg2) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      stablehlo.return %arg5 : tensor<f32>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<?x3xf32>, tensor<?x2xi64>, tensor<?x1xf32>) -> tensor<?x3xf32>
    return %60 : tensor<?x3xf32>
  }
}

