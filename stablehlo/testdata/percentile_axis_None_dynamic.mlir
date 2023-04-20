// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5xf32> {mhlo.sharding = ""}) -> tensor<f32> {
    %0 = stablehlo.constant dense<50> : tensor<i64>
    %1 = call @percentile(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x5xf32>, tensor<i64>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @percentile(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<i64>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.constant dense<1.000000e+02> : tensor<f64>
    %2 = stablehlo.divide %0, %1 : tensor<f64>
    %3 = call @quantile(%arg0, %arg1, %2) : (tensor<i64>, tensor<?x5xf32>, tensor<f64>) -> tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @quantile(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<f64>) -> tensor<f32> {
    %0 = stablehlo.constant dense<5> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_reshape %arg1, %3 : (tensor<?x5xf32>, tensor<1xi32>) -> tensor<?xf32>
    %5 = call @isnan(%arg0, %4) : (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>
    %6 = stablehlo.constant dense<false> : tensor<i1>
    %7 = stablehlo.reduce(%5 init: %6) across dimensions = [0] : (tensor<?xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg3: tensor<i1>, %arg4: tensor<i1>)  {
      %43 = stablehlo.or %arg3, %arg4 : tensor<i1>
      stablehlo.return %43 : tensor<i1>
    }
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %9 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %10 = call @_where(%arg0, %8, %9, %4) : (tensor<i64>, tensor<1xi1>, tensor<f64>, tensor<?xf32>) -> tensor<?xf32>
    %11 = "stablehlo.sort"(%10) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %43 = stablehlo.bitcast_convert %arg3 : (tensor<f32>) -> tensor<i32>
      %44 = stablehlo.bitcast_convert %arg3 : (tensor<f32>) -> tensor<ui32>
      %45 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %46 = stablehlo.compare  EQ, %arg3, %45,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %47 = stablehlo.constant dense<0> : tensor<i32>
      %48 = stablehlo.select %46, %47, %43 : tensor<i1>, tensor<i32>
      %49 = stablehlo.compare  NE, %arg3, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %50 = stablehlo.constant dense<2143289344> : tensor<i32>
      %51 = stablehlo.select %49, %50, %48 : tensor<i1>, tensor<i32>
      %52 = stablehlo.constant dense<2147483647> : tensor<ui32>
      %53 = stablehlo.subtract %52, %44 : tensor<ui32>
      %54 = stablehlo.bitcast_convert %53 : (tensor<ui32>) -> tensor<i32>
      %55 = stablehlo.constant dense<0> : tensor<i32>
      %56 = stablehlo.compare  LT, %51, %55,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %57 = stablehlo.select %56, %54, %51 : tensor<i1>, tensor<i32>
      %58 = stablehlo.bitcast_convert %arg4 : (tensor<f32>) -> tensor<i32>
      %59 = stablehlo.bitcast_convert %arg4 : (tensor<f32>) -> tensor<ui32>
      %60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %61 = stablehlo.compare  EQ, %arg4, %60,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %62 = stablehlo.constant dense<0> : tensor<i32>
      %63 = stablehlo.select %61, %62, %58 : tensor<i1>, tensor<i32>
      %64 = stablehlo.compare  NE, %arg4, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %65 = stablehlo.constant dense<2143289344> : tensor<i32>
      %66 = stablehlo.select %64, %65, %63 : tensor<i1>, tensor<i32>
      %67 = stablehlo.constant dense<2147483647> : tensor<ui32>
      %68 = stablehlo.subtract %67, %59 : tensor<ui32>
      %69 = stablehlo.bitcast_convert %68 : (tensor<ui32>) -> tensor<i32>
      %70 = stablehlo.constant dense<0> : tensor<i32>
      %71 = stablehlo.compare  LT, %66, %70,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %72 = stablehlo.select %71, %69, %66 : tensor<i1>, tensor<i32>
      %73 = stablehlo.compare  LT, %57, %72,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %73 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<?xf32>) -> tensor<?xf32>
    %12 = stablehlo.constant dense<5> : tensor<i64>
    %13 = stablehlo.multiply %arg0, %12 : tensor<i64>
    %14 = stablehlo.convert %13 : (tensor<i64>) -> tensor<f64>
    %15 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %16 = stablehlo.subtract %14, %15 : tensor<f64>
    %17 = stablehlo.multiply %arg2, %16 : tensor<f64>
    %18 = stablehlo.floor %17 : tensor<f64>
    %19 = stablehlo.ceil %17 : tensor<f64>
    %20 = stablehlo.subtract %17, %18 : tensor<f64>
    %21 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %22 = stablehlo.subtract %21, %20 : tensor<f64>
    %23 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %24 = stablehlo.subtract %14, %23 : tensor<f64>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %26 = stablehlo.clamp %25, %18, %24 : tensor<f64>
    %27 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %28 = stablehlo.subtract %14, %27 : tensor<f64>
    %29 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %30 = stablehlo.clamp %29, %19, %28 : tensor<f64>
    %31 = stablehlo.convert %26 : (tensor<f64>) -> tensor<i64>
    %32 = stablehlo.convert %30 : (tensor<f64>) -> tensor<i64>
    %33 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %34 = "stablehlo.gather"(%11, %33) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %35 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %36 = "stablehlo.gather"(%11, %35) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %37 = stablehlo.convert %34 : (tensor<f32>) -> tensor<f64>
    %38 = stablehlo.multiply %37, %22 : tensor<f64>
    %39 = stablehlo.convert %36 : (tensor<f32>) -> tensor<f64>
    %40 = stablehlo.multiply %39, %20 : tensor<f64>
    %41 = stablehlo.add %38, %40 : tensor<f64>
    %42 = stablehlo.convert %41 : (tensor<f64>) -> tensor<f32>
    return %42 : tensor<f32>
  }
  func.func private @isnan(%arg0: tensor<i64>, %arg1: tensor<?xf32>) -> tensor<?xi1> {
    %0 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<1xi1>, %arg2: tensor<f64>, %arg3: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<f32>
    %1 = stablehlo.reshape %arg1 : (tensor<1xi1>) -> tensor<i1>
    %2 = stablehlo.constant dense<5> : tensor<i64>
    %3 = stablehlo.multiply %arg0, %2 : tensor<i64>
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
    %7 = stablehlo.constant dense<5> : tensor<i64>
    %8 = stablehlo.multiply %arg0, %7 : tensor<i64>
    %9 = stablehlo.convert %8 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %0, %10, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %12 = stablehlo.select %6, %11, %arg3 : tensor<?xi1>, tensor<?xf32>
    return %12 : tensor<?xf32>
  }
}

