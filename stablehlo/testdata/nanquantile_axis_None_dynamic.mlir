// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5xf32> {mhlo.sharding = ""}) -> tensor<f32> {
    %0 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %1 = call @nanquantile(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x5xf32>, tensor<f64>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @nanquantile(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<f64>) -> tensor<f32> {
    %0 = stablehlo.constant dense<5> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_reshape %arg1, %3 : (tensor<?x5xf32>, tensor<1xi32>) -> tensor<?xf32>
    %5 = call @isnan(%arg0, %4) : (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>
    %6 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %7 = call @_where(%arg0, %5, %6, %4) : (tensor<i64>, tensor<?xi1>, tensor<f64>, tensor<?xf32>) -> tensor<?xf32>
    %8 = "stablehlo.sort"(%7) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %59 = stablehlo.bitcast_convert %arg3 : (tensor<f32>) -> tensor<i32>
      %60 = stablehlo.bitcast_convert %arg3 : (tensor<f32>) -> tensor<ui32>
      %61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %62 = stablehlo.compare  EQ, %arg3, %61,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %63 = stablehlo.constant dense<0> : tensor<i32>
      %64 = stablehlo.select %62, %63, %59 : tensor<i1>, tensor<i32>
      %65 = stablehlo.compare  NE, %arg3, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %66 = stablehlo.constant dense<2143289344> : tensor<i32>
      %67 = stablehlo.select %65, %66, %64 : tensor<i1>, tensor<i32>
      %68 = stablehlo.constant dense<2147483647> : tensor<ui32>
      %69 = stablehlo.subtract %68, %60 : tensor<ui32>
      %70 = stablehlo.bitcast_convert %69 : (tensor<ui32>) -> tensor<i32>
      %71 = stablehlo.constant dense<0> : tensor<i32>
      %72 = stablehlo.compare  LT, %67, %71,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %73 = stablehlo.select %72, %70, %67 : tensor<i1>, tensor<i32>
      %74 = stablehlo.bitcast_convert %arg4 : (tensor<f32>) -> tensor<i32>
      %75 = stablehlo.bitcast_convert %arg4 : (tensor<f32>) -> tensor<ui32>
      %76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %77 = stablehlo.compare  EQ, %arg4, %76,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %78 = stablehlo.constant dense<0> : tensor<i32>
      %79 = stablehlo.select %77, %78, %74 : tensor<i1>, tensor<i32>
      %80 = stablehlo.compare  NE, %arg4, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %81 = stablehlo.constant dense<2143289344> : tensor<i32>
      %82 = stablehlo.select %80, %81, %79 : tensor<i1>, tensor<i32>
      %83 = stablehlo.constant dense<2147483647> : tensor<ui32>
      %84 = stablehlo.subtract %83, %75 : tensor<ui32>
      %85 = stablehlo.bitcast_convert %84 : (tensor<ui32>) -> tensor<i32>
      %86 = stablehlo.constant dense<0> : tensor<i32>
      %87 = stablehlo.compare  LT, %82, %86,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %88 = stablehlo.select %87, %85, %82 : tensor<i1>, tensor<i32>
      %89 = stablehlo.compare  LT, %73, %88,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %89 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<?xf32>) -> tensor<?xf32>
    %9 = call @isnan_0(%arg0, %8) : (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>
    %10 = stablehlo.not %9 : tensor<?xi1>
    %11 = stablehlo.convert %10 : (tensor<?xi1>) -> tensor<?xi32>
    %12 = stablehlo.convert %11 : (tensor<?xi32>) -> tensor<?xf64>
    %13 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.reduce(%12 init: %13) across dimensions = [0] : (tensor<?xf64>, tensor<f64>) -> tensor<f64>
     reducer(%arg3: tensor<f64>, %arg4: tensor<f64>)  {
      %59 = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %59 : tensor<f64>
    }
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
    %25 = stablehlo.minimum %18, %24 : tensor<f64>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %27 = stablehlo.maximum %26, %25 : tensor<f64>
    %28 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %29 = stablehlo.subtract %14, %28 : tensor<f64>
    %30 = stablehlo.minimum %19, %29 : tensor<f64>
    %31 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %32 = stablehlo.maximum %31, %30 : tensor<f64>
    %33 = stablehlo.convert %27 : (tensor<f64>) -> tensor<i64>
    %34 = stablehlo.convert %32 : (tensor<f64>) -> tensor<i64>
    %35 = stablehlo.constant dense<5> : tensor<i64>
    %36 = stablehlo.multiply %arg0, %35 : tensor<i64>
    %37 = stablehlo.convert %36 : tensor<i64>
    %38 = stablehlo.constant dense<0> : tensor<i64>
    %39 = stablehlo.compare  LT, %33, %38,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %40 = stablehlo.add %33, %37 : tensor<i64>
    %41 = stablehlo.select %39, %40, %33 : tensor<i1>, tensor<i64>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %43 = "stablehlo.gather"(%8, %42) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %44 = stablehlo.constant dense<5> : tensor<i64>
    %45 = stablehlo.multiply %arg0, %44 : tensor<i64>
    %46 = stablehlo.convert %45 : tensor<i64>
    %47 = stablehlo.constant dense<0> : tensor<i64>
    %48 = stablehlo.compare  LT, %34, %47,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %49 = stablehlo.add %34, %46 : tensor<i64>
    %50 = stablehlo.select %48, %49, %34 : tensor<i1>, tensor<i64>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %52 = "stablehlo.gather"(%8, %51) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %53 = stablehlo.convert %43 : (tensor<f32>) -> tensor<f64>
    %54 = stablehlo.multiply %53, %22 : tensor<f64>
    %55 = stablehlo.convert %52 : (tensor<f32>) -> tensor<f64>
    %56 = stablehlo.multiply %55, %20 : tensor<f64>
    %57 = stablehlo.add %54, %56 : tensor<f64>
    %58 = stablehlo.convert %57 : (tensor<f64>) -> tensor<f32>
    return %58 : tensor<f32>
  }
  func.func private @isnan(%arg0: tensor<i64>, %arg1: tensor<?xf32>) -> tensor<?xi1> {
    %0 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<?xi1>, %arg2: tensor<f64>, %arg3: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<f32>
    %1 = stablehlo.constant dense<5> : tensor<i64>
    %2 = stablehlo.multiply %arg0, %1 : tensor<i64>
    %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %0, %4, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %6 = stablehlo.select %arg1, %5, %arg3 : tensor<?xi1>, tensor<?xf32>
    return %6 : tensor<?xf32>
  }
  func.func private @isnan_0(%arg0: tensor<i64>, %arg1: tensor<?xf32>) -> tensor<?xi1> {
    %0 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
}

