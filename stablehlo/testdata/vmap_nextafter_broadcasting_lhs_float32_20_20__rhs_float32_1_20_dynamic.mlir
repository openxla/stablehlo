// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xf32> {mhlo.sharding = ""}, %arg2: tensor<?x1x20xf32> {mhlo.sharding = ""}) -> tensor<?x20x20xf32> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<2147483647> : tensor<i32>
    %5 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %6 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %7 = stablehlo.constant dense<20> : tensor<1xi32>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.concatenate %9, %7, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %arg2, %10, dims = [0, 1, 2] : (tensor<?x1x20xf32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %12 = stablehlo.bitcast_convert %arg1 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %13 = stablehlo.bitcast_convert %11 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %14 = stablehlo.compare  NE, %arg1, %arg1 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %15 = stablehlo.compare  NE, %11, %11 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %16 = stablehlo.or %14, %15 : tensor<?x20x20xi1>
    %17 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %18 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
    %19 = stablehlo.concatenate %18, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %20 = stablehlo.dynamic_broadcast_in_dim %6, %19, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %21 = stablehlo.bitcast_convert %20 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %22 = stablehlo.get_dimension_size %12, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %23 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
    %24 = stablehlo.concatenate %23, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %25 = stablehlo.dynamic_broadcast_in_dim %5, %24, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %26 = stablehlo.get_dimension_size %12, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %27 = stablehlo.reshape %26 : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.concatenate %27, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %29 = stablehlo.dynamic_broadcast_in_dim %4, %28, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %30 = stablehlo.and %12, %29 : tensor<?x20x20xi32>
    %31 = stablehlo.and %13, %29 : tensor<?x20x20xi32>
    %32 = stablehlo.compare  EQ, %arg1, %11 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %33 = stablehlo.get_dimension_size %12, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %34 = stablehlo.reshape %33 : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.concatenate %34, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %36 = stablehlo.dynamic_broadcast_in_dim %3, %35, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %37 = stablehlo.compare  EQ, %30, %36 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %38 = stablehlo.compare  EQ, %31, %36 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %39 = stablehlo.and %12, %25 : tensor<?x20x20xi32>
    %40 = stablehlo.and %13, %25 : tensor<?x20x20xi32>
    %41 = stablehlo.get_dimension_size %12, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %42 = stablehlo.reshape %41 : (tensor<i32>) -> tensor<1xi32>
    %43 = stablehlo.concatenate %42, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %44 = stablehlo.dynamic_broadcast_in_dim %2, %43, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %45 = stablehlo.or %40, %44 : tensor<?x20x20xi32>
    %46 = stablehlo.compare  NE, %39, %40 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %47 = stablehlo.compare  GT, %30, %31 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %48 = stablehlo.or %47, %46 : tensor<?x20x20xi1>
    %49 = stablehlo.get_dimension_size %12, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %50 = stablehlo.reshape %49 : (tensor<i32>) -> tensor<1xi32>
    %51 = stablehlo.concatenate %50, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %52 = stablehlo.dynamic_broadcast_in_dim %1, %51, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %53 = stablehlo.select %48, %52, %44 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %54 = stablehlo.add %12, %53 : tensor<?x20x20xi32>
    %55 = stablehlo.select %38, %13, %45 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %56 = stablehlo.select %37, %55, %54 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %57 = stablehlo.select %32, %13, %56 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %58 = stablehlo.select %16, %21, %57 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %59 = stablehlo.bitcast_convert %58 : (tensor<?x20x20xi32>) -> tensor<?x20x20xf32>
    return %59 : tensor<?x20x20xf32>
  }
}

