// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xf32> {mhlo.sharding = ""}, %arg2: tensor<?x20x20xf32> {mhlo.sharding = ""}) -> tensor<?x20x20xf32> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<2147483647> : tensor<i32>
    %5 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %6 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %7 = stablehlo.bitcast_convert %arg1 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %8 = stablehlo.bitcast_convert %arg2 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %9 = stablehlo.compare  NE, %arg1, %arg1 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %10 = stablehlo.compare  NE, %arg2, %arg2 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %11 = stablehlo.or %9, %10 : tensor<?x20x20xi1>
    %12 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %13 = stablehlo.reshape %12 : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.concatenate %13, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %15 = stablehlo.dynamic_broadcast_in_dim %6, %14, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %17 = stablehlo.get_dimension_size %7, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %18 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
    %19 = stablehlo.concatenate %18, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %20 = stablehlo.dynamic_broadcast_in_dim %5, %19, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %21 = stablehlo.get_dimension_size %7, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %22 = stablehlo.reshape %21 : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.concatenate %22, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %24 = stablehlo.dynamic_broadcast_in_dim %4, %23, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %25 = stablehlo.and %7, %24 : tensor<?x20x20xi32>
    %26 = stablehlo.and %8, %24 : tensor<?x20x20xi32>
    %27 = stablehlo.compare  EQ, %arg1, %arg2 : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %28 = stablehlo.get_dimension_size %7, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %29 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.concatenate %29, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %31 = stablehlo.dynamic_broadcast_in_dim %3, %30, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %32 = stablehlo.compare  EQ, %25, %31 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %33 = stablehlo.compare  EQ, %26, %31 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %34 = stablehlo.and %7, %20 : tensor<?x20x20xi32>
    %35 = stablehlo.and %8, %20 : tensor<?x20x20xi32>
    %36 = stablehlo.get_dimension_size %7, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %37 = stablehlo.reshape %36 : (tensor<i32>) -> tensor<1xi32>
    %38 = stablehlo.concatenate %37, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %39 = stablehlo.dynamic_broadcast_in_dim %2, %38, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %40 = stablehlo.or %35, %39 : tensor<?x20x20xi32>
    %41 = stablehlo.compare  NE, %34, %35 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %42 = stablehlo.compare  GT, %25, %26 : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %43 = stablehlo.or %42, %41 : tensor<?x20x20xi1>
    %44 = stablehlo.get_dimension_size %7, dim = 0 : (tensor<?x20x20xi32>) -> tensor<i32>
    %45 = stablehlo.reshape %44 : (tensor<i32>) -> tensor<1xi32>
    %46 = stablehlo.concatenate %45, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %47 = stablehlo.dynamic_broadcast_in_dim %1, %46, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %48 = stablehlo.select %43, %47, %39 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %49 = stablehlo.add %7, %48 : tensor<?x20x20xi32>
    %50 = stablehlo.select %33, %8, %40 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %51 = stablehlo.select %32, %50, %49 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %52 = stablehlo.select %27, %8, %51 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %53 = stablehlo.select %11, %16, %52 : tensor<?x20x20xi1>, tensor<?x20x20xi32>
    %54 = stablehlo.bitcast_convert %53 : (tensor<?x20x20xi32>) -> tensor<?x20x20xf32>
    return %54 : tensor<?x20x20xf32>
  }
}

