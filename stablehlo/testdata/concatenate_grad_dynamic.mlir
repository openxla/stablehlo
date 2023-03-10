// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<?x?x5xf32> {mhlo.sharding = ""}) -> tensor<?x?x5xf32> {
    %0 = stablehlo.cosine %arg2 : tensor<?x?x5xf32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.constant dense<2> : tensor<i64>
    %3 = stablehlo.multiply %arg0, %2 : tensor<i64>
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.constant dense<5> : tensor<1xi32>
    %9 = stablehlo.concatenate %5, %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %10 = stablehlo.dynamic_broadcast_in_dim %1, %9, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x5xf32>
    %11 = stablehlo.constant dense<0> : tensor<1xi32>
    %12 = stablehlo.constant dense<0> : tensor<1xi32>
    %13 = stablehlo.constant dense<0> : tensor<1xi32>
    %14 = stablehlo.concatenate %11, %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %15 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %18 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
    %19 = stablehlo.constant dense<5> : tensor<1xi32>
    %20 = stablehlo.concatenate %16, %18, %19, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %21 = stablehlo.constant dense<1> : tensor<1xi32>
    %22 = stablehlo.constant dense<1> : tensor<1xi32>
    %23 = stablehlo.constant dense<1> : tensor<1xi32>
    %24 = stablehlo.concatenate %21, %22, %23, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %25 = stablehlo.real_dynamic_slice %10, %14, %20, %24 : (tensor<?x?x5xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x?x5xf32>
    %26 = stablehlo.constant dense<2> : tensor<i64>
    %27 = stablehlo.multiply %arg0, %26 : tensor<i64>
    %28 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %29 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.constant dense<0> : tensor<1xi32>
    %31 = stablehlo.constant dense<0> : tensor<1xi32>
    %32 = stablehlo.concatenate %29, %30, %31, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %33 = stablehlo.convert %27 : (tensor<i64>) -> tensor<i32>
    %34 = stablehlo.reshape %33 : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %36 = stablehlo.reshape %35 : (tensor<i32>) -> tensor<1xi32>
    %37 = stablehlo.constant dense<5> : tensor<1xi32>
    %38 = stablehlo.concatenate %34, %36, %37, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %39 = stablehlo.constant dense<1> : tensor<1xi32>
    %40 = stablehlo.constant dense<1> : tensor<1xi32>
    %41 = stablehlo.constant dense<1> : tensor<1xi32>
    %42 = stablehlo.concatenate %39, %40, %41, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %43 = stablehlo.real_dynamic_slice %10, %32, %38, %42 : (tensor<?x?x5xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x?x5xf32>
    %44 = stablehlo.multiply %43, %0 : tensor<?x?x5xf32>
    %45 = stablehlo.add %25, %44 : tensor<?x?x5xf32>
    return %45 : tensor<?x?x5xf32>
  }
}

