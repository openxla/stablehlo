// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<i32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %arg2, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.add %arg2, %0 : tensor<i32>
    %4 = stablehlo.select %2, %3, %arg2 : tensor<i1>, tensor<i32>
    %5 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.convert %4 : tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.convert %5 : tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.concatenate %7, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %11 = stablehlo.convert %4 : tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.convert %5 : tensor<i32>
    %14 = stablehlo.reshape %13 : (tensor<i32>) -> tensor<1xi32>
    %15 = stablehlo.concatenate %12, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.constant dense<2> : tensor<1xi32>
    %19 = stablehlo.concatenate %17, %18, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %20 = stablehlo.add %15, %19 : tensor<2xi32>
    %21 = stablehlo.constant dense<1> : tensor<1xi32>
    %22 = stablehlo.constant dense<1> : tensor<1xi32>
    %23 = stablehlo.concatenate %21, %22, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %24 = stablehlo.real_dynamic_slice %arg1, %10, %20, %23 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x2xf32>
    return %24 : tensor<?x2xf32>
  }
}

