// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<2xi32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = "stablehlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %1 = stablehlo.reshape %0 : (tensor<1xi32>) -> tensor<i32>
    %2 = "stablehlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.compare  LT, %1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.add %1, %4 : tensor<i32>
    %8 = stablehlo.select %6, %7, %1 : tensor<i1>, tensor<i32>
    %9 = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.compare  LT, %3, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.constant dense<4> : tensor<i32>
    %12 = stablehlo.add %3, %11 : tensor<i32>
    %13 = stablehlo.select %10, %12, %3 : tensor<i1>, tensor<i32>
    %14 = stablehlo.convert %8 : tensor<i32>
    %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
    %16 = stablehlo.convert %13 : tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.concatenate %15, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = stablehlo.convert %8 : tensor<i32>
    %20 = stablehlo.reshape %19 : (tensor<i32>) -> tensor<1xi32>
    %21 = stablehlo.convert %13 : tensor<i32>
    %22 = stablehlo.reshape %21 : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.concatenate %20, %22, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %24 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<2> : tensor<1xi32>
    %27 = stablehlo.concatenate %25, %26, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %28 = stablehlo.add %23, %27 : tensor<2xi32>
    %29 = stablehlo.constant dense<1> : tensor<1xi32>
    %30 = stablehlo.constant dense<1> : tensor<1xi32>
    %31 = stablehlo.concatenate %29, %30, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %32 = stablehlo.real_dynamic_slice %arg1, %18, %28, %31 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x2xf32>
    return %32 : tensor<?x2xf32>
  }
}

