// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<2> : tensor<i64>
    %1 = call @_roll(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x4xf32>, tensor<i64>) -> tensor<?x4xf32>
    return %1 : tensor<?x4xf32>
  }
  func.func private @_roll(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<i64>) -> tensor<?x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.maximum %3, %5 : tensor<i32>
    %7 = call @remainder(%arg0, %4, %6) : (tensor<i64>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %8 = stablehlo.concatenate %arg1, %arg1, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
    %9 = stablehlo.subtract %3, %7 : tensor<i32>
    %10 = stablehlo.constant dense<2> : tensor<i64>
    %11 = stablehlo.multiply %arg0, %10 : tensor<i64>
    %12 = stablehlo.convert %11 : (tensor<i64>) -> tensor<i32>
    %13 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.compare  LT, %9, %13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %15 = stablehlo.add %9, %12 : tensor<i32>
    %16 = stablehlo.select %14, %15, %9 : tensor<i1>, tensor<i32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.constant dense<4> : tensor<i32>
    %19 = stablehlo.add %17, %18 : tensor<i32>
    %20 = stablehlo.constant dense<false> : tensor<i1>
    %21 = stablehlo.constant dense<0> : tensor<i32>
    %22 = stablehlo.select %20, %19, %21 : tensor<i1>, tensor<i32>
    %23 = stablehlo.convert %16 : tensor<i32>
    %24 = stablehlo.reshape %23 : (tensor<i32>) -> tensor<1xi32>
    %25 = stablehlo.convert %22 : tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.concatenate %24, %26, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %28 = stablehlo.convert %16 : tensor<i32>
    %29 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.convert %22 : tensor<i32>
    %31 = stablehlo.reshape %30 : (tensor<i32>) -> tensor<1xi32>
    %32 = stablehlo.concatenate %29, %31, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %33 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %34 = stablehlo.reshape %33 : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.constant dense<4> : tensor<1xi32>
    %36 = stablehlo.concatenate %34, %35, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %37 = stablehlo.add %32, %36 : tensor<2xi32>
    %38 = stablehlo.constant dense<1> : tensor<1xi32>
    %39 = stablehlo.constant dense<1> : tensor<1xi32>
    %40 = stablehlo.concatenate %38, %39, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %41 = stablehlo.real_dynamic_slice %8, %27, %37, %40 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    return %41 : tensor<?x4xf32>
  }
  func.func private @remainder(%arg0: tensor<i64>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  EQ, %arg2, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = call @_where(%arg0, %1, %2, %arg2) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = stablehlo.remainder %arg1, %3 : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.compare  NE, %4, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %4, %7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.compare  LT, %3, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.compare  NE, %8, %10,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %12 = stablehlo.and %11, %6 : tensor<i1>
    %13 = stablehlo.add %4, %3 : tensor<i32>
    %14 = stablehlo.select %12, %13, %4 : tensor<i1>, tensor<i32>
    return %14 : tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg1, %arg2, %arg3 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}

