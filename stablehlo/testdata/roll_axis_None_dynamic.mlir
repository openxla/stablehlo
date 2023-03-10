// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<2> : tensor<i64>
    %1 = call @_roll(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x4xf32>, tensor<i64>) -> tensor<?x4xf32>
    return %1 : tensor<?x4xf32>
  }
  func.func private @_roll(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<i64>) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<4> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_reshape %arg1, %3 : (tensor<?x4xf32>, tensor<1xi32>) -> tensor<?xf32>
    %5 = call @_roll_0(%arg0, %4, %arg2) : (tensor<i64>, tensor<?xf32>, tensor<i64>) -> tensor<?xf32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.constant dense<4> : tensor<1xi32>
    %9 = stablehlo.concatenate %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = stablehlo.dynamic_reshape %5, %9 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x4xf32>
    return %10 : tensor<?x4xf32>
  }
  func.func private @_roll_0(%arg0: tensor<i64>, %arg1: tensor<?xf32>, %arg2: tensor<i64>) -> tensor<?xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %3 = stablehlo.constant dense<4> : tensor<i64>
    %4 = stablehlo.multiply %arg0, %3 : tensor<i64>
    %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.constant dense<1> : tensor<i32>
    %8 = stablehlo.maximum %5, %7 : tensor<i32>
    %9 = call @remainder(%arg0, %6, %8) : (tensor<i64>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %10 = stablehlo.concatenate %arg1, %arg1, dim = 0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %11 = stablehlo.subtract %5, %9 : tensor<i32>
    %12 = stablehlo.constant dense<8> : tensor<i64>
    %13 = stablehlo.multiply %arg0, %12 : tensor<i64>
    %14 = stablehlo.convert %13 : (tensor<i64>) -> tensor<i32>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.compare  LT, %11, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %17 = stablehlo.add %11, %14 : tensor<i32>
    %18 = stablehlo.select %16, %17, %11 : tensor<i1>, tensor<i32>
    %19 = stablehlo.constant dense<4> : tensor<i64>
    %20 = stablehlo.multiply %arg0, %19 : tensor<i64>
    %21 = stablehlo.convert %18 : tensor<i32>
    %22 = stablehlo.reshape %21 : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.convert %18 : tensor<i32>
    %24 = stablehlo.reshape %23 : (tensor<i32>) -> tensor<1xi32>
    %25 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.add %24, %26 : tensor<1xi32>
    %28 = stablehlo.constant dense<1> : tensor<1xi32>
    %29 = stablehlo.real_dynamic_slice %10, %22, %27, %28 : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
    return %29 : tensor<?xf32>
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

