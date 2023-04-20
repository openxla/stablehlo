// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x5xf32> {mhlo.sharding = ""}) -> tensor<?x5xi32> {
    %0 = call @argmax(%arg0, %arg1) : (tensor<i64>, tensor<?x4x5xf32>) -> tensor<?x5xi32>
    return %0 : tensor<?x5xi32>
  }
  func.func private @argmax(%arg0: tensor<i64>, %arg1: tensor<?x4x5xf32>) -> tensor<?x5xi32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<4> : tensor<1xi32>
    %3 = stablehlo.constant dense<5> : tensor<1xi32>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_iota %4, dim = 1 : (tensor<3xi32>) -> tensor<?x4x5xi32>
    %6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.constant dense<0> : tensor<i32>
    %8:2 = stablehlo.reduce(%arg1 init: %6), (%5 init: %7) across dimensions = [1] : (tensor<?x4x5xf32>, tensor<?x4x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<?x5xf32>, tensor<?x5xi32>)
     reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %9 = stablehlo.compare  GT, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = stablehlo.compare  NE, %arg2, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %11 = stablehlo.or %9, %10 : tensor<i1>
      %12 = stablehlo.compare  EQ, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %13 = stablehlo.compare  LT, %arg3, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %14 = stablehlo.and %12, %13 : tensor<i1>
      %15 = stablehlo.or %11, %14 : tensor<i1>
      %16 = stablehlo.select %11, %arg2, %arg4 : tensor<i1>, tensor<f32>
      %17 = stablehlo.select %15, %arg3, %arg5 : tensor<i1>, tensor<i32>
      stablehlo.return %16, %17 : tensor<f32>, tensor<i32>
    }
    return %8#1 : tensor<?x5xi32>
  }
}

