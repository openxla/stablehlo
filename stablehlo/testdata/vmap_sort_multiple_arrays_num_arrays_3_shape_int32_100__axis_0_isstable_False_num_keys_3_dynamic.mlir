// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x100xi32> {mhlo.sharding = ""}, %arg2: tensor<?x100xi32> {mhlo.sharding = ""}, %arg3: tensor<?x100xf32> {mhlo.sharding = ""}) -> (tensor<?x100xi32>, tensor<?x100xi32>, tensor<?x100xf32>) {
    %0:3 = "stablehlo.sort"(%arg1, %arg2, %arg3) ({
    ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<i32>, %arg8: tensor<f32>, %arg9: tensor<f32>):
      %1 = stablehlo.bitcast_convert %arg8 : (tensor<f32>) -> tensor<i32>
      %2 = stablehlo.bitcast_convert %arg8 : (tensor<f32>) -> tensor<ui32>
      %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4 = stablehlo.compare  EQ, %arg8, %3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.constant dense<0> : tensor<i32>
      %6 = stablehlo.select %4, %5, %1 : tensor<i1>, tensor<i32>
      %7 = stablehlo.compare  NE, %arg8, %arg8,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.constant dense<2143289344> : tensor<i32>
      %9 = stablehlo.select %7, %8, %6 : tensor<i1>, tensor<i32>
      %10 = stablehlo.constant dense<2147483647> : tensor<ui32>
      %11 = stablehlo.subtract %10, %2 : tensor<ui32>
      %12 = stablehlo.bitcast_convert %11 : (tensor<ui32>) -> tensor<i32>
      %13 = stablehlo.constant dense<0> : tensor<i32>
      %14 = stablehlo.compare  LT, %9, %13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %15 = stablehlo.select %14, %12, %9 : tensor<i1>, tensor<i32>
      %16 = stablehlo.bitcast_convert %arg9 : (tensor<f32>) -> tensor<i32>
      %17 = stablehlo.bitcast_convert %arg9 : (tensor<f32>) -> tensor<ui32>
      %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %19 = stablehlo.compare  EQ, %arg9, %18,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %20 = stablehlo.constant dense<0> : tensor<i32>
      %21 = stablehlo.select %19, %20, %16 : tensor<i1>, tensor<i32>
      %22 = stablehlo.compare  NE, %arg9, %arg9,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %23 = stablehlo.constant dense<2143289344> : tensor<i32>
      %24 = stablehlo.select %22, %23, %21 : tensor<i1>, tensor<i32>
      %25 = stablehlo.constant dense<2147483647> : tensor<ui32>
      %26 = stablehlo.subtract %25, %17 : tensor<ui32>
      %27 = stablehlo.bitcast_convert %26 : (tensor<ui32>) -> tensor<i32>
      %28 = stablehlo.constant dense<0> : tensor<i32>
      %29 = stablehlo.compare  LT, %24, %28,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %30 = stablehlo.select %29, %27, %24 : tensor<i1>, tensor<i32>
      %31 = stablehlo.compare  LT, %15, %30,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %32 = stablehlo.compare  LT, %arg6, %arg7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %33 = stablehlo.compare  EQ, %arg6, %arg7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %34 = stablehlo.and %33, %31 : tensor<i1>
      %35 = stablehlo.or %32, %34 : tensor<i1>
      %36 = stablehlo.compare  LT, %arg4, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %37 = stablehlo.compare  EQ, %arg4, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %38 = stablehlo.and %37, %35 : tensor<i1>
      %39 = stablehlo.or %36, %38 : tensor<i1>
      stablehlo.return %39 : tensor<i1>
    }) {dimension = 1 : i64} : (tensor<?x100xi32>, tensor<?x100xi32>, tensor<?x100xf32>) -> (tensor<?x100xi32>, tensor<?x100xi32>, tensor<?x100xf32>)
    return %0#0, %0#1, %0#2 : tensor<?x100xi32>, tensor<?x100xi32>, tensor<?x100xf32>
  }
}

