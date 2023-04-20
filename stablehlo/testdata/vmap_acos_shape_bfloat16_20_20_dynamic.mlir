// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<-1.000000e+00> : tensor<bf16>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<20> : tensor<1xi32>
    %4 = stablehlo.constant dense<20> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %0, %5, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %7 = stablehlo.compare  NE, %arg1, %6,  FLOAT : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %8 = stablehlo.multiply %arg1, %arg1 : tensor<?x20x20xbf16>
    %9 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %10 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.constant dense<20> : tensor<1xi32>
    %13 = stablehlo.constant dense<20> : tensor<1xi32>
    %14 = stablehlo.concatenate %11, %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %15 = stablehlo.dynamic_broadcast_in_dim %9, %14, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %16 = stablehlo.subtract %15, %8 : tensor<?x20x20xbf16>
    %17 = stablehlo.sqrt %16 : tensor<?x20x20xbf16>
    %18 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %19 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %20 = stablehlo.reshape %19 : (tensor<i32>) -> tensor<1xi32>
    %21 = stablehlo.constant dense<20> : tensor<1xi32>
    %22 = stablehlo.constant dense<20> : tensor<1xi32>
    %23 = stablehlo.concatenate %20, %21, %22, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %24 = stablehlo.dynamic_broadcast_in_dim %18, %23, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %25 = stablehlo.add %24, %arg1 : tensor<?x20x20xbf16>
    %26 = stablehlo.atan2 %17, %25 : tensor<?x20x20xbf16>
    %27 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %28 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %29 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.constant dense<20> : tensor<1xi32>
    %31 = stablehlo.constant dense<20> : tensor<1xi32>
    %32 = stablehlo.concatenate %29, %30, %31, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %33 = stablehlo.dynamic_broadcast_in_dim %27, %32, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %34 = stablehlo.multiply %33, %26 : tensor<?x20x20xbf16>
    %35 = stablehlo.constant dense<3.140630e+00> : tensor<bf16>
    %36 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %37 = stablehlo.reshape %36 : (tensor<i32>) -> tensor<1xi32>
    %38 = stablehlo.constant dense<20> : tensor<1xi32>
    %39 = stablehlo.constant dense<20> : tensor<1xi32>
    %40 = stablehlo.concatenate %37, %38, %39, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %41 = stablehlo.dynamic_broadcast_in_dim %35, %40, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %42 = stablehlo.select %7, %34, %41 : tensor<?x20x20xi1>, tensor<?x20x20xbf16>
    return %42 : tensor<?x20x20xbf16>
  }
}

