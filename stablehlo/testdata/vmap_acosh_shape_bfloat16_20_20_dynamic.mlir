// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %3 = stablehlo.constant dense<3.389530e+38> : tensor<bf16>
    %4 = stablehlo.constant dense<0x7FC0> : tensor<bf16>
    %5 = stablehlo.constant dense<-1.000000e+00> : tensor<bf16>
    %6 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %7, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = stablehlo.dynamic_broadcast_in_dim %5, %8, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %10 = stablehlo.compare  LT, %arg1, %9 : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %11 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.concatenate %12, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.dynamic_broadcast_in_dim %4, %13, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %15 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.concatenate %16, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %18 = stablehlo.dynamic_broadcast_in_dim %3, %17, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %19 = stablehlo.sqrt %18 : tensor<?x20x20xbf16>
    %20 = stablehlo.compare  GE, %arg1, %19 : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %21 = stablehlo.log %arg1 : tensor<?x20x20xbf16>
    %22 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %23 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
    %24 = stablehlo.concatenate %23, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %25 = stablehlo.dynamic_broadcast_in_dim %2, %24, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %26 = stablehlo.log %25 : tensor<?x20x20xbf16>
    %27 = stablehlo.add %21, %26 : tensor<?x20x20xbf16>
    %28 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %29 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.concatenate %29, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %31 = stablehlo.dynamic_broadcast_in_dim %1, %30, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %32 = stablehlo.add %31, %arg1 : tensor<?x20x20xbf16>
    %33 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %34 = stablehlo.reshape %33 : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.concatenate %34, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %36 = stablehlo.dynamic_broadcast_in_dim %5, %35, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %37 = stablehlo.add %36, %arg1 : tensor<?x20x20xbf16>
    %38 = stablehlo.multiply %32, %37 : tensor<?x20x20xbf16>
    %39 = stablehlo.sqrt %38 : tensor<?x20x20xbf16>
    %40 = stablehlo.add %arg1, %39 : tensor<?x20x20xbf16>
    %41 = stablehlo.log %40 : tensor<?x20x20xbf16>
    %42 = stablehlo.select %20, %27, %41 : tensor<?x20x20xi1>, tensor<?x20x20xbf16>
    %43 = stablehlo.select %10, %14, %42 : tensor<?x20x20xi1>, tensor<?x20x20xbf16>
    return %43 : tensor<?x20x20xbf16>
  }
}

