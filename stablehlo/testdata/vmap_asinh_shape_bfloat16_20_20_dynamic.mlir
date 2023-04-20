// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %3 = stablehlo.constant dense<3.389530e+38> : tensor<bf16>
    %4 = stablehlo.sign %arg1 : tensor<?x20x20xbf16>
    %5 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %6 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %7, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = stablehlo.dynamic_broadcast_in_dim %3, %8, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %10 = stablehlo.sqrt %9 : tensor<?x20x20xbf16>
    %11 = stablehlo.compare  GE, %5, %10 : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %12 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %13 = stablehlo.log %12 : tensor<?x20x20xbf16>
    %14 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
    %16 = stablehlo.concatenate %15, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %17 = stablehlo.dynamic_broadcast_in_dim %2, %16, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %18 = stablehlo.log %17 : tensor<?x20x20xbf16>
    %19 = stablehlo.add %13, %18 : tensor<?x20x20xbf16>
    %20 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %21 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %22 = stablehlo.reshape %21 : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.concatenate %22, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %24 = stablehlo.dynamic_broadcast_in_dim %1, %23, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %25 = stablehlo.compare  LE, %20, %24 : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %26 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %27 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %28 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %29 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %30 = stablehlo.reshape %29 : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.concatenate %30, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %32 = stablehlo.dynamic_broadcast_in_dim %1, %31, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %33 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %34 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %35 = stablehlo.multiply %33, %34 : tensor<?x20x20xbf16>
    %36 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %37 = stablehlo.reshape %36 : (tensor<i32>) -> tensor<1xi32>
    %38 = stablehlo.concatenate %37, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %39 = stablehlo.dynamic_broadcast_in_dim %1, %38, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %40 = stablehlo.add %35, %39 : tensor<?x20x20xbf16>
    %41 = stablehlo.sqrt %40 : tensor<?x20x20xbf16>
    %42 = stablehlo.add %32, %41 : tensor<?x20x20xbf16>
    %43 = stablehlo.divide %28, %42 : tensor<?x20x20xbf16>
    %44 = stablehlo.multiply %27, %43 : tensor<?x20x20xbf16>
    %45 = stablehlo.add %26, %44 : tensor<?x20x20xbf16>
    %46 = stablehlo.log_plus_one %45 : tensor<?x20x20xbf16>
    %47 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %48 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %49 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %50 = stablehlo.multiply %48, %49 : tensor<?x20x20xbf16>
    %51 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %52 = stablehlo.reshape %51 : (tensor<i32>) -> tensor<1xi32>
    %53 = stablehlo.concatenate %52, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %54 = stablehlo.dynamic_broadcast_in_dim %1, %53, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %55 = stablehlo.add %50, %54 : tensor<?x20x20xbf16>
    %56 = stablehlo.sqrt %55 : tensor<?x20x20xbf16>
    %57 = stablehlo.add %47, %56 : tensor<?x20x20xbf16>
    %58 = stablehlo.log %57 : tensor<?x20x20xbf16>
    %59 = stablehlo.select %25, %46, %58 : tensor<?x20x20xi1>, tensor<?x20x20xbf16>
    %60 = stablehlo.select %11, %19, %59 : tensor<?x20x20xi1>, tensor<?x20x20xbf16>
    %61 = stablehlo.multiply %4, %60 : tensor<?x20x20xbf16>
    return %61 : tensor<?x20x20xbf16>
  }
}

