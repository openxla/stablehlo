// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<1x?x16xf32> {mhlo.sharding = ""}, %arg2: tensor<4x16x16xf32> {mhlo.sharding = ""}) -> tensor<1x?x16xf32> {
    %0 = stablehlo.constant dense<-1> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.constant dense<2> : tensor<i64>
    %3 = stablehlo.divide %1, %2 : tensor<i64>
    %4 = stablehlo.remainder %1, %2 : tensor<i64>
    %5 = stablehlo.sign %1 : tensor<i64>
    %6 = stablehlo.sign %2 : tensor<i64>
    %7 = stablehlo.compare  NE, %5, %6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.constant dense<0> : tensor<i64>
    %9 = stablehlo.compare  NE, %4, %8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %10 = stablehlo.and %7, %9 : tensor<i1>
    %11 = stablehlo.constant dense<1> : tensor<i64>
    %12 = stablehlo.subtract %3, %11 : tensor<i64>
    %13 = stablehlo.select %10, %12, %3 : tensor<i1>, tensor<i64>
    %14 = stablehlo.multiply %2, %13 : tensor<i64>
    %15 = stablehlo.subtract %1, %14 : tensor<i64>
    %16 = stablehlo.constant dense<2> : tensor<i64>
    %17 = stablehlo.add %15, %16 : tensor<i64>
    %18 = stablehlo.constant dense<2> : tensor<i64>
    %19 = stablehlo.divide %17, %18 : tensor<i64>
    %20 = stablehlo.remainder %17, %18 : tensor<i64>
    %21 = stablehlo.sign %17 : tensor<i64>
    %22 = stablehlo.sign %18 : tensor<i64>
    %23 = stablehlo.compare  NE, %21, %22,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %24 = stablehlo.constant dense<0> : tensor<i64>
    %25 = stablehlo.compare  NE, %20, %24,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %26 = stablehlo.and %23, %25 : tensor<i1>
    %27 = stablehlo.constant dense<1> : tensor<i64>
    %28 = stablehlo.subtract %19, %27 : tensor<i64>
    %29 = stablehlo.select %26, %28, %19 : tensor<i1>, tensor<i64>
    %30 = stablehlo.multiply %18, %29 : tensor<i64>
    %31 = stablehlo.subtract %17, %30 : tensor<i64>
    %32 = stablehlo.constant dense<-1> : tensor<i64>
    %33 = stablehlo.multiply %arg0, %32 : tensor<i64>
    %34 = stablehlo.constant dense<2> : tensor<i64>
    %35 = stablehlo.divide %33, %34 : tensor<i64>
    %36 = stablehlo.remainder %33, %34 : tensor<i64>
    %37 = stablehlo.sign %33 : tensor<i64>
    %38 = stablehlo.sign %34 : tensor<i64>
    %39 = stablehlo.compare  NE, %37, %38,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %40 = stablehlo.constant dense<0> : tensor<i64>
    %41 = stablehlo.compare  NE, %36, %40,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %42 = stablehlo.and %39, %41 : tensor<i1>
    %43 = stablehlo.constant dense<1> : tensor<i64>
    %44 = stablehlo.subtract %35, %43 : tensor<i64>
    %45 = stablehlo.select %42, %44, %35 : tensor<i1>, tensor<i64>
    %46 = stablehlo.multiply %34, %45 : tensor<i64>
    %47 = stablehlo.subtract %33, %46 : tensor<i64>
    %48 = stablehlo.constant dense<-1> : tensor<i64>
    %49 = stablehlo.multiply %arg0, %48 : tensor<i64>
    %50 = stablehlo.constant dense<2> : tensor<i64>
    %51 = stablehlo.divide %49, %50 : tensor<i64>
    %52 = stablehlo.remainder %49, %50 : tensor<i64>
    %53 = stablehlo.sign %49 : tensor<i64>
    %54 = stablehlo.sign %50 : tensor<i64>
    %55 = stablehlo.compare  NE, %53, %54,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %56 = stablehlo.constant dense<0> : tensor<i64>
    %57 = stablehlo.compare  NE, %52, %56,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %58 = stablehlo.and %55, %57 : tensor<i1>
    %59 = stablehlo.constant dense<1> : tensor<i64>
    %60 = stablehlo.subtract %51, %59 : tensor<i64>
    %61 = stablehlo.select %58, %60, %51 : tensor<i1>, tensor<i64>
    %62 = stablehlo.multiply %50, %61 : tensor<i64>
    %63 = stablehlo.subtract %49, %62 : tensor<i64>
    %64 = stablehlo.constant dense<2> : tensor<i64>
    %65 = stablehlo.add %63, %64 : tensor<i64>
    %66 = stablehlo.constant dense<2> : tensor<i64>
    %67 = stablehlo.divide %65, %66 : tensor<i64>
    %68 = stablehlo.remainder %65, %66 : tensor<i64>
    %69 = stablehlo.sign %65 : tensor<i64>
    %70 = stablehlo.sign %66 : tensor<i64>
    %71 = stablehlo.compare  NE, %69, %70,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %72 = stablehlo.constant dense<0> : tensor<i64>
    %73 = stablehlo.compare  NE, %68, %72,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %74 = stablehlo.and %71, %73 : tensor<i1>
    %75 = stablehlo.constant dense<1> : tensor<i64>
    %76 = stablehlo.subtract %67, %75 : tensor<i64>
    %77 = stablehlo.select %74, %76, %67 : tensor<i1>, tensor<i64>
    %78 = stablehlo.multiply %66, %77 : tensor<i64>
    %79 = stablehlo.subtract %65, %78 : tensor<i64>
    %80 = stablehlo.constant dense<-1> : tensor<i64>
    %81 = stablehlo.multiply %77, %80 : tensor<i64>
    %82 = stablehlo.constant dense<2> : tensor<i64>
    %83 = stablehlo.add %47, %82 : tensor<i64>
    %84 = stablehlo.add %83, %81 : tensor<i64>
    %85 = stablehlo.convert %29 : (tensor<i64>) -> tensor<i32>
    %86 = stablehlo.reshape %85 : (tensor<i32>) -> tensor<1xi32>
    %87 = stablehlo.convert %84 : (tensor<i64>) -> tensor<i32>
    %88 = stablehlo.reshape %87 : (tensor<i32>) -> tensor<1xi32>
    %89 = stablehlo.concatenate %86, %88, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %90 = stablehlo.reshape %89 : (tensor<2xi32>) -> tensor<1x2xi32>
    %91 = stablehlo.concatenate %90, dim = 0 : (tensor<1x2xi32>) -> tensor<1x2xi32>
    %92 = "stablehlo.dynamic_conv"(%arg1, %arg2, %91) {batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>, feature_group_count = 1 : i64, window_strides = dense<2> : tensor<1xi64>} : (tensor<1x?x16xf32>, tensor<4x16x16xf32>, tensor<1x2xi32>) -> tensor<1x?x16xf32>
    return %92 : tensor<1x?x16xf32>
  }
}

