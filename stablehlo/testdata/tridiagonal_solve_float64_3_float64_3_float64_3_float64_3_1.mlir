// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x1xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3x1xf64>)
    %1 = call @expected() : () -> tensor<3x1xf64>
    %2 = stablehlo.transpose %0#3, dims = [1, 0] : (tensor<3x1xf64>) -> tensor<1x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<1x3xf64>) -> tensor<3x1xf64>
    %4 = stablehlo.slice %0#2 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %0#1 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.divide %5, %7 : tensor<f64>
    %9 = stablehlo.slice %0#1 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %10 = stablehlo.slice %0#1 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %11 = stablehlo.slice %0#2 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %12 = stablehlo.slice %0#2 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %13 = stablehlo.slice %0#0 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %14 = stablehlo.slice %0#0 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %15 = stablehlo.reshape %9 : (tensor<0xf64>) -> tensor<0x32xf64>
    %16 = stablehlo.reshape %11 : (tensor<0xf64>) -> tensor<0x32xf64>
    %17 = stablehlo.reshape %13 : (tensor<0xf64>) -> tensor<0x32xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %18 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<0x32xf64>
    %19 = stablehlo.reshape %18 : (tensor<0x32xf64>) -> tensor<0xf64>
    %20 = stablehlo.slice %10 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.slice %12 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
    %24 = stablehlo.slice %14 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26:2 = call @None(%8, %21, %23, %25) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %27 = stablehlo.slice %10 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %28 = stablehlo.reshape %27 : (tensor<1xf64>) -> tensor<f64>
    %29 = stablehlo.slice %12 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.slice %14 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
    %33:2 = call @None(%26#0, %28, %30, %32) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %34 = stablehlo.slice %10 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %12 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.slice %14 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %39 = stablehlo.reshape %38 : (tensor<1xf64>) -> tensor<f64>
    %40:2 = call @None(%33#0, %35, %37, %39) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %41 = stablehlo.broadcast_in_dim %26#1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %42 = stablehlo.broadcast_in_dim %33#1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %43 = stablehlo.broadcast_in_dim %40#1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %44 = stablehlo.concatenate %41, %42, %43, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %45 = stablehlo.concatenate %19, %44, dim = 0 : (tensor<0xf64>, tensor<3xf64>) -> tensor<3xf64>
    %46 = stablehlo.slice %3 [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %47 = stablehlo.reshape %46 : (tensor<1x1xf64>) -> tensor<1xf64>
    %48 = stablehlo.slice %0#1 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %49 = stablehlo.divide %47, %48 : tensor<1xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %50 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %51 = stablehlo.slice %45 [0:2] : (tensor<3xf64>) -> tensor<2xf64>
    %52 = call @append(%50, %51) : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
    %53 = stablehlo.slice %3 [0:0, 0:1] : (tensor<3x1xf64>) -> tensor<0x1xf64>
    %54 = stablehlo.slice %3 [0:3, 0:1] : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %55 = stablehlo.slice %0#1 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %56 = stablehlo.slice %0#1 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %57 = stablehlo.slice %52 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %58 = stablehlo.slice %52 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %59 = stablehlo.slice %0#0 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %60 = stablehlo.slice %0#0 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %61 = stablehlo.reshape %53 : (tensor<0x1xf64>) -> tensor<0x32x1xf64>
    %62 = stablehlo.reshape %55 : (tensor<0xf64>) -> tensor<0x32xf64>
    %63 = stablehlo.reshape %57 : (tensor<0xf64>) -> tensor<0x32xf64>
    %64 = stablehlo.reshape %59 : (tensor<0xf64>) -> tensor<0x32xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %65 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<0x32x1xf64>
    %66 = stablehlo.reshape %65 : (tensor<0x32x1xf64>) -> tensor<0x1xf64>
    %67 = stablehlo.slice %54 [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %68 = stablehlo.reshape %67 : (tensor<1x1xf64>) -> tensor<1xf64>
    %69 = stablehlo.slice %56 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %70 = stablehlo.reshape %69 : (tensor<1xf64>) -> tensor<f64>
    %71 = stablehlo.slice %58 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %72 = stablehlo.reshape %71 : (tensor<1xf64>) -> tensor<f64>
    %73 = stablehlo.slice %60 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75:2 = call @None_0(%49, %68, %70, %72, %74) : (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %76 = stablehlo.slice %54 [1:2, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %77 = stablehlo.reshape %76 : (tensor<1x1xf64>) -> tensor<1xf64>
    %78 = stablehlo.slice %56 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.slice %58 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %81 = stablehlo.reshape %80 : (tensor<1xf64>) -> tensor<f64>
    %82 = stablehlo.slice %60 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84:2 = call @None_0(%75#0, %77, %79, %81, %83) : (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %85 = stablehlo.slice %54 [2:3, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %86 = stablehlo.reshape %85 : (tensor<1x1xf64>) -> tensor<1xf64>
    %87 = stablehlo.slice %56 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %88 = stablehlo.reshape %87 : (tensor<1xf64>) -> tensor<f64>
    %89 = stablehlo.slice %58 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %90 = stablehlo.reshape %89 : (tensor<1xf64>) -> tensor<f64>
    %91 = stablehlo.slice %60 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %92 = stablehlo.reshape %91 : (tensor<1xf64>) -> tensor<f64>
    %93:2 = call @None_0(%84#0, %86, %88, %90, %92) : (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %94 = stablehlo.broadcast_in_dim %75#1, dims = [1] : (tensor<1xf64>) -> tensor<1x1xf64>
    %95 = stablehlo.broadcast_in_dim %84#1, dims = [1] : (tensor<1xf64>) -> tensor<1x1xf64>
    %96 = stablehlo.broadcast_in_dim %93#1, dims = [1] : (tensor<1xf64>) -> tensor<1x1xf64>
    %97 = stablehlo.concatenate %94, %95, %96, dim = 0 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<3x1xf64>
    %98 = stablehlo.concatenate %66, %97, dim = 0 : (tensor<0x1xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %99 = stablehlo.compare  LT, %c, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<-1> : tensor<i64>
    %c_4 = stablehlo.constant dense<3> : tensor<i64>
    %100 = stablehlo.add %c_3, %c_4 : tensor<i64>
    %c_5 = stablehlo.constant dense<-1> : tensor<i64>
    %101 = stablehlo.select %99, %100, %c_5 : tensor<i1>, tensor<i64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_7 = stablehlo.constant dense<0> : tensor<i64>
    %102 = stablehlo.compare  LT, %c_6, %c_7,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_8 = stablehlo.constant dense<0> : tensor<i64>
    %c_9 = stablehlo.constant dense<1> : tensor<i64>
    %103 = stablehlo.add %c_8, %c_9 : tensor<i64>
    %c_10 = stablehlo.constant dense<0> : tensor<i64>
    %104 = stablehlo.select %102, %103, %c_10 : tensor<i1>, tensor<i64>
    %105 = stablehlo.dynamic_slice %98, %101, %104, sizes = [1, 1] : (tensor<3x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
    %106 = stablehlo.reshape %105 : (tensor<1x1xf64>) -> tensor<1xf64>
    %107 = stablehlo.reverse %98, dims = [0] : tensor<3x1xf64>
    %108 = stablehlo.reverse %45, dims = [0] : tensor<3xf64>
    %109 = stablehlo.slice %107 [0:0, 0:1] : (tensor<3x1xf64>) -> tensor<0x1xf64>
    %110 = stablehlo.slice %107 [0:3, 0:1] : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %111 = stablehlo.slice %108 [0:0] : (tensor<3xf64>) -> tensor<0xf64>
    %112 = stablehlo.slice %108 [0:3] : (tensor<3xf64>) -> tensor<3xf64>
    %113 = stablehlo.reshape %109 : (tensor<0x1xf64>) -> tensor<0x32x1xf64>
    %114 = stablehlo.reshape %111 : (tensor<0xf64>) -> tensor<0x32xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %115 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<0x32x1xf64>
    %116 = stablehlo.reshape %115 : (tensor<0x32x1xf64>) -> tensor<0x1xf64>
    %117 = stablehlo.slice %110 [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %118 = stablehlo.reshape %117 : (tensor<1x1xf64>) -> tensor<1xf64>
    %119 = stablehlo.slice %112 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %120 = stablehlo.reshape %119 : (tensor<1xf64>) -> tensor<f64>
    %121:2 = call @None_1(%106, %118, %120) : (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %122 = stablehlo.slice %110 [1:2, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %123 = stablehlo.reshape %122 : (tensor<1x1xf64>) -> tensor<1xf64>
    %124 = stablehlo.slice %112 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126:2 = call @None_1(%121#0, %123, %125) : (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %127 = stablehlo.slice %110 [2:3, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1x1xf64>) -> tensor<1xf64>
    %129 = stablehlo.slice %112 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %130 = stablehlo.reshape %129 : (tensor<1xf64>) -> tensor<f64>
    %131:2 = call @None_1(%126#0, %128, %130) : (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %132 = stablehlo.broadcast_in_dim %121#1, dims = [1] : (tensor<1xf64>) -> tensor<1x1xf64>
    %133 = stablehlo.broadcast_in_dim %126#1, dims = [1] : (tensor<1xf64>) -> tensor<1x1xf64>
    %134 = stablehlo.broadcast_in_dim %131#1, dims = [1] : (tensor<1xf64>) -> tensor<1x1xf64>
    %135 = stablehlo.concatenate %132, %133, %134, dim = 0 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<3x1xf64>
    %136 = stablehlo.concatenate %116, %135, dim = 0 : (tensor<0x1xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %137 = stablehlo.reverse %136, dims = [0] : tensor<3x1xf64>
    %138 = stablehlo.transpose %137, dims = [1, 0] : (tensor<3x1xf64>) -> tensor<1x3xf64>
    %139 = stablehlo.transpose %138, dims = [1, 0] : (tensor<1x3xf64>) -> tensor<3x1xf64>
    stablehlo.custom_call @check.expect_close(%139, %1) {has_side_effect = true} : (tensor<3x1xf64>, tensor<3x1xf64>) -> ()
    return %139 : tensor<3x1xf64>
  }
  func.func private @inputs() -> (tensor<3xf64> {mhlo.layout_mode = "default"}, tensor<3xf64> {mhlo.layout_mode = "default"}, tensor<3xf64> {mhlo.layout_mode = "default"}, tensor<3x1xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    %cst_1 = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3x1xf64>
    return %cst, %cst_0, %cst_1, %cst_2 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3x1xf64>
  }
  func.func private @expected() -> (tensor<3x1xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.5714285714285714], [0.4285714285714286], [-0.2857142857142857]]> : tensor<3x1xf64>
    return %cst : tensor<3x1xf64>
  }
  func.func private @None(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %0 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
    %1 = stablehlo.subtract %arg1, %0 : tensor<f64>
    %2 = stablehlo.divide %arg2, %1 : tensor<f64>
    %3 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
    %4 = stablehlo.subtract %arg1, %3 : tensor<f64>
    %5 = stablehlo.divide %arg2, %4 : tensor<f64>
    return %2, %5 : tensor<f64>, tensor<f64>
  }
  func.func private @append(%arg0: tensor<1xf64> {mhlo.layout_mode = "default"}, %arg1: tensor<2xf64> {mhlo.layout_mode = "default"}) -> (tensor<3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
  func.func private @None_0(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>) {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<1xf64>
    %2 = stablehlo.subtract %arg1, %1 : tensor<1xf64>
    %3 = stablehlo.multiply %arg4, %arg3 : tensor<f64>
    %4 = stablehlo.subtract %arg2, %3 : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.divide %2, %5 : tensor<1xf64>
    %7 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %8 = stablehlo.multiply %7, %arg0 : tensor<1xf64>
    %9 = stablehlo.subtract %arg1, %8 : tensor<1xf64>
    %10 = stablehlo.multiply %arg4, %arg3 : tensor<f64>
    %11 = stablehlo.subtract %arg2, %10 : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %13 = stablehlo.divide %9, %12 : tensor<1xf64>
    return %6, %13 : tensor<1xf64>, tensor<1xf64>
  }
  func.func private @None_1(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<1xf64>
    %2 = stablehlo.subtract %arg1, %1 : tensor<1xf64>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %4 = stablehlo.multiply %3, %arg0 : tensor<1xf64>
    %5 = stablehlo.subtract %arg1, %4 : tensor<1xf64>
    return %2, %5 : tensor<1xf64>, tensor<1xf64>
  }
}
