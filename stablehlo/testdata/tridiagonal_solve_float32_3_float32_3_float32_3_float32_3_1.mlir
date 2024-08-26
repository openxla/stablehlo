// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x1xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>)
    %1 = call @expected() : () -> tensor<3x1xf32>
    %2 = stablehlo.transpose %0#3, dims = [1, 0] : (tensor<3x1xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<1x3xf32>) -> tensor<3x1xf32>
    %4 = stablehlo.slice %0#2 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %4 : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.slice %0#1 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
    %8 = stablehlo.divide %5, %7 : tensor<f32>
    %9 = stablehlo.slice %0#1 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %10 = stablehlo.slice %0#1 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %11 = stablehlo.slice %0#2 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %12 = stablehlo.slice %0#2 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %13 = stablehlo.slice %0#0 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %14 = stablehlo.slice %0#0 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %15 = stablehlo.reshape %9 : (tensor<0xf32>) -> tensor<0x32xf32>
    %16 = stablehlo.reshape %11 : (tensor<0xf32>) -> tensor<0x32xf32>
    %17 = stablehlo.reshape %13 : (tensor<0xf32>) -> tensor<0x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<0x32xf32>
    %19 = stablehlo.reshape %18 : (tensor<0x32xf32>) -> tensor<0xf32>
    %20 = stablehlo.slice %10 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %21 = stablehlo.reshape %20 : (tensor<1xf32>) -> tensor<f32>
    %22 = stablehlo.slice %12 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1xf32>) -> tensor<f32>
    %24 = stablehlo.slice %14 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %25 = stablehlo.reshape %24 : (tensor<1xf32>) -> tensor<f32>
    %26:2 = call @None(%8, %21, %23, %25) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %27 = stablehlo.slice %10 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %28 = stablehlo.reshape %27 : (tensor<1xf32>) -> tensor<f32>
    %29 = stablehlo.slice %12 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
    %31 = stablehlo.slice %14 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
    %33:2 = call @None(%26#0, %28, %30, %32) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %34 = stablehlo.slice %10 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %35 = stablehlo.reshape %34 : (tensor<1xf32>) -> tensor<f32>
    %36 = stablehlo.slice %12 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %37 = stablehlo.reshape %36 : (tensor<1xf32>) -> tensor<f32>
    %38 = stablehlo.slice %14 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %39 = stablehlo.reshape %38 : (tensor<1xf32>) -> tensor<f32>
    %40:2 = call @None(%33#0, %35, %37, %39) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %41 = stablehlo.broadcast_in_dim %26#1, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %42 = stablehlo.broadcast_in_dim %33#1, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %43 = stablehlo.broadcast_in_dim %40#1, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %44 = stablehlo.concatenate %41, %42, %43, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %45 = stablehlo.concatenate %19, %44, dim = 0 : (tensor<0xf32>, tensor<3xf32>) -> tensor<3xf32>
    %46 = stablehlo.slice %3 [0:1, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %47 = stablehlo.reshape %46 : (tensor<1x1xf32>) -> tensor<1xf32>
    %48 = stablehlo.slice %0#1 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %49 = stablehlo.divide %47, %48 : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %51 = stablehlo.slice %45 [0:2] : (tensor<3xf32>) -> tensor<2xf32>
    %52 = call @append(%50, %51) : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
    %53 = stablehlo.slice %3 [0:0, 0:1] : (tensor<3x1xf32>) -> tensor<0x1xf32>
    %54 = stablehlo.slice %3 [0:3, 0:1] : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %55 = stablehlo.slice %0#1 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %56 = stablehlo.slice %0#1 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %57 = stablehlo.slice %52 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %58 = stablehlo.slice %52 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %59 = stablehlo.slice %0#0 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %60 = stablehlo.slice %0#0 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %61 = stablehlo.reshape %53 : (tensor<0x1xf32>) -> tensor<0x32x1xf32>
    %62 = stablehlo.reshape %55 : (tensor<0xf32>) -> tensor<0x32xf32>
    %63 = stablehlo.reshape %57 : (tensor<0xf32>) -> tensor<0x32xf32>
    %64 = stablehlo.reshape %59 : (tensor<0xf32>) -> tensor<0x32xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<0x32x1xf32>
    %66 = stablehlo.reshape %65 : (tensor<0x32x1xf32>) -> tensor<0x1xf32>
    %67 = stablehlo.slice %54 [0:1, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x1xf32>) -> tensor<1xf32>
    %69 = stablehlo.slice %56 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %70 = stablehlo.reshape %69 : (tensor<1xf32>) -> tensor<f32>
    %71 = stablehlo.slice %58 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %72 = stablehlo.reshape %71 : (tensor<1xf32>) -> tensor<f32>
    %73 = stablehlo.slice %60 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %74 = stablehlo.reshape %73 : (tensor<1xf32>) -> tensor<f32>
    %75:2 = call @None_0(%49, %68, %70, %72, %74) : (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %76 = stablehlo.slice %54 [1:2, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x1xf32>) -> tensor<1xf32>
    %78 = stablehlo.slice %56 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %79 = stablehlo.reshape %78 : (tensor<1xf32>) -> tensor<f32>
    %80 = stablehlo.slice %58 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %81 = stablehlo.reshape %80 : (tensor<1xf32>) -> tensor<f32>
    %82 = stablehlo.slice %60 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %83 = stablehlo.reshape %82 : (tensor<1xf32>) -> tensor<f32>
    %84:2 = call @None_0(%75#0, %77, %79, %81, %83) : (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %85 = stablehlo.slice %54 [2:3, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %86 = stablehlo.reshape %85 : (tensor<1x1xf32>) -> tensor<1xf32>
    %87 = stablehlo.slice %56 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %88 = stablehlo.reshape %87 : (tensor<1xf32>) -> tensor<f32>
    %89 = stablehlo.slice %58 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %90 = stablehlo.reshape %89 : (tensor<1xf32>) -> tensor<f32>
    %91 = stablehlo.slice %60 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %92 = stablehlo.reshape %91 : (tensor<1xf32>) -> tensor<f32>
    %93:2 = call @None_0(%84#0, %86, %88, %90, %92) : (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %94 = stablehlo.broadcast_in_dim %75#1, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %95 = stablehlo.broadcast_in_dim %84#1, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %96 = stablehlo.broadcast_in_dim %93#1, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %97 = stablehlo.concatenate %94, %95, %96, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    %98 = stablehlo.concatenate %66, %97, dim = 0 : (tensor<0x1xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
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
    %105 = stablehlo.dynamic_slice %98, %101, %104, sizes = [1, 1] : (tensor<3x1xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
    %106 = stablehlo.reshape %105 : (tensor<1x1xf32>) -> tensor<1xf32>
    %107 = stablehlo.reverse %98, dims = [0] : tensor<3x1xf32>
    %108 = stablehlo.reverse %45, dims = [0] : tensor<3xf32>
    %109 = stablehlo.slice %107 [0:0, 0:1] : (tensor<3x1xf32>) -> tensor<0x1xf32>
    %110 = stablehlo.slice %107 [0:3, 0:1] : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %111 = stablehlo.slice %108 [0:0] : (tensor<3xf32>) -> tensor<0xf32>
    %112 = stablehlo.slice %108 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
    %113 = stablehlo.reshape %109 : (tensor<0x1xf32>) -> tensor<0x32x1xf32>
    %114 = stablehlo.reshape %111 : (tensor<0xf32>) -> tensor<0x32xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %115 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<0x32x1xf32>
    %116 = stablehlo.reshape %115 : (tensor<0x32x1xf32>) -> tensor<0x1xf32>
    %117 = stablehlo.slice %110 [0:1, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %118 = stablehlo.reshape %117 : (tensor<1x1xf32>) -> tensor<1xf32>
    %119 = stablehlo.slice %112 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %120 = stablehlo.reshape %119 : (tensor<1xf32>) -> tensor<f32>
    %121:2 = call @None_1(%106, %118, %120) : (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %122 = stablehlo.slice %110 [1:2, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %123 = stablehlo.reshape %122 : (tensor<1x1xf32>) -> tensor<1xf32>
    %124 = stablehlo.slice %112 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %125 = stablehlo.reshape %124 : (tensor<1xf32>) -> tensor<f32>
    %126:2 = call @None_1(%121#0, %123, %125) : (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %127 = stablehlo.slice %110 [2:3, 0:1] : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %128 = stablehlo.reshape %127 : (tensor<1x1xf32>) -> tensor<1xf32>
    %129 = stablehlo.slice %112 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %130 = stablehlo.reshape %129 : (tensor<1xf32>) -> tensor<f32>
    %131:2 = call @None_1(%126#0, %128, %130) : (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %132 = stablehlo.broadcast_in_dim %121#1, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %133 = stablehlo.broadcast_in_dim %126#1, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %134 = stablehlo.broadcast_in_dim %131#1, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %135 = stablehlo.concatenate %132, %133, %134, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    %136 = stablehlo.concatenate %116, %135, dim = 0 : (tensor<0x1xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
    %137 = stablehlo.reverse %136, dims = [0] : tensor<3x1xf32>
    %138 = stablehlo.transpose %137, dims = [1, 0] : (tensor<3x1xf32>) -> tensor<1x3xf32>
    %139 = stablehlo.transpose %138, dims = [1, 0] : (tensor<1x3xf32>) -> tensor<3x1xf32>
    stablehlo.custom_call @check.expect_close(%139, %1) {has_side_effect = true} : (tensor<3x1xf32>, tensor<3x1xf32>) -> ()
    return %139 : tensor<3x1xf32>
  }
  func.func private @inputs() -> (tensor<3xf32> {mhlo.layout_mode = "default"}, tensor<3xf32> {mhlo.layout_mode = "default"}, tensor<3xf32> {mhlo.layout_mode = "default"}, tensor<3x1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 0.000000e+00]> : tensor<3xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3x1xf32>
    return %cst, %cst_0, %cst_1, %cst_2 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>
  }
  func.func private @expected() -> (tensor<3x1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.571428597], [0.428571403], [-0.285714298]]> : tensor<3x1xf32>
    return %cst : tensor<3x1xf32>
  }
  func.func private @None(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.multiply %arg3, %arg0 : tensor<f32>
    %1 = stablehlo.subtract %arg1, %0 : tensor<f32>
    %2 = stablehlo.divide %arg2, %1 : tensor<f32>
    %3 = stablehlo.multiply %arg3, %arg0 : tensor<f32>
    %4 = stablehlo.subtract %arg1, %3 : tensor<f32>
    %5 = stablehlo.divide %arg2, %4 : tensor<f32>
    return %2, %5 : tensor<f32>, tensor<f32>
  }
  func.func private @append(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<2xf32> {mhlo.layout_mode = "default"}) -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func private @None_0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %1 = stablehlo.multiply %0, %arg0 : tensor<1xf32>
    %2 = stablehlo.subtract %arg1, %1 : tensor<1xf32>
    %3 = stablehlo.multiply %arg4, %arg3 : tensor<f32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.divide %2, %5 : tensor<1xf32>
    %7 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %8 = stablehlo.multiply %7, %arg0 : tensor<1xf32>
    %9 = stablehlo.subtract %arg1, %8 : tensor<1xf32>
    %10 = stablehlo.multiply %arg4, %arg3 : tensor<f32>
    %11 = stablehlo.subtract %arg2, %10 : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1xf32>
    return %6, %13 : tensor<1xf32>, tensor<1xf32>
  }
  func.func private @None_1(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %1 = stablehlo.multiply %0, %arg0 : tensor<1xf32>
    %2 = stablehlo.subtract %arg1, %1 : tensor<1xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.multiply %3, %arg0 : tensor<1xf32>
    %5 = stablehlo.subtract %arg1, %4 : tensor<1xf32>
    return %2, %5 : tensor<1xf32>, tensor<1xf32>
  }
}
