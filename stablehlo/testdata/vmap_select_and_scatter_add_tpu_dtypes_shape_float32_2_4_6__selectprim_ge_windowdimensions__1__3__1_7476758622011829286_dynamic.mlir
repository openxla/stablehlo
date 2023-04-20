// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x1x6xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2x4x6xf32> {mhlo.sharding = ""}) -> tensor<?x2x4x6xf32> {
    %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.pad %arg2, %0, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<?x2x4x6xf32>, tensor<f32>) -> tensor<?x2x4x6xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%1, %arg1, %2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %21 = stablehlo.compare  GE, %arg3, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %21 : tensor<i1>
    }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %21 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }) {window_dimensions = dense<[1, 1, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 1]> : tensor<4xi64>} : (tensor<?x2x4x6xf32>, tensor<?x2x1x6xf32>, tensor<f32>) -> tensor<?x2x4x6xf32>
    %4 = stablehlo.constant dense<0> : tensor<1xi32>
    %5 = stablehlo.constant dense<0> : tensor<1xi32>
    %6 = stablehlo.constant dense<0> : tensor<1xi32>
    %7 = stablehlo.constant dense<0> : tensor<1xi32>
    %8 = stablehlo.concatenate %4, %5, %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %9 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.constant dense<2> : tensor<1xi32>
    %12 = stablehlo.constant dense<4> : tensor<1xi32>
    %13 = stablehlo.constant dense<6> : tensor<1xi32>
    %14 = stablehlo.concatenate %10, %11, %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %15 = stablehlo.constant dense<1> : tensor<1xi32>
    %16 = stablehlo.constant dense<1> : tensor<1xi32>
    %17 = stablehlo.constant dense<1> : tensor<1xi32>
    %18 = stablehlo.constant dense<1> : tensor<1xi32>
    %19 = stablehlo.concatenate %15, %16, %17, %18, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %20 = stablehlo.real_dynamic_slice %3, %8, %14, %19 : (tensor<?x2x4x6xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x2x4x6xf32>
    return %20 : tensor<?x2x4x6xf32>
  }
}

