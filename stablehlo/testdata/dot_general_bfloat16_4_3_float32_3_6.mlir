// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xbf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xbf16> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.421880e+00, 2.593750e+00, -5.000000e+00], [5.585940e-01, -3.562500e+00, 7.734380e-01], [1.632810e+00, 1.968750e+00, 4.406250e+00], [1.359380e+00, -3.453130e+00, 4.031250e+00]]> : tensor<4x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-2.57161117, 2.6343317, -0.814311087, 2.12331724, -5.14184856, -3.27703524], [0.977633655, -4.08951759, 1.00713694, 2.052948, -0.754616558, -2.36108756], [7.15495253, -4.87104321, 0.566327512, 2.41133928, -0.611945331, 4.01912785]]> : tensor<3x6xf32>
    return %cst, %cst_0 : tensor<4x3xbf16>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-24.4392929, 4.73367596, 2.56709456, -13.9975891, 18.6972027, -15.0061054], [0.614602804, 12.2729807, -3.60477519, -4.2625351, -0.657184541, 9.68938732], [29.2522678, -25.2129021, 3.1485641, 18.1336842, -12.5777102, 7.71010684], [21.9717274, -1.93373334, -2.30171609, 5.518010e+00, -6.85081958, 19.9005203]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
