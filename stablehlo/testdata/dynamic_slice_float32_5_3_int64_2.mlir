// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x1xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<5x3xf32>, tensor<2xi64>)
    %1 = call @expected() : () -> tensor<2x1xf32>
    %2 = stablehlo.slice %0#1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %0#1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %6 = stablehlo.compare  LT, %3, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %7 = stablehlo.add %3, %c_0 : tensor<i64>
    %8 = stablehlo.select %6, %7, %3 : tensor<i1>, tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %9 = stablehlo.compare  LT, %5, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<3> : tensor<i64>
    %10 = stablehlo.add %5, %c_2 : tensor<i64>
    %11 = stablehlo.select %9, %10, %5 : tensor<i1>, tensor<i64>
    %12 = stablehlo.dynamic_slice %0#0, %8, %11, sizes = [2, 1] : (tensor<5x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x1xf32>
    stablehlo.custom_call @check.expect_close(%12, %1) {has_side_effect = true} : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
    return %12 : tensor<2x1xf32>
  }
  func.func private @inputs() -> (tensor<5x3xf32> {mhlo.layout_mode = "default"}, tensor<2xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.56213808, -6.8185358, -1.571560e+00], [-1.97369742, -5.09079027, -0.138681203], [2.03843045, 3.21314955, -1.21502125], [-3.25714731, -0.58797425, 2.16309762], [-4.6540575, -0.637134611, -2.76275158]]> : tensor<5x3xf32>
    %c = stablehlo.constant dense<1> : tensor<2xi64>
    return %cst, %c : tensor<5x3xf32>, tensor<2xi64>
  }
  func.func private @expected() -> (tensor<2x1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.09079027], [3.21314955]]> : tensor<2x1xf32>
    return %cst : tensor<2x1xf32>
  }
}
