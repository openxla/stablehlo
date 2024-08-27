// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<5x3xf32>, tensor<3x1xf32>, tensor<2xi64>)
    %1 = call @expected() : () -> tensor<5x3xf32>
    %2 = stablehlo.slice %0#2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %0#2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
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
    %12 = stablehlo.dynamic_update_slice %0#0, %0#1, %8, %11 : (tensor<5x3xf32>, tensor<3x1xf32>, tensor<i64>, tensor<i64>) -> tensor<5x3xf32>
    stablehlo.custom_call @check.expect_close(%12, %1) {has_side_effect = true} : (tensor<5x3xf32>, tensor<5x3xf32>) -> ()
    return %12 : tensor<5x3xf32>
  }
  func.func private @inputs() -> (tensor<5x3xf32> {mhlo.layout_mode = "default"}, tensor<3x1xf32> {mhlo.layout_mode = "default"}, tensor<2xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.37886965, 2.52819586, -1.66359174], [-1.99620152, 1.19541264, 2.62602615], [-1.68388045, 2.40509582, 0.507822573], [-3.40378928, -1.23367071, -1.15847433], [1.35954201, 3.23837733, 0.490934104]]> : tensor<5x3xf32>
    %cst_0 = stablehlo.constant dense<[[-3.86802173], [-0.476641953], [0.915991306]]> : tensor<3x1xf32>
    %c = stablehlo.constant dense<1> : tensor<2xi64>
    return %cst, %cst_0, %c : tensor<5x3xf32>, tensor<3x1xf32>, tensor<2xi64>
  }
  func.func private @expected() -> (tensor<5x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.37886965, 2.52819586, -1.66359174], [-1.99620152, -3.86802173, 2.62602615], [-1.68388045, -0.476641953, 0.507822573], [-3.40378928, 0.915991306, -1.15847433], [1.35954201, 3.23837733, 0.490934104]]> : tensor<5x3xf32>
    return %cst : tensor<5x3xf32>
  }
}
