// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<15xf64>
    %1 = call @expected() : () -> tensor<i32>
    %2 = call @argmin(%0) : (tensor<15xf64>) -> tensor<i32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<i32>, tensor<i32>) -> ()
    return %2 : tensor<i32>
  }
  func.func private @inputs() -> (tensor<15xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-6.6145288704962102, 2.0119804700961526, -3.6775164625496188, 1.2795419045177521, -4.2897148259564668, 3.6975934526314296, 2.4851511280239285, -4.6241596218609216, -0.69549046122738556, -0.96656582605412599, 2.8127515641438134, -1.52730336273762, -4.6440421602568795, -5.0595577787465729, 1.7164292320467918]> : tensor<15xf64>
    return %cst : tensor<15xf64>
  }
  func.func private @expected() -> (tensor<i32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    return %c : tensor<i32>
  }
  func.func private @argmin(%arg0: tensor<15xf64>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<15xi32>
    %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [0] : (tensor<15xf64>, tensor<15xi32>, tensor<f64>, tensor<i32>) -> (tensor<f64>, tensor<i32>)
     reducer(%arg1: tensor<f64>, %arg3: tensor<f64>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  LT, %arg1, %arg3,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f64>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f64>, tensor<i32>
    }
    return %1#1 : tensor<i32>
  }
}
