// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3xcomplex<f64>>
    %cst = stablehlo.constant dense<(0xFFF0000000000000,0.000000e+00)> : tensor<complex<f64>>
    %2 = stablehlo.reduce(%0 init: %cst) across dimensions = [0] : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
     reducer(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>)  {
      %3 = stablehlo.real %arg0 : (tensor<complex<f64>>) -> tensor<f64>
      %4 = stablehlo.real %arg1 : (tensor<complex<f64>>) -> tensor<f64>
      %5 = stablehlo.compare  EQ, %3, %4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %6 = stablehlo.compare  GT, %3, %4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %7 = stablehlo.imag %arg0 : (tensor<complex<f64>>) -> tensor<f64>
      %8 = stablehlo.imag %arg1 : (tensor<complex<f64>>) -> tensor<f64>
      %9 = stablehlo.compare  GT, %7, %8,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %10 = stablehlo.select %5, %9, %6 : tensor<i1>, tensor<i1>
      %11 = stablehlo.select %10, %arg0, %arg1 : tensor<i1>, tensor<complex<f64>>
      stablehlo.return %11 : tensor<complex<f64>>
    }
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> ()
    return %2 : tensor<3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(2.7669833167087874,-0.94169600484289995), (-2.3719987378004435,3.3066471553557681), (-5.8535964049536684,-7.1686460856967784)], [(0.1492160216132154,1.7900065296260239), (-5.2811226168470782,2.9759432261118173), (1.0735368743006275,-0.012922089414236853)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(2.7669833167087874,-0.94169600484289995), (-2.3719987378004435,3.3066471553557681), (1.0735368743006275,-0.012922089414236853)]> : tensor<3xcomplex<f64>>
    return %cst : tensor<3xcomplex<f64>>
  }
}
