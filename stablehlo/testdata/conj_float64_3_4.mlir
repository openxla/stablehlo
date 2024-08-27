// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xf64>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f64>>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x4xf64>
    %3 = stablehlo.complex %0, %2 : tensor<3x4xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    return %3 : tensor<3x4xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<3x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.8963581247357544, -3.9022169915354232, 2.6657789104597041, 1.7520647517365588], [-5.4495198723021723, -0.30357298841749769, 4.478750797787062, 1.0383215526922465], [-0.040330506564937703, -1.2216473191328965, -2.1613784962585871, -1.0604084893165211]]> : tensor<3x4xf64>
    return %cst : tensor<3x4xf64>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.8963581247357544,0.000000e+00), (-3.9022169915354232,0.000000e+00), (2.6657789104597041,0.000000e+00), (1.7520647517365588,0.000000e+00)], [(-5.4495198723021723,0.000000e+00), (-0.30357298841749769,0.000000e+00), (4.478750797787062,0.000000e+00), (1.0383215526922465,0.000000e+00)], [(-0.040330506564937703,0.000000e+00), (-1.2216473191328965,0.000000e+00), (-2.1613784962585871,0.000000e+00), (-1.0604084893165211,0.000000e+00)]]> : tensor<3x4xcomplex<f64>>
    return %cst : tensor<3x4xcomplex<f64>>
  }
}
