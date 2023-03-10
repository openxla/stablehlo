// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3xcomplex<f32>>
    %2 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
     reducer(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>)  {
      %5 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-1.21593916,2.39128661), (4.6942029,-2.61144495), (2.65929818,1.29460669)], [(3.53997397,2.71496224), (0.639440417,5.67562914), (2.82789803,2.696340e+00)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(2.32403469,5.10624886), (5.33364344,3.06418419), (5.48719597,3.99094677)]> : tensor<3xcomplex<f32>>
    return %0 : tensor<3xcomplex<f32>>
  }
}
