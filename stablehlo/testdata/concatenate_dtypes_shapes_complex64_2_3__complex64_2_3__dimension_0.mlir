// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x3xcomplex<f32>>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xcomplex<f32>>, tensor<4x3xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-0.779149591,2.24533319), (-6.139320e-01,-2.5766747), (3.52656031,0.627551138)], [(-2.56904387,5.45755339), (3.33605719,-2.4610548), (-0.596523345,-5.46911526)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(1.08402872,-6.815310e+00), (-1.01068747,-1.54408216), (1.82507873,-3.64951444)], [(3.51390457,-3.61418915), (3.15557718,-1.19769979), (1.49958754,2.75437403)]]> : tensor<2x3xcomplex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.779149591,2.24533319), (-6.139320e-01,-2.5766747), (3.52656031,0.627551138)], [(-2.56904387,5.45755339), (3.33605719,-2.4610548), (-0.596523345,-5.46911526)], [(1.08402872,-6.815310e+00), (-1.01068747,-1.54408216), (1.82507873,-3.64951444)], [(3.51390457,-3.61418915), (3.15557718,-1.19769979), (1.49958754,2.75437403)]]> : tensor<4x3xcomplex<f32>>
    return %0 : tensor<4x3xcomplex<f32>>
  }
}
