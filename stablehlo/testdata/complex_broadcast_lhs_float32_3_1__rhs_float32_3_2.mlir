// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x1xf32>, tensor<3x2xf32>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.complex %2, %0#1 : tensor<3x2xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x1xf32>, tensor<3x2xf32>) {
    %0 = stablehlo.constant dense<[[2.94493532], [3.31010818], [-2.49504519]]> : tensor<3x1xf32>
    %1 = stablehlo.constant dense<[[-4.1446991, -2.1704638], [0.0342655852, -0.920183658], [-3.45004177, 1.29564202]]> : tensor<3x2xf32>
    return %0, %1 : tensor<3x1xf32>, tensor<3x2xf32>
  }
  func.func private @expected() -> tensor<3x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(2.94493532,-4.1446991), (2.94493532,-2.1704638)], [(3.31010818,0.0342655852), (3.31010818,-0.920183658)], [(-2.49504519,-3.45004177), (-2.49504519,1.29564202)]]> : tensor<3x2xcomplex<f32>>
    return %0 : tensor<3x2xcomplex<f32>>
  }
}
