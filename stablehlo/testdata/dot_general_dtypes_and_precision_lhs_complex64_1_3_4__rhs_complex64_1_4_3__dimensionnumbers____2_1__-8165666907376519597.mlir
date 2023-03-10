// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<1xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) -> tensor<1xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xcomplex<f32>>, tensor<1xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[[(-7.18300104,-6.97948503), (1.11816895,-1.68824482), (0.351038933,-1.47352505), (-1.7140733,1.44326186)], [(3.89045668,0.470868319), (0.0152339945,0.472217381), (2.40766764,0.801624596), (3.15675092,2.18012929)], [(4.30926561,1.42080581), (-0.633524239,7.88391495), (-1.77237165,-6.42983341), (1.06225789,-0.833369135)]]]> : tensor<1x3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[[(-2.48327208,0.727223039), (-1.61065125,-2.11209106), (-3.60946941,-3.68373322)], [(-1.35685849,6.40633774), (6.80504656,-1.167629), (-3.41774035,-2.26014376)], [(-0.716802418,-1.70860243), (-0.404517055,2.62042308), (7.56154155,-1.31832159)], [(0.10888844,2.74405527), (2.74797273,5.36415911), (1.38076365,-2.68314981)]]]> : tensor<1x4x3xcomplex<f32>>
    return %0, %1 : tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<1xcomplex<f32>> {
    %0 = stablehlo.constant dense<(1.59982443,-56.1978073)> : tensor<1xcomplex<f32>>
    return %0 : tensor<1xcomplex<f32>>
  }
}
