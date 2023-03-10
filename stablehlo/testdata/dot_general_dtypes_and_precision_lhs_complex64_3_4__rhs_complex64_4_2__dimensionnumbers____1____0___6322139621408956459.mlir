// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-3.633900e+00,-2.73841524), (-1.01464272,-3.38973975), (-1.44648659,2.85447693), (2.32500863,-4.27428484)], [(1.24592507,2.89404058), (-1.74034238,5.50661182), (5.48683214,2.64402699), (-1.69929516,-2.32236099)], [(0.0781846195,-1.59205067), (3.94340324,-3.26957226), (-1.91819739,-0.0289337598), (2.03918433,-6.37816905)]]> : tensor<3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(0.62618345,-5.94444561), (2.10416317,4.13166666)], [(1.205120e+00,-1.65275991), (-2.08400941,-3.84396696)], [(-3.90580726,-1.84156084), (-3.48822308,3.87119603)], [(-2.58177376,-4.11647701), (2.16107559,0.694972336)]]> : tensor<4x2xcomplex<f32>>
    return %0, %1 : tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-38.0702934,10.4578362), (-5.25716496,-32.9895554)], [(3.25330544,-3.52211189), (-15.9746265,12.2691517)], [(-34.1479187,-0.201185226), (1.59871054,-31.0627136)]]> : tensor<3x2xcomplex<f32>>
    return %0 : tensor<3x2xcomplex<f32>>
  }
}

