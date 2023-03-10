// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xcomplex<f32>>
    %1 = call @expected() : () -> tensor<4x5xcomplex<f32>>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xcomplex<f32>>, tensor<4x5xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-4.71631432,-1.94354713), (4.43350458,-2.53362727), (2.45866489,-1.16547585), (2.59781742,-5.81839037), (-4.92355967,-3.57777333)], [(-2.84755611,-0.075127989), (-0.860104322,-2.77400589), (-3.330420e+00,-1.55811322), (2.28729272,-1.76461184), (3.49587321,-1.11559117)], [(-0.753502905,-0.328481287), (4.11367559,-0.278485417), (-0.303381652,-0.715796649), (-1.69689929,-0.486559868), (-2.96901274,1.25158262)], [(5.65013456,-2.44169617), (-3.09247184,1.44848275), (-6.513300e-01,4.15570211), (-1.51833832,-2.69817472), (3.77089787,-4.01752281)]]> : tensor<4x5xcomplex<f32>>
    return %0 : tensor<4x5xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(5.65013456,-2.44169617), (-3.09247184,1.44848275), (-6.513300e-01,4.15570211), (-1.51833832,-2.69817472), (3.77089787,-4.01752281)], [(-0.753502905,-0.328481287), (4.11367559,-0.278485417), (-0.303381652,-0.715796649), (-1.69689929,-0.486559868), (-2.96901274,1.25158262)], [(-2.84755611,-0.075127989), (-0.860104322,-2.77400589), (-3.330420e+00,-1.55811322), (2.28729272,-1.76461184), (3.49587321,-1.11559117)], [(-4.71631432,-1.94354713), (4.43350458,-2.53362727), (2.45866489,-1.16547585), (2.59781742,-5.81839037), (-4.92355967,-3.57777333)]]> : tensor<4x5xcomplex<f32>>
    return %0 : tensor<4x5xcomplex<f32>>
  }
}
