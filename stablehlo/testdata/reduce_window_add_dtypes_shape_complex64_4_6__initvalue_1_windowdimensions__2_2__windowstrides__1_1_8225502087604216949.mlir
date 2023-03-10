// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-2.49180388,0.592961729), (-5.27897358,-4.61703825), (-0.0688779652,-3.0083549), (0.243147939,1.31957912), (1.78620791,-2.08340144), (1.43844557,-2.20513606)], [(0.977789759,4.14434433), (-5.11616039,1.90628994), (-0.985685408,-0.372159481), (-2.34477711,-2.02637458), (-2.63399959,3.51459169), (0.21685636,0.720242917)], [(1.18660629,-2.3718648), (-0.0890328139,2.33344865), (2.32858706,-4.26655579), (-1.73226166,3.31702304), (5.90053463,-3.65677786), (-0.0811745449,-2.50534773)], [(-2.18609715,-0.689088284), (-1.37583792,-2.89551735), (12.5187092,2.35316014), (0.588996172,-2.76524115), (-0.414378077,4.23072863), (0.55397445,-7.4746685)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-10.9091482,2.02655792), (-10.4496975,-6.09126234), (-2.15619254,-4.087310e+00), (-1.94942069,0.724394798), (1.807510e+00,-0.0537028909)], [(-2.040797,6.01221848), (-2.86229134,-0.398976803), (-1.73413706,-3.34806657), (0.18949604,1.1484623), (4.40221691,-1.92729092)], [(-1.46436155,-3.62302184), (14.3824253,-2.47546458), (14.70403,-1.36161375), (5.34289074,1.12573266), (6.95895671,-9.40606498)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

