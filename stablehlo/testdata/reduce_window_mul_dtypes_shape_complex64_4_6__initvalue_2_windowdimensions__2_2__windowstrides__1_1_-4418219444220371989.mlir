// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-2.54936981,2.61797476), (2.14343834,1.05360913), (0.319945276,1.34392071), (-0.724535405,4.69509935), (1.34563041,0.993077397), (2.24038029,-6.33834314)], [(2.08194041,2.7286396), (-0.349532545,-1.36605585), (2.99909568,-4.15860558), (0.802382767,-0.285785377), (4.86077785,2.13434887), (-1.45279205,-1.73665404)], [(-3.34435272,-4.16117573), (4.64262295,-2.70356441), (-4.50974417,2.91285777), (-4.16819239,-3.15037799), (4.75487614,4.10513496), (-1.73615813,-3.1921804)], [(-11.341733,3.28769088), (-3.39032745,-1.30051959), (4.08700657,0.846687555), (-1.4292295,1.9282943), (-1.485440e+00,-1.32828617), (-2.29744649,0.267305762)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-27.1122684,80.0077744), (26.8382168,-39.4447212), (-11.5022821,56.1572304), (-54.4739532,46.8523521), (-207.995316,-1.725960e+02)], [(-238.707642,141.726089), (311.743896,-277.035431), (85.4718475,-229.607834), (-41.3608131,-293.923126), (-547.451782,37.7343521)], [(-2214.14014,-1071.23218), (754.220214,-442.170837), (-445.717407,342.338318), (-126.513069,-287.344482), (154.965561,-142.340057)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

