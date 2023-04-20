// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(0x7F800000,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f32>>) -> tensor<complex<f32>>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %6 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %7 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %8 = stablehlo.compare  EQ, %6, %7,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = stablehlo.compare  LT, %6, %7,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %11 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %12 = stablehlo.compare  LT, %10, %11,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %13 = stablehlo.select %8, %12, %9 : tensor<i1>, tensor<i1>
      %14 = stablehlo.select %13, %arg0, %arg1 : tensor<i1>, tensor<complex<f32>>
      stablehlo.return %14 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-2.34767509,-2.6795392), (1.68114674,0.748172283), (-3.24897337,-2.63342643), (6.15362358,2.53630209), (2.59759474,-3.46653676), (-0.10211727,-1.11211717)], [(-3.07834888,-3.27384233), (3.45906806,6.75625229), (-1.48907936,2.40694094), (-0.214463472,4.22144556), (1.68167877,3.455935), (0.49865672,-0.572191775)], [(-2.89851499,-2.19572258), (-0.949407577,9.67435264), (-0.264912397,-7.360500e-02), (2.62598419,2.9400363), (-1.41885817,1.5909946), (-3.50827193,-1.39164877)], [(-2.26320958,-1.63305056), (-0.689091563,-1.8990618), (4.559800e+00,0.616205036), (4.21033478,4.56388903), (1.16108644,2.56657529), (-0.705174208,5.02120399)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-3.07834888,-3.27384233), (-3.24897337,-2.63342643), (-3.24897337,-2.63342643), (-0.214463472,4.22144556), (-0.10211727,-1.11211717)], [(-3.07834888,-3.27384233), (-1.48907936,2.40694094), (-1.48907936,2.40694094), (-1.41885817,1.5909946), (-3.50827193,-1.39164877)], [(-2.89851499,-2.19572258), (-0.949407577,9.67435264), (-0.264912397,-7.360500e-02), (-1.41885817,1.5909946), (-3.50827193,-1.39164877)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

