// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>)
    %2 = call @expected() : () -> tensor<5xcomplex<f32>>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %6 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %7 = stablehlo.compare  EQ, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  GT, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %10 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %11 = stablehlo.compare  GT, %9, %10,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %12 = stablehlo.select %7, %11, %8 : tensor<i1>, tensor<i1>
      %13 = stablehlo.select %12, %arg0, %arg1 : tensor<i1>, tensor<complex<f32>>
      stablehlo.return %13 : tensor<complex<f32>>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xcomplex<f32>>, tensor<2x1xi32>, tensor<2xcomplex<f32>>) -> tensor<5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(-1.59970856,-0.744870126), (0.146364465,-1.67213547), (0.68349725,3.99860787), (-3.8709712,4.81220818), (-1.58440804,-4.04845667)]> : tensor<5xcomplex<f32>>
    %1 = stablehlo.constant dense<[(6.660250e+00,-4.39738035), (2.35097814,0.796183705)]> : tensor<2xcomplex<f32>>
    return %0, %1 : tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(6.660250e+00,-4.39738035), (0.146364465,-1.67213547), (2.35097814,0.796183705), (-3.8709712,4.81220818), (-1.58440804,-4.04845667)]> : tensor<5xcomplex<f32>>
    return %0 : tensor<5xcomplex<f32>>
  }
}

