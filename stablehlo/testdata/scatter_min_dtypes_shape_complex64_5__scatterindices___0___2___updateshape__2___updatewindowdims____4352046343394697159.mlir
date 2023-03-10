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
      %8 = stablehlo.compare  LT, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %10 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %11 = stablehlo.compare  LT, %9, %10,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %12 = stablehlo.select %7, %11, %8 : tensor<i1>, tensor<i1>
      %13 = stablehlo.select %12, %arg0, %arg1 : tensor<i1>, tensor<complex<f32>>
      stablehlo.return %13 : tensor<complex<f32>>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xcomplex<f32>>, tensor<2x1xi32>, tensor<2xcomplex<f32>>) -> tensor<5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(-5.17213869,-5.02383947), (4.06953478,-2.44848466), (-2.26946092,-0.534447074), (-2.79616499,0.426540554), (-9.039880e-01,1.41251624)]> : tensor<5xcomplex<f32>>
    %1 = stablehlo.constant dense<[(-3.56527686,4.35403872), (-1.92323709,-1.68484616)]> : tensor<2xcomplex<f32>>
    return %0, %1 : tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-5.17213869,-5.02383947), (4.06953478,-2.44848466), (-2.26946092,-0.534447074), (-2.79616499,0.426540554), (-9.039880e-01,1.41251624)]> : tensor<5xcomplex<f32>>
    return %0 : tensor<5xcomplex<f32>>
  }
}

