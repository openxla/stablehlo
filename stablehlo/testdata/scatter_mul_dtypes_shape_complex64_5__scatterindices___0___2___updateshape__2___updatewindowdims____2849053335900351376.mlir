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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xcomplex<f32>>, tensor<2x1xi32>, tensor<2xcomplex<f32>>) -> tensor<5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(-0.171597868,-0.730450451), (4.56433725,1.85803211), (-2.16899896,-1.0819515), (5.06871414,2.06354403), (1.96776199,-3.49827337)]> : tensor<5xcomplex<f32>>
    %1 = stablehlo.constant dense<[(-3.16152859,-2.91531062), (5.81820536,-2.0535903)]> : tensor<2xcomplex<f32>>
    return %0, %1 : tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-1.58697832,2.80960107), (4.56433725,1.85803211), (-14.8415661,-1.84078097), (5.06871414,2.06354403), (1.96776199,-3.49827337)]> : tensor<5xcomplex<f32>>
    return %0 : tensor<5xcomplex<f32>>
  }
}

