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
      %5 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xcomplex<f32>>, tensor<2x1xi32>, tensor<2xcomplex<f32>>) -> tensor<5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(0.464579195,4.22281742), (6.29518843,-3.11532831), (2.44829416,-2.62646723), (1.089329,3.21574783), (-2.39107037,3.05258918)]> : tensor<5xcomplex<f32>>
    %1 = stablehlo.constant dense<[(-0.949137508,-4.3596015), (3.02920651,1.62838078)]> : tensor<2xcomplex<f32>>
    return %0, %1 : tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-0.484558314,-0.136784077), (6.29518843,-3.11532831), (5.47750092,-0.998086452), (1.089329,3.21574783), (-2.39107037,3.05258918)]> : tensor<5xcomplex<f32>>
    return %0 : tensor<5xcomplex<f32>>
  }
}

