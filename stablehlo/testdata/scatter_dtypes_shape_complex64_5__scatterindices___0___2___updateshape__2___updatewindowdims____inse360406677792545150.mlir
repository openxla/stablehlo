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
      stablehlo.return %arg1 : tensor<complex<f32>>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xcomplex<f32>>, tensor<2x1xi32>, tensor<2xcomplex<f32>>) -> tensor<5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(-0.556826949,0.162270024), (-1.89213169,-2.97715926), (-0.22804518,-4.15038824), (1.28760445,1.29979658), (-3.89201522,-0.194024757)]> : tensor<5xcomplex<f32>>
    %1 = stablehlo.constant dense<[(1.60175288,-1.77837968), (0.524038076,-0.367077351)]> : tensor<2xcomplex<f32>>
    return %0, %1 : tensor<5xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(1.60175288,-1.77837968), (-1.89213169,-2.97715926), (0.524038076,-0.367077351), (1.28760445,1.29979658), (-3.89201522,-0.194024757)]> : tensor<5xcomplex<f32>>
    return %0 : tensor<5xcomplex<f32>>
  }
}

