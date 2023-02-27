module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x1xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_ctrsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<4x1xcomplex<f32>>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-3.16801548,5.9716711), (-1.05663657,1.80671155), (-7.88379669,-0.683486223), (-2.30481315,-0.813904821)], [(-3.4569695,-2.14629507), (4.23830843,2.80844975), (-2.1607976,1.38738799), (-2.46207952,-5.68866682)], [(-5.453650e+00,-0.316764981), (-0.282942742,-1.87895584), (-1.743460e+00,-0.64714396), (0.798629879,1.74304938)], [(2.64352846,4.32777262), (1.91385329,3.01114678), (2.80594158,2.22159648), (-5.93554688,-3.74831271)]]> : tensor<4x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(1.35696566,0.474310935)], [(0.0996178686,0.577913463)], [(1.04837811,1.38058341)], [(1.35553324,6.48330069)]]> : tensor<4x1xcomplex<f32>>
    return %0, %1 : tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-1.18372202,1.88309872)], [(-0.978154063,-0.988538146)], [(-0.770918608,-1.47241509)], [(-6.563900e-01,-0.677771747)]]> : tensor<4x1xcomplex<f32>>
    return %0 : tensor<4x1xcomplex<f32>>
  }
}
