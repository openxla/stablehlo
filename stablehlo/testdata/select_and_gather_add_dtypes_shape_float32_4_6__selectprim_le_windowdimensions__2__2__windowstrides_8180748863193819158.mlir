// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[2.35791802, -2.90793347, -0.326863915, -2.02140021, 2.5673728, 3.2670176], [-1.03312874, -4.11909628, 0.970142841, 1.46392369, -3.33775425, 5.80515575], [4.84745455, -0.748723149, -2.57780719, 2.0550456, -3.7078855, -6.67396545], [3.41743708, -1.98370206, -5.07251024, -0.882639288, 3.41479564, -1.68553114]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[5.95014095, 0.689487159, -0.998706519, -4.60249281, 1.67418647, 3.15904117], [2.17690873, 5.46976376, -0.0405632704, -1.73958302, 4.11347771, -0.0490497053], [5.36788511, 3.02579331, 2.34428811, -2.65503931, 3.16358232, 0.968262612], [-1.09945095, 0.0977144166, -3.79372096, 0.79287827, -5.57099295, -1.59796596]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-2.90793347, -0.326863915, -2.02140021, -2.02140021, 5.80515575], [-1.03312874, 0.970142841, 2.0550456, 2.0550456, 5.80515575], [3.41743708, -5.07251024, -5.07251024, 3.41479564, 3.41479564]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

