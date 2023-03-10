// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  GE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[-0.31220004, 2.98553729, 6.76927233, 4.41568851, 3.09804702, 1.36154675], [-1.17713666, 1.06314135, -3.76613045, 3.26746893, -0.829056739, 0.511934042], [-1.11380517, -3.4756155, 0.523748159, 1.13009441, 2.68522286, -0.43022117], [-0.121281229, 1.82162297, 3.41242838, -6.20822048, 2.25823402, -1.15179467]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[-4.34269524, -6.350050e-02, -2.16051722, -1.07253969, 2.91439533, -2.42550015], [1.22520196, -1.12437391, 4.24425268, 2.11868358, 4.11740589, 1.64276886], [-1.66720653, 0.734271049, -2.05781484, -4.68229437, -0.313324928, -0.563751638], [2.83365226, -2.31927252, 1.03329945, -1.80155492, -2.61366701, 1.28511763]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-1.17713666, -3.76613045, -3.76613045, -0.829056739, -0.829056739], [-1.17713666, -3.76613045, -3.76613045, -0.829056739, -0.829056739], [-0.121281229, 3.41242838, 3.41242838, 2.68522286, -1.15179467]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

