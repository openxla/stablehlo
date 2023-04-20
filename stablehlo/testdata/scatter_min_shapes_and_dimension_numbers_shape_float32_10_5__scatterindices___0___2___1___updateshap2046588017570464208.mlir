// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0], [2], [1]]> : tensor<3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<10x5xf32>, tensor<3x3xf32>)
    %2 = call @expected() : () -> tensor<10x5xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<10x5xf32>, tensor<3x1xi32>, tensor<3x3xf32>) -> tensor<10x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<10x5xf32>, tensor<10x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<10x5xf32>, tensor<3x3xf32>) {
    %0 = stablehlo.constant dense<[[-3.3934319, 1.99264336, -1.94391739, -1.37355232, 4.19274235], [-3.5442121, -3.25669765, 3.58125377, -2.15336418, 4.23629427], [0.738987207, -2.15688276, 1.69032097, 0.429829657, -4.0355835], [-2.17226553, 0.0400138162, -0.518476963, 0.189603537, -0.0461742245], [-2.20804811, -4.82117319, 0.207664207, 2.44345093, -1.86209011], [2.17553163, -3.44394636, 2.43954444, -3.68349433, 1.44664085], [3.25226688, 3.6171124, 2.31334853, 0.639582276, -2.76516676], [0.85384804, -2.64964318, 2.30812907, -0.021793291, 7.467460e-01], [1.49462938, -2.50380921, 2.46098804, -1.63531125, 0.872237384], [3.12620711, -4.20555449, -1.12006426, -2.45239973, 6.85851431]]> : tensor<10x5xf32>
    %1 = stablehlo.constant dense<[[-4.38032675, 0.162817597, 2.92452455], [1.32878149, -3.33167171, -1.90544951], [-2.47449136, 1.81515825, 1.1862272]]> : tensor<3x3xf32>
    return %0, %1 : tensor<10x5xf32>, tensor<3x3xf32>
  }
  func.func private @expected() -> tensor<10x5xf32> {
    %0 = stablehlo.constant dense<[[-4.38032675, 0.162817597, -1.94391739, -1.37355232, 4.19274235], [-3.5442121, -3.25669765, 1.1862272, -2.15336418, 4.23629427], [0.738987207, -3.33167171, -1.90544951, 0.429829657, -4.0355835], [-2.17226553, 0.0400138162, -0.518476963, 0.189603537, -0.0461742245], [-2.20804811, -4.82117319, 0.207664207, 2.44345093, -1.86209011], [2.17553163, -3.44394636, 2.43954444, -3.68349433, 1.44664085], [3.25226688, 3.6171124, 2.31334853, 0.639582276, -2.76516676], [0.85384804, -2.64964318, 2.30812907, -0.021793291, 7.467460e-01], [1.49462938, -2.50380921, 2.46098804, -1.63531125, 0.872237384], [3.12620711, -4.20555449, -1.12006426, -2.45239973, 6.85851431]]> : tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
  }
}

