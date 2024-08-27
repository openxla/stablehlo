// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, 0, 1], [4, 0, -2], [0, 1, 2], [6, -4, -2]]> : tensor<4x3xi32>
    %cst = stablehlo.constant dense<[[1.56930411, -1.16874635, 3.76845527, -1.01788259, -1.01884103, 3.28657198], [2.01026917, 3.72412753, -2.2192049, 5.39131403, -0.4193663, 1.57746553], [0.684506536, -1.44958484, 3.107656, -3.92279816, 3.49145579, 6.2946577]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xi32>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.45410156, 0.887907862, -4.42925453, -1.88703299, 5.52913761, -0.278486252], [4.90820313, -1.77581573, 8.85850906, 3.77406597, -11.0582752, 0.556972504], [3.37928224, 0.824957847, 3.9961071, -2.45428228, 6.56354522, 14.1667805], [0.00573515892, -19.009819, 25.2722397, -19.8269558, -11.4184933, 0.820255279]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
