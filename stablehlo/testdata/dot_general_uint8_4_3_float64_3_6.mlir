// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui8>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui8>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xui8> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 0, 5], [0, 3, 5], [5, 2, 0], [2, 1, 1]]> : tensor<4x3xui8>
    %cst = stablehlo.constant dense<[[-4.3886811303306077, -7.1534168587462892, -6.6619289284585239, 2.1838990687896942, -2.7471214742479537, -1.2088552070550003], [-1.326909111780642, -3.2791645718163505, -2.3148246916008381, -0.050698021974274678, -1.011258651905828, 1.4946544363617982], [1.3584636024988166, -0.43609970000415194, -1.6081534488018772, 0.25021662296343494, -1.283272445359728, -2.5618008962455652]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xui8>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.4036368821634753, -9.3339153587670491, -14.70269617246791, 3.434982183606869, -9.1634837010465943, -14.017859688282826], [2.8115906771521573, -12.01799221546981, -14.9852413188119, 1.0989890488943508, -9.4501381825161239, -8.3250411721424307], [-24.597223875214322, -42.325413437364148, -37.939294025494291, 10.818099299999922, -15.758124675051425, -3.0549671625514052], [-8.7458077699430419, -18.022097989313082, -17.246835997319764, 4.5673167385685485, -7.7887740457614631, -3.4848568739937678]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
