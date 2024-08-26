// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui8>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui8>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xui8> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1, 0], [3, 0, 1], [1, 4, 4], [2, 2, 0]]> : tensor<4x3xui8>
    %cst = stablehlo.constant dense<[[-2.08029175, 0.35927242, -9.61650753, 2.60842323, 5.47147703, -3.35032439], [-0.427300513, 4.1255827, -3.22211885, -3.61885571, -0.167576313, -4.04876328], [3.73630929, 7.04818058, -5.21696234, -1.65469015, 3.07341361, -5.56366348]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xui8>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.427300513, 4.1255827, -3.22211885, -3.61885571, -0.167576313, -4.04876328], [-2.50456595, 8.12599754, -34.0664825, 6.17057943, 19.4878445, -15.6146364], [11.1557436, 45.0543251, -43.3728333, -18.4857597, 17.0948257, -41.8000336], [-5.0151844, 8.969710e+00, -25.6772537, -2.02086496, 10.6078014, -14.7981758]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
