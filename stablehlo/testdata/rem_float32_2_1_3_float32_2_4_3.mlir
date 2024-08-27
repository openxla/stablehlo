// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.remainder %2, %0#1 : tensor<2x4x3xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> ()
    return %3 : tensor<2x4x3xf32>
  }
  func.func private @inputs() -> (tensor<2x1x3xf32> {mhlo.layout_mode = "default"}, tensor<2x4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.616980791, 0.706572294, -4.73949623]], [[1.67496932, -3.13438964, 1.29330683]]]> : tensor<2x1x3xf32>
    %cst_0 = stablehlo.constant dense<[[[-0.0259232726, 5.1987915, 7.33156443], [3.09692621, 0.534493804, 2.14104652], [-4.89970493, -0.817309558, -1.55429268], [4.97919846, -4.20424652, -0.378917187]], [[2.69760561, -1.81906879, 0.697872698], [-2.84189224, 2.68079877, 0.158963606], [-1.83596051, -1.92423916, -0.504996121], [-1.42828655, -3.60045314, 4.78329611]]]> : tensor<2x4x3xf32>
    return %cst, %cst_0 : tensor<2x1x3xf32>, tensor<2x4x3xf32>
  }
  func.func private @expected() -> (tensor<2x4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.0207455214, 0.706572294, -4.73949623], [-0.616980791, 0.17207849, -0.457403183], [-0.616980791, 0.706572294, -0.0766181945], [-0.616980791, 0.706572294, -0.192489982]], [[1.67496932, -1.31532085, 0.595434129], [1.67496932, -0.45359087, 0.0215979815], [1.67496932, -1.21015048, 0.283314586], [0.246682763, -3.13438964, 1.29330683]]]> : tensor<2x4x3xf32>
    return %cst : tensor<2x4x3xf32>
  }
}
