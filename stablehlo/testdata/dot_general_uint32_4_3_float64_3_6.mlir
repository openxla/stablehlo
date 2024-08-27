// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui32>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui32>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xui32> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 1], [2, 3, 0], [1, 1, 2], [2, 5, 2]]> : tensor<4x3xui32>
    %cst = stablehlo.constant dense<[[-1.9953186906909837, -0.50157252284448006, -0.79635879080979732, 0.21403786664838789, 0.73126735457670344, -5.9600624798166253], [0.024793196488969351, 2.2386166767780376, 7.5791390610633184, -1.917465254482789, 3.5785937664589147, -0.10897992326768993], [-0.31251176211982479, -2.8238665191207772, -3.4659455093336562, -3.5310183223266565, 0.19344930249099634, -2.427682737610712]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xui32>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.3031491435017921, -3.8270115648097374, -5.0586630909532513, -3.1029425890298805, 1.6559840116444033, -14.347807697243962], [-3.9162577919150592, 5.7127049846451525, 21.144699601570359, -5.3243200301515916, 12.198316008530151, -12.247064729436321], [-2.5955490184416639, -3.910688884307997, -0.14911074841379079, -8.7654640324877136, 4.6967597260176115, -10.92440787830574], [-4.4916949231767695, 4.5422052999596723, 29.371086705029686, -16.221287183770482, 19.742402146429974, -17.320390051193122]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
