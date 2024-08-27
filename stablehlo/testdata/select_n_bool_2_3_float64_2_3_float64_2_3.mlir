// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>)
    %1 = call @expected() : () -> tensor<2x3xf64>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %2 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xi1> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<2x3xi1>
    %cst = stablehlo.constant dense<[[1.6545636564404331, -5.4821347976364159, -2.0086517908148904], [0.69937796660157336, -3.0890795754926761, 0.16769550090612578]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<[[-1.7115590272350207, -2.9520499628953853, -0.27612026716671112], [2.7204079321798487, 3.1137611728964445, -2.6623861275903633]]> : tensor<2x3xf64>
    return %c, %cst, %cst_0 : tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.7115590272350207, -2.9520499628953853, -0.27612026716671112], [2.7204079321798487, 3.1137611728964445, -2.6623861275903633]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
