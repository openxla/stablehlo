// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %1 = call @expected() : () -> tensor<1x125xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<1x125xbf16>, tensor<1xi64>, tensor<1xbf16>) -> tensor<1x125xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> ()
    return %2 : tensor<1x125xbf16>
  }
  func.func private @inputs() -> (tensor<1x125xbf16> {mhlo.layout_mode = "default"}, tensor<1xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x07C066403E3F93C0AF40493E87C086C027C063BFB5BFD4BF96400B3FB83F8C40963F29C0B2BFB14090BF783F9ABFCC3EA2BF0740633F57404AC0013F6CBF73401E4006402D3F12C038C06A40913FA640E6BF80409EC003C02EBF9AC0803FDDBFAD4093C04C4090BFDABE28C07EC075C0BE3F41C0E1BF52C084C09A3F62C0453F44BFA8C02C409B3F86402D40F140A7C0864066405DC04140E7BF37BEC53F51BE57C0AAC04DC0273F20405C3F893E7E3F29BF66C048BF9A3F8E404A4028C06FBF9FBF613F9A406F3F1CC001C0573F9BBF8BC086C0FCC0B4BFD13EB44031406FBFA53FB73F343E94C04D3EC1BF803FB13FC9C048C04B3F28401A41"> : tensor<1x125xbf16>
    %cst_0 = stablehlo.constant dense<9.062500e-01> : tensor<1xbf16>
    return %cst, %cst_0 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> (tensor<1x125xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xF5BF66403E3F93C0AF40493E87C086C027C063BFB5BFD4BF96400B3FB83F8C40963F29C0B2BFB14090BF783F9ABFCC3EA2BF0740633F57404AC0013F6CBF73401E4006402D3F12C038C06A40913FA640E6BF80409EC003C02EBF9AC0803FDDBFAD4093C04C4090BFDABE28C07EC075C0BE3F41C0E1BF52C084C09A3F62C0453F44BFA8C02C409B3F86402D40F140A7C0864066405DC04140E7BF37BEC53F51BE57C0AAC04DC0273F20405C3F893E7E3F29BF66C048BF9A3F8E404A4028C06FBF9FBF613F9A406F3F1CC001C0573F9BBF8BC086C0FCC0B4BFD13EB44031406FBFA53FB73F343E94C04D3EC1BF803FB13FC9C048C04B3F28401A41"> : tensor<1x125xbf16>
    return %cst : tensor<1x125xbf16>
  }
}
