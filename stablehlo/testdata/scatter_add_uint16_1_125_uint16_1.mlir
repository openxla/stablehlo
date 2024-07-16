// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %1 = call @expected() : () -> tensor<1x125xui16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    return %2 : tensor<1x125xui16>
  }
  func.func private @inputs() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}, tensor<1xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0700000002000700040003000200010005000200030001000000010001000600010002000200010000000000010001000900020002000200010001000100000002000400030000000500080000000000020000000200020002000200000000000300010000000200050001000A000100000001000000020000000200020000000300010006000000000003000100010002000000020003000400020000000000020000000700000003000000020003000100000005000100020000000200000002000400010000000000000000000300040002000200030003000100000000000100010004000100010004000100040004000000020005000300"> : tensor<1x125xui16>
    %c_0 = stablehlo.constant dense<0> : tensor<1xui16>
    return %c, %c_0 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0700000002000700040003000200010005000200030001000000010001000600010002000200010000000000010001000900020002000200010001000100000002000400030000000500080000000000020000000200020002000200000000000300010000000200050001000A000100000001000000020000000200020000000300010006000000000003000100010002000000020003000400020000000000020000000700000003000000020003000100000005000100020000000200000002000400010000000000000000000300040002000200030003000100000000000100010004000100010004000100040004000000020005000300"> : tensor<1x125xui16>
    return %c : tensor<1x125xui16>
  }
}
