// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7x4xui32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5x7x4xui32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @expected() : () -> (tensor<5x7x4xui32>, tensor<5x7x4xui32>)
    %1 = stablehlo.iota dim = 0 : tensor<5x7x4xui64>
    %2 = stablehlo.iota dim = 1 : tensor<5x7x4xui64>
    %3 = stablehlo.iota dim = 2 : tensor<5x7x4xui64>
    %c = stablehlo.constant dense<28> : tensor<ui64>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<5x7x4xui64>
    %5 = stablehlo.multiply %4, %1 : tensor<5x7x4xui64>
    %c_0 = stablehlo.constant dense<4> : tensor<ui64>
    %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<5x7x4xui64>
    %7 = stablehlo.multiply %6, %2 : tensor<5x7x4xui64>
    %c_1 = stablehlo.constant dense<1> : tensor<ui64>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<5x7x4xui64>
    %9 = stablehlo.multiply %8, %3 : tensor<5x7x4xui64>
    %10 = stablehlo.add %5, %7 : tensor<5x7x4xui64>
    %11 = stablehlo.add %10, %9 : tensor<5x7x4xui64>
    %c_2 = stablehlo.constant dense<32> : tensor<ui64>
    %12 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<5x7x4xui64>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<5x7x4xui64>
    %14 = stablehlo.convert %11 : (tensor<5x7x4xui64>) -> tensor<5x7x4xui32>
    %15 = stablehlo.convert %13 : (tensor<5x7x4xui64>) -> tensor<5x7x4xui32>
    stablehlo.custom_call @check.expect_eq(%15, %0#0) {has_side_effect = true} : (tensor<5x7x4xui32>, tensor<5x7x4xui32>) -> ()
    stablehlo.custom_call @check.expect_eq(%14, %0#1) {has_side_effect = true} : (tensor<5x7x4xui32>, tensor<5x7x4xui32>) -> ()
    return %15, %14 : tensor<5x7x4xui32>, tensor<5x7x4xui32>
  }
  func.func private @expected() -> (tensor<5x7x4xui32> {mhlo.layout_mode = "default"}, tensor<5x7x4xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<5x7x4xui32>
    %c_0 = stablehlo.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B000000"> : tensor<5x7x4xui32>
    return %c, %c_0 : tensor<5x7x4xui32>, tensor<5x7x4xui32>
  }
}
